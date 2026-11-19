# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import selective_scan_cuda_oflex

from third_party.build.segman.csm_triton import CrossScanTriton, CrossMergeTriton
from .detail_routing import BoundaryAwareDynamicDetailRouting, PostDetailRefine


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class VSSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=1,
        expansion_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_groups=4,
        **kwargs,    
    ):
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio !=1:
            self.xproj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)
        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)
        
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True) # (K * D)
        
        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)
    
    
    def forward(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        xs = self._cross_scan(x)
        if self.expansion_ratio!=1:
            xs = self.xproj(xs.permute(0,1,3,2).contiguous()).permute(0,1,3,2).contiguous()
        xs = xs.reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)
        
        # y = ys.reshape(B, K, -1, L)
        # yf = F.conv1d(y[:, 0, ...], weight=self.factor1, groups=D)
        # yb = F.conv1d(y[:, 1, ...].flip([-1]), weight=self.factor2, groups=D)
        # y = yf + yb
        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        
        if self.expansion_ratio!=1:
            y = self.yproj(y.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        if to_dtype:
            y = y.to(x.dtype)
        
        return y
     

class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class VSSMBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.token_mixer = VSSM(d_model=embed_dim, k_groups=4, expansion_ratio=1, channel_split=False)
        self.norm2 = LayerNorm2d(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = FFN(embed_dim, embed_dim * 4)

    def forward(self, x):
        x = x + self.cpe1(x)
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.cpe2(x)
        return x + self.mlp(self.norm2(x))


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x


@HEADS.register_module()
class SegMANDecoder(BaseDecodeHead):
    def __init__(self, feat_proj_dim=432, boundary_width=3, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = self.channels
        self.feat_proj_dim = feat_proj_dim
        self.boundary_width = boundary_width
        self._boundary_logit = None

        self.conv_downsample_2 = ConvModule(
                        self.embed_dim, self.embed_dim*2, kernel_size=3, stride=2, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.conv_downsample_4 = ConvModule(
                        self.embed_dim, self.embed_dim*4, kernel_size=5, stride=4, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.linear_c4 = MLP(self.in_channels[-1], self.feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], self.feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], self.feat_proj_dim)
        self.detail_gate = BoundaryAwareDynamicDetailRouting(
            self.in_channels[0], self.embed_dim, self.feat_proj_dim,
            self.conv_cfg, self.norm_cfg, self.act_cfg)

        self.linear_fuse = ConvModule(
                        in_channels=self.feat_proj_dim*3,
                        out_channels=self.embed_dim,
                        kernel_size=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        

        self.reduce_channels = nn.ModuleList([ConvModule(in_channels=self.embed_dim*4*(2**i),
                                out_channels=self.embed_dim,kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)) for i in range(3)])
        vssm_dim = self.embed_dim*3

        self.vssm = VSSMBlock(vssm_dim)

        self.short_path = ConvModule(
                            in_channels=self.embed_dim,
                            out_channels=self.embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(self.embed_dim, self.embed_dim, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        self.proj_out = ConvModule(in_channels=vssm_dim,
                                out_channels=self.feat_proj_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))

        feat_concat_dim = self.embed_dim * 5 + self.feat_proj_dim * 3
        self.cat = ConvModule(
            in_channels=feat_concat_dim, out_channels=self.embed_dim, kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))

        # The original initialization includes this module before ABH installs
        # its signed residual head. It is discarded after loading initial tensors.
        self.post_detail_refine = PostDetailRefine(
            self.in_channels[0], self.embed_dim, self.conv_cfg, self.norm_cfg, self.act_cfg)

    def forward_mlp_decoder(self, inputs):
        _, c2, c3, c4 = inputs
        size = c2.shape[2:]
        c4, c3, c2 = self.linear_c4(c4), self.linear_c3(c3), self.linear_c2(c2)
        c4 = resize(c4, size=size, mode="bilinear", align_corners=False).contiguous()
        c3 = resize(c3, size=size, mode="bilinear", align_corners=False).contiguous()
        context = self.linear_fuse(torch.cat([c4, c3, c2], dim=1))
        return context, c2, c3, c4

    def forward_winssm(self, x, injected, c2, c3, c4):
        out = [self.short_path(injected), 
                  resize(self.image_pool(x),
                        size=x.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners).contiguous()]

        H, W = x.shape[-2:]

        # pad H and W to multiples of 4 for pixel_unshuffle(factor=4)
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # obtain multi scale features
        x_2 = self.conv_downsample_2(x) # 1/2 resolution
        x_4 = self.conv_downsample_4(x) # 1/4 resolution

        # unshuffle all features to size 1/4 resolution (16x16 for 512 input res)
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        # reduce channels
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        multi_x = torch.cat([x_unshuffle,x_2_unshuffle,x_4], dim=1)

        _out = self.vssm(multi_x)

        _out = resize(_out,
                size=(H, W),
                mode='bilinear',
                align_corners=self.align_corners)
        
        projected = self.proj_out(_out)
        out += [_out, c2 + projected, c3 + projected, c4 + projected]
        return self.cat(torch.cat(out, dim=1))

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        context, c2, c3, c4 = self.forward_mlp_decoder(inputs)
        injected, self._boundary_logit = self.detail_gate(context, inputs[0], c2, c3, self.align_corners)
        return self.cls_seg(self.forward_winssm(context, injected, c2, c3, c4))

    def _make_boundary_target(self, seg_label, size):
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        label = seg_label.long()
        valid = label != self.ignore_index
        edge = label.new_zeros(label.shape, dtype=torch.float32)

        diff_h = (label[:, 1:, :] != label[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
        edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], diff_h.float())
        edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], diff_h.float())

        diff_w = (label[:, :, 1:] != label[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], diff_w.float())
        edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], diff_w.float())

        edge = edge.unsqueeze(1)
        if self.boundary_width > 1:
            edge = F.max_pool2d(
                edge,
                kernel_size=self.boundary_width,
                stride=1,
                padding=self.boundary_width // 2)
        edge = F.interpolate(edge, size=size, mode='nearest').clamp_(0, 1)
        valid = F.interpolate(valid.unsqueeze(1).float(), size=size, mode='nearest')
        return edge, valid
