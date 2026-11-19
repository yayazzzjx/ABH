"""Boundary-aware routing and the modules required for ABH initialization."""
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize


class BoundaryAwareDynamicDetailRouting(nn.Module):
    def __init__(self, in_channels, channels, guide_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.detail_proj = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.context_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.c2_detail_proj, self.c2_refine, self.c2_gate, self.c2_scale = (
            self._make_guidance_branch(guide_channels, channels, conv_cfg, norm_cfg, act_cfg))
        self.c3_detail_proj, self.c3_refine, self.c3_gate, self.c3_scale = (
            self._make_guidance_branch(guide_channels, channels, conv_cfg, norm_cfg, act_cfg))
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid())

        self.out_proj = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels, max(channels // 4, 32), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 32), 1, kernel_size=1))
        nn.init.zeros_(self.boundary_head[-1].weight)
        nn.init.zeros_(self.boundary_head[-1].bias)
        self.route_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, 96, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 1, kernel_size=1))
        nn.init.zeros_(self.route_head[-1].weight)
        nn.init.zeros_(self.route_head[-1].bias)
        nn.init.zeros_(self.out_proj.conv.weight)
        if self.out_proj.conv.bias is not None:
            nn.init.zeros_(self.out_proj.conv.bias)

    @staticmethod
    def _make_guidance_branch(in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        proj = ConvModule(
            in_channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False))
        gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.Sigmoid())
        scale = nn.Parameter(torch.zeros(1))
        return proj, refine, gate, scale

    @staticmethod
    def _resize_like(feat, target, align_corners):
        if feat.shape[2:] == target.shape[2:]:
            return feat
        return resize(
            feat,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=align_corners).contiguous()

    def _apply_guidance(self, detail, context, guide, guide_proj, guide_refine,
                        guide_gate, guide_scale, align_corners):
        guide_detail = guide_proj(guide)
        guide_detail = self._resize_like(guide_detail, detail, align_corners)
        delta = guide_refine(guide_detail - detail)
        gate = guide_gate(torch.cat([detail, guide_detail, context], dim=1))
        return detail + torch.tanh(guide_scale) * delta * gate

    def forward(self, x, c1, c2, c3, align_corners):
        context = self.context_proj(x)
        detail = self._resize_like(self.detail_proj(c1), x, align_corners)
        detail = self._apply_guidance(
            detail, context, c2, self.c2_detail_proj, self.c2_refine,
            self.c2_gate, self.c2_scale, align_corners)
        detail = self._apply_guidance(
            detail, context, c3, self.c3_detail_proj, self.c3_refine,
            self.c3_gate, self.c3_scale, align_corners)
        gate = self.gate(torch.cat([detail, context], dim=1))
        boundary_logit = self.boundary_head(detail)
        route_inputs = torch.cat([detail, context, torch.sigmoid(boundary_logit)], dim=1)
        route_scale = 2.0 * torch.sigmoid(self.route_head(route_inputs))
        return x + self.out_proj(detail * gate) * route_scale, boundary_logit


class PostDetailRefine(nn.Module):
    """Preserve the zero-update checkpoint layout and initialization order."""

    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.detail_proj = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.context_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid())
        self.refine = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.refine.conv.weight)
        if self.refine.conv.bias is not None:
            nn.init.zeros_(self.refine.conv.bias)
