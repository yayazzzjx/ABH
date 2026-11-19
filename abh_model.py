"""Final ABH body and its original SegMAN-based decoder."""
import copy
import math
from types import MethodType

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from runtime_helpers import MODEL_CONFIG, OPTIMIZER_CONFIG, sha
from mmseg.models.builder import HEADS
from mmseg.models.losses import accuracy
from third_party.segman import segman_encoder  # registers SegMANEncoder_l
from third_party.segman.segman_decoder import SegMANDecoder
import mmseg.models.decode_heads.decode_head as base_head

# Single-stage guide-only Local Attender for the frozen E0 support interface.
#
# UPLiFT Local Attender inspiration: https://github.com/mwalmer-umd/UPLiFT
# This implementation is original project adaptation. It uses a fixed bilinear
# phase prior and an explicit deterministic gather adjoint, not UPLiFT's decoder.
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def neighbour_indices(height: int, width: int) -> torch.Tensor:
    """Forward-only gather indices with replication at the value-bank border."""
    y = torch.arange(height * 2, dtype=torch.long) // 2
    x = torch.arange(width * 2, dtype=torch.long) // 2
    return torch.stack([
        (y.add(dy).clamp(0, height - 1)[:, None] * width
         + x.add(dx).clamp(0, width - 1)[None, :]).reshape(-1)
        for dy, dx in OFFSETS
    ]).reshape(-1)


def bilinear_phase_prior(height: int, width: int) -> torch.Tensor:
    """Align-corners=False 2x weights indexed by floor(u/2) and 3x3 offsets."""
    phase_y = torch.arange(height * 2) % 2
    phase_x = torch.arange(width * 2) % 2
    axis = torch.tensor(((0.25, 0.75, 0.0), (0.0, 0.75, 0.25)))
    return torch.stack([
        axis[phase_y, dy + 1][:, None] * axis[phase_x, dx + 1][None, :]
        for dy, dx in OFFSETS
    ])[None]


def _gather(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = value.shape
    return value.flatten(2).index_select(2, indices).reshape(
        batch, channels, 9, height * 2, width * 2)


def _replication_shift_adjoint(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Transpose of value[clamp(anchor+offset)], with no scatter atomics."""
    if value.shape[-2] > 1:
        if dy == -1:
            value = torch.cat((value[..., :1, :] + value[..., 1:2, :],
                               value[..., 2:, :], torch.zeros_like(value[..., -1:, :])), -2)
        elif dy == 1:
            value = torch.cat((torch.zeros_like(value[..., :1, :]), value[..., :-2, :],
                               value[..., -2:-1, :] + value[..., -1:, :]), -2)
    if value.shape[-1] > 1:
        if dx == -1:
            value = torch.cat((value[..., :1] + value[..., 1:2],
                               value[..., 2:], torch.zeros_like(value[..., -1:])), -1)
        elif dx == 1:
            value = torch.cat((torch.zeros_like(value[..., :1]), value[..., :-2],
                               value[..., -2:-1] + value[..., -1:]), -1)
    return value


class _DeterministicReassembly(torch.autograd.Function):
    """Fixed-order CUDA/CPU adjoint; first-order scientific training only.

    CUDA index_select/replicate_pad backward ordinarily needs repeated-index
    scatter-add. Here value gradients first sum the four output phases at each
    anchor, then fold replication borders in a fixed slice-add order. No
    backward call enters native gather/padding, and no CPU transfer is needed.
    """
    @staticmethod
    def forward(ctx, value, attention, indices):
        ctx.save_for_backward(value, attention, indices)
        return (_gather(value, indices) * attention[:, None]).sum(2)

    @staticmethod
    @once_differentiable
    def backward(ctx, output_grad):
        value, attention, indices = ctx.saved_tensors
        batch, channels, height, width = value.shape
        grad_value = grad_attention = None
        if ctx.needs_input_grad[0]:
            weighted = output_grad[:, :, None] * attention[:, None]
            # Separate fixed 2x2 phase sums; avoid all repeated-index atomics.
            anchors = weighted.reshape(batch, channels, 9, height, 2, width, 2)
            anchors = (anchors[..., 0, :, 0] + anchors[..., 0, :, 1]
                       + anchors[..., 1, :, 0] + anchors[..., 1, :, 1])
            pieces = [_replication_shift_adjoint(anchors[:, :, k], dy, dx)
                      for k, (dy, dx) in enumerate(OFFSETS)]
            grad_value = pieces[0]
            for piece in pieces[1:]:
                grad_value = grad_value + piece
        if ctx.needs_input_grad[1]:
            grad_attention = (_gather(value, indices) * output_grad[:, :, None]).sum(1)
        return grad_value, grad_attention, None


class LocalAttender(nn.Module):
    """Guide-only 3 × 3 convex readout at twice the spatial resolution."""

    def __init__(self):
        super().__init__()
        self.low_size = (64, 64)
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(32, 9, 1, bias=True)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        height, width = self.low_size
        # mmcv1.7.2 serializes every buffer; persistent tables align strict torch state_dict.
        self.register_buffer("log_phase_prior", (bilinear_phase_prior(height, width)
                                                 + 1e-4).log(), persistent=True)
        self.register_buffer("indices", neighbour_indices(height, width), persistent=True)
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore the retired buffer when loading a checkpoint from the original implementation.
        state_dict.pop(prefix + "phase_permutations", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, value, guide):
        batch, channels, height, width = value.shape
        if tuple((height, width)) != self.low_size or channels != 32:
            raise ValueError("Expected value [B,32," + str(self.low_size) + "]")
        if tuple(guide.shape) != (batch, 32, height * 2, width * 2):
            raise ValueError("Guide/value shape mismatch")
        if value.dtype != torch.float32 or guide.dtype != torch.float32:
            raise ValueError("Frozen FP32 contract")
        if value.device != guide.device:
            raise ValueError("Guide/value device mismatch")
        attention = (self.projection(guide) + self.log_phase_prior).softmax(1)
        return _DeterministicReassembly.apply(value, attention, self.indices)


# Geometry, sampling and supervision targets
NATIVE_INTERPOLATE = F.interpolate


class _NewPathBilinear(torch.autograd.Function):
    """Native CUDA forward; deterministic CPU backward ONLY for new shapes.

    Original baseline resizes still use its qualified R01/R05 adapters.
    No CPU transfer in inference. No change to bilinear interpolation formula.
    """
    @staticmethod
    def forward(ctx, value, height, width):
        ctx.input_size = tuple(value.shape)
        ctx.output_size = (height, width)
        ctx.device = value.device
        return NATIVE_INTERPOLATE(value, size=(height, width), mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad):
        cpu_grad = grad.contiguous().to('cpu')
        result = torch.ops.aten.upsample_bilinear2d_backward(
            cpu_grad, ctx.output_size, ctx.input_size, False, None, None)
        return result.to(ctx.device), None, None


def new_resize(value, size):
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        assert value.dtype == torch.float32, 'Frozen FP32 contract'
        return _NewPathBilinear.apply(value, *size)
    return NATIVE_INTERPOLATE(value, size=size, mode='bilinear', align_corners=False)


NATIVE_GRID_SAMPLE = F.grid_sample


def make_geometry_targets(mask):
    mask = np.ascontiguousarray(mask, dtype=bool)
    height, width = mask.shape
    skeleton = skeletonize(mask, method='zhang')
    result = dict(skeleton=skeleton, has_foreground=bool(mask.any()))
    if not mask.any():
        result.update(vector=np.zeros((2, height, width), np.float32),
                      radius=np.zeros((height, width), np.float32),
                      normalization_radius=np.zeros((height,width),np.float32),
                      band=np.zeros((height, width), bool))
        return result
    radius = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    _, nearest = ndi.distance_transform_edt(~skeleton, return_indices=True)
    yy, xx = np.indices(mask.shape)
    vector=np.stack((nearest[1]-xx, nearest[0]-yy)).astype(np.float32)
    inscribed=radius[nearest[0],nearest[1]].astype(np.float32)
    anchors=(nearest[0]*width+nearest[1]).ravel()
    extent=np.zeros(height*width,np.float32)
    np.maximum.at(extent,anchors[mask.ravel()],np.linalg.norm(vector,axis=0)[mask])
    coverage=np.maximum(inscribed,extent[anchors].reshape(mask.shape))
    result.update(vector=vector,
                  radius=coverage,
                  normalization_radius=inscribed,
                  band=(~mask) & (ndi.distance_transform_edt(~mask) <= 32))
    return result


def class_balanced_region_bce(logit, label):
    zero = logit.sum()*0.0
    parts = []
    for selected, sign in ((label == 1, -1), (label == 0, 1)):
        if selected.any():
            parts.append(F.softplus(sign*logit[selected]).mean())
    return sum(parts)/len(parts) if parts else zero


def geometry_auxiliary_losses(cache, target):
    label = target.squeeze(1)
    q = new_resize(cache['q_logit'], label.shape[-2:]).squeeze(1)
    region = class_balanced_region_bce(q, label)
    proposal = class_balanced_region_bce(cache['target_logit'].squeeze(1), label)
    masks = (label == 1).detach().cpu().numpy()
    targets = [make_geometry_targets(m) for m in masks]
    device = q.device
    skeleton = torch.from_numpy(np.stack([t['skeleton'] for t in targets])).to(device)
    center = F.softplus(-q[skeleton]).mean() if skeleton.any() else q.sum()*0
    vector = torch.from_numpy(np.stack([t['vector'] for t in targets])).to(device)
    radius = torch.from_numpy(np.stack([t['radius'] for t in targets])).to(device)
    normalization_radius=torch.from_numpy(np.stack([t['normalization_radius'] for t in targets])).to(device)
    band = torch.from_numpy(np.stack([t['band'] for t in targets])).to(device) & (label == 0)
    vector_error = F.smooth_l1_loss(
        (cache['vector_full']-vector)/(normalization_radius[:, None]+4),
        torch.zeros_like(vector), reduction='none', beta=1).mean(1)
    radius_error = F.smooth_l1_loss(
        cache['radius_full'].squeeze(1).log1p(), radius.log1p(), reduction='none', beta=1)
    groups = ((label == 1), band)
    def group_mean(error):
        parts = [error[selected].mean() for selected in groups if selected.any()]
        return sum(parts)/len(parts) if parts else error.sum()*0
    return dict(region=region, skeleton=center, vector=group_mean(vector_error),
                radius=group_mean(radius_error), proposal=proposal), int(skeleton.sum())


# Decoder and auxiliary losses
MODULE_SEED = 1000046


_support_interval = (500, 2000)


def set_support_schedule(start, full):
    global _support_interval
    _support_interval = (start, full)


def ramp(step):
    start, full = _support_interval
    return max(0.0, min(1.0, (int(step) - start) / (full - start)))


@HEADS.register_module()
class PlainSecondHeadDecoder(SegMANDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_module = None
        self.register_buffer("support_step", torch.tensor(0, dtype=torch.int64))
        self.support_x = None
        self.support_cache = None

    def init_weights(self):
        super().init_weights()
        if self.support_module is None:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(MODULE_SEED).get_state())
                self.support_module = PlainSecondHeadSupport()
            if hasattr(self,'_params_init_info'):
                for parameter in self.support_module.parameters():
                    self._params_init_info[parameter] = dict(
                        init_info='ABH detail readout initialization',tmp_mean_value=parameter.data.mean().cpu())
                for module in self.support_module.modules():
                    module._params_init_info = self._params_init_info
            _install_plain_verifier(self.support_module)

    def cls_seg(self,feature):
        self.support_x = feature
        return super().cls_seg(feature)

    def forward(self, inputs):
        self.support_x = self.support_cache = None
        coefficient = ramp(self.support_step)
        try:
            base = super().forward(inputs)
            if coefficient == 0:
                return base
            output, cache = self.support_module(self.support_x, inputs[0], base, coefficient)
            if self.training and torch.is_grad_enabled():
                self.support_cache = cache
            return output
        finally:
            self.support_x = None

    def losses(self,seg_logit,seg_label):
        coefficient = ramp(int(self.support_step))
        try:
            if coefficient == 0:
                return super().losses(seg_logit,seg_label)
            assert tuple(seg_logit.shape[-2:])==tuple(seg_label.shape[-2:])==(512,512)
            assert self.support_cache is not None
            label = seg_label.squeeze(1)
            output_ce = self.loss_decode(seg_logit,label,weight=None,ignore_index=self.ignore_index)
            base_ce = self.loss_decode(self.support_cache['full_base'],label,weight=None,
                                       ignore_index=self.ignore_index)
            main_ce = (output_ce+coefficient*base_ce)/(1+coefficient)
            losses = dict(loss_ce=main_ce,
                          acc_seg=accuracy(seg_logit,label,ignore_index=self.ignore_index))
            terms, _ = geometry_auxiliary_losses(self.support_cache,seg_label)
            for name,value in terms.items():
                losses['loss_aev_'+name] = 0.05*coefficient*value
            return losses
        finally:
            self.support_cache = None


def _install_plain_verifier(module):
    """Replace 1,633 parameters without consuming the caller's CPU/CUDA RNG."""
    reference = module.geometry_head.weight
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(MODULE_SEED).get_state())
        verifier = nn.Sequential(nn.Conv2d(32, 48, 1), nn.GELU(), nn.Conv2d(48, 1, 1))
        for layer in (verifier[0], verifier[2]):
            # Match the old verifier's distribution, not its different-shaped tensors.
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)
    verifier.to(device=reference.device, dtype=reference.dtype)
    verifier.train(module.training)
    assert sum(p.numel() for p in module.verifier.parameters()) == 1633
    assert sum(p.numel() for p in verifier.parameters()) == 1633
    info = getattr(module, '_params_init_info', None)
    if info is not None:
        for parameter in module.verifier.parameters():
            info.pop(parameter, None)
        for parameter in verifier.parameters():
            info[parameter] = dict(init_info='Plain second head: isolated RNG, normal std=0.001',
                                   tmp_mean_value=parameter.data.mean().cpu())
        for child in verifier.modules():
            child._params_init_info = info
    module.verifier = verifier
    assert sum(p.numel() for p in module.parameters()) == 14766


class PlainSecondHeadSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_proj = nn.Sequential(nn.Conv2d(96,32,1), nn.GELU())
        self.context_proj = nn.Sequential(nn.Conv2d(224,32,1), nn.GELU())
        self.mix = nn.Sequential(nn.Conv2d(64,32,1), nn.GELU(),
                                 nn.Conv2d(32,32,3,padding=1,groups=32), nn.GELU())
        self.geometry_head = nn.Conv2d(32,4,1)
        self.verifier = nn.Sequential(nn.Conv2d(100,16,1), nn.GELU(), nn.Conv2d(16,1,1))
        for layer in (self.geometry_head, self.verifier[0], self.verifier[2]):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.geometry_head.bias[3] = math.log(math.expm1(0.75))
        assert sum(p.numel() for p in self.parameters()) == 14469
        self.local_attender = LocalAttender()
        assert sum(p.numel() for p in self.parameters()) == 14766

    def forward(self, x, c1, base, coefficient):
        assert tuple(x.shape[1:]) == (224, 64, 64)
        assert tuple(c1.shape[1:]) == (96, 128, 128)
        assert tuple(base.shape[1:]) == (2, 64, 64)
        guide = self.detail_proj(c1)
        value = self.context_proj(x)
        sampled = self.local_attender(value, guide)
        hidden = self.mix(torch.cat((guide, sampled), 1))

        # Preserve all four geometry auxiliary targets, with their original units.
        geo = self.geometry_head(hidden)
        q_logit, vector, radius_raw = geo[:, 0:1], geo[:, 1:3], geo[:, 3:4]
        radius = F.softplus(radius_raw) + 0.25
        vector_full = new_resize(vector * 4, (512, 512))
        radius_full = new_resize(radius * 4, (512, 512))

        # No predicted-coordinate sampling or geometry score enters this logit.
        target_logit = new_resize(self.verifier(hidden), (512, 512))
        full_base = base_head.resize(base, size=(512, 512), mode='bilinear', align_corners=False)
        difference = full_base[:, 1:2] - full_base[:, 0:1]
        delta = coefficient * (0.5 * (target_logit - difference.detach())).clamp(-4, 4)
        output = full_base + torch.cat((-0.5 * delta, 0.5 * delta), 1)
        cache = dict(q_logit=q_logit, vector_full=vector_full, radius_full=radius_full,
                     target_logit=target_logit, full_base=full_base)
        return output, cache


def parameter_groups(m):
 # Preserve historical MMCV longest custom-key matching, including head>norm.
 from mmcv.runner import build_optimizer
 cfg=OPTIMIZER_CONFIG
 opt=build_optimizer(m,copy.deepcopy(cfg))
 for g in opt.param_groups:g['base_lr']=g['lr']
 return opt.param_groups


def boundary_loss(head, label):
    z=F.interpolate(head._boundary_logit,size=label.shape[-2:],mode='bilinear',align_corners=False)
    target,mask=head._make_boundary_target(label,z.shape[-2:])
    target,mask=target.to(z),mask.to(z)
    pos=(target*mask).sum(); neg=((1-target)*mask).sum()
    w=torch.where(target>.5,(neg/(pos+1)).clamp(1,20).detach(),torch.ones_like(target))
    bce=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*w*mask).sum()/mask.sum().clamp_min(1)
    p=z.sigmoid(); dice=1-(2*(p*target*mask).sum()+1)/((p+target)*mask).sum().add(1)
    return .05*(.5*bce+.5*dice)


class ABH(nn.Module):
    def __init__(self, backbone, decode_head, config):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.support_start = config["support_start"]
        self.support_full = config["support_full"]
        self.repair_head = nn.ModuleDict({
            "e": nn.Conv2d(96, 32, 1),
            "y": nn.Conv2d(224, 32, 1),
            "read": nn.Sequential(nn.Conv2d(66, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1)),
        })
        nn.init.zeros_(self.repair_head["read"][-1].weight)
        nn.init.zeros_(self.repair_head["read"][-1].bias)

    def reset(self, step):
        set_support_schedule(self.support_start, self.support_full)
        self.decode_head.support_step.fill_(step)

    def correction(self, native):
        size = (512, 512)
        logits = F.interpolate(native, size=size, mode="bilinear", align_corners=False)
        margin = logits[:, 1:2] - logits[:, 0:1]
        shallow = self.repair_head["e"](self.decode_head._repair_e)
        detail = F.interpolate(self.repair_head["y"](self.decode_head._repair_y),
                               size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        score = F.interpolate(margin, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        uncertainty = 4 * score.sigmoid() * (1 - score.sigmoid())
        residual = self.repair_head["read"](torch.cat((shallow, detail, score, uncertainty), 1))
        residual = F.interpolate(residual, size=size, mode="bilinear", align_corners=False)
        coefficient = ramp(self.decode_head.support_step)
        corrected = margin + coefficient * 4 * residual.tanh()
        return torch.cat((-.5 * corrected, .5 * corrected), 1), coefficient

    def loss(self, images, labels, step):
        self.reset(step)
        head = self.decode_head
        try:
            native = head(self.backbone(images))
            losses = {key: value for key, value in head.losses(native, labels.long()).items()
                      if key.startswith("loss_")}
            if step > self.support_start:
                logits, coefficient = self.correction(native)
                losses["loss_repair"] = .5 * coefficient * F.cross_entropy(
                    logits, labels.squeeze(1).long(), ignore_index=head.ignore_index)
            return losses
        finally:
            head._repair_e = head._repair_y = head.support_cache = None

    def logits(self, images):
        head = self.decode_head
        try:
            return self.correction(head(self.backbone(images)))[0]
        finally:
            head._repair_e = head._repair_y = head.support_cache = None


def build(config, *, encoder_weights=None, initial_core=None):
    """Construct ABH with the original state keys and final decoder paths."""
    from mmcv.cnn.utils import revert_sync_batchnorm
    from mmseg.models import build_segmentor

    original = build_segmentor(copy.deepcopy(MODEL_CONFIG))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        original.init_weights()
        if initial_core is not None:
            if sha(initial_core) != "6a9895bf1ad42752a76e32973eb053155bd232865885f123d158270654f892cb":
                raise ValueError("Unexpected zero-update initialization checkpoint")
            canonical = torch.load(initial_core, map_location="cpu", weights_only=True)["core_state"]
            state = original.state_dict()
            expected = {name for name in state if not name.startswith("decode_head.support_")}
            if set(canonical) != expected:
                raise ValueError("Zero-update initialization has incompatible keys")
            original.load_state_dict({**state, **canonical}, strict=True)
        if encoder_weights is not None:
            weights = torch.load(encoder_weights, map_location="cpu", weights_only=True)
            weights = weights.get("state_dict_ema", weights)
            # The ImageNet checkpoint includes a classifier. The encoder-only
            # export uses the same keys and tensors, excluding that classifier.
            target = original.backbone.state_dict()
            original.backbone.load_state_dict({key: weights[key] for key in target}, strict=True)
    finally:
        torch.set_num_threads(threads)

    backbone = revert_sync_batchnorm(original.backbone)
    head = revert_sync_batchnorm(original.decode_head)
    del head.post_detail_refine
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(854)
        model = ABH(backbone, head, config)

    prior_loss = head.losses

    def losses(self, logits, labels):
        result = prior_loss(logits, labels)
        result["loss_boundary"] = boundary_loss(self, labels)
        return result

    head.losses = MethodType(losses, head)
    head.register_forward_pre_hook(lambda module, args: setattr(head, "_repair_e", args[0][0]))
    prior_cls = head.cls_seg

    def capture(self, features):
        self._repair_y = features
        return prior_cls(features)

    head.cls_seg = MethodType(capture, head)
    if sum(parameter.numel() for parameter in model.parameters()) != 94_245_101:
        raise RuntimeError("Unexpected ABH parameter count")
    model.reset(0)
    return model
