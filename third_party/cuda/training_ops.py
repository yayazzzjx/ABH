"""Interpolation adapters used by the qualified ABH training path."""
import torch
import torch.nn.functional as F
import mmseg.models.decode_heads.decode_head as base_head

import bicubic_double_ext
import bicubic_float_ext
import r01_resize_backward_ext
import r05_resize_backward_ext


EXPECTED_INPUT_SHAPE = (2, 2, 64, 64)


EXPECTED_OUTPUT_SIZE = (512, 512)


def _check_r01(input, size, mode, align_corners):
    if not isinstance(input, torch.Tensor):
        raise TypeError("R01 accepts a torch.Tensor only")
    if not input.is_cuda:
        raise RuntimeError("R01 accepts CUDA input only")
    if input.dtype != torch.float32:
        raise RuntimeError("R01 accepts float32 input only")
    if tuple(input.shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"R01 input shape must be {EXPECTED_INPUT_SHAPE}, got {tuple(input.shape)}")
    if not input.is_contiguous():
        raise RuntimeError("R01 requires contiguous input")
    if tuple(size) != EXPECTED_OUTPUT_SIZE:
        raise RuntimeError(f"R01 output size must be {EXPECTED_OUTPUT_SIZE}, got {tuple(size)}")
    if mode != "bilinear" or align_corners is not False:
        raise RuntimeError("R01 supports bilinear with align_corners=False only")


class _R01Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size, mode, align_corners):
        _check_r01(input, size, mode, align_corners)
        return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        if not grad_output.is_cuda or grad_output.dtype != torch.float32:
            raise RuntimeError("R01 upstream gradient must be CUDA float32")
        if tuple(grad_output.shape) != (2, 2, 512, 512):
            raise RuntimeError("R01 upstream gradient shape must be (2, 2, 512, 512)")
        if not grad_output.is_contiguous():
            raise RuntimeError("R01 requires contiguous upstream gradient")
        grad_input = r01_resize_backward_ext.r01_resize_backward(grad_output)
        return grad_input, None, None, None


def _resize_r01(input, size, mode="bilinear", align_corners=False):
    _check_r01(input, size, mode, align_corners)
    return _R01Bilinear.apply(input, tuple(size), mode, align_corners)


REAL_CALLS = {
    ((2, 672, 16, 16), (64, 64)),
    ((2, 2, 64, 64), (512, 512)),
}


def _check_r05(input, size, mode, align_corners):
    if not isinstance(input, torch.Tensor):
        raise TypeError("R05 input must be a tensor")
    if not input.is_cuda or input.dtype != torch.float32 or input.dim() != 4:
        raise RuntimeError("R05 public dispatch requires CUDA float32 NCHW")
    if not input.is_contiguous():
        raise RuntimeError("R05 public dispatch requires contiguous input")
    if mode != "bilinear" or align_corners is not False:
        raise RuntimeError("R05 public dispatch requires bilinear, align_corners=False")
    if (tuple(input.shape), tuple(size)) not in REAL_CALLS:
        raise RuntimeError("R05 public dispatch only accepts frozen S1/S2 shapes")


class _R05Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size, mode, align_corners):
        _check_r05(input, size, mode, align_corners)
        ctx.input_shape = tuple(input.shape)
        ctx.output_size = tuple(size)
        return F.interpolate(
            input, size=ctx.output_size, mode=mode,
            align_corners=align_corners,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        if not grad_output.is_cuda or grad_output.dtype != torch.float32:
            raise RuntimeError("R05 upstream gradient must be CUDA float32")
        grad_output = grad_output.contiguous()
        _, _, ih, iw = ctx.input_shape
        oh, ow = ctx.output_size
        if tuple(grad_output.shape) != (
            ctx.input_shape[0], ctx.input_shape[1], oh, ow
        ):
            raise RuntimeError("R05 upstream gradient shape mismatch")
        grad_input = r05_resize_backward_ext.r05_resize_backward(
            grad_output, ih, iw, oh, ow
        )
        return grad_input, None, None, None


def _resize_r05(input, size, mode="bilinear", align_corners=False):
    _check_r05(input, size, mode, align_corners)
    return _R05Bilinear.apply(
        input, tuple(size), mode, align_corners
    )


REAL_STRIDED_CALLS = {
    "SIG01_BILINEAR_16_TO_64": {
        "input_shape": (2, 432, 16, 16),
        "input_stride": (110592, 1, 6912, 432),
        "output_size": (64, 64),
    },
    "SIG02_BILINEAR_32_TO_64": {
        "input_shape": (2, 432, 32, 32),
        "input_stride": (442368, 1, 13824, 432),
        "output_size": (64, 64),
    },
}


def _check_strided(input: torch.Tensor, size, mode, align_corners) -> str:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if (input.device.type != "cuda" or input.device.index != 0
            or input.dtype != torch.float32 or input.dim() != 4):
        raise RuntimeError("adapter requires CUDA:0 float32 NCHW-rank-4 input")
    if mode != "bilinear" or align_corners is not False:
        raise RuntimeError("adapter requires bilinear align_corners=False")
    size = tuple(size)
    for name, spec in REAL_STRIDED_CALLS.items():
        if (tuple(input.shape) == spec["input_shape"]
                and tuple(input.stride()) == spec["input_stride"]
                and size == spec["output_size"]):
            return name
    raise RuntimeError(
        f"call is outside the frozen two-signature allowlist: "
        f"shape={tuple(input.shape)} stride={tuple(input.stride())} size={size}"
    )


class _RealShapeBilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size, mode, align_corners):
        name = _check_strided(input, size, mode, align_corners)
        ctx.signature = name
        ctx.input_shape = tuple(input.shape)
        ctx.output_size = tuple(size)
        ctx.mode = mode
        ctx.align_corners = align_corners
        return F.interpolate(
            input,
            size=ctx.output_size,
            mode=ctx.mode,
            align_corners=ctx.align_corners,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        if (grad_output.device.type != "cuda"
                or grad_output.device.index != 0
                or grad_output.dtype != torch.float32
                or tuple(grad_output.shape)
                != (ctx.input_shape[0], ctx.input_shape[1], *ctx.output_size)):
            raise RuntimeError("upstream gradient identity mismatch")
        # The frozen R05 r05_resize_backward_ext accepts contiguous NCHW gradients. Keep this
        # conversion inside backward so qualification times the full adapter ABI.
        grad_contiguous = grad_output.contiguous()
        grad_input = r05_resize_backward_ext.r05_resize_backward(
            grad_contiguous,
            ctx.input_shape[2],
            ctx.input_shape[3],
            ctx.output_size[0],
            ctx.output_size[1],
        )
        return grad_input, None, None, None


def _resize_strided(input, size, mode="bilinear", align_corners=False):
    _check_strided(input, size, mode, align_corners)
    return _RealShapeBilinear.apply(input, tuple(size), mode, align_corners)


POOL_SHAPE = (2, 224, 1, 1)


POOL_TARGET = (64, 64)


def _check_pool(input, size, mode, align_corners):
    if (not isinstance(input, torch.Tensor) or not input.is_cuda
            or input.dtype != torch.float32 or not input.is_contiguous()
            or tuple(input.shape) != POOL_SHAPE or tuple(size) != POOL_TARGET
            or mode != 'bilinear' or align_corners is not False):
        raise RuntimeError('R05P dispatch accepts only frozen real pool-resize signature')


class _PoolResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size, mode, align_corners):
        _check_pool(input, size, mode, align_corners)
        return F.interpolate(input, size=POOL_TARGET, mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        if (not grad_output.is_cuda or grad_output.dtype != torch.float32
                or tuple(grad_output.shape) != (2, 224, 64, 64)):
            raise RuntimeError('R05P upstream gradient identity mismatch')
        grad_input = r05_resize_backward_ext.r05_resize_backward(grad_output.contiguous(), 1, 1, 64, 64)
        return grad_input, None, None, None


def _resize_pool(input, size, mode='bilinear', align_corners=False):
    _check_pool(input, size, mode, align_corners)
    return _PoolResize.apply(input, tuple(size), mode, align_corners)


def precision_for_ordinal(ordinal):
    if ordinal == 2:
        return 'double'
    if ordinal in (1, 3, 4):
        return 'float'
    raise RuntimeError(f'unregistered bicubic call ordinal: {ordinal}')


def module_for_ordinal(ordinal):
    return bicubic_double_ext if precision_for_ordinal(ordinal) == 'double' else bicubic_float_ext


ORIGINAL_INTERPOLATE = F.interpolate


class _DeterministicBicubic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, out_h, out_w, ordinal):
        if input.dtype != torch.float32 or input.device.type != 'cuda':
            raise RuntimeError('frozen bicubic repair supports CUDA float32 only')
        if tuple(input.shape) != (1, 20, 49, 49) or (out_h, out_w) != (256, 256):
            raise RuntimeError('bicubic repair called outside frozen real shapes')
        precision_for_ordinal(ordinal)
        ctx.ordinal = ordinal
        ctx.input_h, ctx.input_w = input.shape[-2:]
        return ORIGINAL_INTERPOLATE(input, size=(out_h, out_w), mode='bicubic',
                                    align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        contiguous = grad_output.contiguous()
        grad_input = module_for_ordinal(ctx.ordinal).bicubic_backward(
            contiguous, ctx.input_h, ctx.input_w)
        return grad_input, None, None, None


def _resize_bicubic(input, size, mode, align_corners, ordinal):
    if mode != 'bicubic' or align_corners is not False or tuple(input.shape) != (1, 20, 49, 49) or tuple(size) != (256, 256):
        raise RuntimeError('attempt adapter only accepts the four frozen bicubic calls')
    return _DeterministicBicubic.apply(input, int(size[0]), int(size[1]), ordinal)


BICUBIC_MODULES = {f"backbone.layers.6.blocks.{index}.token_mixer": index + 1
                   for index in range(4)}


class FastR01PathRepairHook:
    def before_run(self, runner):
        self.calls = []
        self.original = base_head.resize

        def repaired(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
            if input.requires_grad:
                if scale_factor is not None:
                    raise RuntimeError("Unexpected loss resize scale factor")
                result = _resize_r01(input, size, mode, align_corners)
                self.calls.append(result.shape)
                return result
            return self.original(input, size, scale_factor, mode, align_corners, warning)

        base_head.resize = repaired

    def after_run(self, runner):
        base_head.resize = self.original


class FastR05PathRepairHook:
    def before_run(self, runner):
        self.original = F.interpolate
        self.inside = False
        self.calls = []
        targets = {
            ((2, 672, 16, 16), (64, 64)): ("S1", _resize_r05),
            ((2, 224, 1, 1), (64, 64)): ("POOL", _resize_pool),
            ((2, 432, 16, 16), (64, 64)): ("REAL16", _resize_strided),
            ((2, 432, 32, 32), (64, 64)): ("REAL32", _resize_strided),
        }

        def dispatch(*args, **kwargs):
            if self.inside:
                return self.original(*args, **kwargs)
            source = args[0] if args else kwargs.get("input")
            size = kwargs.get("size", args[1] if len(args) > 1 else None)
            key = (tuple(source.shape), tuple(size)) if size is not None else None
            target = targets.get(key)
            if target and source.requires_grad:
                mode = kwargs.get("mode", args[3] if len(args) > 3 else "nearest")
                align = kwargs.get("align_corners", args[4] if len(args) > 4 else None)
                scale = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)
                if (scale is not None or kwargs.get("antialias", False)
                        or kwargs.get("recompute_scale_factor") not in (None, False)):
                    raise RuntimeError("Unexpected interpolation options")
                name, resize = target
                self.inside = True
                try:
                    output = resize(source, tuple(size), mode, align)
                finally:
                    self.inside = False
                self.calls.append(name)
                return output
            return self.original(*args, **kwargs)

        F.interpolate = dispatch

    def after_run(self, runner):
        F.interpolate = self.original


class ComposedDispatchHook:
    def before_run(self, runner):
        self.context = None
        self.calls = []
        self.handles = []
        self.original = F.interpolate
        model = getattr(runner.model, "module", runner.model)
        modules = dict(model.named_modules())
        for name in BICUBIC_MODULES:
            module = modules[name]

            def enter(mod, args, name=name):
                if self.context is not None:
                    raise RuntimeError("Nested bicubic context")
                self.context = name

            def leave(mod, args, output):
                self.context = None

            self.handles.extend([module.register_forward_pre_hook(enter), module.register_forward_hook(leave)])

        def dispatch(input, *args, **kwargs):
            mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
            if mode == "bicubic" and torch.is_grad_enabled() and input.requires_grad:
                size = kwargs.get("size", args[0] if args else None)
                align = kwargs.get("align_corners", args[3] if len(args) > 3 else None)
                scale = kwargs.get("scale_factor", args[1] if len(args) > 1 else None)
                if (scale is not None or kwargs.get("antialias", False)
                        or kwargs.get("recompute_scale_factor") not in (None, False)):
                    raise RuntimeError("Unexpected bicubic options")
                if self.context not in BICUBIC_MODULES or not input.is_contiguous():
                    raise RuntimeError("Unexpected bicubic input or module")
                ordinal = BICUBIC_MODULES[self.context]
                output = _resize_bicubic(input, tuple(size), mode, align, ordinal)
                row = dict(module=self.context, precision=precision_for_ordinal(ordinal), backward_seen=False)

                def incoming(gradient):
                    row["backward_seen"] = True
                    return gradient

                output.register_hook(incoming)
                self.calls.append(row)
                return output
            return self.original(input, *args, **kwargs)

        F.interpolate = dispatch

    def after_run(self, runner):
        F.interpolate = self.original
        for handle in self.handles:
            handle.remove()
