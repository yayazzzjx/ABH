#include <torch/extension.h>
#include <cstdint>

void bicubic_backward_cuda_launcher(const torch::Tensor&, torch::Tensor&,
                                    int64_t, int64_t, int64_t, int64_t);

torch::Tensor bicubic_backward(torch::Tensor grad_output,
                               int64_t input_h, int64_t input_w) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.scalar_type() == at::kFloat,
                "bicubic backward requires CUDA float32 upstream gradient");
    TORCH_CHECK(grad_output.dim() == 4 && grad_output.is_contiguous(),
                "bicubic backward requires contiguous NCHW upstream gradient");
    TORCH_CHECK(grad_output.size(0) == 1 && grad_output.size(1) == 20 &&
                grad_output.size(2) == 256 && grad_output.size(3) == 256 &&
                input_h == 49 && input_w == 49,
                "unsupported shape; frozen real scope is [1,20,49,49] to [1,20,256,256]");
    auto grad_input = torch::empty({1, 20, input_h, input_w}, grad_output.options());
    bicubic_backward_cuda_launcher(grad_output, grad_input,
                                   input_h, input_w, 256, 256);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bicubic_backward", &bicubic_backward,
               "Fixed-order deterministic bicubic backward for frozen B0 shapes");
}
