#include <torch/extension.h>
#include <cstdint>

void r01_resize_backward_cuda_launcher(const torch::Tensor& grad_output,
                                       torch::Tensor& grad_input);

torch::Tensor r01_resize_backward(torch::Tensor grad_output) {
    TORCH_CHECK(grad_output.is_cuda(), "R01 upstream gradient must be CUDA");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat,
                "R01 upstream gradient must be float32");
    TORCH_CHECK(grad_output.is_contiguous(),
                "R01 upstream gradient must be contiguous");
    TORCH_CHECK(grad_output.dim() == 4 &&
                grad_output.size(0) == 2 && grad_output.size(1) == 2 &&
                grad_output.size(2) == 512 && grad_output.size(3) == 512,
                "R01 supports only grad_output shape (2, 2, 512, 512)");

    auto grad_input = torch::empty({2, 2, 64, 64}, grad_output.options());
    r01_resize_backward_cuda_launcher(grad_output, grad_input);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("r01_resize_backward", &r01_resize_backward,
               "R01 deterministic bilinear resize backward (fixed gather order)");
}
