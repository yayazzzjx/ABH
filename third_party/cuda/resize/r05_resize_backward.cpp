#include <torch/extension.h>
#include <cstdint>
#include <limits>

void r05_resize_backward_cuda_launcher(
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w);

static void check_tensors(const torch::Tensor& grad_output,
                          const torch::Tensor& grad_input) {
    TORCH_CHECK(grad_output.is_cuda() && grad_input.is_cuda(),
                "R05 gradients must be CUDA tensors");
    TORCH_CHECK(grad_output.device() == grad_input.device(),
                "R05 gradient devices differ");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat &&
                grad_input.scalar_type() == at::kFloat,
                "R05 gradients must be float32");
    TORCH_CHECK(grad_output.dim() == 4 && grad_input.dim() == 4 &&
                grad_output.is_contiguous() && grad_input.is_contiguous(),
                "R05 gradients must be contiguous NCHW");
    TORCH_CHECK(grad_output.size(0) == grad_input.size(0) &&
                grad_output.size(1) == grad_input.size(1),
                "R05 batch/channel dimensions differ");
    TORCH_CHECK(grad_output.numel() > 0 && grad_input.numel() > 0,
                "R05 empty gradients are unsupported");
}

void r05_resize_backward_into(torch::Tensor grad_output,
                              torch::Tensor grad_input) {
    check_tensors(grad_output, grad_input);
    const int64_t input_h = grad_input.size(2);
    const int64_t input_w = grad_input.size(3);
    const int64_t output_h = grad_output.size(2);
    const int64_t output_w = grad_output.size(3);
    TORCH_CHECK(input_h > 0 && input_w > 0 && output_h > 0 && output_w > 0,
                "R05 dimensions must be positive");
    TORCH_CHECK(input_h <= std::numeric_limits<int64_t>::max() / input_w &&
                output_h <= std::numeric_limits<int64_t>::max() / output_w,
                "R05 spatial product overflow");
    r05_resize_backward_cuda_launcher(
        grad_output, grad_input, input_h, input_w, output_h, output_w);
}

torch::Tensor r05_resize_backward(torch::Tensor grad_output,
                                  int64_t input_h, int64_t input_w,
                                  int64_t output_h, int64_t output_w) {
    TORCH_CHECK(input_h > 0 && input_w > 0 && output_h > 0 && output_w > 0,
                "R05 dimensions must be positive");
    TORCH_CHECK(grad_output.dim() == 4 &&
                grad_output.size(2) == output_h &&
                grad_output.size(3) == output_w,
                "R05 output shape mismatch");
    auto grad_input = torch::empty(
        {grad_output.size(0), grad_output.size(1), input_h, input_w},
        grad_output.options());
    r05_resize_backward_into(grad_output, grad_input);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("r05_resize_backward", &r05_resize_backward,
               "R05 generalized deterministic bilinear backward");
    module.def("r05_resize_backward_into", &r05_resize_backward_into,
               "R05 debug-only prefilled output coverage probe");
}
