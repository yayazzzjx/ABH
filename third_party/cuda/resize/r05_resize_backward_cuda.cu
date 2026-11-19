#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int64_t conservative_begin(
    int64_t input_index, int64_t input_size, int64_t output_size) {
    const float low =
        (static_cast<float>(input_index) - 0.5f) *
        static_cast<float>(output_size) /
        static_cast<float>(input_size) - 0.5f;
    int64_t begin = static_cast<int64_t>(floorf(low)) - 2;
    if (begin < 0) begin = 0;
    if (begin > output_size) begin = output_size;
    return begin;
}

__device__ __forceinline__ int64_t conservative_end(
    int64_t input_index, int64_t input_size, int64_t output_size) {
    const float high =
        (static_cast<float>(input_index) + 1.5f) *
        static_cast<float>(output_size) /
        static_cast<float>(input_size) - 0.5f;
    int64_t end = static_cast<int64_t>(ceilf(high)) + 2;
    if (end < 0) end = 0;
    if (end > output_size) end = output_size;
    return end;
}

__global__ void r05_gather_backward_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t total) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t input_plane = input_h * input_w;
    const int64_t spatial = idx % input_plane;
    const int64_t input_y = spatial / input_w;
    const int64_t input_x = spatial % input_w;
    const int64_t batch_channel = idx / input_plane;

    const int64_t y_begin =
        conservative_begin(input_y, input_h, output_h);
    const int64_t y_end =
        conservative_end(input_y, input_h, output_h);
    const int64_t x_begin =
        conservative_begin(input_x, input_w, output_w);
    const int64_t x_end =
        conservative_end(input_x, input_w, output_w);

    const float scale_y =
        static_cast<float>(input_h) / static_cast<float>(output_h);
    const float scale_x =
        static_cast<float>(input_w) / static_cast<float>(output_w);

    float accumulator = 0.0f;
    for (int64_t output_y = y_begin; output_y < y_end; ++output_y) {
        float source_y =
            (static_cast<float>(output_y) + 0.5f) * scale_y - 0.5f;
        source_y = source_y < 0.0f ? 0.0f : source_y;
        source_y = source_y > static_cast<float>(input_h - 1)
                        ? static_cast<float>(input_h - 1)
                        : source_y;
        const int64_t y_low = static_cast<int64_t>(floorf(source_y));
        const int64_t y_high =
            y_low + 1 < input_h ? y_low + 1 : input_h - 1;
        const float y_lambda = source_y - static_cast<float>(y_low);
        float weight_y = 0.0f;
        if (y_low == input_y) weight_y += 1.0f - y_lambda;
        if (y_high == input_y) weight_y += y_lambda;
        if (weight_y == 0.0f) continue;

        for (int64_t output_x = x_begin; output_x < x_end; ++output_x) {
            float source_x =
                (static_cast<float>(output_x) + 0.5f) * scale_x - 0.5f;
            source_x = source_x < 0.0f ? 0.0f : source_x;
            source_x = source_x > static_cast<float>(input_w - 1)
                            ? static_cast<float>(input_w - 1)
                            : source_x;
            const int64_t x_low = static_cast<int64_t>(floorf(source_x));
            const int64_t x_high =
                x_low + 1 < input_w ? x_low + 1 : input_w - 1;
            const float x_lambda = source_x - static_cast<float>(x_low);
            float weight_x = 0.0f;
            if (x_low == input_x) weight_x += 1.0f - x_lambda;
            if (x_high == input_x) weight_x += x_lambda;
            if (weight_x == 0.0f) continue;

            const int64_t offset =
                (batch_channel * output_h + output_y) * output_w + output_x;
            accumulator += grad_output[offset] * weight_y * weight_x;
        }
    }
    grad_input[idx] = accumulator;
}

}  // namespace

void r05_resize_backward_cuda_launcher(
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w) {
    c10::cuda::CUDAGuard device_guard(grad_output.device());
    const int64_t total = grad_input.numel();
    const int64_t blocks64 = total / kThreads + (total % kThreads != 0);
    TORCH_CHECK(blocks64 <= 2147483647LL, "R05 launch grid is too large");
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(grad_output.get_device());
    r05_gather_backward_kernel<<<static_cast<int>(blocks64), kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        input_h, input_w, output_h, output_w, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
