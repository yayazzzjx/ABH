#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
constexpr double kCubicA = -0.75;

__device__ __forceinline__ double cubic_weight(double distance) {
    const double x = fabs(distance);
    if (x <= 1.0) {
        return ((kCubicA + 2.0) * x - (kCubicA + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((kCubicA * x - 5.0 * kCubicA) * x + 8.0 * kCubicA) * x
               - 4.0 * kCubicA;
    }
    return 0.0;
}

__device__ __forceinline__ int64_t begin_output(int64_t input_index,
                                                 int64_t input_size,
                                                 int64_t output_size) {
    const double low = (static_cast<double>(input_index) - 1.5) *
                       static_cast<double>(output_size) /
                       static_cast<double>(input_size) - 0.5;
    int64_t value = static_cast<int64_t>(floor(low)) - 2;
    return value < 0 ? 0 : (value > output_size ? output_size : value);
}

__device__ __forceinline__ int64_t end_output(int64_t input_index,
                                               int64_t input_size,
                                               int64_t output_size) {
    const double high = (static_cast<double>(input_index) + 2.5) *
                        static_cast<double>(output_size) /
                        static_cast<double>(input_size) - 0.5;
    int64_t value = static_cast<int64_t>(ceil(high)) + 2;
    return value < 0 ? 0 : (value > output_size ? output_size : value);
}

__device__ __forceinline__ double weight_for_input(double source,
                                                   int64_t input_index,
                                                   int64_t input_size) {
    const int64_t base = static_cast<int64_t>(floor(source));
    double total = 0.0;
    #pragma unroll
    for (int tap = -1; tap <= 2; ++tap) {
        const int64_t raw_index = base + tap;
        const int64_t clamped = raw_index < 0 ? 0 :
            (raw_index >= input_size ? input_size - 1 : raw_index);
        if (clamped == input_index) {
            total += cubic_weight(source - static_cast<double>(raw_index));
        }
    }
    return total;
}

__global__ void bicubic_gather_backward_kernel(const float* __restrict__ grad_output,
                                                float* __restrict__ grad_input,
                                                int64_t input_h, int64_t input_w,
                                                int64_t output_h, int64_t output_w,
                                                int64_t total) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t plane = input_h * input_w;
    const int64_t spatial = index % plane;
    const int64_t input_y = spatial / input_w;
    const int64_t input_x = spatial % input_w;
    const int64_t batch_channel = index / plane;
    const int64_t y_begin = begin_output(input_y, input_h, output_h);
    const int64_t y_end = end_output(input_y, input_h, output_h);
    const int64_t x_begin = begin_output(input_x, input_w, output_w);
    const int64_t x_end = end_output(input_x, input_w, output_w);
    const double scale_y = static_cast<double>(input_h) / static_cast<double>(output_h);
    const double scale_x = static_cast<double>(input_w) / static_cast<double>(output_w);
    double accumulator = 0.0;
    for (int64_t oy = y_begin; oy < y_end; ++oy) {
        const double source_y = (static_cast<double>(oy) + 0.5) * scale_y - 0.5;
        const double wy = weight_for_input(source_y, input_y, input_h);
        if (wy == 0.0) continue;
        for (int64_t ox = x_begin; ox < x_end; ++ox) {
            const double source_x = (static_cast<double>(ox) + 0.5) * scale_x - 0.5;
            const double wx = weight_for_input(source_x, input_x, input_w);
            if (wx == 0.0) continue;
            const int64_t go_index = (batch_channel * output_h + oy) * output_w + ox;
            accumulator += (static_cast<double>(grad_output[go_index]) * wy) * wx;
        }
    }
    grad_input[index] = static_cast<float>(accumulator);
}
}  // namespace

void bicubic_backward_cuda_launcher(const torch::Tensor& grad_output,
                                    torch::Tensor& grad_input,
                                    int64_t input_h, int64_t input_w,
                                    int64_t output_h, int64_t output_w) {
    c10::cuda::CUDAGuard guard(grad_output.device());
    const int64_t total = grad_input.numel();
    const int64_t blocks = total / kThreads + (total % kThreads != 0);
    TORCH_CHECK(blocks <= 2147483647LL, "bicubic backward grid too large");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
    bicubic_gather_backward_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
        input_h, input_w, output_h, output_w, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
