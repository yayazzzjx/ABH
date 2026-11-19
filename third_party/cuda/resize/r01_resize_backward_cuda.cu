#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
constexpr int kBatch = 2;
constexpr int kChannels = 2;
constexpr int kInputH = 64;
constexpr int kInputW = 64;
constexpr int kOutputH = 512;
constexpr int kOutputW = 512;
constexpr int kInputElements = kBatch * kChannels * kInputH * kInputW;
constexpr int kThreads = 256;

__device__ __forceinline__ int ceil_div_positive(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

__global__ void r01_gather_backward_kernel(const float* __restrict__ grad_output,
                                           float* __restrict__ grad_input) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kInputElements) {
        return;
    }

    const int spatial = idx % (kInputH * kInputW);
    const int input_y = spatial / kInputW;
    const int input_x = spatial % kInputW;
    const int batch_channel = idx / (kInputH * kInputW);

    const int y_begin = input_y == 0 ? 0 :
        ceil_div_positive((2 * input_y - 1) * kOutputH - kInputH,
                          2 * kInputH);
    const int y_end = input_y == kInputH - 1 ? kOutputH :
        ceil_div_positive((2 * input_y + 3) * kOutputH - kInputH,
                          2 * kInputH);
    const int x_begin = input_x == 0 ? 0 :
        ceil_div_positive((2 * input_x - 1) * kOutputW - kInputW,
                          2 * kInputW);
    const int x_end = input_x == kInputW - 1 ? kOutputW :
        ceil_div_positive((2 * input_x + 3) * kOutputW - kInputW,
                          2 * kInputW);

    float accumulator = 0.0f;
    const float scale_y = static_cast<float>(kInputH) / static_cast<float>(kOutputH);
    const float scale_x = static_cast<float>(kInputW) / static_cast<float>(kOutputW);

    for (int output_y = y_begin; output_y < y_end; ++output_y) {
        float source_y = (static_cast<float>(output_y) + 0.5f) * scale_y - 0.5f;
        source_y = source_y < 0.0f ? 0.0f : source_y;
        const int y_low = static_cast<int>(floorf(source_y));
        const int y_high_raw = y_low + 1;
        const int y_high = y_high_raw < kInputH ? y_high_raw : kInputH - 1;
        const float y_lambda = source_y - static_cast<float>(y_low);
        float weight_y = 0.0f;
        if (y_low == input_y) {
            weight_y += 1.0f - y_lambda;
        }
        if (y_high == input_y) {
            weight_y += y_lambda;
        }
        if (weight_y == 0.0f) {
            continue;
        }

        for (int output_x = x_begin; output_x < x_end; ++output_x) {
            float source_x = (static_cast<float>(output_x) + 0.5f) * scale_x - 0.5f;
            source_x = source_x < 0.0f ? 0.0f : source_x;
            const int x_low = static_cast<int>(floorf(source_x));
            const int x_high_raw = x_low + 1;
            const int x_high = x_high_raw < kInputW ? x_high_raw : kInputW - 1;
            const float x_lambda = source_x - static_cast<float>(x_low);
            float weight_x = 0.0f;
            if (x_low == input_x) {
                weight_x += 1.0f - x_lambda;
            }
            if (x_high == input_x) {
                weight_x += x_lambda;
            }
            if (weight_x == 0.0f) {
                continue;
            }

            const float weight = weight_y * weight_x;
            const int64_t offset = static_cast<int64_t>(batch_channel) *
                kOutputH * kOutputW + output_y * kOutputW + output_x;
            accumulator += grad_output[offset] * weight;
        }
    }

    grad_input[idx] = accumulator;
}
}  // namespace

void r01_resize_backward_cuda_launcher(const torch::Tensor& grad_output,
                                       torch::Tensor& grad_input) {
    c10::cuda::CUDAGuard device_guard(grad_output.device());
    const int blocks = (kInputElements + kThreads - 1) / kThreads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
    r01_gather_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(), grad_input.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
