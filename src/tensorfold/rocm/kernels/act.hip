#include "act.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace tf {
namespace rocm {

__device__ float warp_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

__device__ inline float widen(float v) { return v; }
__device__ inline float widen(__half v) { return __half2float(v); }
__device__ inline float widen(hip_bfloat16 v) { return static_cast<float>(v); }
__device__ inline void narrow(float v, float* out) { *out = v; }
__device__ inline void narrow(float v, __half* out) { *out = __float2half_rn(v); }
__device__ inline void narrow(float v, hip_bfloat16* out) { *out = hip_bfloat16(v); }

// fp32 sum of squares by one wave (lane l adds l, l + 32, ... then the xor tree), so bits do not depend on blockDim.
constexpr int kRmsThreads = 256;
constexpr int kRmsWidth = 8192;

template <typename T>
__global__ void __launch_bounds__(kRmsThreads) rms_kernel(const T* __restrict__ x, const float* __restrict__ weight,
                                                         T* __restrict__ y, int width, float eps) {
    __shared__ float row_v[kRmsWidth];
    __shared__ float inv_s;
    const int row = blockIdx.x;
    const T* in = x + static_cast<long long>(row) * width;
    for (int i = threadIdx.x; i < width; i += kRmsThreads) row_v[i] = widen(in[i]);
    __syncthreads();
    if (threadIdx.x < 32) {
        float sum = 0.f;
        for (int i = threadIdx.x; i < width; i += 32) sum += row_v[i] * row_v[i];
        const float inv = rsqrtf(warp_sum(sum) / static_cast<float>(width) + eps);
        if (threadIdx.x == 0) inv_s = inv;
    }
    __syncthreads();
    const float inv = inv_s;
    T* out = y + static_cast<long long>(row) * width;
    for (int i = threadIdx.x; i < width; i += kRmsThreads) {
        float value = row_v[i] * inv;
        if (weight != nullptr) value *= weight[i];
        narrow(value, out + i);
    }
}

// Length 1. Taps run in order, then the state shifts toward the new sample.
__global__ void conv_decode_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                   float* __restrict__ state, float* __restrict__ y, int channels, int kernel) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (channel >= channels) return;
    const float* row = weight + static_cast<long long>(channel) * kernel;
    float history[7];
    int kept = kernel - 1;
    float acc = 0.f;
    for (int tap = 0; tap < kept; ++tap) {
        history[tap] = state[(static_cast<long long>(batch) * kept + tap) * channels + channel];
        acc += history[tap] * row[tap];
    }
    float sample = x[static_cast<long long>(batch) * channels + channel];
    acc += sample * row[kept];
    float gate = 1.f / (1.f + expf(-acc));
    y[(static_cast<long long>(batch) * channels) + channel] = acc * gate;
    for (int tap = 0; tap < kept - 1; ++tap) {
        state[(static_cast<long long>(batch) * kept + tap) * channels + channel] = history[tap + 1];
    }
    if (kept > 0) state[(static_cast<long long>(batch) * kept + kept - 1) * channels + channel] = sample;
}

// A prompt's causal conv, the host loop's arithmetic in one pass: out = 0, then each tap's product rounded and added
// in tap order (the build keeps fp-contract off), then silu as torch's x / (1 + exp(-x)). Row t's window is the
// state's rows then x's; the new state is the window's last rows and stays on the host.
template <typename T>
__global__ void conv_prefill_kernel(const T* __restrict__ x, const float* __restrict__ weight,
                                    const float* __restrict__ state, float* __restrict__ y, int length, int channels,
                                    int kernel) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int batch = blockIdx.z;
    if (channel >= channels) return;
    const float* row = weight + static_cast<long long>(channel) * kernel;
    const int kept = kernel - 1;
    float acc = 0.f;
    for (int tap = 0; tap < kernel; ++tap) {
        const int at = t + tap;                       // the window's row
        const float v = at < kept ? state[(static_cast<long long>(batch) * kept + at) * channels + channel]
                                  : static_cast<float>(x[(static_cast<long long>(batch) * length + at - kept) *
                                                         channels + channel]);
        acc = acc + v * row[tap];
    }
    y[(static_cast<long long>(batch) * length + t) * channels + channel] = acc / (1.f + expf(-acc));
}

__global__ void rope_decode_kernel(const float* __restrict__ x, float* __restrict__ y, int width, int rotary,
                                   int pos, float theta, const int* __restrict__ pos_dev) {
    if (pos_dev != nullptr) pos = *pos_dev;
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const float* in = x + static_cast<long long>(row) * width;
    float* out = y + static_cast<long long>(row) * width;
    int half = rotary >> 1;
    for (int i = lane; i < half; i += 32) {
        float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(half));
        float ang = static_cast<float>(pos) * freq;
        float c = cosf(ang);
        float s = sinf(ang);
        float x1 = in[i];
        float x2 = in[half + i];
        out[i] = x1 * c - x2 * s;
        out[half + i] = x1 * s + x2 * c;
    }
    for (int i = rotary + lane; i < width; i += 32) out[i] = in[i];
}

template <typename T>
__global__ void __launch_bounds__(256) moe_router_kernel(const T* __restrict__ x, const float* __restrict__ rows,
                                                         float* __restrict__ logits, int d, int e) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row = blockIdx.y;
    if (expert >= e) return;
    const T* xr = x + static_cast<long long>(row) * d;
    const float* w = rows + static_cast<long long>(expert) * d;
    float acc = 0.f;
    for (int i = lane; i < d; i += 32) acc = fmaf(widen(xr[i]), w[i], acc);
    acc = warp_sum(acc);
    if (lane == 0) logits[static_cast<long long>(row) * e + expert] = acc;
}

// The same sums as moe_router_kernel (each lane's fmaf chain over k = lane, lane + 32, ..., then warp_sum), with the
// expert's row held in registers across a tile of x rows: a prompt reads the router once a tile, not once a row,
// and a decode step issues the row's loads up front.
constexpr int kRouterTile = 32;

template <typename T, int PER>
__global__ void __launch_bounds__(256) moe_router_rows(const T* __restrict__ x, const float* __restrict__ rows,
                                                       float* __restrict__ logits, int d, int e, int r) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (expert >= e) return;
    const float* w = rows + static_cast<long long>(expert) * d;
    float wr[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) wr[j] = w[lane + 32 * j];
    const int r0 = blockIdx.y * kRouterTile;
    const int r1 = r0 + kRouterTile < r ? r0 + kRouterTile : r;
    for (int row = r0; row < r1; ++row) {
        const T* xr = x + static_cast<long long>(row) * d;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc = fmaf(widen(xr[lane + 32 * j]), wr[j], acc);
        acc = warp_sum(acc);
        if (lane == 0) logits[static_cast<long long>(row) * e + expert] = acc;
    }
}

constexpr int kSelectMax = 1024;

__device__ inline float round_bf16(float v) { return static_cast<float>(hip_bfloat16(v)); }

__global__ void __launch_bounds__(32) moe_select_kernel(const float* __restrict__ logits, int* __restrict__ pick,
                                                        float* __restrict__ wts, int* __restrict__ items,
                                                        int* __restrict__ members, int capacity, int experts,
                                                        int top_k) {
    __shared__ float v[kSelectMax];
    __shared__ float ex[32];
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    for (int i = lane; i < experts; i += 32) v[i] = l[i];
    __syncthreads();
    float top = 0.f, total = 0.f;
    for (int k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int i = lane; i < experts; i += 32) {
            const float value = v[i];
            if (value > best || (value == best && i < idx)) {
                best = value;
                idx = i;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor(best, mask, 32);
            const int at = __shfl_xor(idx, mask, 32);
            if (other > best || (other == best && at < idx)) {
                best = other;
                idx = at;
            }
        }
        if (k == 0) top = best;
        const float e = expf(best - top);
        total += e;
        if (lane == 0) {
            pick[row * slots + k] = idx;
            ex[k] = e;
            v[idx] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane == 0) {
        for (int k = 0; k < top_k; ++k) wts[row * slots + k] = round_bf16(ex[k] / total);
        const float gate = round_bf16(l[experts]);
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = lane; k < capacity; k += 32) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

template <typename T>
__global__ void moe_act_kernel(const float* __restrict__ both, T* __restrict__ out, int width, float limit,
                               long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long p = i / width;
    const int c = static_cast<int>(i - p * width);
    float g = both[p * 2 * width + c];
    float u = both[p * 2 * width + width + c];
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    narrow(g / (1.f + expf(-g)) * u, out + i);
}

template <typename T>
__global__ void moe_combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, T* __restrict__ out,
                                   int slots, int d) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float acc = 0.f;
    for (int s = 0; s < slots; ++s) {
        acc += y[(static_cast<long long>(row) * slots + s) * d + c] * wts[row * slots + s];
    }
    narrow(acc, out + static_cast<long long>(row) * d + c);
}

template <typename T>
__global__ void gdn_gate_kernel(const T* __restrict__ a, const T* __restrict__ b, const float* __restrict__ a_log,
                                const float* __restrict__ dt_bias, float* __restrict__ gate, float* __restrict__ beta,
                                int count, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int h = i % heads;
    beta[i] = 1.f / (1.f + expf(-widen(b[i])));
    const float x = widen(a[i]) + dt_bias[h];
    const float soft = x > 20.f ? x : log1pf(expf(x));
    gate[i] = expf(-expf(a_log[h]) * soft);
}

}  // namespace rocm
}  // namespace tf

void rms_launch(const void* x, const float* weight, void* y, int kind, int rows, int width, float eps,
                hipStream_t stream) {
    if (kind == 1) {
        tf::rocm::rms_kernel<__half><<<rows, tf::rocm::kRmsThreads, 0, stream>>>(static_cast<const __half*>(x), weight,
                                                               static_cast<__half*>(y), width, eps);
    } else if (kind == 2) {
        tf::rocm::rms_kernel<hip_bfloat16><<<rows, tf::rocm::kRmsThreads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(x), weight,
                                                                     static_cast<hip_bfloat16*>(y), width, eps);
    } else {
        tf::rocm::rms_kernel<float><<<rows, tf::rocm::kRmsThreads, 0, stream>>>(static_cast<const float*>(x), weight,
                                                              static_cast<float*>(y), width, eps);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("rms kernel: ") + hipGetErrorString(err));
}

void conv_decode_launch(const float* x, const float* weight, float* state, float* y, int batch, int channels,
                        int kernel, hipStream_t stream) {
    dim3 block(128);
    dim3 grid((channels + 127) / 128, batch);
    tf::rocm::conv_decode_kernel<<<grid, block, 0, stream>>>(x, weight, state, y, channels, kernel);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("conv kernel: ") + hipGetErrorString(err));
}

void conv_prefill_launch(const void* x, int kind, const float* weight, const float* state, float* y, int batch,
                         int length, int channels, int kernel, hipStream_t stream) {
    dim3 block(128);
    dim3 grid((channels + 127) / 128, length, batch);
    if (kind == 2) {
        tf::rocm::conv_prefill_kernel<<<grid, block, 0, stream>>>(static_cast<const hip_bfloat16*>(x), weight, state,
                                                                   y, length, channels, kernel);
    } else if (kind == 1) {
        tf::rocm::conv_prefill_kernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(x), weight, state, y,
                                                                   length, channels, kernel);
    } else {
        tf::rocm::conv_prefill_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(x), weight, state, y,
                                                                   length, channels, kernel);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("conv prefill kernel: ") + hipGetErrorString(err));
}

void rope_decode_launch(const float* x, float* y, int rows, int width, int rotary, int pos, float theta,
                        hipStream_t stream, const int* pos_dev) {
    tf::rocm::rope_decode_kernel<<<rows, 32, 0, stream>>>(x, y, width, rotary, pos, theta, pos_dev);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("rope kernel: ") + hipGetErrorString(err));
}

template <typename T>
static bool router_rows_launch(const void* x, const float* rows, float* logits, int r, int d, int e,
                               hipStream_t stream) {
    const dim3 grid((e + 7) / 8, (r + tf::rocm::kRouterTile - 1) / tf::rocm::kRouterTile);
    const T* xt = static_cast<const T*>(x);
    switch (d) {
        case 1024: tf::rocm::moe_router_rows<T, 32><<<grid, 256, 0, stream>>>(xt, rows, logits, d, e, r); return true;
        case 2048: tf::rocm::moe_router_rows<T, 64><<<grid, 256, 0, stream>>>(xt, rows, logits, d, e, r); return true;
        case 4096: tf::rocm::moe_router_rows<T, 128><<<grid, 256, 0, stream>>>(xt, rows, logits, d, e, r); return true;
        default: return false;
    }
}

void moe_router_launch(const void* x, int kind, const float* rows, float* logits, int r, int d, int e,
                       hipStream_t stream) {
    if (kind == 1 ? router_rows_launch<__half>(x, rows, logits, r, d, e, stream)
                  : router_rows_launch<hip_bfloat16>(x, rows, logits, r, d, e, stream)) {
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) throw std::runtime_error(std::string("moe router kernel: ") + hipGetErrorString(err));
        return;
    }
    dim3 grid((e + 7) / 8, r);
    if (kind == 1) {
        tf::rocm::moe_router_kernel<__half><<<grid, 256, 0, stream>>>(static_cast<const __half*>(x), rows, logits, d,
                                                                       e);
    } else {
        tf::rocm::moe_router_kernel<hip_bfloat16><<<grid, 256, 0, stream>>>(static_cast<const hip_bfloat16*>(x), rows,
                                                                           logits, d, e);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("moe router kernel: ") + hipGetErrorString(err));
}

void moe_select_launch(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int r,
                       int experts, int top_k, hipStream_t stream) {
    if (experts > tf::rocm::kSelectMax || top_k < 1 || top_k > 31) throw std::runtime_error("moe select: shape");
    tf::rocm::moe_select_kernel<<<r, 32, 0, stream>>>(logits, pick, wts, items, members, capacity, experts, top_k);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("moe select kernel: ") + hipGetErrorString(err));
}

void moe_act_launch(const float* both, void* out, int kind, int pairs, int width, float limit, hipStream_t stream) {
    const long long total = static_cast<long long>(pairs) * width;
    const int blocks = static_cast<int>((total + 255) / 256);
    if (kind == 1) {
        tf::rocm::moe_act_kernel<__half><<<blocks, 256, 0, stream>>>(both, static_cast<__half*>(out), width, limit,
                                                                     total);
    } else {
        tf::rocm::moe_act_kernel<hip_bfloat16><<<blocks, 256, 0, stream>>>(both, static_cast<hip_bfloat16*>(out), width,
                                                                           limit, total);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("moe act kernel: ") + hipGetErrorString(err));
}

void moe_combine_launch(const float* y, const float* wts, void* out, int kind, int r, int slots, int d,
                        hipStream_t stream) {
    dim3 grid((d + 255) / 256, r);
    if (kind == 1) {
        tf::rocm::moe_combine_kernel<__half><<<grid, 256, 0, stream>>>(y, wts, static_cast<__half*>(out), slots, d);
    } else if (kind == 2) {
        tf::rocm::moe_combine_kernel<hip_bfloat16><<<grid, 256, 0, stream>>>(y, wts, static_cast<hip_bfloat16*>(out),
                                                                             slots, d);
    } else {
        tf::rocm::moe_combine_kernel<float><<<grid, 256, 0, stream>>>(y, wts, static_cast<float*>(out), slots, d);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("moe combine kernel: ") + hipGetErrorString(err));
}

void gdn_gate_launch(const void* a, const void* b, int kind, const float* a_log, const float* dt_bias, float* gate,
                     float* beta, int count, int heads, hipStream_t stream) {
    const int blocks = (count + 255) / 256;
    if (kind == 1) {
        tf::rocm::gdn_gate_kernel<__half><<<blocks, 256, 0, stream>>>(static_cast<const __half*>(a),
                                                                      static_cast<const __half*>(b), a_log, dt_bias,
                                                                      gate, beta, count, heads);
    } else if (kind == 2) {
        tf::rocm::gdn_gate_kernel<hip_bfloat16><<<blocks, 256, 0, stream>>>(
            static_cast<const hip_bfloat16*>(a), static_cast<const hip_bfloat16*>(b), a_log, dt_bias, gate, beta, count,
            heads);
    } else {
        tf::rocm::gdn_gate_kernel<float><<<blocks, 256, 0, stream>>>(static_cast<const float*>(a),
                                                                     static_cast<const float*>(b), a_log, dt_bias,
                                                                     gate, beta, count, heads);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("gdn gate kernel: ") + hipGetErrorString(err));
}
