// Gated DeltaNet: a wave per (batch, value head, value row) walks the sequence, lane sums as the Python reference.

#include "gated_delta.hpp"
#include "wave.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace tf {
namespace rocm {

// Every lane of a wave enters each sum.
__device__ inline float lane0_sum(float x) { return wave_lane0_sum(x); }

template <int DK>
__global__ void __launch_bounds__(32)
gated_delta_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                   const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                   float* __restrict__ y, int length, int key_heads, int value_heads, int dv) {
    constexpr int SPAN = (DK + 31) / 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (row >= dv) return;
    const int key_head = head / (value_heads / key_heads);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row =
        ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t = 0; t < length; ++t) {
        const size_t step = (static_cast<size_t>(batch) * length + t) * value_heads + head;
        const float decay = gate[step];
        const float step_beta = beta[step];
        const float value = v[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
                              static_cast<size_t>(head) * dv + row];
        const size_t key =
            ((static_cast<size_t>(batch) * length + t) * key_heads + key_head) * DK;

        float remembered = 0.f;
        float ks[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) {
                ks[i] = k[key + idx];
                s[i] = s[i] * decay;
                remembered = remembered + s[i] * ks[i];
            }
        }
        const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) s[i] = s[i] + ks[i] * delta;
        }

        float readout = 0.f;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) readout = readout + s[i] * q[key + idx];
        }
        // Every lane has to enter the shuffle. Lane 0 owns the stored sum.
        readout = lane0_sum(readout);
        if (lane == 0) {
            y[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
              static_cast<size_t>(head) * dv + row] = readout;
        }
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

// Value rows share a staged tile of k and q, in the wave kernel's order (-ffp-contract=off keeps it).
template <int DK, int ROWS, int STEPS>
__global__ void __launch_bounds__(32 * ROWS)
gated_delta_tile(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                 const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                 float* __restrict__ y, int length, int key_heads, int value_heads, int dv) {
    constexpr int SPAN = (DK + 31) / 32;
    constexpr int WIDTH = 32 * ROWS;
    const int lane = threadIdx.x;
    const int local = threadIdx.y;
    const int row = blockIdx.x * ROWS + local;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int key_head = head / (value_heads / key_heads);
    const int tid = local * 32 + lane;

    __shared__ float ks[STEPS][DK];
    __shared__ float qs[STEPS][DK];
    __shared__ float gs[STEPS];
    __shared__ float bs[STEPS];
    __shared__ float vs[STEPS][ROWS];

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row = ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t0 = 0; t0 < length; t0 += STEPS) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
        for (int i = tid; i < nvec; i += WIDTH) {
            const int elem = i * 4;
            const int step = elem / DK;
            const int col = elem - step * DK;
            const size_t at =
                ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + col;
            const float4 key = *reinterpret_cast<const float4*>(k + at);
            const float4 query = *reinterpret_cast<const float4*>(q + at);
            ks[step][col] = key.x;
            ks[step][col + 1] = key.y;
            ks[step][col + 2] = key.z;
            ks[step][col + 3] = key.w;
            qs[step][col] = query.x;
            qs[step][col + 1] = query.y;
            qs[step][col + 2] = query.z;
            qs[step][col + 3] = query.w;
        }
        for (int i = tid; i < n; i += WIDTH) {
            const size_t step = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[step];
            bs[i] = beta[step];
        }
        for (int i = tid; i < n * ROWS; i += WIDTH) {
            const int step = i / ROWS;
            const int which = i - step * ROWS;
            vs[step][which] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * dv +
                                blockIdx.x * ROWS + which];
        }
        __syncthreads();

        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt];
            const float step_beta = bs[tt];
            const float value = vs[tt][local];
            float remembered = 0.f;
            float held[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) {
                    held[i] = ks[tt][idx];
                    s[i] = s[i] * decay;
                    remembered = remembered + s[i] * held[i];
                }
            }
            const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) s[i] = s[i] + held[i] * delta;
            }
            float readout = 0.f;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) readout = readout + s[i] * qs[tt][idx];
            }
            readout = lane0_sum(readout);
            if (lane == 0) {
                y[((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row] = readout;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

}  // namespace rocm
}  // namespace tf

void gated_delta_launch(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* state, float* y, int batch, int length, int key_heads, int value_heads, int dk,
                        int dv, void* stream) {
    if ((dk != 16 && dk != 128) || dv < 1 || key_heads < 1 || value_heads % key_heads != 0 || batch < 1 ||
        length < 1) {
        throw std::invalid_argument("gated delta wants dk 16 or 128 and value heads a multiple of key heads");
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err;
    // Eight rows share one k/q tile. A short dv stays on the one-row wave, which is the same scan.
    if (dv % 8 == 0) {
        dim3 block(32, 8);
        dim3 grid(dv / 8, value_heads, batch);
        if (dk == 128) {
            tf::rocm::gated_delta_tile<128, 8, 16><<<grid, block, 0, hip_stream>>>(
                q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv);
        } else {
            tf::rocm::gated_delta_tile<16, 8, 16><<<grid, block, 0, hip_stream>>>(
                q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv);
        }
    } else if (dk == 128) {
        dim3 block(32);
        dim3 grid(dv, value_heads, batch);
        tf::rocm::gated_delta_kernel<128><<<grid, block, 0, hip_stream>>>(
            q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv);
    } else {
        dim3 block(32);
        dim3 grid(dv, value_heads, batch);
        tf::rocm::gated_delta_kernel<16><<<grid, block, 0, hip_stream>>>(
            q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv);
    }
    err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("gated delta kernel: ") + hipGetErrorString(err));
}
