// RDNA4's FP8 prompt matmul: prefill_glue's e4m3 rows times the weight's 4-bit codes as e4m3 (qmm_groups._nibbles8),
// on v_wmma_f32_16x16x16_fp8_fp8. out = a * (sum over groups, in order, of s * dot64(x8, w8) + xs . b): the Triton
// _gemm8's arithmetic, which Triton 3.6-3.8 lowers to at most ~130 of the fp8 WMMA's ~325 TFLOPS on gfx1201.
//
// A row's bits depend only on its own inputs: every output runs the same WMMA sequence (fixed 16-input steps, groups
// in order, one epilogue) whatever tile, block or call holds it, so any prompt chunking gives the same bits.
//
// Credits:
// - The arithmetic it implements (e4m3 rows with group sums and a row scale, codes exact in e4m3, per-group scales
//   and the bias on the group sums) is TensorFold's FP8 prompt path by Ash Hart (github.com/ashhart/TensorFold:
//   prefill_glue.py, qmm.prefill_matmul8), as ported to RDNA4 in Triton by jkuepker (github.com/jkuepker/TensorFold,
//   PR ashhart/TensorFold#100: qmm_groups._nibbles8 and _gemm8, whose grouped tile order this kernel keeps).
// - The group-major weight layout (tile_at), the gfx12 WMMA fragment mapping (lane c = row/column, lane half h = 8 of
//   a step's 16 inputs) and bf16_value/bf16_round come from jkuepker's qmm_rocm.cu and attention_rocm.cu in the same
//   PR.
// - Ideas, not code, from vLLM-radiance (codeberg.org/StillDeadcode/vllm-radiance, mirrored at
//   github.com/magiccodingman/vllm-radiance; radiance_mxfp4_fp8.hip): that hand-issued fp8 WMMA is the way past
//   Triton's fp8 dot on gfx1201, staging operands through LDS rather than reading fragments from global memory, and
//   padding LDS rows by 8 bytes so 16 lanes reading rows 64 bytes apart stop colliding on banks. That repository
//   carries no license, so none of its code is copied here.

#ifndef __HIPCC__
#error "qmm8_rocm.cu is RDNA4's FP8 prompt matmul"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

namespace {

typedef int int2v __attribute__((ext_vector_type(2)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int GK = 64;             // inputs a group (one scale and bias)

__device__ __forceinline__ float bf16_value(unsigned short v) {
    return __uint_as_float(static_cast<unsigned>(v) << 16);
}

__device__ __forceinline__ unsigned short bf16_round(float f) {
    const unsigned u = __float_as_uint(f);
    return static_cast<unsigned short>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Group-major layout (qmm_groups.to_groups): the scale or bias of output col, group g.
__device__ __forceinline__ size_t tile_at(int g, int kg, int col) {
    return (static_cast<size_t>(col >> 4) * kg + g) * 16 + (col & 15);
}

// A block: BM x BN outputs, WM x WN waves (each MT x NT fragments of 16 x 16), GPS groups of inputs staged at once
// (one barrier a stage), NB LDS buffers. LDS rows are padded 8 bytes so 16 lanes reading 8 bytes from 16 rows do not
// share banks. The tiling never changes a row's arithmetic: every output folds the same 16-input WMMA steps, groups
// in order, so the variants differ in speed only.
template <int BM, int BN, int WM, int WN, int GPS, int NB>
struct Cfg {
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int MT = BM / WM / 16, NT = BN / WN / 16;
    static constexpr int SK = GPS * GK;                 // inputs a stage
    static constexpr int PITCH = SK + 8;
    static constexpr int XP = BM * SK / 16 / THREADS;   // 16-byte pieces a thread loads a stage
    static constexpr int WP = BN * SK / 16 / THREADS;
    static_assert(BM * SK % (16 * THREADS) == 0 && BN * SK % (16 * THREADS) == 0, "whole pieces a thread");
};

template <int ROWS, int SK, int THREADS, int P>
__device__ __forceinline__ void fetch(uint4 (&v)[P], const uint8_t* __restrict__ src, int rows, int row0, int k,
                                      int k0) {
    constexpr int PER = SK / 16;                         // pieces a row
#pragma unroll
    for (int i = 0; i < P; ++i) {
        const int at = threadIdx.x + i * THREADS;
        // past the last row: read the last row again (branch-free loads keep the load counter waits exact); a padding
        // row only feeds outputs that are never written, since each output reads its own row and column alone
        const int r = min(row0 + at / PER, rows - 1), kk = min(k0 + (at % PER) * 16, k - 16);
        v[i] = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * k + kk);
    }
}

template <int SK, int PITCH, int THREADS, int P>
__device__ __forceinline__ void stage(uint8_t* lds, const uint4 (&v)[P]) {
    constexpr int PER = SK / 16;
#pragma unroll
    for (int i = 0; i < P; ++i) {
        const int at = threadIdx.x + i * THREADS;
        uint2* dst = reinterpret_cast<uint2*>(lds + (at / PER) * PITCH + (at % PER) * 16);   // 8-byte aligned rows
        dst[0] = make_uint2(v[i].x, v[i].y);
        dst[1] = make_uint2(v[i].z, v[i].w);
    }
}

// A workgroup barrier that waits only for LDS traffic: __syncthreads() also drains the block's outstanding global
// loads (the next stage's prefetch) at every stage. The idea is radiance's (radiance_mxfp4_fp8.hip), the code ours.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

template <bool LDSBAR>
__device__ __forceinline__ void barrier() {
    if constexpr (LDSBAR) lds_barrier();
    else __syncthreads();
}

template <int BM, int BN, int WM, int WN, int GPS, int NB, bool LDSBAR>
__global__ void __launch_bounds__(32 * WM * WN) gemm8_kernel(
        const uint8_t* __restrict__ x8, const unsigned short* __restrict__ xs, const float* __restrict__ a,
        const uint8_t* __restrict__ w8, const unsigned short* __restrict__ scales,
        const unsigned short* __restrict__ biases, int m, int n, int k, int group, void* __restrict__ out, bool f32) {
    using C = Cfg<BM, BN, WM, WN, GPS, NB>;
    constexpr int MT = C::MT, NT = C::NT, SK = C::SK, PITCH = C::PITCH, T = C::THREADS;
    __shared__ uint8_t xl[NB][BM * PITCH];
    __shared__ uint8_t wl[NB][BN * PITCH];
    const int kg = k / GK, stages = (kg + GPS - 1) / GPS;
    // grouped order: `group` row blocks at a time down each column block, so their inputs and weights meet in cache
    const int blocks_m = (m + BM - 1) / BM, blocks_n = (n + BN - 1) / BN;
    const int pid = blockIdx.x, per = group * blocks_n, first = (pid / per) * group;
    const int size = min(blocks_m - first, group);
    const int row0 = (first + (pid % per) % size) * BM, col0 = ((pid % per) / size) * BN;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int h = lane >> 4, c = lane & 15;
    const int wr = (wave / WN) * (MT * 16), wc = (wave % WN) * (NT * 16);   // the wave's corner in the tile

    float8 acc[MT][NT];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int u = 0; u < NT; ++u) acc[t][u] = float8{0, 0, 0, 0, 0, 0, 0, 0};

    uint4 xn[C::XP], wn[C::WP];
    fetch<BM, SK, T>(xn, x8, m, row0, k, 0);
    fetch<BN, SK, T>(wn, w8, n, col0, k, 0);
    for (int st = 0; st < stages; ++st) {
        const int buf = NB == 1 ? 0 : st % NB;
        if (NB == 1) barrier<LDSBAR>();                  // one buffer: the last stage's reads are done
        stage<SK, PITCH, T>(xl[buf], xn);
        stage<SK, PITCH, T>(wl[buf], wn);
        barrier<LDSBAR>();                               // staged (with NB 2, also: the buffer before it is free)
        // this stage's scales first, then the next stage's operands: the scales' wait (a load counter, in order)
        // then leaves the prefetch in flight under the WMMAs
        unsigned short sc16[GPS][NT];
#pragma unroll
        for (int j = 0; j < GPS; ++j)
#pragma unroll
            for (int u = 0; u < NT; ++u) {
                const int col = min(col0 + wc + 16 * u + c, n - 1), g = min(st * GPS + j, kg - 1);
                sc16[j][u] = scales[tile_at(g, kg, col)];
            }
        // the next stage's reads fly under this one's WMMAs; after the last stage they re-read its slab (unused): a
        // branch here would join two paths, and the compiler then drains every load before the WMMAs
        fetch<BM, SK, T>(xn, x8, m, row0, k, (st + 1) * SK);
        fetch<BN, SK, T>(wn, w8, n, col0, k, (st + 1) * SK);
        asm volatile("" ::: "memory");                   // keep them issued here: the compiler sinks them to their use
#pragma unroll
        for (int j = 0; j < GPS; ++j) {
            const int g = st * GPS + j;
            if (g >= kg) break;
            float8 p[MT][NT];
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int u = 0; u < NT; ++u) p[t][u] = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < GK / 16; ++s) {          // 16 inputs a step: lane half h holds 8 of them
                const int kk = j * GK + 16 * s + 8 * h;
                int2v af[MT], bf[NT];
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const uint2 v = *reinterpret_cast<const uint2*>(xl[buf] + (wr + 16 * t + c) * PITCH + kk);
                    af[t] = int2v{static_cast<int>(v.x), static_cast<int>(v.y)};
                }
#pragma unroll
                for (int u = 0; u < NT; ++u) {
                    const uint2 v = *reinterpret_cast<const uint2*>(wl[buf] + (wc + 16 * u + c) * PITCH + kk);
                    bf[u] = int2v{static_cast<int>(v.x), static_cast<int>(v.y)};
                }
#pragma unroll
                for (int t = 0; t < MT; ++t)
#pragma unroll
                    for (int u = 0; u < NT; ++u)
                        p[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[t], bf[u], p[t][u]);
            }
#pragma unroll
            for (int u = 0; u < NT; ++u) {               // the group's scale, then on to the next group
                const float sc = bf16_value(sc16[j][u]);
#pragma unroll
                for (int t = 0; t < MT; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[t][u][i] = fmaf(p[t][u][i], sc, acc[t][u][i]);
            }
        }
    }
    // the bias on the group sums: (rows x groups) . (groups x cols) on bf16 WMMA, 16 groups a step, in order
    for (int g0 = 0; g0 < kg; g0 += 16) {
        short8 bx[NT];
#pragma unroll
        for (int u = 0; u < NT; ++u) {
            const int col = col0 + wc + 16 * u + c;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int g = g0 + 8 * h + j;
                bx[u][j] = col < n && g < kg ? static_cast<short>(biases[tile_at(g, kg, col)]) : short(0);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int row = row0 + wr + 16 * t + c;
            short8 ax;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int g = g0 + 8 * h + j;
                ax[j] = row < m && g < kg ? static_cast<short>(xs[static_cast<size_t>(row) * kg + g]) : short(0);
            }
#pragma unroll
            for (int u = 0; u < NT; ++u)
                acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ax, bx[u], acc[t][u]);
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + wr + 16 * t + 8 * h + i;
            if (row >= m) break;
            const float scale = a[row];
#pragma unroll
            for (int u = 0; u < NT; ++u) {
                const int col = col0 + wc + 16 * u + c;
                if (col >= n) continue;
                const size_t at = static_cast<size_t>(row) * n + col;
                const float v = acc[t][u][i] * scale;
                if (f32) static_cast<float*>(out)[at] = v;
                else static_cast<unsigned short*>(out)[at] = bf16_round(v);
            }
        }
}

template <int BM, int BN, int WM, int WN, int GPS, int NB, bool LDSBAR = false>
void launch(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
            const at::Tensor& scales, const at::Tensor& biases, int m, int n, int k, int group, at::Tensor& out) {
    const int blocks = ((m + BM - 1) / BM) * ((n + BN - 1) / BN);
    gemm8_kernel<BM, BN, WM, WN, GPS, NB, LDSBAR><<<blocks, 32 * WM * WN, 0, at::cuda::getCurrentCUDAStream()>>>(
        x8.data_ptr<uint8_t>(), reinterpret_cast<const unsigned short*>(xs.data_ptr()), a.data_ptr<float>(),
        w8.data_ptr<uint8_t>(), reinterpret_cast<const unsigned short*>(scales.data_ptr()),
        reinterpret_cast<const unsigned short*>(biases.data_ptr()), m, n, k, group, out.data_ptr(),
        out.scalar_type() == at::kFloat);
}

}  // namespace

// variant: the tiling (speed only, never bits); 0 is the default
void gemm8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
           const at::Tensor& scales, const at::Tensor& biases, int n, int group, at::Tensor& out, int variant) {
    const int m = static_cast<int>(x8.size(0)), k = static_cast<int>(x8.size(1));
    TORCH_CHECK(k % GK == 0 && w8.size(1) == k && w8.size(0) >= n, "gemm8: (M, K) e4m3 rows against (N, K) codes");
    TORCH_CHECK(xs.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kFloat, "gemm8: bf16 group sums, fp32 a");
    TORCH_CHECK(x8.is_contiguous() && w8.is_contiguous() && xs.is_contiguous() && out.is_contiguous(),
                "gemm8 takes contiguous tensors");
    if (m == 0 || n == 0) return;
    switch (variant) {
        case 0: launch<128, 128, 2, 4, 1, 2>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 1: launch<128, 128, 2, 4, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 2: launch<128, 128, 2, 4, 2, 1, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 3: launch<256, 128, 4, 4, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 8: launch<128, 128, 4, 2, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        default: TORCH_CHECK(false, "gemm8: unknown variant");
    }
}
