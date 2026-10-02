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

// The 4-bit codes of 16 inputs (words w0 = inputs 0-7 and w1 = 8-15 of a 16-input block, nibble of input i at
// 4 * (i % 8 / 2) + 16 * (i % 2)) as e4m3 bytes in prefill_glue's stored order, where position 4a + j holds input
// 2a + j % 2 + 8 * (j / 2): exactly _nibbles8's bytes, so the products and their bits are unchanged.
__device__ __forceinline__ uint4 codes16(unsigned w0, unsigned w1) {
    // e4m3 of 0..7 and of 8..15, a byte each, for v_perm_b32's byte select
    constexpr unsigned LO0 = 0x44403800u, LO1 = 0x4E4C4A48u, HI0 = 0x53525150u, HI1 = 0x57565554u;
    unsigned out[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const unsigned x = (w0 >> (4 * a)) & 0x000F000Fu, y = (w1 >> (4 * a)) & 0x000F000Fu;
        const unsigned q = __builtin_amdgcn_perm(y, x, 0x06040200u);    // bytes: inputs 2a, 2a+1, 2a+8, 2a+9
        const unsigned sel = q & 0x07070707u;
        const unsigned lo = __builtin_amdgcn_perm(LO1, LO0, sel), hi = __builtin_amdgcn_perm(HI1, HI0, sel);
        const unsigned big = ((q >> 3) & 0x01010101u) * 0xFFu;          // 0xFF in each byte whose code is 8 or more
        out[a] = (lo & ~big) | (hi & big);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}

// A stage's weight codes straight from the group-major words: thread t takes column t / 2, half t % 2 (32 inputs:
// four words, 16 bytes), so 16 columns read 512 contiguous bytes. Columns past n re-read the last one; in a block of
// more than 2 BN threads the rest repeat the first ones' reads (no branch) and stage nothing.
template <int BN, int THREADS>
__device__ __forceinline__ uint4 fetch_words(const unsigned* __restrict__ words, int n, int col0, int kg, int g) {
    static_assert(THREADS % (BN * 2) == 0, "whole half columns a thread");
    const int t = threadIdx.x % (BN * 2);
    const int col = min(col0 + (t >> 1), n - 1), half = t & 1;
    return *reinterpret_cast<const uint4*>(words + tile_at(min(g, kg - 1), kg, col) * 8 + 4 * half);
}

template <int BN, int PITCH>
__device__ __forceinline__ void stage_words(uint8_t* lds, const uint4& v) {
    if (threadIdx.x >= BN * 2) return;
    const int col = threadIdx.x >> 1, half = threadIdx.x & 1;
    const uint4 b0 = codes16(v.x, v.y), b1 = codes16(v.z, v.w);
    uint2* dst = reinterpret_cast<uint2*>(lds + col * PITCH + 32 * half);
    dst[0] = make_uint2(b0.x, b0.y);
    dst[1] = make_uint2(b0.z, b0.w);
    dst[2] = make_uint2(b1.x, b1.y);
    dst[3] = make_uint2(b1.z, b1.w);
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

// W4: the weight operand is the group-major 4-bit words, widened to e4m3 while staged (one stage = one group);
// otherwise it is _nibbles8's (N, K) e4m3 codes.
template <int BM, int BN, int WM, int WN, int GPS, int NB, bool LDSBAR, bool W4>
__global__ void __launch_bounds__(32 * WM * WN) gemm8_kernel(
        const uint8_t* __restrict__ x8, const unsigned short* __restrict__ xs, const float* __restrict__ a,
        const void* __restrict__ wsrc, const unsigned short* __restrict__ scales,
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

    static_assert(!W4 || GPS == 1, "W4 stages one group at a time");
    const uint8_t* w8 = static_cast<const uint8_t*>(wsrc);
    const unsigned* words = static_cast<const unsigned*>(wsrc);
    uint4 xn[C::XP], wn[C::WP], wq;
    fetch<BM, SK, T>(xn, x8, m, row0, k, 0);
    if constexpr (W4) wq = fetch_words<BN, T>(words, n, col0, kg, 0);
    else fetch<BN, SK, T>(wn, w8, n, col0, k, 0);
    for (int st = 0; st < stages; ++st) {
        const int buf = NB == 1 ? 0 : st % NB;
        if (NB == 1) barrier<LDSBAR>();                  // one buffer: the last stage's reads are done
        stage<SK, PITCH, T>(xl[buf], xn);
        if constexpr (W4) stage_words<BN, PITCH>(wl[buf], wq);
        else stage<SK, PITCH, T>(wl[buf], wn);
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
        if constexpr (W4) wq = fetch_words<BN, T>(words, n, col0, kg, st + 1);
        else fetch<BN, SK, T>(wn, w8, n, col0, k, (st + 1) * SK);
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

template <int BM, int BN, int WM, int WN, int GPS, int NB, bool LDSBAR = false, bool W4 = false>
void launch(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
            const at::Tensor& scales, const at::Tensor& biases, int m, int n, int k, int group, at::Tensor& out) {
    const int blocks = ((m + BM - 1) / BM) * ((n + BN - 1) / BN);
    gemm8_kernel<BM, BN, WM, WN, GPS, NB, LDSBAR, W4><<<blocks, 32 * WM * WN, 0, at::cuda::getCurrentCUDAStream()>>>(
        x8.data_ptr<uint8_t>(), reinterpret_cast<const unsigned short*>(xs.data_ptr()), a.data_ptr<float>(),
        w8.data_ptr(), reinterpret_cast<const unsigned short*>(scales.data_ptr()),
        reinterpret_cast<const unsigned short*>(biases.data_ptr()), m, n, k, group, out.data_ptr(),
        out.scalar_type() == at::kFloat);
}

// ------------------------------------------------------------------------------------------------- tiled activations
// The prompt rows arrive in WMMA fragment order (prefill_glue.py with TILED: 16 rows x 16 positions a 256-byte
// fragment, lane l's 8 bytes at 8 l), so each lane loads its A fragment straight from global memory into the register
// the WMMA reads, and only the weights go through LDS. Ideas from radiance's A-tiled prefill kernel (its notes measure
// staging A through LDS at a quarter of the run time): fragment-ordered activations, wave-uniform bases in scalar
// registers with 32-bit lane offsets, LDS-only barriers, a 256-row tile. The products, their order and the epilogue
// are the staged kernel's, so are the bits.
__device__ __forceinline__ void lds_fence_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// TN 16-column fragments a wave (BN = 32 TN), GPS groups a slab, SEQ: one partial sum live at a time.
template <int TN, int GPS, bool SEQ>
__global__ void __launch_bounds__(256) gemm8_tiled_kernel(
        const uint8_t* __restrict__ x8t, const unsigned short* __restrict__ xs, const float* __restrict__ a,
        const unsigned* __restrict__ words, const unsigned short* __restrict__ scales,
        const unsigned short* __restrict__ biases, int m, int n, int k, void* __restrict__ out, bool f32) {
    constexpr int WN = 2, TM = 4, BM = 256, BN = WN * TN * 16;
    constexpr int LBK = GPS * GK, NS = LBK / 16, PITCH = LBK + 8;
    constexpr int PIECES = BN * GPS * 2, P = (PIECES + 255) / 256;   // 16-byte word pieces a slab, a thread
    __shared__ uint8_t wl[BN * PITCH];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave / WN, wn = wave % WN;
    const int h = lane >> 4, c = lane & 15;
    const int kg = k / GK, ksteps = k / 16, mt_last = (m + 15) / 16 - 1;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = wm * TM * 16, wc = wn * TN * 16;

    const uint8_t* abase[TM];                            // a wave's fragment rows: uniform, in scalar registers
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mt = min(row0 / 16 + wm * TM + i, mt_last);   // past m: the last tile again (outputs dropped)
        abase[i] = x8t + __builtin_amdgcn_readfirstlane(mt * ksteps * 256);
    }
    const unsigned aoff = lane * 8;
    int colc[TN];
#pragma unroll
    for (int u = 0; u < TN; ++u) colc[u] = min(col0 + wc + 16 * u + c, n - 1);

    float8 acc[TM][TN];
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int u = 0; u < TN; ++u) acc[t][u] = float8{0, 0, 0, 0, 0, 0, 0, 0};

    const int stages = (kg + GPS - 1) / GPS;
    for (int st = 0; st < stages; ++st) {
        // the weights first: staging waits on them alone (the load counter is in order), the A fragments land later
        uint4 wv[P];
#pragma unroll
        for (int it = 0; it < P; ++it) {
            const int q = min(tid + it * 256, PIECES - 1);
            const int j = q / (BN * 2), r = q % (BN * 2);
            const int g = min(st * GPS + j, kg - 1), col = min(col0 + (r >> 1), n - 1);
            wv[it] = *reinterpret_cast<const uint4*>(words + tile_at(g, kg, col) * 8 + 4 * (r & 1));
        }
        int2v af[TM][NS];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const unsigned ks = min(st * NS + s, ksteps - 1);
                af[i][s] = *reinterpret_cast<const int2v*>(abase[i] + (aoff + ks * 256u));
            }
        unsigned short sc16[GPS][TN];
#pragma unroll
        for (int j = 0; j < GPS; ++j)
#pragma unroll
            for (int u = 0; u < TN; ++u) sc16[j][u] = scales[tile_at(min(st * GPS + j, kg - 1), kg, colc[u])];
        lds_fence_barrier();                             // the last slab's weight reads are done
#pragma unroll
        for (int it = 0; it < P; ++it) {
            const int q = tid + it * 256;
            if (q < PIECES) {
                const int j = q / (BN * 2), r = q % (BN * 2);
                const uint4 b0 = codes16(wv[it].x, wv[it].y), b1 = codes16(wv[it].z, wv[it].w);
                uint2* dst = reinterpret_cast<uint2*>(wl + (r >> 1) * PITCH + j * GK + 32 * (r & 1));
                dst[0] = make_uint2(b0.x, b0.y);
                dst[1] = make_uint2(b0.z, b0.w);
                dst[2] = make_uint2(b1.x, b1.y);
                dst[3] = make_uint2(b1.z, b1.w);
            }
        }
        lds_fence_barrier();                             // staged
#pragma unroll
        for (int j = 0; j < GPS; ++j) {
            if (st * GPS + j >= kg) break;
            int2v bf[GK / 16][TN];
#pragma unroll
            for (int s = 0; s < GK / 16; ++s)
#pragma unroll
                for (int u = 0; u < TN; ++u) {
                    const uint2 v = *reinterpret_cast<const uint2*>(wl + (wc + 16 * u + c) * PITCH + j * GK + 16 * s + 8 * h);
                    bf[s][u] = int2v{static_cast<int>(v.x), static_cast<int>(v.y)};
                }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (SEQ) {
#pragma unroll
                for (int u = 0; u < TN; ++u) {
                    const float sc = bf16_value(sc16[j][u]);
#pragma unroll
                    for (int t = 0; t < TM; ++t) {
                        float8 p = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                        for (int s = 0; s < GK / 16; ++s)
                            p = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[t][j * 4 + s], bf[s][u], p);
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[t][u][e] = fmaf(p[e], sc, acc[t][u][e]);
                    }
                }
            } else {
                float8 p[TM][TN];
#pragma unroll
                for (int t = 0; t < TM; ++t)
#pragma unroll
                    for (int u = 0; u < TN; ++u) p[t][u] = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < GK / 16; ++s)
#pragma unroll
                    for (int t = 0; t < TM; ++t)
#pragma unroll
                        for (int u = 0; u < TN; ++u)
                            p[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[t][j * 4 + s], bf[s][u],
                                                                                           p[t][u]);
#pragma unroll
                for (int u = 0; u < TN; ++u) {
                    const float sc = bf16_value(sc16[j][u]);
#pragma unroll
                    for (int t = 0; t < TM; ++t)
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[t][u][e] = fmaf(p[t][u][e], sc, acc[t][u][e]);
                }
            }
        }
    }
    // the bias on the group sums, as the staged kernel
    for (int g0 = 0; g0 < kg; g0 += 16) {
        short8 bx[TN];
#pragma unroll
        for (int u = 0; u < TN; ++u) {
            const int col = col0 + wc + 16 * u + c;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int g = g0 + 8 * h + j;
                bx[u][j] = col < n && g < kg ? static_cast<short>(biases[tile_at(g, kg, col)]) : short(0);
            }
        }
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            const int row = row0 + wr + 16 * t + c;
            short8 ax;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int g = g0 + 8 * h + j;
                ax[j] = row < m && g < kg ? static_cast<short>(xs[static_cast<size_t>(row) * kg + g]) : short(0);
            }
#pragma unroll
            for (int u = 0; u < TN; ++u)
                acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ax, bx[u], acc[t][u]);
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = row0 + wr + 16 * t + 8 * h + e;
            if (row >= m) break;
            const float scale = a[row];
#pragma unroll
            for (int u = 0; u < TN; ++u) {
                const int col = col0 + wc + 16 * u + c;
                if (col >= n) continue;
                const size_t at = static_cast<size_t>(row) * n + col;
                const float v = acc[t][u][e] * scale;
                if (f32) static_cast<float*>(out)[at] = v;
                else static_cast<unsigned short*>(out)[at] = bf16_round(v);
            }
        }
}

template <int TN, int GPS, bool SEQ>
void launch_tiled(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& words,
                  const at::Tensor& scales, const at::Tensor& biases, int m, int n, int k, at::Tensor& out) {
    constexpr int BN = 32 * TN;
    const dim3 grid((n + BN - 1) / BN, (m + 255) / 256);
    gemm8_tiled_kernel<TN, GPS, SEQ><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x8.data_ptr<uint8_t>(), reinterpret_cast<const unsigned short*>(xs.data_ptr()), a.data_ptr<float>(),
        reinterpret_cast<const unsigned*>(words.data_ptr()), reinterpret_cast<const unsigned short*>(scales.data_ptr()),
        reinterpret_cast<const unsigned short*>(biases.data_ptr()), m, n, k, out.data_ptr(),
        out.scalar_type() == at::kFloat);
}

}  // namespace

// variant: the tiling (speed only, never bits); 0 is the default
void gemm8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
           const at::Tensor& scales, const at::Tensor& biases, int n, int group, at::Tensor& out, int variant) {
    const int m = static_cast<int>(x8.size(0)), k = static_cast<int>(x8.size(1));
    const bool w4 = w8.scalar_type() == at::kInt;      // group-major words, widened in the kernel
    TORCH_CHECK(k % GK == 0 && (w4 ? w8.numel() * 8 >= static_cast<int64_t>(n) * k
                                   : w8.size(1) == k && w8.size(0) >= n),
                "gemm8: (M, K) e4m3 rows against (N, K) codes or group-major 4-bit words");
    TORCH_CHECK(xs.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kFloat, "gemm8: bf16 group sums, fp32 a");
    TORCH_CHECK(x8.is_contiguous() && w8.is_contiguous() && xs.is_contiguous() && out.is_contiguous(),
                "gemm8 takes contiguous tensors");
    TORCH_CHECK(w4 == (variant >= 10), "gemm8: variants 10 and up take the words, the others _nibbles8's codes");
    if (m == 0 || n == 0) return;
    switch (variant) {
        case 0: launch<128, 128, 2, 4, 1, 2>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 1: launch<128, 128, 2, 4, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 2: launch<128, 128, 2, 4, 2, 1, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 3: launch<256, 128, 4, 4, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 8: launch<128, 128, 4, 2, 1, 2, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 10: launch<128, 128, 2, 4, 1, 2, false, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 11: launch<128, 128, 2, 4, 1, 2, true, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 12: launch<256, 128, 4, 4, 1, 2, false, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 13: launch<256, 128, 4, 4, 1, 2, true, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        case 14: launch<128, 64, 2, 2, 1, 2, false, true>(x8, xs, a, w8, scales, biases, m, n, k, group, out); break;
        default: TORCH_CHECK(false, "gemm8: unknown variant");
    }
}

// x8: (row tiles, K / 16, 256) e4m3 fragments (prefill_glue's TILED rows); words: group-major 4-bit words.
void gemm8_tiled(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& words,
                 const at::Tensor& scales, const at::Tensor& biases, int n, at::Tensor& out, int variant) {
    const int m = static_cast<int>(xs.size(0)), k = static_cast<int>(x8.size(1)) * 16;
    TORCH_CHECK(x8.dim() == 3 && x8.size(2) == 256 && x8.size(0) == (m + 15) / 16 && k % GK == 0 &&
                xs.size(1) == k / GK && words.scalar_type() == at::kInt && words.numel() * 8 >= static_cast<int64_t>(n) * k,
                "gemm8_tiled: (M / 16, K / 16, 256) fragments, (M, K / 64) group sums, group-major words");
    TORCH_CHECK(xs.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kFloat, "gemm8_tiled: bf16 sums, fp32 a");
    TORCH_CHECK(x8.is_contiguous() && words.is_contiguous() && xs.is_contiguous() && out.is_contiguous(),
                "gemm8_tiled takes contiguous tensors");
    if (m == 0 || n == 0) return;
    switch (variant) {
        case 0: launch_tiled<2, 1, false>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        case 1: launch_tiled<2, 2, false>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        case 2: launch_tiled<4, 1, true>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        case 3: launch_tiled<4, 2, true>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        case 4: launch_tiled<2, 1, true>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        case 5: launch_tiled<2, 2, true>(x8, xs, a, words, scales, biases, m, n, k, out); break;
        default: TORCH_CHECK(false, "gemm8_tiled: unknown variant");
    }
}
