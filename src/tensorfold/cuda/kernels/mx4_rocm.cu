// RDNA4 matmuls on MXFP4 weights (mx4_load.Mx4: e2m1 codes, an e8m0 scale per 32 inputs).
//
// ``prompt_kernel``: FP8 prompt rows (prefill_glue's TILED fragments) times the weights on v_wmma_f32_16x16x16_fp8_fp8.
// Each output column has a reference exponent (its largest group scale); a group's codes widen to e4m3 as value *
// 2^-(reference - scale) while they are staged, so the K loop is WMMAs alone and the column's 2^(reference - 127)
// and the row's scale come once, in the epilogue. Exact while the shift d is at most 8 (e4m3's subnormals reach
// 2^-9); larger shifts round to nearest even. A row's bits depend only on its own row and column: fixed 16-input
// steps in order whatever tile, block or call holds it, so any prompt chunking gives the same bits.
//
// Credits: the folded-scale design (the weight exponent pushed into the e4m3 byte, one reference exponent a column,
// subnormal shifts), the A-fragment loads straight from global memory and the 256-row tile are ideas from
// vLLM-radiance's MXFP4 kernels (codeberg.org/StillDeadcode/vllm-radiance, radiance_mxfp4_fp8.hip), which carry no
// license; no code is taken from them, and the shift table below is computed with torch's e4m3 rounding. The
// fragment-ordered activations and the WMMA layout are ours and jkuepker's (qmm8_rocm.cu, qmm_rocm.cu).

#ifndef __HIPCC__
#error "mx4_rocm.cu is RDNA4's MXFP4 matmul"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>

namespace {

typedef int int2v __attribute__((ext_vector_type(2)));
typedef float float8 __attribute__((ext_vector_type(8)));

// e4m3 bytes of the e2m1 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6) times 2^-d, four a word, d = 0..15
// (torch.float8_e4m3fn rounding; exact through d = 8).
__constant__ unsigned SHIFTED[16][2] = {
    {0x3C383000u, 0x4C484440u}, {0x34302800u, 0x44403C38u}, {0x2C282000u, 0x3C383430u},
    {0x24201800u, 0x34302C28u}, {0x1C181000u, 0x2C282420u}, {0x14100800u, 0x24201C18u},
    {0x0C080400u, 0x1C181410u}, {0x06040200u, 0x14100C08u}, {0x03020100u, 0x0C080604u},
    {0x02010000u, 0x06040302u}, {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
    {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u}};

__device__ __forceinline__ unsigned short bf16_round(float f) {
    const unsigned u = __float_as_uint(f);
    return static_cast<unsigned short>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// A 16-input block's codes (bytes u0 = inputs 0-7, u1 = 8-15; low nibble the even input) as e4m3 in prefill_glue's
// stored order, where position 4a + j holds input 2a + j % 2 + 8 (j / 2), each code shifted by the table row (lo, hi).
__device__ __forceinline__ uint4 widen16(unsigned u0, unsigned u1, unsigned lo, unsigned hi) {
    unsigned out[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        // bytes a of u0 (inputs 2a, 2a + 1) twice, then byte a of u1 (inputs 2a + 8, 2a + 9) twice
        const unsigned x = __builtin_amdgcn_perm(u1, u0, a | (a << 8) | ((a + 4) << 16) | ((a + 4) << 24));
        const unsigned codes = __builtin_amdgcn_perm((x >> 4) & 0x0F0F0F0Fu, x & 0x0F0F0F0Fu, 0x07020500u);
        out[a] = __builtin_amdgcn_perm(hi, lo, codes & 0x07070707u) | ((codes & 0x08080808u) << 4);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}

// TN 16-column fragments a wave (BN = 32 TN), 8 waves: 4 along M (4 fragments each), 2 along N; a slab is 64 inputs.
template <int TN>
__global__ void __launch_bounds__(256) prompt_kernel(
        const uint8_t* __restrict__ x8t, const float* __restrict__ a, const uint8_t* __restrict__ wt,
        const uint8_t* __restrict__ sct, const int* __restrict__ ref, int m, int n, int k, void* __restrict__ out,
        bool f32) {
    constexpr int WN = 2, TM = 4, BM = 256, BN = WN * TN * 16, SK = 64, NS = SK / 16, PITCH = SK + 8;
    constexpr int PIECES = BN * 2;                       // a column's two 32-input groups a slab, 16 bytes each
    static_assert(PIECES <= 256, "one piece a thread");
    __shared__ __align__(16) uint8_t wl[BN * PITCH];
    __shared__ unsigned table[16][2];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave / WN, wn = wave % WN, h = lane >> 4, c = lane & 15;
    const int kg = k / 32, ksteps = k / 16, mt_last = (m + 15) / 16 - 1;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN, wr = wm * TM * 16, wc = wn * TN * 16;
    if (tid < 32) table[tid >> 1][tid & 1] = SHIFTED[tid >> 1][tid & 1];

    const uint8_t* abase[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mt = min(row0 / 16 + wm * TM + i, mt_last);  // past m: the last tile again (outputs dropped)
        abase[i] = x8t + __builtin_amdgcn_readfirstlane(mt * ksteps * 256);
    }
    const unsigned aoff = lane * 8;
    const int pcol = min(col0 + tid % BN, n - 1), pg = tid / BN;   // this thread's weight piece: column, group
    const int pref = ref[pcol];

    float8 acc[TM][TN];
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int u = 0; u < TN; ++u) acc[t][u] = float8{0, 0, 0, 0, 0, 0, 0, 0};

    const int slabs = k / SK;
    for (int sl = 0; sl < slabs; ++sl) {
        uint4 wv = make_uint4(0, 0, 0, 0);
        int d = 0;
        if (tid < PIECES) {                              // the weights first: staging waits on them alone
            const int g = 2 * sl + pg;
            wv = *reinterpret_cast<const uint4*>(wt + ((static_cast<size_t>(pcol >> 4) * kg + g) * 16 + (pcol & 15)) * 16);
            d = min(max(pref - static_cast<int>(sct[static_cast<size_t>(g) * n + pcol]), 0), 15);
        }
        int2v af[TM][NS];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int s = 0; s < NS; ++s)
                af[i][s] = *reinterpret_cast<const int2v*>(abase[i] + (aoff + static_cast<unsigned>(sl * NS + s) * 256u));
        lds_barrier();                                   // the last slab's weight reads are done (and the table)
        if (tid < PIECES) {
            const unsigned lo = table[d][0], hi = table[d][1];
            const uint4 b0 = widen16(wv.x, wv.y, lo, hi), b1 = widen16(wv.z, wv.w, lo, hi);
            uint2* dst = reinterpret_cast<uint2*>(wl + (tid % BN) * PITCH + 32 * pg);
            dst[0] = make_uint2(b0.x, b0.y);
            dst[1] = make_uint2(b0.z, b0.w);
            dst[2] = make_uint2(b1.x, b1.y);
            dst[3] = make_uint2(b1.z, b1.w);
        }
        lds_barrier();
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            int2v bf[TN];
#pragma unroll
            for (int u = 0; u < TN; ++u) {
                const uint2 v = *reinterpret_cast<const uint2*>(wl + (wc + 16 * u + c) * PITCH + 16 * s + 8 * h);
                bf[u] = int2v{static_cast<int>(v.x), static_cast<int>(v.y)};
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int t = 0; t < TM; ++t)
#pragma unroll
                for (int u = 0; u < TN; ++u)
                    acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[t][s], bf[u], acc[t][u]);
        }
    }
    float cs[TN];
#pragma unroll
    for (int u = 0; u < TN; ++u) {
        const int col = min(col0 + wc + 16 * u + c, n - 1);
        const int e = ref[col] - 127;                   // 2^e exactly (e8m0 references are 1..254 in practice)
        cs[u] = __int_as_float((e + 127) << 23);
    }
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + wr + 16 * t + 8 * h + i;
            if (row >= m) break;
            const float rs = a[row];
#pragma unroll
            for (int u = 0; u < TN; ++u) {
                const int col = col0 + wc + 16 * u + c;
                if (col >= n) continue;
                const size_t at = static_cast<size_t>(row) * n + col;
                const float v = acc[t][u][i] * cs[u] * rs;
                if (f32) static_cast<float*>(out)[at] = v;
                else static_cast<unsigned short*>(out)[at] = bf16_round(v);
            }
        }
}

// ------------------------------------------------------------------------------------------------------ decode rows
// ``decode_kernel``: up to 16 MT bf16 rows (decode, a draft window's verify rows) times the weights on bf16 WMMA. A
// group's codes widen to bf16 as value * 2^-(reference - scale), exact for any shift (bf16 keeps fp32's exponent), so
// the K loop is WMMAs alone; the column's 2^(reference - 127) comes at the end. K splits into slices fixed by the
// weight's shape (never by the row count), whose sums ``reduce_kernel`` adds in slice order, so a row's bits are the
// same in any call. Memory bound: a block's 8 waves take 16 columns each, every lane reading its column's 16 bytes a
// group.
typedef short short8 __attribute__((ext_vector_type(8)));

// bf16 high and low bytes of the e2m1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, four a word
constexpr unsigned BH0 = 0x3F3F3F00u, BH1 = 0x40404040u, BL0 = 0xC0800000u, BL1 = 0xC0804000u;

// Four codes (one a byte) as two words of bf16 pairs, each value shifted down by d binades (zeros stay zero).
__device__ __forceinline__ uint2 bf16x4(unsigned codes, unsigned shift) {
    const unsigned mag = codes & 0x07070707u;
    const unsigned hb = __builtin_amdgcn_perm(BH1, BH0, mag) | ((codes & 0x08080808u) << 4);
    const unsigned lb = __builtin_amdgcn_perm(BL1, BL0, mag);
    unsigned w[2] = {__builtin_amdgcn_perm(hb, lb, 0x05010400u), __builtin_amdgcn_perm(hb, lb, 0x07030602u)};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned nz = (((w[j] & 0x7FFF7FFFu) + 0x7FFF7FFFu) & 0x80008000u) >> 15;   // 1 per nonzero half
        w[j] -= shift & (nz * 0xFFFFu);
    }
    return make_uint2(w[0], w[1]);
}

// Bytes b (inputs 2b, 2b + 1) of a group's 16: inputs 8h .. 8h + 7 of step s are bytes 8s + 4h .. + 3, in order.
__device__ __forceinline__ short8 wfrag(const uint4& g, int s, int h, unsigned shift) {
    const unsigned word = s == 0 ? (h == 0 ? g.x : g.y) : (h == 0 ? g.z : g.w);
    // nibbles in input order: byte j's low then high
    const unsigned lo = word & 0x0F0F0F0Fu, hi = (word >> 4) & 0x0F0F0F0Fu;
    const uint2 a = bf16x4(__builtin_amdgcn_perm(hi, lo, 0x05010400u), shift);   // inputs 0-3
    const uint2 b = bf16x4(__builtin_amdgcn_perm(hi, lo, 0x07030602u), shift);   // inputs 4-7
    return __builtin_bit_cast(short8, make_uint4(a.x, a.y, b.x, b.y));
}

constexpr int DGS = 16;                                  // groups (512 inputs) staged a step

template <int MT>
__global__ void __launch_bounds__(256) decode_kernel(
        const unsigned short* __restrict__ x, int m, const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct,
        const int* __restrict__ ref, int n, int k, int gps, float* __restrict__ part, unsigned short* __restrict__ out16,
        float* __restrict__ out32, int slices) {
    __shared__ __align__(16) unsigned short xl[MT * 16][DGS * 32 + 8];
    const int kg = k / 32, slice = blockIdx.y, g0 = slice * gps, g1 = min(kg, g0 + gps);
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = lane >> 4, c = lane & 15;
    const int col = blockIdx.x * 128 + wave * 16 + c, colc = min(col, n - 1);
    const int r = ref[colc];
    float8 acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int gb = g0; gb < g1; gb += DGS) {
        const int groups = min(DGS, g1 - gb);
        __syncthreads();
        for (int i = threadIdx.x; i < MT * 16 * DGS * 4; i += 256) {   // 16-byte pieces of the rows' inputs
            const int row = i / (DGS * 4), pc = i % (DGS * 4);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row < m && pc < groups * 4)
                v = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * k + gb * 32 + pc * 8);
            *reinterpret_cast<uint2*>(&xl[row][pc * 8]) = make_uint2(v.x, v.y);
            *reinterpret_cast<uint2*>(&xl[row][pc * 8 + 4]) = make_uint2(v.z, v.w);
        }
        __syncthreads();
        for (int j = 0; j < groups; ++j) {
            const int g = gb + j;
            const uint4 gw = *reinterpret_cast<const uint4*>(wt + ((static_cast<size_t>(colc >> 4) * kg + g) * 16 + (colc & 15)) * 16);
            const unsigned d = static_cast<unsigned>(r - static_cast<int>(sct[static_cast<size_t>(g) * n + colc]));
            const unsigned shift = (d << 7) | (d << 23);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const short8 b = wfrag(gw, s, h, shift);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const short8 av = *reinterpret_cast<const short8*>(&xl[16 * t + c][j * 32 + 16 * s + 8 * h]);
                    acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, b, acc[t]);
                }
            }
        }
    }
    if (col >= n) return;
    const float cs = __int_as_float(r << 23);
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = 16 * t + 8 * h + i;
            if (row >= m) break;
            const float v = acc[t][i] * cs;
            if (slices > 1) part[(static_cast<size_t>(slice) * m + row) * n + col] = v;
            else if (out32 != nullptr) out32[static_cast<size_t>(row) * n + col] = v;
            else out16[static_cast<size_t>(row) * n + col] = bf16_round(v);
        }
}

// ``b16_kernel``: the same rows times a bf16 (N, K) weight as stored (the MXFP4 checkpoint's head; the drafter's rows
// of it): no widening, the same slices and order, so a row's bits are the same in any call.
template <int MT>
__global__ void __launch_bounds__(256) b16_kernel(
        const unsigned short* __restrict__ x, int m, const unsigned short* __restrict__ wb, int n, int k, int gps,
        float* __restrict__ part, unsigned short* __restrict__ out16, float* __restrict__ out32, int slices) {
    __shared__ __align__(16) unsigned short xl[MT * 16][DGS * 32 + 8];
    const int kg = k / 32, slice = blockIdx.y, g0 = slice * gps, g1 = min(kg, g0 + gps);
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = lane >> 4, c = lane & 15;
    const int col = blockIdx.x * 128 + wave * 16 + c, colc = min(col, n - 1);
    const unsigned short* wrow = wb + static_cast<size_t>(colc) * k;
    float8 acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int gb = g0; gb < g1; gb += DGS) {
        const int groups = min(DGS, g1 - gb);
        __syncthreads();
        for (int i = threadIdx.x; i < MT * 16 * DGS * 4; i += 256) {
            const int row = i / (DGS * 4), pc = i % (DGS * 4);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row < m && pc < groups * 4)
                v = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * k + gb * 32 + pc * 8);
            *reinterpret_cast<uint2*>(&xl[row][pc * 8]) = make_uint2(v.x, v.y);
            *reinterpret_cast<uint2*>(&xl[row][pc * 8 + 4]) = make_uint2(v.z, v.w);
        }
        __syncthreads();
        for (int j = 0; j < groups; ++j) {
            const int g = gb + j;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const short8 b = *reinterpret_cast<const short8*>(wrow + g * 32 + 16 * s + 8 * h);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const short8 av = *reinterpret_cast<const short8*>(&xl[16 * t + c][j * 32 + 16 * s + 8 * h]);
                    acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, b, acc[t]);
                }
            }
        }
    }
    if (col >= n) return;
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = 16 * t + 8 * h + i;
            if (row >= m) break;
            const float v = acc[t][i];
            if (slices > 1) part[(static_cast<size_t>(slice) * m + row) * n + col] = v;
            else if (out32 != nullptr) out32[static_cast<size_t>(row) * n + col] = v;
            else out16[static_cast<size_t>(row) * n + col] = bf16_round(v);
        }
}

__global__ void reduce_kernel(const float* __restrict__ part, int slices, size_t total, float* __restrict__ out32,
                              unsigned short* __restrict__ out16) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = part[i];
    for (int s = 1; s < slices; ++s) v = v + part[s * total + i];
    if (out32 != nullptr) out32[i] = v;
    else out16[i] = bf16_round(v);
}

}  // namespace

// x8: (row tiles, K / 16, 256) e4m3 fragments; a: (M,) row scales; wt: the weight's 16-byte groups, 16 columns a tile
// ((N / 16, K / 32, 16, 16) bytes); sct: (K / 32, N) e8m0 scales; ref: (N,) int32 reference exponents.
void prompt_mx4(const at::Tensor& x8, const at::Tensor& a, const at::Tensor& wt, const at::Tensor& sct,
                const at::Tensor& ref, int n, at::Tensor& out, int tn) {
    const int m = static_cast<int>(a.size(0)), k = static_cast<int>(x8.size(1)) * 16;
    TORCH_CHECK(x8.dim() == 3 && x8.size(2) == 256 && x8.size(0) == (m + 15) / 16 && k % 64 == 0,
                "prompt_mx4: (M / 16, K / 16, 256) fragments, K a multiple of 64");
    TORCH_CHECK(wt.numel() == static_cast<int64_t>((n + 15) / 16) * 16 * (k / 2) && sct.size(0) == k / 32 &&
                sct.size(1) == n && ref.numel() == n && ref.scalar_type() == at::kInt && a.scalar_type() == at::kFloat,
                "prompt_mx4: tiled fp4 words, (K / 32, N) scales, (N,) int32 references, fp32 row scales");
    TORCH_CHECK(x8.is_contiguous() && wt.is_contiguous() && sct.is_contiguous() && out.is_contiguous(),
                "prompt_mx4 takes contiguous tensors");
    if (m == 0 || n == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((n + 32 * tn - 1) / (32 * tn), (m + 255) / 256);
    const bool f32 = out.scalar_type() == at::kFloat;
#define LAUNCH(TN)                                                                                                \
    prompt_kernel<TN><<<grid, 256, 0, stream>>>(x8.data_ptr<uint8_t>(), a.data_ptr<float>(), wt.data_ptr<uint8_t>(), \
                                                sct.data_ptr<uint8_t>(), ref.data_ptr<int>(), m, n, k, out.data_ptr(), f32)
    if (tn == 2) LAUNCH(2);
    else LAUNCH(4);
#undef LAUNCH
}

// The K slices of an (n outputs, kg groups) weight: enough blocks to fill the GPU, fixed by the shape alone.
int decode_slices(int n, int kg) {
    const int blocks = (n + 127) / 128;
    int slices = std::max(1, std::min(kg / 16, (256 + blocks - 1) / blocks));
    return slices;
}

// x: (M, K) bf16 rows, M <= 32; out: (M, N) bf16 or fp32; part: (slices, M, N) fp32 scratch.
void decode_mx4(const at::Tensor& x, const at::Tensor& wt, const at::Tensor& sct, const at::Tensor& ref, int n,
                at::Tensor& out, at::Tensor& part) {
    const int m = static_cast<int>(x.size(0)), k = static_cast<int>(x.size(1)), kg = k / 32;
    TORCH_CHECK(m <= 32 && k % 32 == 0 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && out.is_contiguous(),
                "decode_mx4: up to 32 contiguous bf16 rows, K a multiple of 32");
    if (m == 0 || n == 0) return;
    const int slices = decode_slices(n, kg), gps = (kg + slices - 1) / slices;
    TORCH_CHECK(slices == 1 || part.numel() >= static_cast<int64_t>(slices) * m * n, "decode_mx4: scratch too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((n + 127) / 128, slices);
    float* o32 = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    auto* o16 = o32 == nullptr ? reinterpret_cast<unsigned short*>(out.data_ptr()) : nullptr;
    const auto* xp = reinterpret_cast<const unsigned short*>(x.data_ptr());
    if (m <= 16)
        decode_kernel<1><<<grid, 256, 0, stream>>>(xp, m, wt.data_ptr<uint8_t>(), sct.data_ptr<uint8_t>(),
                                                   ref.data_ptr<int>(), n, k, gps, part.data_ptr<float>(), o16, o32, slices);
    else
        decode_kernel<2><<<grid, 256, 0, stream>>>(xp, m, wt.data_ptr<uint8_t>(), sct.data_ptr<uint8_t>(),
                                                   ref.data_ptr<int>(), n, k, gps, part.data_ptr<float>(), o16, o32, slices);
    if (slices > 1) {
        const size_t total = static_cast<size_t>(m) * n;
        reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(part.data_ptr<float>(), slices, total, o32, o16);
    }
}

// x: (M, K) bf16 rows, M <= 32; w: (N, K) bf16 as stored; out and part as decode_mx4's.
void decode_b16(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, at::Tensor& part) {
    const int m = static_cast<int>(x.size(0)), k = static_cast<int>(x.size(1)), n = static_cast<int>(w.size(0));
    TORCH_CHECK(m <= 32 && k % 32 == 0 && w.size(1) == k && x.scalar_type() == at::kBFloat16 &&
                w.scalar_type() == at::kBFloat16 && x.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
                "decode_b16: up to 32 contiguous bf16 rows against a contiguous bf16 (N, K) weight, K a multiple of 32");
    if (m == 0 || n == 0) return;
    const int kg = k / 32, slices = decode_slices(n, kg), gps = (kg + slices - 1) / slices;
    TORCH_CHECK(slices == 1 || part.numel() >= static_cast<int64_t>(slices) * m * n, "decode_b16: scratch too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((n + 127) / 128, slices);
    float* o32 = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    auto* o16 = o32 == nullptr ? reinterpret_cast<unsigned short*>(out.data_ptr()) : nullptr;
    const auto* xp = reinterpret_cast<const unsigned short*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const unsigned short*>(w.data_ptr());
    if (m <= 16) {                                       // braces: torch's hipify mangles an else-line launch
        b16_kernel<1><<<grid, 256, 0, stream>>>(xp, m, wp, n, k, gps, part.data_ptr<float>(), o16, o32, slices);
    } else {
        b16_kernel<2><<<grid, 256, 0, stream>>>(xp, m, wp, n, k, gps, part.data_ptr<float>(), o16, o32, slices);
    }
    if (slices > 1) {
        const size_t total = static_cast<size_t>(m) * n;
        reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(part.data_ptr<float>(), slices, total, o32, o16);
    }
}
