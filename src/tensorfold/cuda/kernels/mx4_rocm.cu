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
