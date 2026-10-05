// gfx11 / gfx12 WMMA tiles; odd rows replay in a second WMMA's even rows so a row's bits ignore its neighbours.

#include "affine_api.hpp"
#include "affine_wmma.hpp"

#include <cstring>
#include <stdexcept>
#include <string>


namespace tf {
namespace rocm {

#if TENSORFOLD_RDNA_WMMA

template <bool kEight>
__global__ void __launch_bounds__(32) affine_wmma_kernel(Affine a) {
    using namespace rocwmma;
    int n0 = blockIdx.x * kTile;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];  // logical row r parked at physical row r ^ 1
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kTile];
    __shared__ float bcol[kTile];

    for (int e = lane; e < kTile * kTile; e += 32) acc[e] = 0.f;
    // Odd rows of this tile are past M: the single WMMA already uses row 0's rounding.
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        // The whole group is in flight before the scale sum, so the K steps do not each wait on HBM.
        uint2 b0, b1, b2, b3, b4, b5, b6, b7;
        if constexpr (kEight) {
            b0 = load_b8(a.words, words, n0, base, a.n, lane);
            b1 = load_b8(a.words, words, n0, base + 16, a.n, lane);
            b2 = a.group > 32 ? load_b8(a.words, words, n0, base + 32, a.n, lane) : make_uint2(0u, 0u);
            b3 = a.group > 48 ? load_b8(a.words, words, n0, base + 48, a.n, lane) : make_uint2(0u, 0u);
            b4 = a.group > 64 ? load_b8(a.words, words, n0, base + 64, a.n, lane) : make_uint2(0u, 0u);
            b5 = a.group > 80 ? load_b8(a.words, words, n0, base + 80, a.n, lane) : make_uint2(0u, 0u);
            b6 = a.group > 96 ? load_b8(a.words, words, n0, base + 96, a.n, lane) : make_uint2(0u, 0u);
            b7 = a.group > 112 ? load_b8(a.words, words, n0, base + 112, a.n, lane) : make_uint2(0u, 0u);
        }
        write_xsum(x, a.scale, a.bias, xsum, scol, bcol, lane, m0, n0, base, group, a.group, groups, a.m, a.k,
                   a.n);
        __syncthreads();

        FragC part_even, part_odd;
        fill_fragment(part_even, 0.f);
        fill_fragment(part_odd, 0.f);
        if constexpr (kEight) {
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b0, m0, base, a.m, a.k, lane,
                                          replay, true);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b1, m0, base + 16, a.m, a.k, lane,
                                          replay, true);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b2, m0, base + 32, a.m, a.k, lane,
                                          replay, a.group > 32);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b3, m0, base + 48, a.m, a.k, lane,
                                          replay, a.group > 48);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b4, m0, base + 64, a.m, a.k, lane,
                                          replay, a.group > 64);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b5, m0, base + 80, a.m, a.k, lane,
                                          replay, a.group > 80);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b6, m0, base + 96, a.m, a.k, lane,
                                          replay, a.group > 96);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b7, m0, base + 112, a.m, a.k, lane,
                                          replay, a.group > 112);
        } else {
            for (int step = 0; step < a.group; step += kTile) {
                int k0 = base + step;
                for (int e = lane; e < kTile * kTile; e += 32) {
                    int r = e / kTile;
                    int c = e % kTile;
                    int grow = m0 + r;
                    int gcol = k0 + c;
                    hip_bfloat16 xv = hip_bfloat16(0.f);
                    if (grow < a.m && gcol < a.k) xv = x[static_cast<long long>(grow) * a.k + gcol];
                    As[r * kTile + c] = xv;
                    if (replay) AsOdd[(r ^ 1) * kTile + c] = xv;

                    int ncol = n0 + r;
                    int nk = k0 + c;
                    hip_bfloat16 qv = hip_bfloat16(0.f);
                    if (ncol < a.n && nk < a.k) {
                        const uint32_t* wrow = a.words + static_cast<long long>(ncol) * words;
                        qv = hip_bfloat16(code_bf16(affine_code(wrow, nk, a.bits, words)));
                    }
                    Bs[r * kTile + c] = qv;  // col-major (k, n): index n * 16 + k
                }
                __syncthreads();
                mma_tile<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, replay);
                __syncthreads();
            }
        }

        store_matrix_sync(dot_even, part_even, kTile, mem_row_major);
        if (replay) store_matrix_sync(dot_odd, part_odd, kTile, mem_row_major);
        __syncthreads();
        // The replay's even physical row r ^ 1 holds logical row r.
        fma_group(acc, dot_even, dot_odd, xsum, scol, bcol, replay, lane);
        __syncthreads();
    }

    for (int e = lane; e < kTile * kTile; e += 32) {
        int grow = m0 + e / kTile;
        int gcol = n0 + e % kTile;
        if (grow < a.m && gcol < a.n) a.out[static_cast<long long>(grow) * a.n + gcol] = acc[e];
    }
}

// A wave owns several 16-column tiles of one row tile, sharing each K step's activation load.
constexpr int kCols = 4;

template <typename FragA, typename FragB, typename FragC>
__device__ void cols_step(FragC* even, FragC* odd, hip_bfloat16* As, hip_bfloat16* AsOdd,
                          hip_bfloat16 Bs[][kTile * kTile], const hip_bfloat16* x, const uint2* packed, int m0,
                          int k0, int m, int k, int lane, bool replay) {
    fill_a8(As, AsOdd, x, m0, k0, m, k, lane, replay);
#pragma unroll
    for (int col = 0; col < kCols; ++col) store_b8(Bs[col], packed[col], lane);
    __syncthreads();
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
        mma_tile<FragA, FragB, FragC>(even[col], odd[col], As, AsOdd, Bs[col], replay);
    }
    __syncthreads();
}

template <int Group>
__global__ void __launch_bounds__(32) affine_wmma_cols(Affine a) {
    using namespace rocwmma;
    constexpr int kSteps = Group / kTile;
    const int lane = threadIdx.x;
    const int nbase = blockIdx.x * (kCols * kTile);
    const int m0 = blockIdx.y * kTile;
    const int words = a.k * a.bits / 32;
    const int groups = a.k / Group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    const bool replay = m0 + 1 < a.m;

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kCols][kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kCols][kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kCols][kTile];
    __shared__ float bcol[kCols][kTile];

    for (int col = 0; col < kCols; ++col) {
        for (int e = lane; e < kTile * kTile; e += 32) acc[col][e] = 0.f;
    }

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        const int base = group * Group;
        // Every code load for this group is issued before the first WMMA, so the misses overlap.
        uint2 stage[kSteps][kCols];
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                stage[step][col] = load_b8(a.words, words, nbase + col * kTile, base + step * kTile, a.n, lane);
            }
        }
        if (lane < kTile) {
            float sum = 0.f;
            if (m0 + lane < a.m) {
                const hip_bfloat16* xrow = x + static_cast<long long>(m0 + lane) * a.k + base;
                for (int t = 0; t < Group; ++t) sum += static_cast<float>(xrow[t]);
            }
            xsum[lane] = sum;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                const int gcol = nbase + col * kTile + lane;
                const long long at = static_cast<long long>(gcol) * groups + group;
                scol[col][lane] = gcol < a.n ? a.scale[at] : 0.f;
                bcol[col][lane] = gcol < a.n ? a.bias[at] : 0.f;
            }
        }
        __syncthreads();

        FragC part_even[kCols], part_odd[kCols];
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            fill_fragment(part_even[col], 0.f);
            fill_fragment(part_odd[col], 0.f);
        }
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
            cols_step<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, stage[step], m0, base + step * kTile,
                                           a.m, a.k, lane, replay);
        }
        for (int col = 0; col < kCols; ++col) {
            store_matrix_sync(dot_even, part_even[col], kTile, mem_row_major);
            if (replay) store_matrix_sync(dot_odd, part_odd[col], kTile, mem_row_major);
            __syncthreads();
            fma_group(acc[col], dot_even, dot_odd, xsum, scol[col], bcol[col], replay, lane);
            __syncthreads();
        }
    }

    for (int col = 0; col < kCols; ++col) {
        for (int e = lane; e < kTile * kTile; e += 32) {
            const int grow = m0 + e / kTile;
            const int gcol = nbase + col * kTile + e % kTile;
            if (grow < a.m && gcol < a.n) a.out[static_cast<long long>(grow) * a.n + gcol] = acc[col][e];
        }
    }
}

hipError_t launch_affine_wmma(const Affine& a, hipStream_t stream) {
    // Tiles past M or N are filled with zeros, so a single row uses the same WMMA as a full tile.
    if (a.m < 1 || a.n < 1 || a.group % kTile || a.k % a.group) return hipErrorInvalidValue;
    if (a.bits == 8 && a.m > kTile && a.n > kTile && (a.group == 32 || a.group == 64 || a.group == 128)) {
        dim3 block(32);
        dim3 grid((a.n + kCols * kTile - 1) / (kCols * kTile), (a.m + kTile - 1) / kTile);
        if (a.group == 32) {
            affine_wmma_cols<32><<<grid, block, 0, stream>>>(a);
        } else if (a.group == 64) {
            affine_wmma_cols<64><<<grid, block, 0, stream>>>(a);
        } else {
            affine_wmma_cols<128><<<grid, block, 0, stream>>>(a);
        }
        return hipGetLastError();
    }
    dim3 block(32);
    dim3 grid((a.n + kTile - 1) / kTile, (a.m + kTile - 1) / kTile);
    if (a.bits == 8) {
        affine_wmma_kernel<true><<<grid, block, 0, stream>>>(a);
    } else {
        affine_wmma_kernel<false><<<grid, block, 0, stream>>>(a);
    }
    return hipGetLastError();
}

#else

hipError_t launch_affine_wmma(const Affine&, hipStream_t) { return hipErrorNotSupported; }

#endif

hipError_t launch_affine(const Affine& a, int schedule, hipStream_t stream) {
    if (a.fp16) {
        if (schedule == 2) return hipErrorNotSupported;
#if TENSORFOLD_RDNA_WMMA
        return hipErrorNotSupported;
#else
        return launch_affine_dot2(a, schedule == 1, stream);
#endif
    }
#if TENSORFOLD_RDNA_WMMA
    // auto runs the BF16 dot2 tiles at every width; WMMA (schedule 2) is opt-in, at every row count.
    if (schedule == 0 && (a.group == 32 || a.group == 64 || a.group == 128) && a.k % a.group == 0 &&
        a.k % 16 == 0) {
        return launch_affine_dot2_bf16(a, stream);
    }
    bool tile = schedule != 1 && a.group % 16 == 0 && a.k % a.group == 0;
#else
    bool tile = false;
#endif
    if (schedule == 3) return launch_affine_gemv_cols(a, stream);
    if (schedule == 2 && !tile) return hipErrorNotSupported;
    if (tile) return launch_affine_wmma(a, stream);
    return launch_affine_gemv(a, stream);
}

}  // namespace rocm
}  // namespace tf

void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half) {
    tf::rocm::Affine a{x, static_cast<const uint32_t*>(words), {scale, scale_kind}, {bias, scale_kind},
                       static_cast<float*>(out), m, n, k, bits, group, fp16};
    if (out_half == 1) {
        if (!fp16 || schedule != 0 || splits > 1) {
            throw std::runtime_error("affine kernel: fp16 output is the decode tile");
        }
        a.out = nullptr;
        a.out16 = static_cast<__half*>(out);
    } else if (out_half == 2) {
        if (fp16 || schedule != 0 || splits > 1 || m > 8) {
            throw std::runtime_error("affine kernel: bf16 output is the gfx11 / gfx12 decode tile");
        }
        a.out = nullptr;
        a.outb16 = static_cast<hip_bfloat16*>(out);
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = (partial != nullptr && splits > 1)
                         ? tf::rocm::launch_affine_dot2_split(a, partial, splits, hip_stream)
                         : tf::rocm::launch_affine(a, schedule, hip_stream);
    if (err != hipSuccess) throw std::runtime_error(std::string("affine kernel: ") + hipGetErrorString(err));
}
