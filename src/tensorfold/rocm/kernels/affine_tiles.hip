// Dot2 tiles for every width and both activation types: one row, 2-8 rows, the GEMM tile, and routed plans.

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "affine_dot2.hpp"

namespace tf {
namespace rocm {


// Groups a wave owns each round. Fewer rows leave room for more loads in flight; at most 16 code words a lane.
constexpr int lane_groups(int bits, int rows, int pieces) {
    int per = rows <= 2 ? 4 : rows <= 4 ? 2 : 1;
    while (per > 1 && per * pieces * bits > 8) per /= 2;
    return per;
}

template <typename T, int BITS, int ROWS, int PIECES>
__global__ void __launch_bounds__(kLaneCols * kLaneWaves) affine_dot2_lanes(Affine a) {
    if (!take_item(a, blockIdx.z)) return;
    constexpr int kGroup = 32 * PIECES;
    constexpr int kPer = lane_groups(BITS, ROWS, PIECES);
    constexpr int kRound = kLaneWaves * kPer;
    __shared__ typename T::elem xs[ROWS][kRound * kGroup];
    __shared__ float part[kRound][ROWS][2][kLaneCols];
    const int lane = threadIdx.x % kLaneCols;
    const int wave = threadIdx.x / kLaneCols;
    const int col = blockIdx.x * kLaneCols + lane;
    const int groups = a.k / kGroup;
    const int words_row = a.k * BITS / 32;
    // A lane past N reads the last column and stores nothing, so every lane reaches every barrier.
    const uint32_t* wrow = a.words + static_cast<long long>(col < a.n ? col : a.n - 1) * words_row;
    const typename T::elem* x = static_cast<const typename T::elem*>(a.x);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int g0 = 0; g0 < groups; g0 += kRound) {
        constexpr int kSpan = kRound * kGroup;
        for (int e = threadIdx.x; e < ROWS * kSpan; e += blockDim.x) {
            const int r = e / kSpan;
            const int kk = e - r * kSpan;
            const int gk = g0 * kGroup + kk;
            xs[r][kk] = (r < a.m && gk < a.k) ? x[x_row(a, r) * a.k + gk] : T::zero();
        }
        // 6- and 8-bit lanes load every code word of their groups up front; narrower widths load a piece at a time.
        constexpr bool kAhead = BITS >= 6;
        uint32_t w[kPer][PIECES][BITS];
        if constexpr (kAhead) {
#pragma unroll
            for (int j = 0; j < kPer; ++j) {
                const int g = g0 + wave * kPer + j;
                if (g < groups) {
#pragma unroll
                    for (int p = 0; p < PIECES; ++p) load_piece<BITS>(wrow + (g * PIECES + p) * BITS, w[j][p]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int local = wave * kPer + j;
            if (g0 + local >= groups) continue;
            float dot[ROWS];
            float sum[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                dot[r] = 0.f;
                sum[r] = 0.f;
            }
#pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                if constexpr (!kAhead) load_piece<BITS>(wrow + ((g0 + local) * PIECES + p) * BITS, w[j][p]);
                const int xk = local * kGroup + 32 * p;
#pragma unroll
                for (int t = 0; t < 32; t += 2) {
                    const typename T::pair q = T::two(T::template code<BITS>(w[j][p], t),
                                                      T::template code<BITS>(w[j][p], t + 1));
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const typename T::pair xp = T::two(xs[r][xk + t], xs[r][xk + t + 1]);
                        dot[r] = T::dot(xp, q, dot[r]);
                        sum[r] += T::lo(xp);
                        sum[r] += T::hi(xp);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                part[local][r][0][lane] = dot[r];
                part[local][r][1][lane] = sum[r];
            }
        }
        __syncthreads();
        if (wave == 0 && col < a.n) {
            const int round = groups - g0 < kRound ? groups - g0 : kRound;
            for (int j = 0; j < round; ++j) {
                const long long at = static_cast<long long>(col) * groups + g0 + j;
                const float scale = a.scale[at];
                const float bias = a.bias[at];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[r] = fmaf(part[j][r][0][lane], scale, acc[r]);
                    acc[r] = fmaf(part[j][r][1][lane], bias, acc[r]);
                }
            }
        }
        // The next round restages xs: wave 0's fold above only reads part.
    }
    if (wave == 0 && col < a.n) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= a.m) continue;
            if (a.out16 != nullptr) {
                a.out16[out_row(a, r) * a.n + col] = __float2half_rn(acc[r]);
            } else if (a.outb16 != nullptr) {
                a.outb16[out_row(a, r) * a.n + col] = hip_bfloat16(acc[r]);
            } else {
                a.out[out_row(a, r) * a.n + col] = acc[r];
            }
        }
    }
}

// One-row tile: a lane owns a group, so a wave reads a weight row contiguously; lane c folds column c in order.
// ROWS = 2 runs two rows on each code it decodes: every row keeps its own dot chain, group sum and fold, the m = 1
// arithmetic, so a row's bits do not depend on the other row (a verify step of two tokens reads the weights once).
constexpr int kRowWaves = 4;
constexpr int kRowCols = 32;

template <typename T, int BITS, int PIECES, int ROWS>
__device__ __forceinline__ void row_tile(const Affine& a, int lpc, int bx) {
    constexpr int kGroup = 32 * PIECES;
    __shared__ float dots[ROWS][kRowWaves][kRowCols][33];
    __shared__ float sums[ROWS][kRowWaves][32];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int slots = 32 / lpc;                 // columns a pass
    const int slot = lane / lpc;
    const int j = lane % lpc;
    const int col0 = (bx * kRowWaves + wave) * kRowCols;
    const int mine = col0 + lane;               // the column this lane folds
    const int groups = a.k / kGroup;
    const long long words_row = static_cast<long long>(a.k) * BITS / 32;
    const typename T::pair* x[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        x[r] = reinterpret_cast<const typename T::pair*>(static_cast<const typename T::elem*>(a.x) +
                                                         x_row(a, r < a.m ? r : 0) * a.k);
    }
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    // bf16 tables a whole number of rounds wide: the folding lane loads its column's scales and biases for the round
    // in 16-byte pieces before the passes, not one by one after the barrier. The fold's arithmetic is unchanged.
    const bool vec = a.scale.kind == kScaleBF16 && a.bias.kind == kScaleBF16 && lpc >= 8 && groups % lpc == 0 &&
                     ((reinterpret_cast<uintptr_t>(a.scale.p) | reinterpret_cast<uintptr_t>(a.bias.p)) & 15) == 0;
    for (int g0 = 0; g0 < groups; g0 += lpc) {
        const int g = g0 + j;
        const bool live = g < groups;
        uint32_t sw[16], bw[16];
        if (vec && mine < a.n) {
            const long long at0 = static_cast<long long>(mine) * groups + g0;
            const uint4* sp = reinterpret_cast<const uint4*>(static_cast<const hip_bfloat16*>(a.scale.p) + at0);
            const uint4* bp = reinterpret_cast<const uint4*>(static_cast<const hip_bfloat16*>(a.bias.p) + at0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                if (v < lpc / 8) {
                    const uint4 sq = sp[v], bq = bp[v];
                    sw[4 * v] = sq.x; sw[4 * v + 1] = sq.y; sw[4 * v + 2] = sq.z; sw[4 * v + 3] = sq.w;
                    bw[4 * v] = bq.x; bw[4 * v + 1] = bq.y; bw[4 * v + 2] = bq.z; bw[4 * v + 3] = bq.w;
                }
            }
        }
        typename T::pair xv[ROWS][PIECES * 16];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < PIECES * 16; ++i) {
                xv[r][i] = live ? x[r][g * (kGroup / 2) + i] : T::two(T::zero(), T::zero());
                sum += T::lo(xv[r][i]);
                sum += T::hi(xv[r][i]);
            }
            if (slot == 0) sums[r][wave][j] = sum;
        }
#pragma unroll 2
        for (int pass = 0; pass < kRowCols / slots; ++pass) {
            const int c = pass * slots + slot;
            const int col = col0 + c;
            float dot[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) dot[r] = 0.f;
            if (live && col < a.n) {
                const uint32_t* wrow = a.words + static_cast<long long>(col) * words_row +
                                       static_cast<long long>(g) * PIECES * BITS;
#pragma unroll
                for (int p = 0; p < PIECES; ++p) {
                    uint32_t w[BITS];
                    load_piece<BITS>(wrow + p * BITS, w);
#pragma unroll
                    for (int t = 0; t < 32; t += 2) {
                        const typename T::pair q =
                            T::two(T::template code<BITS>(w, t), T::template code<BITS>(w, t + 1));
#pragma unroll
                        for (int r = 0; r < ROWS; ++r) dot[r] = T::dot(xv[r][p * 16 + t / 2], q, dot[r]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) dots[r][wave][c][j] = dot[r];
        }
        __syncthreads();
        if (mine < a.n && vec) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < lpc) {
                    const float scale = __uint_as_float(((sw[i >> 1] >> (16 * (i & 1))) & 0xffffu) << 16);
                    const float bias = __uint_as_float(((bw[i >> 1] >> (16 * (i & 1))) & 0xffffu) << 16);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        acc[r] = fmaf(dots[r][wave][lane][i], scale, acc[r]);
                        acc[r] = fmaf(sums[r][wave][i], bias, acc[r]);
                    }
                }
            }
        } else if (mine < a.n) {
            const int round = groups - g0 < lpc ? groups - g0 : lpc;
            for (int i = 0; i < round; ++i) {
                const long long at = static_cast<long long>(mine) * groups + g0 + i;
                const float scale = a.scale[at];
                const float bias = a.bias[at];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[r] = fmaf(dots[r][wave][lane][i], scale, acc[r]);
                    acc[r] = fmaf(sums[r][wave][i], bias, acc[r]);
                }
            }
        }
        __syncthreads();
    }
    if (mine < a.n) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= a.m) continue;
            if (a.out16 != nullptr) {
                a.out16[out_row(a, r) * a.n + mine] = __float2half_rn(acc[r]);
            } else if (a.outb16 != nullptr) {
                a.outb16[out_row(a, r) * a.n + mine] = hip_bfloat16(acc[r]);
            } else {
                a.out[out_row(a, r) * a.n + mine] = acc[r];
            }
        }
    }
}

template <typename T, int BITS, int PIECES, int ROWS = 1>
__global__ void __launch_bounds__(32 * kRowWaves) affine_dot2_row(Affine a, int lpc) {
    if (!take_item(a, blockIdx.z)) return;
    row_tile<T, BITS, PIECES, ROWS>(a, lpc, blockIdx.x);
}

// Up to four products of one x in one launch: block b runs the row tile of the part whose blocks hold b, the code a
// solo launch of that part runs, so each output has its solo bits. The narrow ones (the 32-wide gates of the linear
// attention) no longer cost a launch and a lone latency-bound wave each.
struct AffineRows {
    Affine part[4];
    int first[5];  // part p owns blocks first[p] .. first[p + 1] - 1
    int count;
};

template <typename T, int BITS, int PIECES, int ROWS>
__global__ void __launch_bounds__(32 * kRowWaves) affine_dot2_rows_group(AffineRows g, int lpc) {
    const int b = blockIdx.x;
    const int p = b >= g.first[3] && g.count > 3 ? 3 : b >= g.first[2] && g.count > 2 ? 2 : b >= g.first[1] ? 1 : 0;
    row_tile<T, BITS, PIECES, ROWS>(g.part[p], lpc, b - g.first[p]);
}

// Lanes a column: the groups rounded up to a power of two, at most 32; under four groups keeps the column tile.
inline int row_lanes(int groups) {
    if (groups < 4) return 0;
    int lpc = 4;
    while (lpc < groups && lpc < 32) lpc <<= 1;
    return lpc;
}

template <typename T, int BITS>
hipError_t launch_row_as(const Affine& a, int lpc, hipStream_t stream, int items) {
    const dim3 block(32 * kRowWaves);
    const dim3 grid((a.n + kRowWaves * kRowCols - 1) / (kRowWaves * kRowCols), 1, items);
    if (a.m == 2) {
        switch (a.group) {
            case 32: affine_dot2_row<T, BITS, 1, 2><<<grid, block, 0, stream>>>(a, lpc); break;
            case 64: affine_dot2_row<T, BITS, 2, 2><<<grid, block, 0, stream>>>(a, lpc); break;
            default: affine_dot2_row<T, BITS, 4, 2><<<grid, block, 0, stream>>>(a, lpc); break;
        }
        return hipGetLastError();
    }
    switch (a.group) {
        case 32: affine_dot2_row<T, BITS, 1><<<grid, block, 0, stream>>>(a, lpc); break;
        case 64: affine_dot2_row<T, BITS, 2><<<grid, block, 0, stream>>>(a, lpc); break;
        default: affine_dot2_row<T, BITS, 4><<<grid, block, 0, stream>>>(a, lpc); break;
    }
    return hipGetLastError();
}

template <typename T, int BITS>
hipError_t launch_rows_group_as(const AffineRows& g, int lpc, hipStream_t stream) {
    const dim3 grid(g.first[g.count]);
    const dim3 block(32 * kRowWaves);
    const Affine& a = g.part[0];
    if (a.m == 2) {
        switch (a.group) {
            case 32: affine_dot2_rows_group<T, BITS, 1, 2><<<grid, block, 0, stream>>>(g, lpc); break;
            case 64: affine_dot2_rows_group<T, BITS, 2, 2><<<grid, block, 0, stream>>>(g, lpc); break;
            default: affine_dot2_rows_group<T, BITS, 4, 2><<<grid, block, 0, stream>>>(g, lpc); break;
        }
    } else {
        switch (a.group) {
            case 32: affine_dot2_rows_group<T, BITS, 1, 1><<<grid, block, 0, stream>>>(g, lpc); break;
            case 64: affine_dot2_rows_group<T, BITS, 2, 1><<<grid, block, 0, stream>>>(g, lpc); break;
            default: affine_dot2_rows_group<T, BITS, 4, 1><<<grid, block, 0, stream>>>(g, lpc); break;
        }
    }
    return hipGetLastError();
}

template <int BITS>
hipError_t launch_row(const Affine& a, int lpc, hipStream_t stream, int items) {
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) return launch_row_as<DotBF16, BITS>(a, lpc, stream, items);
#endif
    return launch_row_as<DotF16, BITS>(a, lpc, stream, items);
}

// One or two rows on the row tile; false when the shape keeps the column tile.
bool launch_affine_dot2_row(const Affine& a, hipStream_t stream, int items, hipError_t* err) {
    if (a.m < 1 || a.m > 2 || a.group % 32 || a.group > kLaneGroupMax || a.k % a.group) return false;
    const int lpc = row_lanes(a.k / a.group);
    if (lpc == 0) return false;
    switch (a.bits) {
        case 2: *err = launch_row<2>(a, lpc, stream, items); break;
        case 3: *err = launch_row<3>(a, lpc, stream, items); break;
        case 4: *err = launch_row<4>(a, lpc, stream, items); break;
        case 5: *err = launch_row<5>(a, lpc, stream, items); break;
        case 6: *err = launch_row<6>(a, lpc, stream, items); break;
        case 8: *err = launch_row<8>(a, lpc, stream, items); break;
        default: return false;
    }
    return true;
}

template <typename T, int BITS, int PIECES>
hipError_t launch_lanes_as(const Affine& a, hipStream_t stream, int items) {
    const dim3 block(kLaneCols * kLaneWaves);
    const dim3 grid((a.n + kLaneCols - 1) / kLaneCols, 1, items);
    if (a.m == 1) {
        affine_dot2_lanes<T, BITS, 1, PIECES><<<grid, block, 0, stream>>>(a);
    } else if (a.m == 2) {
        affine_dot2_lanes<T, BITS, 2, PIECES><<<grid, block, 0, stream>>>(a);
    } else if (a.m <= 4) {
        affine_dot2_lanes<T, BITS, 4, PIECES><<<grid, block, 0, stream>>>(a);
    } else {
        affine_dot2_lanes<T, BITS, kLaneRows, PIECES><<<grid, block, 0, stream>>>(a);
    }
    return hipGetLastError();
}

template <int BITS, int PIECES>
hipError_t launch_lanes_rows(const Affine& a, hipStream_t stream, int items = 1) {
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) return launch_lanes_as<DotBF16, BITS, PIECES>(a, stream, items);
#endif
    return launch_lanes_as<DotF16, BITS, PIECES>(a, stream, items);
}

template <int BITS>
hipError_t launch_lanes(const Affine& a, hipStream_t stream, int items) {
    switch (a.group) {
        case 32: return launch_lanes_rows<BITS, 1>(a, stream, items);
        case 64: return launch_lanes_rows<BITS, 2>(a, stream, items);
        default: return launch_lanes_rows<BITS, 4>(a, stream, items);
    }
}

hipError_t launch_affine_dot2_lanes(const Affine& a, hipStream_t stream, int items) {
    if (a.m < 1 || a.m > kLaneRows || a.n < 1 || a.group % 32 || a.group > kLaneGroupMax || a.k % a.group) {
        return hipErrorInvalidValue;
    }
    hipError_t row = hipSuccess;
    if (launch_affine_dot2_row(a, stream, items, &row)) return row;
    switch (a.bits) {
        case 2: return launch_lanes<2>(a, stream, items);
        case 3: return launch_lanes<3>(a, stream, items);
        case 4: return launch_lanes<4>(a, stream, items);
        case 5: return launch_lanes<5>(a, stream, items);
        case 6: return launch_lanes<6>(a, stream, items);
        case 8: return launch_lanes<8>(a, stream, items);
        default: return hipErrorInvalidValue;
    }
}

// The parts share x, m, k, bits, group and the scale type. hipErrorNotSupported when a solo launch of one of them
// would not take the row tile, so the caller keeps the solo launches.
hipError_t launch_affine_dot2_rows_group(const Affine* parts, int count, hipStream_t stream) {
#if TENSORFOLD_RDNA_WMMA
    if (count < 1 || count > 4) return hipErrorInvalidValue;
    const Affine& a = parts[0];
    if (a.fp16 || a.m < 1 || a.m > 2 || a.group % 32 || a.group > kLaneGroupMax || a.k % a.group || a.k % 16) {
        return hipErrorNotSupported;
    }
    const int lpc = row_lanes(a.k / a.group);
    if (lpc == 0) return hipErrorNotSupported;
    AffineRows g{};
    g.count = count;
    for (int p = 0; p < count; ++p) {
        const Affine& b = parts[p];
        if (b.m != a.m || b.k != a.k || b.bits != a.bits || b.group != a.group || b.fp16 || b.n < 1 ||
            b.scale.kind != a.scale.kind) {
            return hipErrorNotSupported;
        }
        g.part[p] = b;
        g.first[p + 1] = g.first[p] + (b.n + kRowWaves * kRowCols - 1) / (kRowWaves * kRowCols);
    }
    switch (a.bits) {
        case 2: return launch_rows_group_as<DotBF16, 2>(g, lpc, stream);
        case 3: return launch_rows_group_as<DotBF16, 3>(g, lpc, stream);
        case 4: return launch_rows_group_as<DotBF16, 4>(g, lpc, stream);
        case 5: return launch_rows_group_as<DotBF16, 5>(g, lpc, stream);
        case 6: return launch_rows_group_as<DotBF16, 6>(g, lpc, stream);
        case 8: return launch_rows_group_as<DotBF16, 8>(g, lpc, stream);
        default: return hipErrorNotSupported;
    }
#else
    (void)parts;
    (void)count;
    (void)stream;
    return hipErrorNotSupported;
#endif
}

// GEMM tile: 128 x 128, a thread 8 rows x 8 columns strided by 16, K through double-buffered LDS 32 codes a stage.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kBlockLd = kBlockK + 2;  // 17 words a row: the 16 column threads of a wave read 16 banks

template <typename T, int BITS>
__device__ inline void stage_block(typename T::elem* xs, typename T::elem* ws, const Affine& a, int m0, int n0, int k0,
                                   int words_row) {
    const typename T::elem* x = static_cast<const typename T::elem*>(a.x);
    // x: 128 rows x 32 halves, 16 halves (two 16-byte loads) a thread.
    {
        const int r = threadIdx.x >> 1;
        const int half = (threadIdx.x & 1) * 16;
        const int gr = m0 + r;
        uint4 lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 hi = lo;
        if (gr < a.m) {
            const uint4* src = reinterpret_cast<const uint4*>(x + x_row(a, gr) * a.k + k0 + half);
            lo = src[0];
            hi = src[1];
        }
        uint32_t* dst = reinterpret_cast<uint32_t*>(xs + r * kBlockLd + half);
        dst[0] = lo.x; dst[1] = lo.y; dst[2] = lo.z; dst[3] = lo.w;
        dst[4] = hi.x; dst[5] = hi.y; dst[6] = hi.z; dst[7] = hi.w;
    }
    // Codes: one 32-code piece a column, threads 0..127.
    if (threadIdx.x < kBlockN) {
        const int c = threadIdx.x;
        const int gc = n0 + c;
        typename T::elem* dst = ws + c * kBlockLd;
        if (gc < a.n) {
            uint32_t w[BITS];
            load_piece<BITS>(a.words + static_cast<long long>(gc) * words_row + k0 / 32 * BITS, w);
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) =
                    T::two(T::template code<BITS>(w, t), T::template code<BITS>(w, t + 1));
            }
        } else {
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) = T::two(T::zero(), T::zero());
            }
        }
    }
}

// stage_block in two halves, so a stage's global reads are in flight while the tile multiplies the current one: fetch
// into registers before the product loop, put into the other LDS buffer after it.
template <int BITS>
struct StageRegs {
    uint4 lo, hi;
    uint32_t w[BITS];
};

template <typename T, int BITS>
__device__ inline void fetch_block(StageRegs<BITS>& r, const Affine& a, int m0, int n0, int k0, int words_row) {
    const typename T::elem* x = static_cast<const typename T::elem*>(a.x);
    const int row = threadIdx.x >> 1;
    const int half = (threadIdx.x & 1) * 16;
    const int gr = m0 + row;
    r.lo = make_uint4(0u, 0u, 0u, 0u);
    r.hi = r.lo;
    if (gr < a.m) {
        const uint4* src = reinterpret_cast<const uint4*>(x + x_row(a, gr) * a.k + k0 + half);
        r.lo = src[0];
        r.hi = src[1];
    }
    const int gc = n0 + static_cast<int>(threadIdx.x);
    if (threadIdx.x < kBlockN && gc < a.n) {
        load_piece<BITS>(a.words + static_cast<long long>(gc) * words_row + k0 / 32 * BITS, r.w);
    }
}

template <typename T, int BITS>
__device__ inline void put_block(const StageRegs<BITS>& r, typename T::elem* xs, typename T::elem* ws, const Affine& a,
                                 int n0) {
    {
        const int row = threadIdx.x >> 1;
        const int half = (threadIdx.x & 1) * 16;
        uint32_t* dst = reinterpret_cast<uint32_t*>(xs + row * kBlockLd + half);
        dst[0] = r.lo.x; dst[1] = r.lo.y; dst[2] = r.lo.z; dst[3] = r.lo.w;
        dst[4] = r.hi.x; dst[5] = r.hi.y; dst[6] = r.hi.z; dst[7] = r.hi.w;
    }
    if (threadIdx.x < kBlockN) {
        const int c = threadIdx.x;
        typename T::elem* dst = ws + c * kBlockLd;
        if (n0 + c < a.n) {
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) =
                    T::two(T::template code<BITS>(r.w, t), T::template code<BITS>(r.w, t + 1));
            }
        } else {
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) = T::two(T::zero(), T::zero());
            }
        }
    }
}

// A row's group sum of x (what the bias multiplies) is one thread's: thread r adds row r's 32 values a stage in k order
// and keeps the running sum to the group's end, the order every tile uses, so the bits are unchanged; the product loop
// is then dot2s and LDS reads alone, where each of the 16 threads sharing a row used to add the row again.
template <typename T, int BITS>
__global__ void __launch_bounds__(256, 1) affine_dot2_block(Affine a) {
    if (!take_item(a, blockIdx.z)) return;
    if (static_cast<int>(blockIdx.y) * kBlockM >= a.m) return;
    __shared__ typename T::elem xs[2][kBlockM * kBlockLd];
    __shared__ typename T::elem ws[2][kBlockN * kBlockLd];
    __shared__ float rowsum[kBlockM];
    // bf16 scales and biases eight groups at a time, [column][group]: one 16-byte read a column and table instead of
    // every thread reading its eight columns' values one by one at each group's fold. Same values, same bits.
    __shared__ alignas(16) uint16_t ssc[kBlockN * 8];
    __shared__ alignas(16) uint16_t sbi[kBlockN * 8];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int m0 = blockIdx.y * kBlockM;
    const int n0 = blockIdx.x * kBlockN;
    const int words_row = a.k * BITS / 32;
    const int groups = a.k / a.group;
    const int stages = a.k / kBlockK;
    const int per = a.group / kBlockK;
    float acc[8][8];
    float dot[8][8];
    float run = 0.f;                     // thread r < kBlockM: row r's sum of x since the group began
    const bool staged = a.scale.kind == kScaleBF16 && a.bias.kind == kScaleBF16 && groups % 8 == 0 &&
                        ((reinterpret_cast<uintptr_t>(a.scale.p) | reinterpret_cast<uintptr_t>(a.bias.p)) & 15) == 0;
    const int chunk = 8 * per;           // stages a staged slab of eight groups lasts
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = 0.f;
            dot[i][j] = 0.f;
        }
    }
    stage_block<T, BITS>(xs[0], ws[0], a, m0, n0, 0, words_row);
    __syncthreads();
    for (int s = 0; s < stages; ++s) {
        const int cur = s & 1;
        uint4 sq = make_uint4(0u, 0u, 0u, 0u), bq = sq;
        const bool slab = staged && s % chunk == 0 && static_cast<int>(threadIdx.x) < kBlockN;
        if (slab) {
            const int col = n0 + static_cast<int>(threadIdx.x);
            if (col < a.n) {
                const long long at = static_cast<long long>(col) * groups + s / per;
                sq = *reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.scale.p) + at);
                bq = *reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.bias.p) + at);
            }
        }
        StageRegs<BITS> next;
        if (s + 1 < stages) fetch_block<T, BITS>(next, a, m0, n0, (s + 1) * kBlockK, words_row);
        const typename T::elem* xb = xs[cur];
        const typename T::elem* wb = ws[cur];
        if (static_cast<int>(threadIdx.x) < kBlockM) {
#pragma unroll
            for (int t = 0; t < kBlockK; t += 2) {
                const typename T::pair xp = *reinterpret_cast<const typename T::pair*>(xb + threadIdx.x * kBlockLd + t);
                run += T::lo(xp);
                run += T::hi(xp);
            }
        }
#pragma unroll 4
        for (int t = 0; t < kBlockK; t += 2) {
            typename T::pair xv[8];
            typename T::pair wv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xv[i] = *reinterpret_cast<const typename T::pair*>(xb + (ty + 16 * i) * kBlockLd + t);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wv[j] = *reinterpret_cast<const typename T::pair*>(wb + (tx + 16 * j) * kBlockLd + t);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) dot[i][j] = T::dot(xv[i], wv[j], dot[i][j]);
            }
        }
        if (s + 1 < stages) put_block<T, BITS>(next, xs[cur ^ 1], ws[cur ^ 1], a, n0);
        if (slab) {
            reinterpret_cast<uint4*>(ssc)[threadIdx.x] = sq;
            reinterpret_cast<uint4*>(sbi)[threadIdx.x] = bq;
        }
        if (s % per == per - 1) {
            if (static_cast<int>(threadIdx.x) < kBlockM) {
                rowsum[threadIdx.x] = run;
                run = 0.f;
            }
            __syncthreads();
            const int g = s / per;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + tx + 16 * j;
                float scale, bias;
                if (staged) {
                    const int slot = (tx + 16 * j) * 8 + (g & 7);
                    scale = __uint_as_float(static_cast<uint32_t>(ssc[slot]) << 16);
                    bias = __uint_as_float(static_cast<uint32_t>(sbi[slot]) << 16);
                } else {
                    const long long at = static_cast<long long>(col < a.n ? col : 0) * groups + g;
                    scale = a.scale[at];
                    bias = a.bias[at];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i][j] = fmaf(dot[i][j], scale, acc[i][j]);
                    acc[i][j] = fmaf(rowsum[ty + 16 * i], bias, acc[i][j]);
                    dot[i][j] = 0.f;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + ty + 16 * i;
        if (row >= a.m) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = n0 + tx + 16 * j;
            if (col < a.n) a.out[out_row(a, row) * a.n + col] = acc[i][j];
        }
    }
}

template <int BITS>
hipError_t launch_block(const Affine& a, hipStream_t stream, int items = 1) {
    const dim3 block(256);
    const dim3 grid((a.n + kBlockN - 1) / kBlockN, (a.m + kBlockM - 1) / kBlockM, items);
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) {
        affine_dot2_block<DotBF16, BITS><<<grid, block, 0, stream>>>(a);
        return hipGetLastError();
    }
#endif
    affine_dot2_block<DotF16, BITS><<<grid, block, 0, stream>>>(a);
    return hipGetLastError();
}

hipError_t launch_affine_dot2_block(const Affine& a, hipStream_t stream, int items) {
    if (a.m < 1 || a.n < 1 || a.group % kBlockK || a.k % a.group || a.k % 16) return hipErrorInvalidValue;
    switch (a.bits) {
        case 2: return launch_block<2>(a, stream, items);
        case 3: return launch_block<3>(a, stream, items);
        case 4: return launch_block<4>(a, stream, items);
        case 5: return launch_block<5>(a, stream, items);
        case 6: return launch_block<6>(a, stream, items);
        case 8: return launch_block<8>(a, stream, items);
        default: return hipErrorInvalidValue;
    }
}

hipError_t launch_affine_dot2_routed(const Affine& a, int items, hipStream_t stream) {
    // BF16 activations are the gfx11 / gfx12 dot2 (the RDNA3 experts); FP16 the RDNA2 one.
    if ((!a.fp16 && !TENSORFOLD_RDNA_WMMA) || a.route.items == nullptr || a.route.members == nullptr || items < 1 ||
        a.route.x_div < 1) {
        return hipErrorInvalidValue;
    }
    if (a.group != 32 && a.group != 64 && a.group != 128) return hipErrorInvalidValue;
    if (a.m <= kLaneRows) return launch_affine_dot2_lanes(a, stream, items);
    hipError_t lut = ensure_byte_lut(stream);
    if (lut != hipSuccess) return lut;
    return launch_affine_dot2_block(a, stream, items);
}

// gfx11 / gfx12 BF16 x: the decode tiles up to 8 rows, the GEMM tile past them.
hipError_t launch_affine_dot2_bf16(const Affine& a, hipStream_t stream) {
#if TENSORFOLD_RDNA_WMMA
    if (a.fp16 || a.m < 1 || a.n < 1 || a.k % a.group || a.k % 16) return hipErrorInvalidValue;
    if (a.group != 32 && a.group != 64 && a.group != 128) return hipErrorInvalidValue;
    if (a.m <= kLaneRows) return launch_affine_dot2_lanes(a, stream);
    return launch_affine_dot2_block(a, stream);
#else
    (void)a;
    (void)stream;
    return hipErrorNotSupported;
#endif
}

}  // namespace rocm
}  // namespace tf

void affine_routed_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind,
                          void* out, const int* items, int count, const int* members, int x_div, int rows, int n,
                          int k, int bits, int group, int fp16, void* stream) {
    tf::rocm::Affine a{x, static_cast<const uint32_t*>(words), {scale, scale_kind}, {bias, scale_kind},
                       static_cast<float*>(out), rows, n, k, bits, group, fp16};
    a.route.items = items;
    a.route.members = members;
    a.route.x_div = x_div;
    hipError_t err = tf::rocm::launch_affine_dot2_routed(a, count, static_cast<hipStream_t>(stream));
    if (err != hipSuccess) throw std::runtime_error(std::string("affine routed kernel: ") + hipGetErrorString(err));
}

bool affine_rows_group_launch(const void* x, const void* const* words, const void* const* scale,
                              const void* const* bias, int scale_kind, void* const* out, const int* ns, int count,
                              int m, int k, int bits, int group, void* stream) {
    tf::rocm::Affine parts[4];
    for (int p = 0; p < count && p < 4; ++p) {
        parts[p] = tf::rocm::Affine{x, static_cast<const uint32_t*>(words[p]), {scale[p], scale_kind},
                                    {bias[p], scale_kind}, nullptr, m, ns[p], k, bits, group, 0};
        parts[p].outb16 = static_cast<hip_bfloat16*>(out[p]);
    }
    hipError_t err = tf::rocm::launch_affine_dot2_rows_group(parts, count, static_cast<hipStream_t>(stream));
    if (err == hipErrorNotSupported) return false;
    if (err != hipSuccess) throw std::runtime_error(std::string("affine rows group: ") + hipGetErrorString(err));
    return true;
}
