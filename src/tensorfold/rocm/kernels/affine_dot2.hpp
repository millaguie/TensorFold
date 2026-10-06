#pragma once

// Code unpacking and the dot types shared by the dot2 files.

#include <hip/hip_fp16.h>

#include "affine.hpp"
#include "arch.hpp"

namespace tf {
namespace rocm {

constexpr int kLaneCols = 32;
constexpr int kLaneWaves = 8;
constexpr int kLaneRows = 8;
constexpr int kLaneGroupMax = 128;
constexpr int kBlockRows = 64;  // from here the GEMM tile beats the 128-row column tile

hipError_t ensure_byte_lut(hipStream_t stream);
hipError_t launch_affine_dot2_lanes(const Affine& a, hipStream_t stream, int items = 1);
hipError_t launch_affine_dot2_block(const Affine& a, hipStream_t stream, int items = 1);

// 32 codes are exactly BITS words: 16-byte loads when aligned and BITS % 4 == 0, 8-byte for even BITS.
template <int BITS>
__device__ inline void load_piece(const uint32_t* src, uint32_t (&w)[BITS]) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(src);
    if constexpr (BITS % 4 == 0) {
        if ((at & 15u) == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 4; ++i) {
                const uint4 v = reinterpret_cast<const uint4*>(src)[i];
                w[4 * i] = v.x;
                w[4 * i + 1] = v.y;
                w[4 * i + 2] = v.z;
                w[4 * i + 3] = v.w;
            }
            return;
        }
    }
    if constexpr (BITS % 2 == 0) {
        if ((at & 7u) == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 2; ++i) {
                const uint2 v = reinterpret_cast<const uint2*>(src)[i];
                w[2 * i] = v.x;
                w[2 * i + 1] = v.y;
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < BITS; ++i) w[i] = src[i];
}

// Code t of a 32-code piece as FP16. Codes are below 256, exact in BF16 and FP16, so this is code_f16(code).
template <int BITS>
__device__ inline uint32_t piece_bits(const uint32_t (&w)[BITS], int t) {
    const int bit = t * BITS;
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = w[word] >> shift;
    if (shift + BITS > 32) value |= w[word + 1] << (32 - shift);
    return value & ((1u << BITS) - 1u);
}

template <int BITS>
__device__ inline __half piece_code(const uint32_t (&w)[BITS], int t) {
    return __ushort2half_rn(static_cast<unsigned short>(piece_bits<BITS>(w, t)));
}

// torch's float to bf16 cast: round to nearest even, and every NaN to the one quiet NaN 0x7FC0 (hip_bfloat16 keeps
// a NaN's sign and payload), so a tile that rounds its own output matches the cast after an fp32 one.
__device__ inline hip_bfloat16 bf16_like_torch(float v) {
    hip_bfloat16 out(v);
    if (v != v) out.data = 0x7FC0;
    return out;
}

// Activation types of the tiles: FP16 v_dot2_f32_f16 (RDNA2) and BF16 v_dot2_f32_bf16 (gfx11 / gfx12).
struct DotF16 {
    using elem = __half;
    using pair = __half2;
    __device__ static elem zero() { return __float2half(0.f); }
    __device__ static pair two(elem a, elem b) { return __halves2half2(a, b); }
    __device__ static float lo(pair p) { return __low2float(p); }
    __device__ static float hi(pair p) { return __high2float(p); }
    template <int BITS>
    __device__ static elem code(const uint32_t (&w)[BITS], int t) { return piece_code<BITS>(w, t); }
    __device__ static float dot(pair x, pair q, float acc) { return __builtin_amdgcn_fdot2(x, q, acc, false); }
};

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

struct DotBF16 {
    using elem = __bf16;
    using pair = bf16x2;
    __device__ static elem zero() { return static_cast<__bf16>(0.f); }
    __device__ static pair two(elem a, elem b) { return pair{a, b}; }
    __device__ static float lo(pair p) { return static_cast<float>(p.x); }
    __device__ static float hi(pair p) { return static_cast<float>(p.y); }
    // A code is an integer below 256, exact in BF16: the rounding cast is the float's top half, which gfx11 takes
    // with a shift where the cast itself is a software round to nearest even.
    template <int BITS>
    __device__ static elem code(const uint32_t (&w)[BITS], int t) {
        static_assert(BITS <= 8, "codes up to 8 bits are exact in BF16");
        const float f = static_cast<float>(piece_bits<BITS>(w, t));
        return __builtin_bit_cast(__bf16, static_cast<unsigned short>(__float_as_uint(f) >> 16));
    }
    __device__ static float dot(pair x, pair q, float acc) {
#if TF_DEVICE_BF16_DOT2
        return __builtin_amdgcn_fdot2_f32_bf16(x, q, acc, false);
#else
        __builtin_trap();
        return acc;
#endif
    }
};

}  // namespace rocm
}  // namespace tf
