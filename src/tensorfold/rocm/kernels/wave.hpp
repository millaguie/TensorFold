#pragma once

// Wave32 lane sums on the VALU. __shfl_xor and __shfl compile to ds_bpermute, a round trip through LDS; these keep
// the same xor tree (16, 8, 4, 2, 1) on v_permlanex16 and DPP row_xmask, so a sum's bits do not change.

#include <hip/hip_runtime.h>

// The tree and readfirstlane are the old sums' only on wave32 (build.py passes -mno-wavefrontsize64).
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "wave.hpp lane sums are wave32"
#endif

namespace tf {
namespace rocm {

// x from lane (lane ^ M) of a full wave32.
template <int M>
__device__ inline float swap_xor(float x) {
    const int bits = __builtin_bit_cast(int, x);
    if constexpr (M == 16) {
        return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(bits, bits, 0x76543210, 0xfedcba98, false, false));
    } else {
        return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(bits, 0x160 + M, 0xf, 0xf, false));
    }
}

// for (mask = 16; mask; mask >>= 1) x += __shfl_xor(x, mask); return __shfl(x, 0): lane 0's total in every lane.
// Every lane of the wave has to be active.
__device__ inline float wave_lane0_sum(float x) {
    x += swap_xor<16>(x);
    x += swap_xor<8>(x);
    x += swap_xor<4>(x);
    x += swap_xor<2>(x);
    x += swap_xor<1>(x);
    return __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, x)));
}

}  // namespace rocm
}  // namespace tf
