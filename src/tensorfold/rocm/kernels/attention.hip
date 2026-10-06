// Causal attention, fp32 scores and softmax; KV heads are read as head / (heads / kv_heads), never repeated.

#include "attention.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "attention_fa.hip"
#include "wave.hpp"

namespace tf {
namespace rocm {

// Every caller's wave is whole: the key and query loops are uniform across it.
__device__ inline float lane0_sum(float x) { return wave_lane0_sum(x); }

template <typename Cache>
__device__ float row_dot(const float* qreg, const Cache* row, int d, int lane) {
    float partial = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) partial += qreg[i] * static_cast<float>(row[idx]);
    }
    return lane0_sum(partial);
}

template <typename Cache>
__global__ void __launch_bounds__(32)
causal_kernel(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
              float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale, int q_pos0,
              long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    int qi = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (qi >= qlen) return;
    int lane = threadIdx.x;
    int kv = head / (heads / kv_heads);
    int visible = q_pos0 + qi + 1;
    if (visible > span) visible = span;

    const float* qrow = q + ((static_cast<long long>(batch) * heads + head) * qlen + qi) * d;
    float qreg[8] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) qreg[i] = qrow[idx];
    }

    const Cache* kbase = k + batch * k_sb + kv * k_sh;
    const Cache* vbase = v + batch * v_sb + kv * v_sh;
    float max_score = -INFINITY;
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        max_score = fmaxf(max_score, score);
    }
    float denom = 0.f;
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        denom += expf(score - max_score);
    }
    float acc[8] = {};
    for (int key = 0; key < visible; ++key) {
        float score = row_dot(qreg, kbase + key * k_ss, d, lane) * scale;
        float weight = expf(score - max_score) / denom;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane + i * 32;
            if (idx < d) acc[i] += weight * static_cast<float>((vbase + key * v_ss)[idx]);
        }
    }
    float* orow = out + ((static_cast<long long>(batch) * heads + head) * qlen + qi) * d;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        if (idx < d) orow[idx] = acc[i];
    }
}

}  // namespace rocm
}  // namespace tf

// Decode: a warp scores a key, a block per head reduces the softmax, then V streams in 128-key tiles.
template <typename Cache>
__global__ void score_keys(const float* __restrict__ q, const Cache* __restrict__ k, float* __restrict__ scores,
                           int span, int heads, int kv_heads, int d, float scale, int visible, long long k_sb,
                           long long k_sh, long long k_ss, const int* pos) {
    if (pos != nullptr) visible = min(*pos + 1, span);
    int lane = threadIdx.x & 31;
    int key = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int head = blockIdx.y;
    int batch = blockIdx.z;
    const float* qrow = q + (static_cast<long long>(batch) * heads + head) * d;
    float qreg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane + i * 32;
        qreg[i] = idx < d ? qrow[idx] : 0.f;
    }
    float partial = 0.f;
    if (key < visible) {
        int kv = head / (heads / kv_heads);
        const Cache* krow = k + batch * k_sb + static_cast<long long>(kv) * k_sh + static_cast<long long>(key) * k_ss;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane + i * 32;
            if (idx < d) partial += qreg[i] * static_cast<float>(krow[idx]);
        }
    }
    float dot = tf::rocm::lane0_sum(partial);
    if (lane == 0 && key < span) {
        float score = key < visible ? dot * scale : -INFINITY;
        scores[(static_cast<long long>(batch) * heads + head) * span + key] = score;
    }
}

__global__ void softmax_stats(const float* __restrict__ scores, float* __restrict__ stats, int span, int visible,
                              const int* pos) {
    if (pos != nullptr) visible = min(*pos + 1, span);
    int head = blockIdx.x;
    int batch = blockIdx.y;
    const float* row = scores + (static_cast<long long>(batch) * gridDim.x + head) * span;
    float local = -INFINITY;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) local = fmaxf(local, row[i]);
    __shared__ float buf[256];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + step]);
        __syncthreads();
    }
    float max_score = buf[0];
    // Every wave reads the max before thread 0 reuses buf[0] for its sum: a wave held back (preempted by another
    // process's work, for one) otherwise reads a partial sum as the max and the head's softmax goes wrong.
    __syncthreads();
    local = 0.f;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) local += expf(row[i] - max_score);
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (threadIdx.x < step) buf[threadIdx.x] += buf[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* dest = stats + (static_cast<long long>(batch) * gridDim.x + head) * 2;
        dest[0] = max_score;
        dest[1] = buf[0];
    }
}

template <typename Cache>
__global__ void apply_values(const float* __restrict__ scores, const float* __restrict__ stats,
                             const Cache* __restrict__ v, float* __restrict__ partials, int span, int heads,
                             int kv_heads, int d, int visible, long long v_sb, long long v_sh, long long v_ss,
                             const int* pos) {
    if (pos != nullptr) visible = min(*pos + 1, span);
    int tile = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int lane = threadIdx.x;
    int key0 = tile * 128;
    if (lane >= d) return;
    if (key0 >= visible) {  // a tile past the keys in use (the graph's grid covers the whole cache) adds zero
        partials[(((static_cast<long long>(batch) * heads + head) * gridDim.x) + tile) * d + lane] = 0.f;
        return;
    }
    int kv = head / (heads / kv_heads);
    const float* row = scores + (static_cast<long long>(batch) * heads + head) * span;
    float max_score = stats[(static_cast<long long>(batch) * heads + head) * 2];
    float denom = stats[(static_cast<long long>(batch) * heads + head) * 2 + 1];
    const Cache* base = v + batch * v_sb + static_cast<long long>(kv) * v_sh + lane;
    float acc = 0.f;
    int stop = key0 + 128;
    if (stop > visible) stop = visible;
    for (int key = key0; key < stop; ++key) {
        float weight = expf(row[key] - max_score) / denom;
        acc += weight * static_cast<float>(base[static_cast<long long>(key) * v_ss]);
    }
    int tiles = gridDim.x;
    partials[(((static_cast<long long>(batch) * heads + head) * tiles) + tile) * d + lane] = acc;
}

__global__ void sum_partials(const float* __restrict__ partials, float* __restrict__ out, int tiles, int d) {
    int head = blockIdx.x;
    int batch = blockIdx.y;
    int lane = threadIdx.x;
    if (lane >= d) return;
    const float* col = partials + ((static_cast<long long>(batch) * gridDim.x + head) * tiles) * d + lane;
    float acc = 0.f;
    for (int tile = 0; tile < tiles; ++tile) acc += col[static_cast<long long>(tile) * d];
    out[(static_cast<long long>(batch) * gridDim.x + head) * d + lane] = acc;
}

namespace {

struct Walk {
    dim3 score_grid, score_block, head_grid, tile_grid;
    hipStream_t stream;
    const float* q;
    const void* k;
    const void* v;
    float* scores;
    float* stats;
    float* partials;
    int span, heads, kv_heads, d;
    float scale;
    int visible;
    long long k_sb, k_sh, k_ss, v_sb, v_sh, v_ss;
    const int* pos;
};

// Score the keys, reduce the softmax, then stream V, for one cache type.
template <typename Cache>
void decode_walk(const Walk& w) {
    score_keys<Cache><<<w.score_grid, w.score_block, 0, w.stream>>>(
        w.q, static_cast<const Cache*>(w.k), w.scores, w.span, w.heads, w.kv_heads, w.d, w.scale, w.visible, w.k_sb,
        w.k_sh, w.k_ss, w.pos);
    softmax_stats<<<w.head_grid, 256, 0, w.stream>>>(w.scores, w.stats, w.span, w.visible, w.pos);
    apply_values<Cache><<<w.tile_grid, 256, 0, w.stream>>>(
        w.scores, w.stats, static_cast<const Cache*>(w.v), w.partials, w.span, w.heads, w.kv_heads, w.d, w.visible,
        w.v_sb, w.v_sh, w.v_ss, w.pos);
}

}  // namespace

void causal_launch(const float* q, const void* k, const void* v, float* out, int batch, int qlen, int span,
                   int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                   long long k_ss, long long v_sb, long long v_sh, long long v_ss, int cache_kind,
                   float* scores, float* stats, float* partials, void* stream, const int* pos) {
    if (d < 1 || d > 256 || heads < 1 || kv_heads < 1 || heads % kv_heads != 0 || batch < 1 || qlen < 1 ||
        span < 1 || q_pos0 < 0) {
        throw std::invalid_argument("causal attention wants d <= 256 and heads a multiple of kv heads");
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    int visible = q_pos0 + 1;
    if (visible > span) visible = span;
    if (scores != nullptr) {
        if (qlen != 1 || stats == nullptr || partials == nullptr) {
            throw std::invalid_argument("the decode walk is one query with scores, stats and partials");
        }
        // With a device position the keys in use are read on the device and the grid covers the whole cache.
        int tiles = pos != nullptr ? (span + 127) / 128 : (visible + 127) / 128;
        dim3 score_block(128);  // four warps, one key each
        dim3 score_grid((span + 3) / 4, heads, batch);
        dim3 head_grid(heads, batch);
        dim3 tile_grid(tiles, heads, batch);
        const Walk walk{score_grid, score_block, head_grid, tile_grid, hip_stream, q, k, v, scores, stats, partials,
                        span, heads, kv_heads, d, scale, visible, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss, pos};
        if (cache_kind == 0) {
            decode_walk<__half>(walk);
        } else if (cache_kind == 1) {
            decode_walk<hip_bfloat16>(walk);
        } else if (cache_kind == 2) {
            decode_walk<float>(walk);
        } else {
            throw std::invalid_argument("cache kind is fp16, bf16, or fp32");
        }
        sum_partials<<<head_grid, 256, 0, hip_stream>>>(partials, out, tiles, d);
    } else if (!tf::rocm::launch_fa_prefill(q, k, v, out, batch, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb,
                                            k_sh, k_ss, v_sb, v_sh, v_ss, cache_kind, hip_stream)) {
        // Odd head sizes stay on one wave per query. Even sizes take the flash query tile.
        if (cache_kind == 0) {
            tf::rocm::causal_kernel<__half><<<dim3(qlen, heads, batch), 32, 0, hip_stream>>>(
                q, static_cast<const __half*>(k), static_cast<const __half*>(v), out, qlen, span, heads, kv_heads, d,
                scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss);
        } else if (cache_kind == 1) {
            tf::rocm::causal_kernel<hip_bfloat16><<<dim3(qlen, heads, batch), 32, 0, hip_stream>>>(
                q, static_cast<const hip_bfloat16*>(k), static_cast<const hip_bfloat16*>(v), out, qlen, span, heads,
                kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss);
        } else if (cache_kind == 2) {
            tf::rocm::causal_kernel<float><<<dim3(qlen, heads, batch), 32, 0, hip_stream>>>(
                q, static_cast<const float*>(k), static_cast<const float*>(v), out, qlen, span, heads, kv_heads, d,
                scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss);
        } else {
            throw std::invalid_argument("cache kind is fp16, bf16, or fp32");
        }
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("causal attention: ") + hipGetErrorString(err));
}
