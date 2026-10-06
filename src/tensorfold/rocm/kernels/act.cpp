#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "act.hpp"

namespace {

void keep(const at::Tensor& tensor, c10::cuda::CUDAStream stream) {
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}

}  // namespace

void rms(const at::Tensor& x, const at::Tensor& weight, at::Tensor& y, double eps) {
    const auto type = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (type == at::kFloat || type == at::kHalf || type == at::kBFloat16),
                "x: (rows, d) fp32, fp16 or bf16");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes() && y.scalar_type() == type, "y matches x");
    const int64_t rows = x.size(0), width = x.size(1);
    TORCH_CHECK(rows >= 1 && width >= 1 && width <= 8192, "rms width");
    const float* wptr = nullptr;
    if (weight.defined() && weight.numel() > 0) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat &&
                        weight.numel() == width,
                    "rms weight");
        wptr = weight.data_ptr<float>();
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    keep(x, stream);
    keep(y, stream);
    if (wptr != nullptr) keep(weight, stream);
    const int kind = type == at::kHalf ? 1 : type == at::kBFloat16 ? 2 : 0;
    rms_launch(x.data_ptr(), wptr, y.data_ptr(), kind, static_cast<int>(rows), static_cast<int>(width),
               static_cast<float>(eps), stream.stream());
}

void conv_decode(const at::Tensor& x, const at::Tensor& weight, at::Tensor& state, at::Tensor& y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 3 && x.size(1) == 1,
                "x: (batch, 1, channels) fp32");
    const int64_t batch = x.size(0), channels = x.size(2);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat && weight.dim() == 2 &&
                    weight.size(0) == channels && weight.size(1) >= 1 && weight.size(1) <= 8,
                "weight: (channels, kernel) fp32, kernel 1..8");
    const int64_t kernel = weight.size(1);
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat &&
                    state.sizes() == at::IntArrayRef({batch, kernel - 1, channels}),
                "state: (batch, kernel - 1, channels) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kFloat && y.sizes() == x.sizes(), "y");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {x, weight, state, y}) keep(tensor, stream);
    conv_decode_launch(x.data_ptr<float>(), weight.data_ptr<float>(), state.data_ptr<float>(), y.data_ptr<float>(),
                       static_cast<int>(batch), static_cast<int>(channels), static_cast<int>(kernel), stream.stream());
}

void conv_prefill(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& state, at::Tensor& y) {
    const auto type = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 &&
                    (type == at::kFloat || type == at::kHalf || type == at::kBFloat16),
                "x: (batch, length, channels) fp32, fp16 or bf16");
    const int64_t batch = x.size(0), length = x.size(1), channels = x.size(2);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat && weight.dim() == 2 &&
                    weight.size(0) == channels && weight.size(1) >= 1 && weight.size(1) <= 8,
                "weight: (channels, kernel) fp32, kernel 1..8");
    const int64_t kernel = weight.size(1);
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat &&
                    state.sizes() == at::IntArrayRef({batch, kernel - 1, channels}),
                "state: (batch, kernel - 1, channels) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kFloat && y.sizes() == x.sizes(), "y");
    TORCH_CHECK(length <= 65535 && batch <= 65535, "length and batch: at most 65535 a launch");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {x, weight, state, y}) keep(tensor, stream);
    const int kind = type == at::kHalf ? 1 : type == at::kBFloat16 ? 2 : 0;
    conv_prefill_launch(x.data_ptr(), kind, weight.data_ptr<float>(), state.data_ptr<float>(), y.data_ptr<float>(),
                        static_cast<int>(batch), static_cast<int>(length), static_cast<int>(channels),
                        static_cast<int>(kernel), stream.stream());
}

void rope_decode(const at::Tensor& x, at::Tensor& y, int64_t pos, int64_t rotary, double theta,
                 const c10::optional<at::Tensor>& pos_dev) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 2, "x: (rows, d) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes(), "y");
    const int64_t rows = x.size(0), width = x.size(1);
    TORCH_CHECK(rotary > 0 && rotary <= width && rotary % 2 == 0 && width <= 8192, "rotary dim");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    keep(x, stream);
    keep(y, stream);
    const int* at = nullptr;
    if (pos_dev.has_value()) {
        TORCH_CHECK(pos_dev->is_cuda() && pos_dev->scalar_type() == at::kInt && pos_dev->numel() == 1,
                    "pos_dev: one int32 on the device");
        keep(*pos_dev, stream);
        at = pos_dev->data_ptr<int>();
    }
    rope_decode_launch(x.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(width),
                       static_cast<int>(rotary), static_cast<int>(pos), static_cast<float>(theta), stream.stream(), at);
}

int act_kind(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kFloat: return 0;
        case at::kHalf: return 1;
        case at::kBFloat16: return 2;
        default: TORCH_CHECK(false, "fp32, fp16 or bf16");
    }
    return 0;
}

void moe_router(const at::Tensor& x, const at::Tensor& rows, at::Tensor& logits) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
                "x: (R, D) fp16 or bf16");
    TORCH_CHECK(rows.is_cuda() && rows.is_contiguous() &&
                    (rows.scalar_type() == at::kFloat || rows.scalar_type() == at::kBFloat16) && rows.dim() == 2 &&
                    rows.size(1) == x.size(1),
                "rows: (E + 1, D) fp32, or bf16 for D of 1024, 2048 or 4096");
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.scalar_type() == at::kFloat &&
                    logits.size(0) == x.size(0) && logits.size(1) == rows.size(0),
                "logits: (R, E + 1) fp32");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {x, rows, logits}) keep(tensor, stream);
    if (rows.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(moe_router_bf16_launch(x.data_ptr(), act_kind(x), rows.data_ptr(), logits.data_ptr<float>(),
                                           static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                                           static_cast<int>(rows.size(0)), stream.stream()),
                    "bf16 router rows want D of 1024, 2048 or 4096");
        return;
    }
    moe_router_launch(x.data_ptr(), act_kind(x), rows.data_ptr<float>(), logits.data_ptr<float>(),
                      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<int>(rows.size(0)),
                      stream.stream());
}

void moe_select(const at::Tensor& logits, at::Tensor& pick, at::Tensor& wts, int64_t top_k,
                const c10::optional<at::Tensor>& items, const c10::optional<at::Tensor>& members) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.scalar_type() == at::kFloat && logits.dim() == 2,
                "logits: (R, E + 1) fp32");
    const int64_t r = logits.size(0), experts = logits.size(1) - 1, slots = top_k + 1;
    TORCH_CHECK(pick.is_cuda() && pick.is_contiguous() && pick.scalar_type() == at::kInt && pick.numel() >= r * slots,
                "pick: (R, top_k + 1) int32");
    TORCH_CHECK(wts.is_cuda() && wts.is_contiguous() && wts.scalar_type() == at::kFloat && wts.numel() >= r * slots,
                "wts: (R, top_k + 1) fp32");
    int* iptr = nullptr;
    int* mptr = nullptr;
    int capacity = 0;
    c10::cuda::CUDAGuard guard(logits.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (items.has_value()) {
        TORCH_CHECK(r == 1 && members.has_value(), "the plan is written for one row");
        TORCH_CHECK(items->is_cuda() && items->is_contiguous() && items->scalar_type() == at::kInt &&
                        items->size(1) == 3 && items->size(0) >= slots,
                    "items: (capacity, 3) int32");
        TORCH_CHECK(members->is_cuda() && members->scalar_type() == at::kInt && members->numel() >= slots, "members");
        iptr = items->data_ptr<int>();
        mptr = members->data_ptr<int>();
        capacity = static_cast<int>(items->size(0));
        keep(*items, stream);
        keep(*members, stream);
    }
    for (const at::Tensor& tensor : {logits, pick, wts}) keep(tensor, stream);
    moe_select_launch(logits.data_ptr<float>(), pick.data_ptr<int>(), wts.data_ptr<float>(), iptr, mptr, capacity,
                      static_cast<int>(r), static_cast<int>(experts), static_cast<int>(top_k), stream.stream());
}

void moe_act(const at::Tensor& both, at::Tensor& out, double limit) {
    TORCH_CHECK(both.is_cuda() && both.is_contiguous() && both.scalar_type() == at::kFloat && both.dim() == 2 &&
                    both.size(1) % 2 == 0,
                "both: (P, 2 NI) fp32, gate then up");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 && out.size(0) == both.size(0) &&
                    out.size(1) * 2 == both.size(1) && act_kind(out) != 0,
                "out: (P, NI) fp16 or bf16");
    c10::cuda::CUDAGuard guard(both.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    keep(both, stream);
    keep(out, stream);
    moe_act_launch(both.data_ptr<float>(), out.data_ptr(), act_kind(out), static_cast<int>(both.size(0)),
                   static_cast<int>(out.size(1)), static_cast<float>(limit), stream.stream());
}

void moe_combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out) {
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kFloat && y.dim() == 3, "y: (R, S, D) fp32");
    const int64_t r = y.size(0), slots = y.size(1), d = y.size(2);
    TORCH_CHECK(wts.is_cuda() && wts.is_contiguous() && wts.scalar_type() == at::kFloat && wts.numel() >= r * slots,
                "wts: (R, S) fp32");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == r && out.size(1) == d, "out: (R, D)");
    c10::cuda::CUDAGuard guard(y.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {y, wts, out}) keep(tensor, stream);
    moe_combine_launch(y.data_ptr<float>(), wts.data_ptr<float>(), out.data_ptr(), act_kind(out),
                       static_cast<int>(r), static_cast<int>(slots), static_cast<int>(d), stream.stream());
}

void gdn_gate(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_log, const at::Tensor& dt_bias,
              at::Tensor& gate, at::Tensor& beta) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_cuda() && b.is_contiguous() &&
                    a.scalar_type() == b.scalar_type() && a.numel() == b.numel(),
                "a and b: contiguous, one dtype, one size");
    const int64_t heads = a_log.numel();
    TORCH_CHECK(a_log.is_cuda() && a_log.is_contiguous() && a_log.scalar_type() == at::kFloat &&
                    dt_bias.is_cuda() && dt_bias.is_contiguous() && dt_bias.scalar_type() == at::kFloat &&
                    dt_bias.numel() == heads && heads >= 1 && a.numel() % heads == 0,
                "a_log and dt_bias: fp32 (heads), a and b whole rows of heads");
    TORCH_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.scalar_type() == at::kFloat &&
                    gate.numel() == a.numel() && beta.is_cuda() && beta.is_contiguous() &&
                    beta.scalar_type() == at::kFloat && beta.numel() == a.numel(),
                "gate and beta: fp32, a's size");
    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {a, b, a_log, dt_bias, gate, beta}) keep(tensor, stream);
    gdn_gate_launch(a.data_ptr(), b.data_ptr(), act_kind(a), a_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
                    gate.data_ptr<float>(), beta.data_ptr<float>(), static_cast<int>(a.numel()),
                    static_cast<int>(heads), stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_gate", &gdn_gate);
    m.def("moe_router", &moe_router);
    m.def("moe_select", &moe_select);
    m.def("moe_act", &moe_act);
    m.def("moe_combine", &moe_combine);
    m.def("rms", &rms);
    m.def("conv_decode", &conv_decode);
    m.def("conv_prefill", &conv_prefill);
    m.def("rope_decode", &rope_decode);
}
