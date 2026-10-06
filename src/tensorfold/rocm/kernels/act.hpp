#pragma once

#include <hip/hip_runtime.h>

// One row of RMSNorm, the causal conv (a step or a prompt), and a length-1 RoPE.

// kind: 0 fp32, 1 fp16, 2 bf16 for x and y. weight is fp32. The arithmetic is fp32.
void rms_launch(const void* x, const float* weight, void* y, int kind, int rows, int width, float eps,
                hipStream_t stream);
void conv_decode_launch(const float* x, const float* weight, float* state, float* y, int batch, int channels,
                        int kernel, hipStream_t stream);
// A prompt's conv and silu; x kind 0 fp32, 1 fp16, 2 bf16; state (batch, kernel - 1, channels) fp32 is read only.
void conv_prefill_launch(const void* x, int kind, const float* weight, const float* state, float* y, int batch,
                         int length, int channels, int kernel, hipStream_t stream);
// pos_dev, when set, is the position on the device (read when the kernel runs) and pos is ignored.
void rope_decode_launch(const float* x, float* y, int rows, int width, int rotary, int pos, float theta,
                        hipStream_t stream, const int* pos_dev = nullptr);

// Router logits in fp32 by one wave a logit (fixed order), so a row's bits do not depend on R. x kind 1 fp16, 2 bf16.
// The same logits from the router as stored in bf16, for D of 1024, 2048 or 4096; false for another D.
bool moe_router_bf16_launch(const void* x, int kind, const void* rows, float* logits, int r, int d, int e,
                            hipStream_t stream);
void moe_router_launch(const void* x, int kind, const float* rows, float* logits, int r, int d, int e,
                       hipStream_t stream);
// The CUDA pick rule per row; with items set (one row) it also writes the plan: item k = (pick_k, k, 1).
void moe_select_launch(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int r,
                       int experts, int top_k, hipStream_t stream);
// out[p, i] = silu(g) * u from both = [gate | up] fp32, clamped by limit when it is above 0.
void moe_act_launch(const float* both, void* out, int kind, int pairs, int width, float limit, hipStream_t stream);
// out[r, d] = sum over slots in order of y[r, s, d] * wts[r, s] in fp32, rounded once to out kind.
void moe_combine_launch(const float* y, const float* wts, void* out, int kind, int r, int slots, int d,
                        hipStream_t stream);
// Gated DeltaNet beta = sigmoid(b) and gate = exp(-exp(a_log) * softplus(a + dt_bias)) in fp32, one launch.
void gdn_gate_launch(const void* a, const void* b, int kind, const float* a_log, const float* dt_bias, float* gate,
                     float* beta, int count, int heads, hipStream_t stream);
