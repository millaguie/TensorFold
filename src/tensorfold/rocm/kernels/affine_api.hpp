#pragma once

// x (m, k) bf16, or fp16 on RDNA2; words (n, k * bits / 32); scale and bias (n, k / group) of one type; out (m, n).
void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half = 0);
// out_half: out is fp16, rounded by the RDNA2 decode tile (fp16 x, m <= 8, schedule 0, one launch).
int affine_dot2_splits(int m, int n, int k, int group, int mode);
void affine_pair_launch(const void* x, const void* words0, const void* scale0, const void* bias0, void* out0,
                        const void* words1, const void* scale1, const void* bias1, void* out1, int scale_kind, int m,
                        int n, int k, int bits, int group, void* stream);
// Up to four packed products that share x and K. Each column tile matches a solo WMMA launch.
void affine_group_launch(const void* x, const void* const* words, const void* const* scale, const void* const* bias,
                         int scale_kind, void* const* out, const int* ns, int nsides, int m, int k, int bits,
                         int group, void* stream);
// Every item of a routed plan in one launch over stacked (E, n, ...) weights; out (pairs, n) fp32 by pair id.
void affine_routed_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind,
                          void* out, const int* items, int count, const int* members, int x_div, int rows, int n,
                          int k, int bits, int group, int fp16, void* stream);
// Up to four products of one bf16 x with m <= 2 on the gfx11 / gfx12 one-row tile, bf16 out (m, n) each, every
// output with its solo launch's bits. False when a part would not take the row tile alone.
bool affine_rows_group_launch(const void* x, const void* const* words, const void* const* scale,
                              const void* const* bias, int scale_kind, void* const* out, const int* ns, int count,
                              int m, int k, int bits, int group, void* stream);
