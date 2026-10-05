# ROCm

`--backend rocm` serves Qwen3.5 / Qwen3.8 dense and Qwen3.6-35B-A3B MoE MLX affine checkpoints on AMD Radeon GPUs,
behind the same torch server as CUDA. `auto` picks it where `/dev/kfd` exists and the family has a ROCm engine
(`rocm_engine`). Text only.

## Setup

Install a ROCm build of PyTorch, `hipcc` (ROCm) and `ninja`, then TensorFold:

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --backend rocm --name local-model --host 0.0.0.0 --port 8080
```

The first start compiles the HIP kernels for the visible GPU only (one gfx target, wave32); later starts reuse them.
An extension rebuilds when any of its sources, the headers beside them or its flags change. Pin a card with
`HIP_VISIBLE_DEVICES`.

| | |
| --- | --- |
| GPUs measured | Radeon PRO V620 (gfx1030, RDNA2, FP16 activations), Radeon PRO W7800 (gfx1100, RDNA3, BF16 activations) |
| GPUs built for | gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200-1201; RDNA1 is refused |
| Weights | MLX affine 2, 3, 4, 5, 6 and 8 bits, groups 32, 64 and 128, mixed widths per tensor as the config names them; scales and biases fp32, bf16 or fp16 as stored |
| Tested stacks | V620: ROCm 7.14, torch 2.12.0+rocm7.14.0. W7800: ROCm 7.2, torch 2.12.0+rocm7.2 |

Checkpoints measured below: `mlx-community/Qwen3.5-0.8B-MLX-8bit`, `mlx-community/Qwen3.5-4B-3bit`,
`mlx-community/Qwen3.5-9B-MLX-4bit` with the head of `mlx-community/Qwen3.5-9B-MTP-4bit` beside it as
`mtp.safetensors`, `mlx-community/Qwen3.5-9B-6bit`, `mlx-community/Qwen3.5-9B-MLX-8bit`,
`TensorFold/Qwen3.8-27B-MLX-4bit`, `leonsarmiento/Qwen3.8-27B-3bit-mtp-mlx` (3 bits, embeddings at 4, an
unquantized MTP `fc`) and `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP`.

The server takes one request at a time on ROCm. Chat completions, completions, the Responses API, `response_format`
grammars, tool calls, keyed sampling and the prefix cache work as on CUDA.

## Kernels

The weights stay packed on the GPU; each kernel unpacks the codes it reads. Every output is the group sum of
`x * code` in fp32, scaled and biased once per group, with no atomics, and a row's bits do not depend on how many
rows share the launch.

- Projections run the dot2 tiles (`affine_tiles.hip`, `affine_dot2.hip`) at every width: FP16 activations with
  `v_dot2_f32_f16` on RDNA2, BF16 with `v_dot2_f32_bf16` on gfx11 / gfx12. One row takes a tile whose lanes read a
  weight row contiguously, 2-8 rows a column tile, longer prompts a 128x128 GEMM tile. All of them keep one pair chain
  and one group fold, so a row's bits are the same alone, in a batch or in a prefill.
- MoE experts run the same tiles over a routing plan: one launch per expert projection for every routed pair.
- `TENSORFOLD_ROCM_SCHEDULE=wmma` runs gfx11's WMMA tiles instead, at every row count. They are slower than the dot2
  tiles today and are kept for later work.
- Attention: a split-over-keys decode walk, and one prefill kernel for every prefill row whatever its span (a Triton
  FA2 tile on gfx1030 when Triton is installed, the HIP FA2 tile otherwise), so a resumed prompt has a fresh one's
  bits. The KV cache is the activation dtype.
- The Gated DeltaNet recurrence of the linear-attention layers, bit-exact against its PyTorch reference.

## Decode

A request's one-token decode step is captured as a HIP graph after its first step and replayed, on every rank
(`TENSORFOLD_GRAPH=0` keeps it eager). On gfx1150 (Radeon 890M) it stays eager by default: a replayed step aborts
there with a malformed AQL packet unless `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` is set before HIP starts, and the graph
decodes no faster on that part; `TENSORFOLD_GRAPH=1` turns it back on. Decode attention and RoPE read the position on the device, so a replay gives
the eager step's bits.

## MTP drafting

A checkpoint's MTP head (its `mtp.*` tensors, with or without the `language_model.` prefix, or a side
`mtp*.safetensors`) drafts 4 tokens a round; `--no-drafts` turns it off. Each draft is verified with the serial
step's own sampling key, so a drafted reply is the serial reply token for token, greedy or sampled, at any tp.
Verification is one main forward a draft, so drafting does not speed decode up yet.

## Tensor parallel

One process a rank, RCCL between them. `--tp 2`, `4` or `8` are ROCm only. Heads and MLP columns are split per rank;
o, out and down are split by input groups and their fp32 shares are summed before each residual add; the vocabulary is
split for the head. With fewer KV heads than ranks each KV head is kept by the ranks whose query heads read it. A MoE
layer splits by expert: each rank holds `E / tp` routed experts (the shared one on rank 0), every rank routes every
token, and the ranks' fp32 shares are summed like a down projection.

On one host, give each rank its own card. With every card visible, rank r takes card r:

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 1 --master 127.0.0.1 &
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 0 --master 127.0.0.1 --name local-model
```

Rank 0 serves HTTP; the other ranks follow it. Two ranks on one card are refused. `--p2p` / `--no-p2p` force RCCL
peer-to-peer on or off; unset, RCCL decides.

## W4A16 (work in progress)

GPTQ / AWQ W4A16 experts run through grouped int4 kernels on gfx1030, for exports in the MLX layout (stacked
`switch_mlp` experts, an affine embedding), on one rank. Hugging Face GPTQ / AWQ exports (`quantization_config`,
per-expert tensors, unquantized dense layers) are refused at load.

## Measurements

`python -m tensorfold.rocm.bench MODEL_DIR 1024 256 1 --served [--mtp N] [--tp N --rank R --master ADDR]` times the
engine `tensorfold serve` runs, one request at a time: prefill is the prompt over the time to the first token,
decode the other tokens over the time after it, median of two runs. `... 1024 256 8` (without `--served`) is eight
requests in one batched generate.

**V620 (gfx1030), up to 8 cards**, served engine, 1,024-token prompt, 256 tokens, prefill / decode tok/s (MTP: the checkpoint's head drafting 2 a round; batched: 8 requests in one generate)

| Model | tp=1 | tp=2 | tp=4 | tp=8 | tp=1 MTP 2 | tp=2 MTP 2 | tp=4 MTP 2 | tp=8 MTP 2 | tp=1 batched x8 | tp=2 batched x8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B 8-bit | 7,266 / 129 | 10,283 / 104 | 5,893 / 101 | 2,893 / 66.5 | - | - | - | - | 6,891 / 328 | 11,137 / 296 |
| 4B 3-bit | 1,622 / 51.8 | 2,685 / 52.0 | 1,744 / 51.8 | 881 / 39.9 | - | - | - | - | 1,507 / 207 | 2,590 / 219 |
| 9B 4-bit + MTP head | 829 / 28.9 | 1,418 / 39.5 | 1,067 / 44.5 | 545 / 36.2 | 840 / 28.8 | 1,415 / 39.8 | 1,056 / 43.4 | 544 / 36.2 | 840 / 112 | 1,434 / 166 |
| 9B 6-bit | 805 / 27.8 | 1,411 / 37.4 | 1,061 / 41.3 | 543 / 35.7 | - | - | - | - | 798 / 111 | 1,475 / 178 |
| 9B 8-bit | 773 / 19.6 | 1,336 / 31.1 | 1,028 / 38.7 | 544 / 35.3 | - | - | - | - | 781 / 80.6 | 1,367 / 124 |
| 27B 4-bit | 260 / 12.4 | 470 / 15.8 | 402 / 18.1 | 214 / 16.3 | - | - | - | - | 244 / 42.2 | 456 / 73.4 |
| 27B 3-bit + MTP head | 264 / 12.7 | 481 / 15.9 | 404 / 18.7 | 214 / 16.5 | 257 / 12.4 | 471 / 16.1 | 402 / 18.1 | 214 / 16.5 | 257 / 44.8 | 456 / 70.4 |
| 35B-A3B 4-bit + MTP head | 1,217 / 59.5 | 1,807 / 36.4 | 1,493 / 35.4 | 815 / 28.0 | 1,221 / 59.2 | 1,864 / 36.3 | 1,489 / 34.2 | 813 / 28.0 | 1,514 / 128 | 2,052 / 108 |

Mode gates (graphs off/on x serial/drafted, greedy and t=0.7, identical replies): 32/32 pass

The GPTQ-Int4 35B-A3B export does not load (Hugging Face GPTQ exports are refused at load).

**W7800 (gfx1100), up to 2 cards, 180 W**, served engine, 1,024-token prompt, 256 tokens, prefill / decode tok/s (MTP: the checkpoint's head drafting 2 a round; batched: 8 requests in one generate)

| Model | tp=1 | tp=2 | tp=1 MTP 2 | tp=2 MTP 2 | tp=1 batched x8 | tp=2 batched x8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8B 8-bit | 6,911 / 121 | 9,614 / 121 | - | - | 6,848 / 397 | 10,111 / 362 |
| 4B 3-bit | 1,474 / 35.4 | 2,232 / 41.2 | - | - | 1,481 / 201 | 2,350 / 259 |
| 9B 4-bit + MTP head | 800 / 34.2 | 1,296 / 44.7 | 797 / 33.8 | 1,303 / 45.0 | 817 / 124 | 1,357 / 186 |
| 9B 6-bit | 822 / 31.1 | 1,345 / 41.3 | - | - | 834 / 123 | 1,389 / 192 |
| 9B 8-bit | 815 / 32.9 | 1,314 / 42.1 | - | - | 838 / 118 | 1,371 / 176 |
| 27B 4-bit | 235 / 11.7 | 406 / 15.9 | - | - | 240 / 43.3 | 417 / 72.3 |
| 27B 3-bit + MTP head | 243 / 10.1 | 422 / 13.2 | 245 / 10.0 | 424 / 13.3 | 247 / 43.7 | 430 / 73.1 |
| 35B-A3B 4-bit + MTP head | 1,108 / 59.1 | 1,513 / 38.2 | 1,102 / 58.8 | 1,617 / 38.2 | 1,465 / 136 | 2,012 / 149 |

Mode gates (graphs off/on x serial/drafted, greedy and t=0.7, identical replies): 16/16 pass

**W7800 (gfx1100), 180 W, TENSORFOLD_ROCM_SCHEDULE=wmma**, served engine, 1,024-token prompt, 256 tokens, prefill / decode tok/s (MTP: the checkpoint's head drafting 2 a round; batched: 8 requests in one generate)

| Model | tp=1 | tp=2 | tp=1 MTP 2 | tp=2 MTP 2 | tp=1 batched x8 | tp=2 batched x8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8B 8-bit | 3,165 / 48.2 | 5,298 / 62.7 | - | - | 1,957 / 248 | 5,931 / 387 |
| 4B 3-bit | 211 / 6.3 | 415 / 8.6 | - | - | 103 / 32.3 | 333 / 49.3 |
| 9B 4-bit + MTP head | 104 / 3.6 | 205 / 5.3 | 103 / 3.6 | 204 / 5.3 | 52.5 / 17.4 | 105 / 29.3 |
| 9B 6-bit | 99.9 / 3.5 | 201 / 5.2 | - | - | 50.6 / 16.7 | 101 / 28.4 |
| 9B 8-bit | 235 / 8.3 | 479 / 13.9 | - | - | 151 / 35.2 | 283 / 60.3 |
| 27B 4-bit | 29.0 / 1.3 | 58.4 / 2.1 | - | - | - / - | 32.4 / 9.1 |
| 27B 3-bit + MTP head | 29.1 / 1.3 | 59.2 / 2.1 | 28.6 / 1.3 | 59.4 / 2.1 | - / - | 31.2 / 10.1 |
| 35B-A3B 4-bit + MTP head | 431 / 11.2 | 766 / 12.4 | 429 / 11.2 | 763 / 12.4 | - / - | 457 / 55.5 |

Mode gates (graphs off/on x serial/drafted, greedy and t=0.7, identical replies): 16/16 pass

WMMA is opt-in and slower than the dot2 tiles today; `-`: not measured.

