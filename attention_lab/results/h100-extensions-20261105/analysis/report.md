# CUDA decode and structured-attention backward experiments

Ratios are baseline latency / custom latency; values above 1 favor the custom implementation. Medians are independently recomputed from raw timing samples. Geometric means weight each measured pair equally.

## Runs and environment

| Run | Status | Aggregate eligible | GPU | PyTorch | CUDA | Triton |
| --- | --- | --- | --- | --- | --- | --- |
| Decode | completed | True | NVIDIA H100 80GB HBM3 | 2.11.0+cu130 | 13.0 | 3.6.0 |
| Backward | passed | True | NVIDIA H100 80GB HBM3 | 2.11.0+cu130 | 13.0 | 3.6.0 |

Decode evidence: `/Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-extensions-20261105/decode`. Backward evidence: `/Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-extensions-20261105/backward`.

## CUDA decode: fixed-split tiling ablation

30 shape/seed cases, 10 distinct shapes, seeds [211, 212, 213]. Each row compares token-at-a-time and tiled kernels at the same split count; both include the merge kernel.

| Split | Pairs | Wins | Geomean speedup | Min | Max |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 21 | 1.189 | 0.975 | 1.408 |
| 2 | 30 | 21 | 1.194 | 0.981 | 1.406 |
| 4 | 30 | 23 | 1.197 | 0.987 | 1.402 |
| 8 | 30 | 24 | 1.181 | 0.992 | 1.391 |
| 16 | 30 | 24 | 1.187 | 0.991 | 1.577 |
| 32 | 30 | 27 | 1.201 | 0.996 | 1.388 |

Pooled fixed-split comparison: **1.191×** across 180 pairs; 140 wins. Geometric-mean latency reduction: **16.059%**. These include correlated splits/seeds and are not independent statistical trials.

## CUDA decode: best observed configurations

The split is chosen using these same measurements. This is an oracle selection result, not a held-out dispatcher or production speedup.

| Baseline | Paired cases | Missing | Wins | Geomean speedup | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| cuda_token_split_1 | 30 | 0 | 30 | 21.522 | 4.417 | 55.613 |
| cuda_tile_split_1 | 30 | 0 | 30 | 18.102 | 3.605 | 43.199 |
| FlexAttention TRITON_DECODE | 30 | 0 | 27 | 2.061 | 0.437 | 3.986 |
| compiled FlexAttention TRITON | 30 | 0 | 30 | 1.816 | 1.356 | 2.936 |
| PyTorch Flash SDPA | 30 | 0 | 0 | 0.307 | 0.098 | 0.932 |

Best tile versus best token: 1.201× geometric mean. A ratio below 1 against SDPA means the custom kernel is slower than that optimized baseline. The corrected run includes both explicit TRITON_DECODE and regular TRITON FlexAttention backends.

## Training attention: per-case GPU graph replay

| Case | N | D | Mask | Operation | Baseline | Custom ms | Baseline ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 1024 | 64 | causal | backward_only | compiled FlexAttention (Triton backend) | 0.056 | 0.040 | 0.723 |
| 10 | 1024 | 64 | causal | backward_only | pytorch_sdpa_native_causal | 0.056 | 0.036 | 0.644 |
| 10 | 1024 | 64 | causal | forward_backward | compiled FlexAttention (Triton backend) | 0.071 | 0.071 | 0.993 |
| 10 | 1024 | 64 | causal | forward_backward | pytorch_sdpa_native_causal | 0.071 | 0.051 | 0.718 |
| 11 | 1024 | 64 | causal+window(127,None) | backward_only | compiled FlexAttention (Triton backend) | 0.020 | 0.019 | 0.961 |
| 11 | 1024 | 64 | causal+window(127,None) | forward_backward | compiled FlexAttention (Triton backend) | 0.026 | 0.037 | 1.399 |
| 12 | 4096 | 64 | causal+window(255,None) | backward_only | compiled FlexAttention (Triton backend) | 0.027 | 0.024 | 0.903 |
| 12 | 4096 | 64 | causal+window(255,None) | forward_backward | compiled FlexAttention (Triton backend) | 0.034 | 0.047 | 1.371 |
| 13 | 2048 | 128 | causal+same-document | backward_only | compiled FlexAttention (Triton backend) | 0.066 | 0.057 | 0.869 |
| 13 | 2048 | 128 | causal+same-document | forward_backward | compiled FlexAttention (Triton backend) | 0.092 | 0.084 | 0.917 |

## Complete optimizer steps: matched eager CUDA events

This measurement includes parameter restoration, forward/loss, backward, SGD and host submission gaps. It is separate from attention-only graph replay and is not an end-to-end model-training throughput claim.

| Case | N | D | Mask | Baseline | Custom ms | Baseline ms | Speedup | Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 512 | 64 | causal | pytorch_sdpa_native_causal | 1.182 | 0.916 | 0.775 | eager |
| 9 | 512 | 64 | causal | compiled FlexAttention (Triton backend) | 1.182 | 1.249 | 1.056 | eager |
| 14 | 1024 | 128 | causal+window(127,None) | compiled FlexAttention (Triton backend) | 1.161 | 1.305 | 1.124 | eager |

## Correctness

Decode preflight: 582 recorded comparisons, 582 finite; maximum absolute error 0.0039038658142089844. These are tolerance checks, not bitwise equivalence.

Backward study: 15/15 study cases passed (15 planned). Case types: {'gpu_unittests': 1, 'correctness': 8, 'transformer_optimizer_step': 2, 'attention_training': 4}.

GPU unittest suite: 2 tests, status passed, skipped 0. five mask/shape cases each in FP16 and BF16; D64/D128, ragged lengths, repeated labels, asymmetric tiles, plus broadcast upstream gradient

Flash SDPA was faster in all 30 measured pairs, even against each case's best observed custom split: 3.257× faster geometrically. The tiling improvement is an internal-kernel improvement and does not establish a win over Flash SDPA.

Raw output/dQ/dK/dV errors, optimizer gradient/update-delta comparisons, and nonzero-update checks are retained in `summary.json`. Near-zero reference gradients can make relative RMS errors large; absolute errors and the recorded tolerances must be inspected together.

## Scope and limitations

- One H100 and the recorded software versions; no cross-device claim or statistical confidence interval.
- Decode benchmarks use fixed synthetic BF16 shapes and CUDA graph replay; preallocated scratch and warm kernel execution include split/merge work but exclude compilation, allocation and Python dispatch.
- Setup and first-call measurements remain in JSON and are separate from warmed latency.
- Best observed configurations reuse the evaluation measurements for selection.
- Performance ratios from diagnostic, failed or incomplete runs are excluded from all aggregates.
- first derivatives only
- no dropout
- same sequence length for Q/K/V
- one shared document-label sequence across batch and heads
- attention GPU graph timings exclude Python dispatch; complete optimizer steps use matched eager CUDA events including host submission gaps
- Flex buffer donation disabled for retained backward, explicitly not the default donation setting
- correctness uses dense FP32 oracle outside timing
- time budget checked between cases; one compilation can exceed the remaining budget

## Diagnostic runs excluded from aggregates

- `/Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-extensions-20261105/diagnostics/decode-baseline-label`: completed; 0 error records retained in JSON.
- `/Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-extensions-20261105/diagnostics/backward-initial`: failed; 3 error records retained in JSON.
- `/Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-extensions-20261105/diagnostics/backward-stream`: failed; 7 error records retained in JSON.
