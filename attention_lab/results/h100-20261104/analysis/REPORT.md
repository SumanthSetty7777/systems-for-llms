# Attention GPU experiment report

Forward-only CUDA-graph replay timings unless a recorded environment states otherwise. Attention microbenchmarks and single-Transformer-block studies are reported separately. Excludes Python dispatch, mask setup, compilation, backward and full-model execution. Ratios greater than one favor the denominator; geometric means weight paired workload/input-seed median timings equally. No best-only filtering, cross-run pooling, or pooling of default and hinted FlexAttention baselines.

## ablation: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/ablation/study

Status: passed. Recorded 120 / 120 planned cases; passed 120, failed 0. Requested FlexAttention failures/unavailable: 0 / 36.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup

### Ablations

| Ratio | Pairs | Geomean | Range | Regressions | Missing / failed / ambiguous / incompatible |
|---|---:|---:|---:|---:|---:|
| csr_masked/csr_full | 36 | 1.069× | 0.8997–1.379× | 16 | 0 / 0 / 0 / 0 |
| csr_full/interval_split | 24 | 1.443× | 1.19–1.768× | 0 | 0 / 0 / 0 / 0 |
| interval_masked/interval_split | 24 | 0.9983× | 0.8425–1.23× | 9 | 0 / 0 / 0 / 0 |

Document masks are excluded as inapplicable to interval specialization; their counts remain in SUMMARY.json.

### External baselines

| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |
|---|---|---:|---:|---:|---:|---:|
| csr_full:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 36 | 1.409× | 0.7533–2.973× | 9 | 0 / 0 |
| csr_full:64-64-4-2 | sdpa / explicit_boolean_mask  | 24 | 4.609× | 1.545–11.87× | 0 | 0 / 0 |
| csr_full:64-64-4-2 | sdpa / native_causal  | 12 | 0.5148× | 0.4081–0.6276× | 12 | 0 / 0 |
| csr_masked:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 36 | 1.317× | 0.7007–2.993× | 12 | 0 / 0 |
| csr_masked:64-64-4-2 | sdpa / explicit_boolean_mask  | 24 | 4.382× | 1.281–12.3× | 0 | 0 / 0 |
| csr_masked:64-64-4-2 | sdpa / native_causal  | 12 | 0.4669× | 0.3794–0.5641× | 12 | 0 / 0 |
| interval_masked:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 24 | 1.688× | 1.001–3.416× | 0 | 0 / 0 |
| interval_masked:64-64-4-2 | sdpa / explicit_boolean_mask  | 12 | 9.284× | 5.488–16.62× | 0 | 0 / 0 |
| interval_masked:64-64-4-2 | sdpa / native_causal  | 12 | 0.7491× | 0.5434–1.001× | 11 | 0 / 0 |
| interval_split:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 24 | 1.685× | 1.128–2.954× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / explicit_boolean_mask  | 12 | 8.436× | 4.64–16.72× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / native_causal  | 12 | 0.8181× | 0.651–1.007× | 8 | 0 / 0 |

SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

### Setup observations

These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.

| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |
|---|---:|---:|---:|
| csr_full | 0.2014 | 0.8275 | 0.4161–358.8 |
| csr_masked | 0.2014 | 0.8487 | 0.4084–1366 |
| interval_masked | 0.1032 | 0.8394 | 0.4365–424.7 |
| interval_split | 0.09807 | 1.474 | 0.6269–790.3 |

## block: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/block/study

Status: passed. Recorded 8 / 8 planned cases; passed 8, failed 0. Requested FlexAttention failures/unavailable: 0 / 8.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. Batched CUDA graph replay of one full inference block, per block call; excludes Python dispatch, compilation, masks and graph capture. Not whole-model or application latency.

### One-block and isolated-attention comparisons

These measurements use the same BF16 block weights and input seed within each case. They measure one inference block, not whole-model or application latency.

| Baseline | Scope | Pairs | Geomean | Range | Regressions | Failed / unavailable |
|---|---|---:|---:|---:|---:|---:|
| flex_triton / triton_default | isolated_attention_forward | 8 | 1.748× | 1.245–2.757× | 0 | 0 |
| flex_triton / triton_default | single_block_forward | 8 | 1.129× | 1.041–1.271× | 0 | 0 |
| sdpa / explicit_boolean_mask | isolated_attention_forward | 4 | 8.091× | 4.445–15.42× | 0 | 0 |
| sdpa / explicit_boolean_mask | single_block_forward | 4 | 1.679× | 1.316–2.361× | 0 | 0 |
| sdpa / native_causal | isolated_attention_forward | 4 | 0.9345× | 0.7523–1.147× | 2 | 0 |
| sdpa / native_causal | single_block_forward | 4 | 0.9891× | 0.9166–1.105× | 2 | 0 |

Configured tiles were fixed before block evaluation. Full-block ratios and isolated-attention ratios remain separate; illustrative Amdahl estimates and exact per-case source values are in SUMMARY.json.


SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

## confirmation: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/confirmation/study

Status: passed. Recorded 24 / 24 planned cases; passed 24, failed 0. Requested FlexAttention failures/unavailable: 0 / 12.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup

### Confirmation

| Ratio | Pairs | Geomean | Range | Regressions | Missing / failed / ambiguous / incompatible |
|---|---:|---:|---:|---:|---:|
| fixed_default/transferred | 12 | 1.012× | 0.9895–1.036× | 3 | 0 / 0 / 0 / 0 |

### External baselines

| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |
|---|---|---:|---:|---:|---:|---:|
| csr_full:64-64-4-2:fixed_default | flex / triton_default {"BACKEND": "TRITON"} | 3 | 3.142× | 3.118–3.163× | 0 | 0 / 0 |
| csr_full:64-64-4-2:fixed_default | sdpa / explicit_boolean_mask  | 3 | 15.46× | 15.38–15.5× | 0 | 0 / 0 |
| csr_full:64-64-4-4:transferred | flex / triton_default {"BACKEND": "TRITON"} | 3 | 3.189× | 3.168–3.2× | 0 | 0 / 0 |
| csr_full:64-64-4-4:transferred | sdpa / explicit_boolean_mask  | 3 | 15.69× | 15.53–15.78× | 0 | 0 / 0 |
| interval_split:64-64-4-2:fixed_default | flex / triton_default {"BACKEND": "TRITON"} | 9 | 1.596× | 1.088–2.703× | 0 | 0 / 0 |
| interval_split:64-64-4-2:fixed_default | sdpa / explicit_boolean_mask  | 6 | 5.956× | 3.995–8.918× | 0 | 0 / 0 |
| interval_split:64-64-4-2:fixed_default | sdpa / native_causal  | 3 | 0.6964× | 0.6934–0.7005× | 3 | 0 / 0 |
| interval_split:64-64-4-2:transferred | flex / triton_default {"BACKEND": "TRITON"} | 3 | 2.696× | 2.661–2.733× | 0 | 0 / 0 |
| interval_split:64-64-4-2:transferred | sdpa / explicit_boolean_mask  | 3 | 4.063× | 4.048–4.077× | 0 | 0 / 0 |
| interval_split:64-64-4-3:transferred | flex / triton_default {"BACKEND": "TRITON"} | 6 | 1.248× | 1.102–1.403× | 0 | 0 / 0 |
| interval_split:64-64-4-3:transferred | sdpa / explicit_boolean_mask  | 3 | 8.794× | 8.668–8.958× | 0 | 0 / 0 |
| interval_split:64-64-4-3:transferred | sdpa / native_causal  | 3 | 0.7184× | 0.7176–0.719× | 3 | 0 / 0 |

SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

### Setup observations

These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.

| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |
|---|---:|---:|---:|
| csr_full | 5.145 | 1.645 | 1.466–593.4 |
| interval_split | 0.3483 | 2.233 | 0.8496–2047 |

## flex_hints: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/flex_hints/study

Status: passed. Recorded 24 / 24 planned cases; passed 24, failed 0. Requested FlexAttention failures/unavailable: 0 / 24.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup

### External baselines

| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |
|---|---|---:|---:|---:|---:|---:|
| interval_split:64-64-4-2 | flex / triton_contiguous_blocks {"BACKEND": "TRITON", "BLOCKS_ARE_CONTIGUOUS": true} | 24 | 1.661× | 1.033–2.978× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / explicit_boolean_mask  | 12 | 8.484× | 4.643–16.82× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / native_causal  | 12 | 0.8174× | 0.6563–1.013× | 9 | 0 / 0 |

SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

### Setup observations

These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.

| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |
|---|---:|---:|---:|
| interval_split | 0.1018 | 1.467 | 0.6091–1274 |

## smoke: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/smoke/study

Status: passed. Recorded 20 / 20 planned cases; passed 20, failed 0. Requested FlexAttention failures/unavailable: 0 / 2.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup

### External baselines

| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |
|---|---|---:|---:|---:|---:|---:|
| csr_full:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 2 | 2.012× | 1.911–2.118× | 0 | 10 / 0 |
| csr_full:64-64-4-2 | sdpa / explicit_boolean_mask  | 8 | 2.354× | 1.901–2.902× | 0 | 0 / 0 |
| csr_full:64-64-4-2 | sdpa / native_causal  | 4 | 1.043× | 0.8407–1.391× | 2 | 0 / 0 |
| interval_split:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.511× | 2.511–2.511× | 0 | 7 / 0 |
| interval_split:64-64-4-2 | sdpa / explicit_boolean_mask  | 4 | 3.284× | 2.809–4.163× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / native_causal  | 4 | 1.393× | 0.9748–1.799× | 1 | 0 / 0 |

SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

### Setup observations

These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.

| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |
|---|---:|---:|---:|
| csr_full | 0.02717 | 417.9 | 304.1–1563 |
| interval_split | 0.02063 | 715.2 | 509.3–967.6 |

## tuning: /Users/ssetty/dl_repos/systems-for-llms/attention_lab/results/h100-20261104/tuning/study

Status: passed. Recorded 96 / 96 planned cases; passed 96, failed 0. Requested FlexAttention failures/unavailable: 0 / 4.

Environment: NVIDIA H100 80GB HBM3; PyTorch 2.11.0+cu130; Triton 3.6.0; CUDA 13.0. CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup

### External baselines

| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |
|---|---|---:|---:|---:|---:|---:|
| csr_full:128-32-2-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7495× | 0.7495–0.7495× | 1 | 0 / 0 |
| csr_full:128-32-2-2 | sdpa / explicit_boolean_mask  | 1 | 1.62× | 1.62–1.62× | 0 | 0 / 0 |
| csr_full:128-32-2-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7516× | 0.7516–0.7516× | 1 | 0 / 0 |
| csr_full:128-32-2-3 | sdpa / explicit_boolean_mask  | 1 | 1.604× | 1.604–1.604× | 0 | 0 / 0 |
| csr_full:128-32-2-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7524× | 0.7524–0.7524× | 1 | 0 / 0 |
| csr_full:128-32-2-4 | sdpa / explicit_boolean_mask  | 1 | 1.601× | 1.601–1.601× | 0 | 0 / 0 |
| csr_full:128-32-4-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.657× | 1.657–1.657× | 0 | 0 / 0 |
| csr_full:128-32-4-2 | sdpa / explicit_boolean_mask  | 1 | 3.546× | 3.546–3.546× | 0 | 0 / 0 |
| csr_full:128-32-4-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.783× | 1.783–1.783× | 0 | 0 / 0 |
| csr_full:128-32-4-3 | sdpa / explicit_boolean_mask  | 1 | 3.823× | 3.823–3.823× | 0 | 0 / 0 |
| csr_full:128-32-4-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.661× | 1.661–1.661× | 0 | 0 / 0 |
| csr_full:128-32-4-4 | sdpa / explicit_boolean_mask  | 1 | 3.564× | 3.564–3.564× | 0 | 0 / 0 |
| csr_full:128-64-2-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7572× | 0.7572–0.7572× | 1 | 0 / 0 |
| csr_full:128-64-2-2 | sdpa / explicit_boolean_mask  | 1 | 1.616× | 1.616–1.616× | 0 | 0 / 0 |
| csr_full:128-64-2-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7776× | 0.7776–0.7776× | 1 | 0 / 0 |
| csr_full:128-64-2-3 | sdpa / explicit_boolean_mask  | 1 | 1.668× | 1.668–1.668× | 0 | 0 / 0 |
| csr_full:128-64-2-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 0.7666× | 0.7666–0.7666× | 1 | 0 / 0 |
| csr_full:128-64-2-4 | sdpa / explicit_boolean_mask  | 1 | 1.634× | 1.634–1.634× | 0 | 0 / 0 |
| csr_full:128-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.478× | 1.478–1.478× | 0 | 0 / 0 |
| csr_full:128-64-4-2 | sdpa / explicit_boolean_mask  | 1 | 3.155× | 3.155–3.155× | 0 | 0 / 0 |
| csr_full:128-64-4-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.62× | 1.62–1.62× | 0 | 0 / 0 |
| csr_full:128-64-4-3 | sdpa / explicit_boolean_mask  | 1 | 3.476× | 3.476–3.476× | 0 | 0 / 0 |
| csr_full:128-64-4-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.498× | 1.498–1.498× | 0 | 0 / 0 |
| csr_full:128-64-4-4 | sdpa / explicit_boolean_mask  | 1 | 3.239× | 3.239–3.239× | 0 | 0 / 0 |
| csr_full:64-32-2-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.739× | 1.739–1.739× | 0 | 0 / 0 |
| csr_full:64-32-2-2 | sdpa / explicit_boolean_mask  | 1 | 3.728× | 3.728–3.728× | 0 | 0 / 0 |
| csr_full:64-32-2-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.65× | 1.65–1.65× | 0 | 0 / 0 |
| csr_full:64-32-2-3 | sdpa / explicit_boolean_mask  | 1 | 3.574× | 3.574–3.574× | 0 | 0 / 0 |
| csr_full:64-32-2-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.654× | 1.654–1.654× | 0 | 0 / 0 |
| csr_full:64-32-2-4 | sdpa / explicit_boolean_mask  | 1 | 3.583× | 3.583–3.583× | 0 | 0 / 0 |
| csr_full:64-32-4-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.947× | 2.947–2.947× | 0 | 0 / 0 |
| csr_full:64-32-4-2 | sdpa / explicit_boolean_mask  | 1 | 6.365× | 6.365–6.365× | 0 | 0 / 0 |
| csr_full:64-32-4-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.939× | 2.939–2.939× | 0 | 0 / 0 |
| csr_full:64-32-4-3 | sdpa / explicit_boolean_mask  | 1 | 6.303× | 6.303–6.303× | 0 | 0 / 0 |
| csr_full:64-32-4-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.924× | 2.924–2.924× | 0 | 0 / 0 |
| csr_full:64-32-4-4 | sdpa / explicit_boolean_mask  | 1 | 6.31× | 6.31–6.31× | 0 | 0 / 0 |
| csr_full:64-64-2-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.61× | 1.61–1.61× | 0 | 0 / 0 |
| csr_full:64-64-2-2 | sdpa / explicit_boolean_mask  | 1 | 3.492× | 3.492–3.492× | 0 | 0 / 0 |
| csr_full:64-64-2-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.648× | 1.648–1.648× | 0 | 0 / 0 |
| csr_full:64-64-2-3 | sdpa / explicit_boolean_mask  | 1 | 3.573× | 3.573–3.573× | 0 | 0 / 0 |
| csr_full:64-64-2-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 1.66× | 1.66–1.66× | 0 | 0 / 0 |
| csr_full:64-64-2-4 | sdpa / explicit_boolean_mask  | 1 | 3.581× | 3.581–3.581× | 0 | 0 / 0 |
| csr_full:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.948× | 2.948–2.948× | 0 | 0 / 0 |
| csr_full:64-64-4-2 | sdpa / explicit_boolean_mask  | 1 | 6.355× | 6.355–6.355× | 0 | 0 / 0 |
| csr_full:64-64-4-3 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.945× | 2.945–2.945× | 0 | 0 / 0 |
| csr_full:64-64-4-3 | sdpa / explicit_boolean_mask  | 1 | 6.353× | 6.353–6.353× | 0 | 0 / 0 |
| csr_full:64-64-4-4 | flex / triton_default_options_unrecorded options unrecorded | 1 | 2.961× | 2.961–2.961× | 0 | 0 / 0 |
| csr_full:64-64-4-4 | sdpa / explicit_boolean_mask  | 1 | 6.338× | 6.338–6.338× | 0 | 0 / 0 |
| interval_split:128-32-2-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.2257× | 0.01824–1.043× | 2 | 0 / 0 |
| interval_split:128-32-2-2 | sdpa / explicit_boolean_mask  | 2 | 0.9303× | 0.1032–8.389× | 1 | 0 / 0 |
| interval_split:128-32-2-2 | sdpa / native_causal  | 1 | 0.3431× | 0.3431–0.3431× | 1 | 0 / 0 |
| interval_split:128-32-2-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.226× | 0.01729–1.024× | 2 | 0 / 0 |
| interval_split:128-32-2-3 | sdpa / explicit_boolean_mask  | 2 | 0.8927× | 0.09656–8.253× | 1 | 0 / 0 |
| interval_split:128-32-2-3 | sdpa / native_causal  | 1 | 0.3752× | 0.3752–0.3752× | 1 | 0 / 0 |
| interval_split:128-32-2-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.2308× | 0.01774–1.056× | 2 | 0 / 0 |
| interval_split:128-32-2-4 | sdpa / explicit_boolean_mask  | 2 | 0.9287× | 0.1012–8.521× | 1 | 0 / 0 |
| interval_split:128-32-2-4 | sdpa / native_causal  | 1 | 0.3732× | 0.3732–0.3732× | 1 | 0 / 0 |
| interval_split:128-32-4-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.195× | 0.9291–1.525× | 1 | 0 / 0 |
| interval_split:128-32-4-2 | sdpa / explicit_boolean_mask  | 2 | 9.155× | 6.799–12.33× | 0 | 0 / 0 |
| interval_split:128-32-4-2 | sdpa / native_causal  | 1 | 0.5306× | 0.5306–0.5306× | 1 | 0 / 0 |
| interval_split:128-32-4-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.236× | 0.9768–1.564× | 1 | 0 / 0 |
| interval_split:128-32-4-3 | sdpa / explicit_boolean_mask  | 2 | 9.431× | 7.05–12.62× | 0 | 0 / 0 |
| interval_split:128-32-4-3 | sdpa / native_causal  | 1 | 0.5632× | 0.5632–0.5632× | 1 | 0 / 0 |
| interval_split:128-32-4-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.209× | 0.958–1.558× | 1 | 0 / 0 |
| interval_split:128-32-4-4 | sdpa / explicit_boolean_mask  | 2 | 9.176× | 6.753–12.47× | 0 | 0 / 0 |
| interval_split:128-32-4-4 | sdpa / native_causal  | 1 | 0.5434× | 0.5434–0.5434× | 1 | 0 / 0 |
| interval_split:128-64-2-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.02447× | 0.01601–0.04034× | 3 | 0 / 0 |
| interval_split:128-64-2-2 | sdpa / explicit_boolean_mask  | 2 | 0.1727× | 0.09139–0.3264× | 2 | 0 / 0 |
| interval_split:128-64-2-2 | sdpa / native_causal  | 1 | 0.01256× | 0.01256–0.01256× | 1 | 0 / 0 |
| interval_split:128-64-2-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.02554× | 0.0157–0.04296× | 3 | 0 / 0 |
| interval_split:128-64-2-3 | sdpa / explicit_boolean_mask  | 2 | 0.1722× | 0.08772–0.338× | 2 | 0 / 0 |
| interval_split:128-64-2-3 | sdpa / native_causal  | 1 | 0.01367× | 0.01367–0.01367× | 1 | 0 / 0 |
| interval_split:128-64-2-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.02232× | 0.01228–0.04038× | 3 | 0 / 0 |
| interval_split:128-64-2-4 | sdpa / explicit_boolean_mask  | 2 | 0.1493× | 0.07037–0.3167× | 2 | 0 / 0 |
| interval_split:128-64-2-4 | sdpa / native_causal  | 1 | 0.01246× | 0.01246–0.01246× | 1 | 0 / 0 |
| interval_split:128-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.336× | 1.138–1.7× | 0 | 0 / 0 |
| interval_split:128-64-4-2 | sdpa / explicit_boolean_mask  | 2 | 9.802× | 7.006–13.71× | 0 | 0 / 0 |
| interval_split:128-64-4-2 | sdpa / native_causal  | 1 | 0.646× | 0.646–0.646× | 1 | 0 / 0 |
| interval_split:128-64-4-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.206× | 0.9222–1.698× | 1 | 0 / 0 |
| interval_split:128-64-4-3 | sdpa / explicit_boolean_mask  | 2 | 8.435× | 5.225–13.62× | 0 | 0 / 0 |
| interval_split:128-64-4-3 | sdpa / native_causal  | 1 | 0.6342× | 0.6342–0.6342× | 1 | 0 / 0 |
| interval_split:128-64-4-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.152× | 0.8931–1.598× | 1 | 0 / 0 |
| interval_split:128-64-4-4 | sdpa / explicit_boolean_mask  | 2 | 8.096× | 5.087–12.89× | 0 | 0 / 0 |
| interval_split:128-64-4-4 | sdpa / native_causal  | 1 | 0.6125× | 0.6125–0.6125× | 1 | 0 / 0 |
| interval_split:64-32-2-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.9982× | 0.6917–1.853× | 2 | 0 / 0 |
| interval_split:64-32-2-2 | sdpa / explicit_boolean_mask  | 2 | 7.684× | 3.94–14.99× | 0 | 0 / 0 |
| interval_split:64-32-2-2 | sdpa / native_causal  | 1 | 0.4464× | 0.4464–0.4464× | 1 | 0 / 0 |
| interval_split:64-32-2-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.029× | 0.7106–1.769× | 2 | 0 / 0 |
| interval_split:64-32-2-3 | sdpa / explicit_boolean_mask  | 2 | 7.589× | 4.023–14.32× | 0 | 0 / 0 |
| interval_split:64-32-2-3 | sdpa / native_causal  | 1 | 0.4975× | 0.4975–0.4975× | 1 | 0 / 0 |
| interval_split:64-32-2-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.8581× | 0.4418–1.686× | 2 | 0 / 0 |
| interval_split:64-32-2-4 | sdpa / explicit_boolean_mask  | 2 | 5.838× | 2.512–13.57× | 0 | 0 / 0 |
| interval_split:64-32-2-4 | sdpa / native_causal  | 1 | 0.4775× | 0.4775–0.4775× | 1 | 0 / 0 |
| interval_split:64-32-4-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.234× | 0.967–1.943× | 1 | 0 / 0 |
| interval_split:64-32-4-2 | sdpa / explicit_boolean_mask  | 2 | 9.466× | 5.709–15.69× | 0 | 0 / 0 |
| interval_split:64-32-4-2 | sdpa / native_causal  | 1 | 0.5497× | 0.5497–0.5497× | 1 | 0 / 0 |
| interval_split:64-32-4-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.269× | 1.024–1.882× | 0 | 0 / 0 |
| interval_split:64-32-4-3 | sdpa / explicit_boolean_mask  | 2 | 9.56× | 6.007–15.21× | 0 | 0 / 0 |
| interval_split:64-32-4-3 | sdpa / native_causal  | 1 | 0.5787× | 0.5787–0.5787× | 1 | 0 / 0 |
| interval_split:64-32-4-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.28× | 0.994–1.861× | 1 | 0 / 0 |
| interval_split:64-32-4-4 | sdpa / explicit_boolean_mask  | 2 | 9.857× | 6.489–14.97× | 0 | 0 / 0 |
| interval_split:64-32-4-4 | sdpa / native_causal  | 1 | 0.5701× | 0.5701–0.5701× | 1 | 0 / 0 |
| interval_split:64-64-2-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.12× | 0.7886–1.834× | 2 | 0 / 0 |
| interval_split:64-64-2-2 | sdpa / explicit_boolean_mask  | 2 | 8.128× | 4.458–14.82× | 0 | 0 / 0 |
| interval_split:64-64-2-2 | sdpa / native_causal  | 1 | 0.5528× | 0.5528–0.5528× | 1 | 0 / 0 |
| interval_split:64-64-2-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.9684× | 0.5659–1.717× | 2 | 0 / 0 |
| interval_split:64-64-2-3 | sdpa / explicit_boolean_mask  | 2 | 6.693× | 3.228–13.88× | 0 | 0 / 0 |
| interval_split:64-64-2-3 | sdpa / native_causal  | 1 | 0.5337× | 0.5337–0.5337× | 1 | 0 / 0 |
| interval_split:64-64-2-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 0.7496× | 0.3014–1.501× | 2 | 0 / 0 |
| interval_split:64-64-2-4 | sdpa / explicit_boolean_mask  | 2 | 4.55× | 1.705–12.14× | 0 | 0 / 0 |
| interval_split:64-64-2-4 | sdpa / native_causal  | 1 | 0.5353× | 0.5353–0.5353× | 1 | 0 / 0 |
| interval_split:64-64-4-2 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.469× | 1.227–2.06× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / explicit_boolean_mask  | 2 | 10.92× | 7.144–16.68× | 0 | 0 / 0 |
| interval_split:64-64-4-2 | sdpa / native_causal  | 1 | 0.6943× | 0.6943–0.6943× | 1 | 0 / 0 |
| interval_split:64-64-4-3 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.447× | 1.323–1.66× | 0 | 0 / 0 |
| interval_split:64-64-4-3 | sdpa / explicit_boolean_mask  | 2 | 10.03× | 7.499–13.42× | 0 | 0 / 0 |
| interval_split:64-64-4-3 | sdpa / native_causal  | 1 | 0.7897× | 0.7897–0.7897× | 1 | 0 / 0 |
| interval_split:64-64-4-4 | flex / triton_default_options_unrecorded options unrecorded | 3 | 1.193× | 0.8848–1.512× | 1 | 0 / 0 |
| interval_split:64-64-4-4 | sdpa / explicit_boolean_mask  | 2 | 7.828× | 5.025–12.2× | 0 | 0 / 0 |
| interval_split:64-64-4-4 | sdpa / native_causal  | 1 | 0.7308× | 0.7308–0.7308× | 1 | 0 / 0 |

SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.

### Setup observations

These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.

| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |
|---|---:|---:|---:|
| csr_full | 2.051 | 525.8 | 270.1–1166 |
| interval_split | 0.3241 | 1271 | 442.9–7327 |

### Tuning resource observations

These are within-workload associations, not causal explanations or measured occupancy.

- tune-causal: 24 valid configurations, 0 failed. Latency range 0.08344–5.133 ms. n_regs Spearman ρ=-0.06219 (n=24); n_spills Spearman ρ=0.7515 (n=24); shared_memory_bytes Spearman ρ=-0.3918 (n=24)
- tune-document: 24 valid configurations, 0 failed. Latency range 0.02752–0.1087 ms. n_regs Spearman ρ=0.8547 (n=24); n_spills Spearman ρ=0.7404 (n=24); shared_memory_bytes Spearman ρ=-0.09152 (n=24)
- tune-window128: 24 valid configurations, 0 failed. Latency range 0.01061–0.5417 ms. n_regs Spearman ρ=0.01498 (n=24); n_spills Spearman ρ=0.653 (n=24); shared_memory_bytes Spearman ρ=0.3125 (n=24)
- tune-window512: 24 valid configurations, 0 failed. Latency range 0.03334–3.59 ms. n_regs Spearman ρ=-0.3869 (n=24); n_spills Spearman ρ=0.836 (n=24); shared_memory_bytes Spearman ρ=-0.09008 (n=24)

### Offline search replay

OFFLINE replay of fixed GPU-measured latency tables. No new GPU calls; excludes timing noise, compile cost, thermal/order effects and end-to-end search overhead.

| Method | Budget | Trials | Failed | Mean regret | Median regret |
|---|---:|---:|---:|---:|---:|
| evolution | 4 | 80 | 0 | 0.2323 | 0.1095 |
| evolution | 8 | 80 | 0 | 0.082 | 0.05498 |
| evolution | 12 | 80 | 0 | 0.03621 | 0 |
| random | 4 | 80 | 0 | 0.1683 | 0.09127 |
| random | 8 | 80 | 0 | 0.06553 | 0.009897 |
| random | 12 | 80 | 0 | 0.0504 | 0.004214 |

