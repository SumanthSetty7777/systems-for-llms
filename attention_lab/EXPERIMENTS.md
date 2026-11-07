# Structured attention: experiment design and completed study

## Question and implementation

When does lowering mask structure into Triton loop bounds beat generic sparse
schedule traversal, and how much of a kernel gain survives inside a Transformer
block? This is an engineering study of established techniques, not a new
attention algorithm.

The compiler supports causal, sliding-window, and same-document predicates.
Its general lowering emits CSR tile rows and full/partial classifications.
For masks without document constraints, interval lowering derives retained key
ranges analytically and separates masked boundaries from an unmasked interior.
Both paths preserve attention semantics and use FP32 online-softmax accumulation
with FP16/BF16 inputs. These new GPU paths implement forward only; the earlier
standalone FlashAttention project contains a separate backward implementation.

| Variant | Retained tiles | Logical masking | GPU metadata |
|---|---|---|---|
| CSR masked | Compiled sparse schedule | Every retained tile | CSR rows and columns |
| CSR full | Same sparse schedule | Partial tiles only | CSR and full-block flags |
| Interval masked | Analytic key bounds | Every retained tile | No CSR traversal |
| Interval split | Same analytic bounds | Separate boundary loops | No CSR traversal |

Interval paths reject document masks. Full-block on/off ablations retain the
same tiles, so a change in attention semantics cannot create an apparent gain.

## Study sequence

The initial design comprised four stages: smoke, ablation, tuning, and held-out
confirmation. After inspecting those results, two follow-ups were added:
FlexAttention contiguity hints and a Transformer-block benchmark. **The six-stage
study was not all specified in advance.** The campaign command now reproduces
all six stages; follow-up results remain separate from the original runs.

| Stage | Completed / planned | Requested FlexAttention checks | Design |
|---|---:|---:|---|
| Smoke | 20 / 20 | 2 / 2 passed | N128/257, D64/128, FP16/BF16; causal/window/document masks |
| Ablation | 120 / 120 | 36 / 36 passed | N1024/4096 × D64/128 × three masks × three seeds; four interval variants, two document variants |
| Configuration sweep | 96 / 96 | 4 / 4 passed | Four workloads × 24 configurations |
| Held-out confirmation | 24 / 24 | 12 / 12 passed | Four held-out workloads × three fresh seeds × default/transferred configuration |
| Flex contiguity follow-up | 24 / 24 | 24 / 24 passed | Eight causal/window workloads × three seeds; validate and supply contiguity hint |
| Transformer-block follow-up | 8 / 8 | 8 / 8 passed | Eight shapes, one fresh input/weight seed; frozen transferred configuration |
| **Total** | **292 / 292** | **86 / 86 passed** | All required correctness and baseline gates passed |

Main workloads use B1/H8/BF16. Fixed-default ablations use Q64/K64 tiles,
four warps, and two pipeline stages. Packed documents cycle through lengths
97/211/53/389. A window of 128 includes 127 previous positions and the current
position. The smoke stage additionally covers irregular tails and FP16.

The measured configuration space is Q={64,128}, K={32,64}, warps={2,4}, and
stages={2,3,4}. Tuning uses N4096 causal/D64, window128/D64, window512/D128,
and packed causal/D64, with one fixed input seed. Held-out confirmation uses
N257/D64 window128, N4097/D128 window512, N8192/D64 causal, and N8192/D64 packed
causal, with seeds 101/102/103. Only tuning data selects configurations.

The block follow-up covers N1024/4096 × D64/128 × causal/window128, seed 211.
It measures layer norm → QKV projection → attention → output projection and
residual → layer norm → four-times-width GELU MLP and residual. Every variant
uses identical BF16 weights and inputs. Configurations are transferred from
training-family winners before block evaluation; block timings do not select
them. This is one inference block, not a whole model or training measurement.

## Environment and measurement protocol

All final stages used an NVIDIA H100 80GB HBM3 (132 SMs, compute capability 9.0),
Python 3.12.10, PyTorch 2.11.0+cu130, Triton 3.6.0, and CUDA 13.0. Exact environment
and source records accompany every stage in
[results/h100-20261104](results/h100-20261104).

- Correctness uses an independent chunked PyTorch math-SDPA reference. Timings
  are accepted only for finite, numerically close outputs and unchanged inputs.
  The block benchmark also verifies unchanged weights and a finite reference.
- Unrestricted causal attention uses native `is_causal=True` SDPA. Structured
  masks use the identical explicit boolean mask. SDPA’s automatic backend is
  not inspected; native and explicit-mask groups are reported separately.
- FlexAttention constructs masks independently and explicitly requests its
  Triton backend. The later hinted baseline additionally validates that both
  partial and full block lists are contiguous before setting
  `BLOCKS_ARE_CONTIGUOUS=True`. Default and hinted runs are never pooled.
- Five warmup calls precede 20 timing samples. Attention samples replay a CUDA
  graph containing 20 forwards; block samples contain 5 block calls. Results
  report per-call median latency. Graph timing excludes Python dispatch,
  construction, uploads, JIT, and graph capture.
- First-call, host schedule compilation, and Flex mask-setup costs are recorded
  separately. First-call cost includes allocation/upload/validation/JIT/cache
  lookup and is **not pure compiler time**. Setup-amortization break-even points
  have not been established.
- Raw samples, seeds, shapes, masks, tolerances, numerical errors, register/spill/
  shared-memory metadata, and generated-assembly hashes are preserved. Static
  compiler metadata is not Nsight occupancy or stall profiling.
- Paired ratios use matching workload, input seed, configuration where required,
  timing protocol, and environment. Summaries take geometric means over those
  pairs. Ranges are observed minima/maxima, not confidence intervals. All
  configurations and regressions remain visible.

## Findings, including negative results

| Finding | Evidence | Interpretation |
|---|---|---|
| Interval specialization improves over generic CSR traversal | 1.443× geomean, 24 pairs; 1.190–1.768× range | Removing generic traversal is useful on these workloads |
| Gain remains against the stronger Flex baseline | 1.661× geomean versus hinted Flex Triton, 24 pairs; 1.033–2.978× range | A same-semantics compiled baseline, not dense attention alone |
| Loop splitting alone is neutral in aggregate | Interval masked / split: 0.9983×, 24 pairs | Do not attribute the specialization gain solely to skipping interior mask predicates |
| Block-level gains are smaller | 1.129× versus default Flex, eight shapes; isolated attention 1.748× | Other operations limit the benefit; not a whole-model speedup |
| Native causal SDPA remains competitive | 0.818× attention ratio in the ablation; 0.989× full-block ratio | The custom implementation is not universally faster |
| Transferred tuning provides a modest gain | Default / transferred: 1.012×, 12 held-out pairs | Avoid presenting per-shape best tuning numbers as generalizable gains |
| Configuration choices can be catastrophic | Window512/D128 ranges 0.03334–3.5896 ms, or 107.7×; slow configurations show high spill counts | A resource-pressure hypothesis, not causal proof or measured occupancy |

The [complete report](results/h100-20261104/analysis/REPORT.md) and
[JSON analysis](results/h100-20261104/analysis/SUMMARY.json) contain individual
source cases, all workload groups, raw configuration tables, and regressions.
The [figure](results/h100-20261104/gpu_tradeoffs.png) shows complete workload and
configuration sets for its stated comparisons.

Random and evolutionary searches are replayed **offline** over the four
GPU-measured candidate tables, with budgets 4/8/12 and 20 search seeds. These are
480 replay trials, not 480 independent GPU searches. Regret is relative to the
fastest correct configuration in each finite table. Replay excludes online
measurement noise, compilation, thermal/order variation, and search wall time;
it does not establish end-to-end tuning savings or a learned RL policy.

## Execution history and provenance

An early cloud preflight timed out before creating a GPU function. After API
connectivity was restored, smoke passed. The first ablation attempt exposed
PyTorch Dynamo’s default eight-recompilation limit across the mask sweep, causing
required FlexAttention comparisons to fail. The benchmark configured a bounded
larger recompilation limit for its isolated study process and reran the **entire
ablation**. The final report uses that complete passing rerun, not a mixture of
successful cases from the diagnostic attempt. The diagnostic log remains under
[results/diagnostics/attention-ablation-20261104a](results/diagnostics/attention-ablation-20261104a),
outside the final 292-case result directory; the retained final ablation is the
`b` run. The final local correctness and orchestration suite passes all 47 tests.

Each final stage preserves its launch manifest, environment, case JSON, summary,
log, and source SHA-256 values. The two follow-up stages additionally preserve
exact Python source snapshots; the earlier stages have hashes rather than full
snapshots. The analysis also hashes evidence JSON. Source changes between stages
are recorded, and comparisons stay within compatible runs. Stored timing data
is never relabeled as a fresh execution of later code.

`run_campaign.py` executes one GPU stage at a time, with process/container
timeouts and no automatic retries. Required correctness or baseline failures
stop advancement. At November 4 rates, the sum of requested-resource estimates
at **all six stage timeouts is $5.4188**. This is neither an actual charge nor a
billing cap; it excludes image builds, startup, excess CPU/RAM, storage, and other
applications. Reproduction commands are in the [README](README.md).

## Scope and follow-ups

The implementation handles contiguous square self-attention, D64/128, and a
mask shared across batches/heads. Dynamic document labels require recompilation.
GPU backward, arbitrary mask expressions, mask-reuse break-even measurements,
multiple hardware architectures, and profiler-confirmed bottleneck attribution
remain future work. A custom Python mask-lowering pass is not an LLVM/MLIR pass;
this developer tool is not an LLM optimizer or a trained RL policy.

## Primary references

- [NVIDIA attention optimization study](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/): loop splitting, tile-size interactions, and resource pressure.
- [FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/): compiler lowering, sparse-mask hooks, and scheduling. Our baseline is Triton; it is not FA4.
- [KernelFoundry](https://proceedings.mlr.press/v306/wiedemann26a.html): hardware-aware evolutionary optimization. This finite configuration search does not reproduce its LLM/MAP-Elites framework.
- [KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/): profiling-driven optimization; our tool does not claim its agent architecture or results.
