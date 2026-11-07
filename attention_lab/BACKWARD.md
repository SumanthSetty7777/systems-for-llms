# Training extension

`backward.training_attention(q, k, v, spec, document_ids=None)` makes the
structured attention implementation differentiable. Inputs have contiguous
`[batch, heads, sequence, head_dimension]` layout, CUDA FP16 or BF16 dtype, and
head dimension 64 or 128. Q, K, and V must share shape, dtype, and device.

The forward uses the existing interval kernel for causal/window masks and the
existing CSR kernel for document masks. It additionally saves FP32 natural-log
row log-sum-exp. `triton_attention.attention` retains its original tensor return
by default; the new optional `return_lse=True` returns `(output, lse)`.

The backward reconstructs softmax probabilities tile by tile from the saved
normalizer. One query-tile program owns each dQ tile. One key-tile program owns
each dK/dV tile and traverses the transposed CSR schedule. Gradient accumulation
uses FP32 and writes each destination once, with no atomic accumulation. An
extra linear-size kernel computes the row softmax derivative reduction.
Probabilities and score derivatives are cast to the input dtype for tensor-core
matrix multiplications. The implementation is mathematically exact masked
attention, subject to these floating-point approximations.

Full/partial tile flags are preserved when transposing the schedule. Boundary
predicates are checked at token granularity, including noncontiguous repeated
document labels. Causal, left/right window, and document constraints compose by
intersection. Ragged final tiles and different Q/K tile sizes are supported.

Limitations: first derivatives only, no dropout, no GQA/MQA, no unequal Q/K
sequence lengths, and one shared sequence of document labels across all batch
items and heads. Every supported mask includes the diagonal; arbitrary masks
with completely empty rows are outside this interface. Backward currently uses
CSR traversal even for interval masks. It reconstructs score/probability tiles
separately for dQ and dK/dV; this additional work can make it slower than fused
library implementations. The measured results below include those slower cases.

## Validation and experiments

Run CPU tests and, when CUDA is available, GPU correctness tests:

```sh
python -m unittest attention_lab.test_backward -v
```

CPU tests check transpose ownership and flag preservation, and finite-difference
gradcheck an independent dense float64 oracle. CUDA tests compare output and all
three gradients against independently masked dense FP32 attention. They include
FP16/BF16, head dimensions 64/128, irregular lengths, asymmetric tiles, zero
windows, asymmetric windows, repeated document labels, and broadcast upstream
gradients. CPU success does not establish GPU kernel correctness.

On an already allocated CUDA GPU, from the repository root:

```sh
python -m attention_lab.backward_study --stage smoke --output-dir results/backward-smoke --time-budget-seconds 480
python -m attention_lab.backward_study --stage study --output-dir results/backward-study --time-budget-seconds 600
```

These commands do not provision GPUs. The smoke stage runs numerical cases,
then a complete one-block Transformer SGD step, then attention backward and
forward-plus-backward timing. The study first runs both CUDA unit tests, covering
every mask case in FP16 and BF16, then extends the shape and mask coverage.
Comparators are FlexAttention with equivalent masks and native PyTorch SDPA for
unrestricted causal/dense masks. All numerical errors and raw timing samples
are retained, including failures and slower cases.

The Transformer consists of pre-norm attention with two heads, residuals, a
GELU MLP of width 4C, mean squared output loss, and SGD with learning rate 0.01.
FP32 master weights are cast to the activation dtype for linear operators;
SGD updates the FP32 master weights so small updates remain observable.
It is a systems workload, not a language-model convergence experiment. Every
invocation restores identical starting weights; restoration cost is included
for all implementations. Loss, outputs, every weight gradient, and updated
parameter deltas are compared against a library baseline before timing. These
checks use a scale-aware RMS tolerance (5% of reference RMS plus 1e-8), rather
than an absolute tolerance that could accept zero gradients. Gradient norms,
update norms, and fractions of parameters that changed are recorded, and a
zero gradient or zero update for any weight matrix fails validation.

Attention-only timing uses CUDA graph replay and excludes Python dispatch,
mask setup, compilation, and reference calculations. For repeated backward on
the retained graph, AOTAutograd buffer donation is explicitly disabled before
Flex compilation. All Flex results use this setting; they are not default
buffer-donation results. The setting is recorded in every comparison and in
the summary.

Attention inputs, the retained forward graph, warmups, graph capture, and replay
are all created on the same explicit nondefault CUDA stream. The training timer
passes that stream explicitly to `torch.cuda.graph`, because autograd backward
follows the streams of its forward operations. The caller stream is ordered
before and after a successful case. Capture/replay failure aborts the study
after persisting the traceback; it does not continue in potentially invalid
CUDA state. This training timer does not change the existing forward-only timer.

Complete Transformer optimizer steps use matched eager CUDA-event timing for
every implementation, including host submission gaps. Native SDPA's full-step
backward raised `cudaErrorStreamCaptureImplicit` during CUDA-graph capture in
the initial smoke run. Eager timing is the explicitly selected full-step method,
not a per-implementation fallback, and full-step speedups compare the same mode.
Attention and complete-step timings therefore measure distinct scopes and
should not be conflated. Neither scope claims that the entire Python program
was compiled. First-call wall time is reported
separately, and includes compile/cache lookup, execution, and synchronization.
Peak allocated memory is measured for one eager invocation after warmup; live
inputs, saved graphs, and cached masks are counted in the recorded baseline.
Both absolute peak and incremental peak are reported; graph-pool allocation
is excluded. These measurements must not be described as total GPU memory or
whole-application latency.

Every completed case is saved to JSON immediately, along with `summary.json`.
Time budgets are checked between cases, so a single compilation may exceed the
remaining budget. Exit code 0 means all planned cases passed; 1 means a failure;
2 means the run ended early because of its budget. Partial coverage is not a
passed smoke gate. The completed run below establishes correctness and timing
only for its recorded cases, hardware, precision, and software versions.

## Validated H100 results

[The final study](../runs/cohere-backward-study-20261105T183827Z-529f17/study/summary.json)
passed all 15 planned cases on an NVIDIA H100 80GB HBM3 with PyTorch 2.11.0,
Triton 3.6.0, and CUDA 13.0. TF32 was explicitly disabled for both CUDA matmul
and cuDNN. Recorded study time was 77.17 seconds; the runner's wall time was
85.15 seconds. The run includes two passing CUDA unit tests (ten mask/dtype
combinations plus broadcast-gradient coverage), eight standalone numerical
cases, four timed attention workloads, and two complete optimizer workloads.

The following maxima are from the eight numerical cases and four timed custom
attention workloads against dense FP32 references. The unit tests also passed
their assertions but do not emit per-tensor error statistics.

| Dtype | Tensor | Maximum absolute error | Maximum relative RMS error |
| --- | --- | ---: | ---: |
| FP16 | Output | 0.001078 | 0.0250% |
| FP16 | dQ | 0.001268 | 0.0324% |
| FP16 | dK | 0.001513 | 0.0326% |
| FP16 | dV | 0.001705 | 0.0297% |
| BF16 | Output | 0.007830 | 0.2102% |
| BF16 | dQ | 0.010249 | 0.2795% |
| BF16 | dK | 0.014320 | 0.2791% |
| BF16 | dV | 0.014473 | 0.2367% |

Relative RMS excludes dQ/dK for the diagonal-only mask, whose mathematical
reference gradients are zero. Their maximum absolute errors were 7.64e-7 and
9.13e-7 respectively; output and dV matched exactly in that case. Reported
absolute and relative maxima can arise from different workloads.

Attention timings below are BF16, batch 1, two heads, and CUDA graph replay on
the shared explicit stream. Times are medians in microseconds. Ratios divide
Flex time by custom time; a ratio above 1 favors the custom implementation.
Flex buffer donation was disabled as described above.

| Sequence / head dimension | Mask | Custom forward + backward | Flex forward + backward | Flex/custom ratio | Backward-only Flex/custom ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 / 64 | Causal | 71.157 | 70.635 | 0.993x | 0.723x |
| 1024 / 64 | Causal, left window 127 | 26.133 | 36.555 | 1.399x | 0.961x |
| 4096 / 64 | Causal, left window 255 | 34.261 | 46.976 | 1.371x | 0.903x |
| 2048 / 128 | Causal, repeated document labels | 91.989 | 84.352 | 0.917x | 0.869x |

Backward alone was slower than Flex in every timed case. On unrestricted causal
attention, native SDPA forward + backward took 51.104 microseconds, versus
71.157 for the custom implementation (0.718x baseline/custom ratio). No general
training-kernel speedup over Flex or SDPA is established. Incremental allocated
memory for custom and Flex was equal in these four attention measurements.

Complete Transformer SGD steps used the same matched eager timing for all
variants, including restoration of initial weights and host submission gaps.

| Sequence / head dimension | Mask | Custom step (ms) | Flex step (ms) | Flex/custom ratio | Native SDPA step (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 / 64 | Causal | 1.1821 | 1.2487 | 1.056x | 0.9157 |
| 1024 / 128 | Causal, left window 127 | 1.1612 | 1.3053 | 1.124x | Not applicable |

For the 512-token block, custom parameter gradients differed from SDPA by at
most 0.2897% relative RMS; FP32 update deltas differed by at most 0.2988%.
The largest absolute RMS error in an update delta was 2.77e-8. For the
1024-token windowed block versus Flex, the corresponding maxima were 0.0604%,
0.0736%, and 7.16e-10. Every parameter matrix had nonzero gradients and updates;
at least 99.7879% of entries changed in each custom matrix.

These are one-GPU, bounded, synthetic systems measurements, not language-model
quality, convergence, multi-GPU scaling, or production end-to-end measurements.
The optimizer and attention scopes differ and must not be combined into one
speedup. Results apply to the listed shapes and baseline configuration only.
