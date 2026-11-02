# Model size, context and changing bottlenecks

A six-workload study of attention, checkpointing and microbatch size across
three model sizes and two context lengths. The
[complete screen](../results/2026-11-01-scaling-screen/README.md) records
48 attempts, 46 completed cases and two OOMs. Longer training comparisons use the
separate [quality protocol](quality-study.md).

## Why one model is insufficient

For a dense decoder with batch B, context T, width d and feed-forward width f,
leading forward matrix-multiply work per layer is approximately:

- Attention projections: 8 B T d² FLOPs.
- Attention scores and value mixing: 4 B T² d FLOPs for dense full matrices.
- SwiGLU feed-forward layers: 6 B T d f FLOPs.

These counts treat a multiply-add as two FLOPs. Causal kernels can avoid work
on masked entries, and the expressions exclude normalization, softmax, the
vocabulary head, backward, recomputation and optimizer work. They are analytical
guides, not measurements or runtime predictions.

With f proportional to d, projection/feed-forward work grows roughly as d²,
whereas attention mixing grows as d at fixed context. Increasing context instead
raises attention work per token. At fixed total tokens B T, projection/MLP work
is approximately unchanged while dense attention work grows with T. Depth adds
work and saved activations; parameter count alone does not determine the result.
Different shapes also change GPU utilization, memory traffic and kernel choice.
The vocabulary head is another meaningful cost, especially for small models.

Thus each optimization needs a domain of validity: which sizes, contexts,
precisions, batch policies and hardware benefit, and where the ranking changes.

## Controlled family and staged screen

All models use this repository's custom MHA decoder, vocabulary 10,000, untied
embeddings, head dimension 64, RoPE theta 10,000 and the same token files.
Exact counts were checked by constructing the model on PyTorch's meta device.

| Parameters | Width | Layers | Heads | Feed-forward width |
| ---: | ---: | ---: | ---: | ---: |
| 22,696,448 | 512 | 4 | 8 | 1,344 |
| 100,313,856 | 768 | 12 | 12 | 2,048 |
| 324,092,928 | 1,024 | 24 | 16 | 2,752 |

This family grows width and depth together, with slightly different rounding of
feed-forward dimensions. Results establish transfer across these configurations,
not a causal effect of parameter count alone. Add a fixed-width depth ablation
later if the explanation requires separating those effects. These configurations
are not SmolLM2 or Qwen checkpoints. The earlier 22.7M integration pilot used
16 heads and remains a separate workload.

Six configuration files, `configs/scale-{22m,100m,324m}-s{1024,4096}.json`, define
the workloads. Each has eight candidate treatments: eager/SDPA × checkpoint
none/block × microbatch 1/maximum. At 1,024 context the effective batch is eight
sequences; at 4,096 it is two. Both process **8,192 tokens per optimizer update**.
Accumulation preserves the ordered examples within each workload comparison.

The screen has **48 cases**, seed 1337, 30 updates each, with the first five
excluded from steady-state timing. The 100M/1K and 100M/4K workloads ran first,
followed by the smaller and larger model controls.

The original screens used 120-second case timeouts and 30-minute Modal limits.
The completed 324M screens used 180 seconds at 1K and
300 seconds at 4K, with 2,400/3,600-second job limits. Failed and OOM cases remain in the reports. Thirty-step runs screen hardware
behavior; they do not establish model quality. Do not pool validation losses
across context lengths, which change the conditioning task and validation windows.

## Explain the measurements

At representative small, medium and large points, collect separate profiler
traces for forward, backward, attention, feed-forward, vocabulary projection,
optimizer and transfers. Inspect the actual SDPA kernel. Profiled runs are
diagnostic and stay separate from unprofiled timing comparisons; nested inclusive
times must not be added as if they were disjoint component costs. Component
profiling has not yet been executed.

Report complete step time, tokens/second, allocated/reserved peaks, job time,
failures and cost. Compare the best feasible SDPA policy as the framework
baseline. A checkpointing claim must improve on that strong baseline at the
same workload, rather than only beat eager attention.

## Confirm the result and measure quality

Select configurations on development seed 1337. Repeat the important comparison
on three fresh paired seeds (2027, 2028, 2029). Preserve model initialization,
ordered batches, precision, effective token batch and evaluation within each pair.

The longer quality protocol was fixed before development: 100M/1K, three SDPA
policies, 1,200 updates from initialization, and 64 fixed validation windows every
50 updates. Cosine decay depends on that horizon; extending a 30-step checkpoint
would not reproduce the planned run. All three development policies completed.
The predefined rule gives target loss 2.40 from their step-600 losses; save the
numeric target and development hashes before launching confirmation on seeds
2027, 2028 and 2029. See [quality study](quality-study.md) for the complete protocol.
The current validation file is used for these comparisons; a separate final
quality evaluation remains future work, and file hashes do not prove document-level
separation.

Report every seed, endpoint loss, first observed evaluation reaching the target,
unreached targets and full job cost. Do not interpolate an unmeasured crossing.
Keep checkpoints needed for any later quality audit; the current runner retains
only the latest checkpoint. This requires an extension before a checkpoint audit.

This is a **systems scaling study** first. A later model/data scaling-law study
must vary parameter count and token budget independently, include more runs and
held-out configurations, and test extrapolation uncertainty. A few timings or
three model sizes cannot establish a compute-optimal training law.

## Execution status

The first eight 100M/1K cases completed on an L40S in run
`20261101T115005Z-e1fdfb9a`. The
[report and returned raw summaries](../results/2026-11-01-100m-1k/README.md) record
passing comparability, 144.113 seconds of remote function time, and the observed
microbatch/checkpointing interactions. GPU preflight and the pytest invocation
passed their gates. The original invocation returned summaries; its full test logs remain remote.
The 100M/4K screen subsequently completed as run `20261101T123821Z-74cc9b01`,
also with eight completed cases and passing comparability/preflight/pytest gates.
Its [report](../results/2026-11-01-100m-4k/README.md) records 291.461 seconds of
function time and compares both contexts. At the full token microbatch without
checkpointing, the 4K/1K median-step ratio was 3.299x for eager and 1.096x for
SDPA. These are separate invocations with changed sequence shapes and windows;
the reports keep validation losses separate.

The 22M/1K screen also completed as run `20261101T131029Z-aeb8becf`; see its
[report](../results/2026-11-01-22m-1k/README.md). The development campaign then
completed the remaining three workloads and three quality-development policies.
The [complete screen](../results/2026-11-01-scaling-screen/README.md) now contains
48 attempts: 46 completed, two OOMs. At 324M/4K, both eager/no-checkpoint cases failed
before any update; the full matrix is therefore marked insufficient for comparability.
That L40S exposed 47.40 GiB versus 44.42 GiB in the other invocations. Report this
capacity difference rather than assuming identical hardware environments.

The four new campaign jobs each returned complete evidence and a GPU test log with
168 passed. Full per-step/test logs for the original three screens remain remote.
The [quality protocol](quality-study.md) defines a separate 57-case confirmation
that remains unexecuted. The later
[distributed confirmation](../results/2026-11-01-distributed-confirmation/README.md)
repeats distributed comparisons; it does not confirm the single-GPU quality target.

The first screen used the command below; it is retained for reproduction, not as
an instruction to duplicate the completed run. It starts eight 100M/1K cases.
The launcher runs
CUDA/BF16 checks first and saves the summary locally and full results in the Modal
Volume. Stop and inspect that output before moving to the next workload.

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 python scripts/modal_pilot.py \
  --config configs/scale-100m-s1024.json --execute --gpu L40S \
  --microbatches 1 8 --effective-batch-size 8 --seeds 1337 \
  --case-timeout-seconds 120 --job-timeout-seconds 1800
```

The completed 100M/4K screen used this command, retained for reproduction. It
preserves 8,192 tokens/update by changing the effective batch to two sequences:

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 python scripts/modal_pilot.py \
  --config configs/scale-100m-s4096.json --execute --gpu L40S \
  --microbatches 1 2 --effective-batch-size 2 --seeds 1337 \
  --case-timeout-seconds 120 --job-timeout-seconds 1800
```

The completed 22M/1K screen used this command, retained for reproduction:

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 python scripts/modal_pilot.py \
  --config configs/scale-22m-s1024.json --execute --gpu L40S \
  --microbatches 1 8 --effective-batch-size 8 --seeds 1337 \
  --case-timeout-seconds 120 --job-timeout-seconds 1800
```

The 324M/4K screen used the prospectively chosen 300-second case limit. Its two
eager/no-checkpoint cases hit OOM; both eager checkpointed cases completed in about
144 seconds per subprocess. No timeout was observed in this screen. Preserve that
distinction: a timeout would not demonstrate memory infeasibility or numerical failure.

## Retrieve usage and test evidence

The Modal 1.5.1 commands below retrieve usage and original test logs. Usage
reports cover completed intervals, may include other apps, and are not invoices.
Keep billing output under ignored `runs/`. The earlier test logs belong to the
workspace that ran those jobs.

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 modal billing report \
  --for today --resolution h --show-resources --json > runs/billing-2026-11-01.json

uv run --no-project --python 3.12 --with modal==1.5.1 modal volume get \
  systems-for-llms-results 20261101T115005Z-e1fdfb9a/tests.log \
  runs/modal-20261101T115005Z-e1fdfb9a/tests.log

uv run --no-project --python 3.12 --with modal==1.5.1 modal volume get \
  systems-for-llms-results 20261101T123821Z-74cc9b01/tests.log \
  runs/modal-20261101T123821Z-74cc9b01/tests.log
```

Retrieve the per-step metrics and resolved plan separately for a full audit.
Downloading an entire run directory also downloads its large model checkpoints.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): transformer components.
- [FlashAttention](https://arxiv.org/abs/2205.14135): IO and end-to-end timing matter alongside arithmetic work.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361): model, data and compute scaling.
- [Chinchilla](https://arxiv.org/abs/2203.15556): joint model/token allocation under a compute budget.
