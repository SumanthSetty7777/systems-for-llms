# Single-GPU pilot

The completed [size/context screen](../results/2026-11-01-scaling-screen/README.md)
contains 48 attempts: 46 completed and two OOMs. Three longer
[quality development runs](../results/2026-11-01-100m-quality-dev/README.md)
followed. This note describes the original integration pilot and its runner;
the [size/context study](scaling-study.md) and [quality protocol](quality-study.md)
describe those subsequent experiments.

The question is whether faster attention changes the best checkpointing and
microbatch choice, and whether that choice lowers time and cost to a defined
quality. This pilot checks the implementation and measures feasibility. The
longer quality comparison follows it.

## Initial workload

`configs/gpu-pilot.json` uses the existing 22,696,448-parameter model and
TinyStories token arrays. It starts from random weights, uses a 1,024-token
context and runs 60 optimizer updates. This is a familiar workload for the first
hardware check. The later size/context screen uses the controlled model family described in
[model and workload](model-selection.md).

| Variable | Pilot settings |
| --- | --- |
| Attention | Eager; PyTorch SDPA with automatic kernel selection |
| Activation checkpointing | None; each transformer block |
| Sequences per microbatch | 1, 2, 4, 8 |
| Accumulation steps | 8, 4, 2, 1 respectively |
| Effective batch | 8 sequences / 8,192 tokens per optimizer update |
| Precision | BF16 autocast, FP32 parameters and optimizer state |
| Initialization and data seed | 1337, paired across all 16 configurations |
| Measurement | Exclude the first 5 updates from steady-state step summaries |
| Validation | Same 4,096 validation tokens at the end of each run |

Each case processes 491,520 training tokens. All 16 cases together process
7,864,320 tokens, including repeated examples across treatments. These are not
unique-token counts. Sixty updates and one seed cannot establish convergence,
generalization or long-run mixed-precision stability.

## What must remain equal

The runner samples all eight sequence starts once per update and splits that
ordered batch into microbatches. It logs a SHA-256 of those starts at every step.
This prevents a microbatch sweep from silently changing data order. It divides
each microbatch's mean loss by the accumulation count, accumulates gradients,
then clips and updates once. All sequences currently have equal length and no
padding; a variable-length extension will need valid-token weighting.

Checkpointing uses PyTorch's non-reentrant implementation. It is active only
during training with gradients enabled. BF16 runs retain FP32 loss and eager
softmax normalization. BF16 requires native CUDA support; CPU execution is a
separate FP32 harness check.

Before timed GPU runs, execute the test suite on that GPU, including the
CUDA-only output/gradient and BF16 training tests. Compare an FP32 reference
with BF16 on a small fixed workload before using BF16 for quality conclusions.
The original three screen invocations passed their GPU preflight and pytest
exit-code gates; their exact counts remain in remote logs. Each of the four later
development-campaign jobs returned a GPU test log showing 168 tests passed.

## Run and inspect

The pilot command writes a reviewable manifest and resolved configurations by
default. `--execute` starts the cases. They run sequentially in separate
processes, in a recorded shuffled order, so one case's CUDA allocator state
does not carry into another. Each case has a timeout. Failed and out-of-memory
cases stay in the results.

Use a CUDA environment with the dependencies in `pyproject.toml` and make the
data paths available there. Resolved local paths in a saved plan are machine
specific: regenerate the plan on the target host when relocating the data.

```sh
PYTHONPATH=src python -m llmlab.pilot \
  --config configs/gpu-pilot.json --output-dir runs/gpu-plan \
  --microbatches 1 2 4 8 --effective-batch-size 8 \
  --seeds 1337 --warmup-steps 5 --timeout-seconds 120
```

On the same host and source version, `python -m llmlab.pilot --run-plan runs/gpu-plan`
executes that saved plan. Planning flags cannot override a saved plan. The actual
local eight-case CPU harness completed on October 31; all cases used matching
data/tokenizer identities and the same ordered batch stream. That verifies
orchestration, not GPU performance.

The pilot summary reports median and p90 complete optimizer-step time, throughput,
CUDA allocated/reserved peaks, final validation loss and invocation timings.
Step time includes sampling, batch preparation/transfers, forward/backward,
recomputation, clipping, the optimizer and scalar diagnostics. Evaluation,
checkpointing and process overhead are separate costs. CUDA peak memory covers
training and evaluation after model initialization, including persistent state.
Report these scopes when comparing measurements.

SDPA is an interface, not a guarantee that a particular FlashAttention kernel ran.
Inspect a profiler trace before attributing a result to a kernel. Randomized
order alone does not eliminate hardware noise; repeat important timing comparisons.

## Move from a pilot to research results

1. Check paired data hashes, numerical differences, finite losses and memory
   feasibility. Diagnose failures before expanding the matrix.
2. Repeat promising and reference configurations, then sweep sequence length
   and the controlled 22.7M/100.3M/324.1M family.
   Hold model, data, precision and effective batch fixed within each comparison.
3. Choose a reachable held-out loss target and maximum token budget from separate
   development runs. Keep those choices fixed for the main comparison. Evaluate
   at the same token checkpoints; a target is reached at the first measured
   evaluation that meets it. Report runs that never reach it.
4. Repeat central comparisons across at least three initialization/data seeds.
   Report individual runs and variability, plus downstream task checks that are
   informative at this model scale.
5. Account for complete elapsed jobs and actual provider charges, including
   preparation, failed runs, tuning, evaluation and checkpoints. A faster step
   is useful only if quality is preserved and total time/cost improves.

Expected outputs are a memory/throughput table, learning curves against tokens,
time and dollars, and an explanation of where the preferred configuration
changes. A null result, numerical regression or OOM boundary is a valid finding.
The pilot itself is not a novelty claim.

## Modal execution

The completed screens used L40S GPUs. GPU identity, exposed memory and software
versions are recorded with each run. Function-time estimates and provider billing
are separate: startup, image builds, storage and other resources are not included
in a GPU-only rate.

The optional `scripts/modal_pilot.py` launcher lists its upload payload without
contacting Modal by default:

```sh
python scripts/modal_pilot.py --config configs/gpu-pilot.json
```

Use Python 3.12 and Modal 1.5.1 to match the serialized remote function:

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 python scripts/modal_pilot.py \
  --config configs/gpu-pilot.json --execute --gpu L40S \
  --case-timeout-seconds 120 --job-timeout-seconds 1800
```

The wrapper allows one container, zero retries and a short scale-down period.
Preflight runs in a subprocess so its CUDA context is released before the pilot.
Tests must pass before training starts. Logs and checkpoints remain in the
`systems-for-llms-results` Modal Volume under the run ID; small summaries return
to a new local `runs/modal-<run-id>` directory. A partial or non-comparable matrix
is reported as failed, with completed cases retained. Original screens returned small summaries; the updated launcher returned full
per-step/test evidence for all four development-campaign jobs. These files are archived with their reports. Checkpoints
remain remote. The campaign layer retains expected OOM screen outcomes while
stopping on unexpected errors; confirmation requires complete comparable cases.

## Implementation references

- [PyTorch 2.11 checkpointing](https://docs.pytorch.org/docs/2.11/checkpoint.html).
- [PyTorch 2.11 autocast](https://docs.pytorch.org/docs/2.11/amp.html).
- [PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).
- [Modal pricing](https://modal.com/pricing), checked October 31, 2026.
