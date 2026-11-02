# Quality comparison and confirmation

A fixed validation-target comparison for the 100M/1K single-GPU workload.
Development runs are complete; fresh-seed quality confirmation remains deferred.
The [distributed results](../results/2026-11-01-distributed-confirmation/README.md)
cover a separate performance study and do not establish time to this target.

## Evidence so far

The [size/context screen](../results/2026-11-01-scaling-screen/README.md) is complete:
48 attempted cases, 46 completed and two OOMs. Both failures were eager attention
without checkpointing at 324M/4K, before any training update. The five complete
matrices passed comparability; the partial 324M/4K matrix is marked insufficient.
The latter job exposed 47.40 GiB on its L40S, versus 44.42 GiB elsewhere, which
limits capacity comparisons across invocations.

The development campaign also completed three 1,200-update 100M/1K quality runs.
All four campaign jobs returned full evidence and logs showing 168 GPU tests
passed. The [quality development report](../results/2026-11-01-100m-quality-dev/README.md)
records learning curves, timing scopes and the target calculation. The predefined
rule gives **2.40**: ceil(2.3883473575115204 / 0.05) × 0.05. All three policies first
reached it at update 600 / 4,915,200 training tokens.

Checkpointed microbatch 8 reached the target in 90.741 seconds with 2.745 GiB
allocated, versus 76.114 seconds and 6.751 GiB without checkpointing. The
microbatch-1/accumulation-8 baseline took 253.167 seconds with 2.429 GiB allocated.
Its throughput was 48.7% lower than in the short pilot, already in early updates;
the cause is unproven. Treat its apparent 2.79× time-to-target disadvantage as a
development observation awaiting fresh-seed confirmation, not a general claim.

## Deferred confirmation

The saved target and confirmation plan passed local validation on November 1.
The command below runs five jobs sequentially on one L40S at a time. Choose a
fresh output directory; there is no automatic retry or resume. This confirmation
has not been launched.

```sh
uv run --no-project --python 3.12 --with modal==1.5.1 python scripts/run_study.py --stage confirmation --output-dir runs/quality-confirmation --execute
```

Omit `--execute` to inspect the resolved plan and validate its inputs without
contacting Modal. Both modes require
`results/2026-11-01-100m-quality-dev/frozen-target.json`; the target and development
evidence are checked before launching, together with unchanged training source,
data identities, and compatible quality settings. Keep code/configs unchanged
while a campaign runs; documentation and report updates do not change its payload.

| Job | Cases | Updates/case | Seed(s) | Case / job timeout |
| --- | ---: | ---: | --- | --- |
| 100M/1K quality | 3 | 1,200 | 2027 | 600 / 2,400 seconds |
| 100M/1K quality | 3 | 1,200 | 2028 | 600 / 2,400 seconds |
| 100M/1K quality | 3 | 1,200 | 2029 | 600 / 2,400 seconds |
| 100M/1K timing | 24 | 60 | 2027, 2028, 2029 | 180 / 2,400 seconds |
| 100M/4K timing | 24 | 60 | 2027, 2028, 2029 | 180 / 2,400 seconds |

This is **57 new cases**. Quality runs come first so the central time-to-target
result is available before the broader timing repeats. `--stage quality-confirmation`
and `--stage timing-confirmation` also allow the two parts to run separately;
do not use them to duplicate already completed cases from the combined stage.

The five job timeouts total 12,000 seconds. Requested resources are one GPU,
two CPU cores and 8 GiB RAM per job. The launch plan records the resource-rate
estimate; final cost accounting must include startup/builds, excess resources,
storage and any other usage.

Unexpected process errors, comparability failures, missing successful-case evidence,
or changed inputs stop subsequent launches. Completed work is retained. Treat a
`blocked` stage as requiring inspection, not an instruction to rerun the whole
stage in a new directory. The launcher returns `evidence.json` containing full
per-step metrics, metadata, resolved plans, test logs and bounded failure tails;
model checkpoints remain in the Volume.

## Fixed quality protocol

Same 100,313,856-parameter decoder and 1,024-token context as the pilot. Every
policy uses effective batch 8 / 8,192 tokens per update, BF16 autocast and FP32
parameters/loss/optimizer state. Run from initialization for **1,200 updates /
9,830,400 training tokens per case**. Do not extend the earlier 30-step cosine run.

| SDPA policy | Role | Pilot allocated peak | Pilot tokens/s |
| --- | --- | ---: | ---: |
| No checkpoint, microbatch 8 / accumulation 1 | Unconstrained framework baseline | 6.75 GiB | 83,015 |
| No checkpoint, microbatch 1 / accumulation 8 | Selected lower-memory baseline | 2.43 GiB | 40,653 |
| Block checkpoint, microbatch 8 / accumulation 1 | Candidate memory/speed trade-off | 2.75 GiB | 71,084 |

The latter two fit the illustrative **3 GiB allocated-memory budget** selected
from development measurements. This does not mean fitting a physical 3 GiB GPU:
checkpointed microbatch 8 reserved 3.04 GiB, before external CUDA/context overhead.
These are selected policies, not an exhaustive microbatch optimum. Intermediate
microbatches and the particular hardware kernel may change the best choice.

- Development seed 1337; confirmation seeds 2027, 2028, 2029, paired by initialization
  and ordered batches. Keep the numerical/training settings identical across policies.
- Learning rate 0.0003, minimum 0.00003, 50 warmup updates, cosine schedule to update
  1,200. Other optimizer settings stay as in `scale-100m-s1024.json`.
- Evaluate the same 64 fixed validation windows every 50 updates, batch 1:
  65,536 validation tokens at each evaluation. Their selection is independent of
  training seed and microbatch. Checkpoint at update 1,200, identically for all policies.
- Target-selection rule, fixed before development: take the largest of the three
  validation losses at update 600, then round upward to the next multiple of 0.05.
  After development, save that numeric target, this rule and input hashes before
  launching any quality-confirmation seed. If development is incomplete or invalid,
  do not freeze a target from partial curves.
- Report each seed's endpoint loss and paired differences. Report the first observed
  evaluation at or below the frozen target, with its training tokens and elapsed
  time. Preserve unreached targets; do not interpolate an unseen crossing.
- `wall_seconds` at evaluation is measured from runner entry: setup, training and
  earlier evaluation/checkpoint costs are included; Python startup is excluded.
  Also report whole subprocess/job times. A final checkpoint occurs after the final
  evaluation, so it contributes to whole-job time but not that evaluation's crossing.

This tests whether a measured speed/memory advantage survives to a common sampled
validation loss. Three paired seeds allow descriptive variability and paired loss
differences, not an automatic statistical equivalence claim. It does not establish
downstream accuracy or document-deduplicated generalization.

## Confirmation analysis

For each seed, report endpoint loss, first observed target crossing, ordered
batch identity, allocated/reserved peaks and complete job time. Keep unreached
targets visible. Timing repeats use 60 updates, with 55 measured after warmup.
Separate profiler runs can investigate the microbatch-1 timing change; traced
timings must not enter the unprofiled comparison.

The full 12-case quality study would process 117,964,800 tokens including repeated
examples across policies; this is not a unique-token count. The three development
cases are complete. The remaining nine quality cases and 48 timing repeats are
not part of the completed results.
