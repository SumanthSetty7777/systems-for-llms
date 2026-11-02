# GPU screen: 324M parameters, 1K context

November 1, 2026. Run `study-20261101T132912Z-324m-1k-screen`.

All eight configurations completed with passing comparability checks. At microbatch 8 without checkpointing, SDPA measured **4.263x eager throughput** with **52.8% less peak allocated memory**.
This is one invocation per configuration at one seed, not a convergence or quality-equivalence experiment.

## Workload and evidence checks

- Custom 324,092,928-parameter MHA decoder: 24 layers, width 1,024, 16 heads, head dimension 64, feed-forward width 2,752, untied 10,000-token vocabulary/output head, random initialization.
- TinyStories arrays; context 1,024; BF16 autocast with FP32 parameters, optimizer state and loss. Effective batch 8 sequences / 8,192 tokens per update.
- Seed 1337; each completed case trained for 30 updates / 245,760 tokens. Excluding the first five updates leaves 25 timed updates. Validation uses 4 fixed batch(es) of one sequence, totaling 4,096 tokens.
- All 8 completed cases have matching dataset/tokenizer identities and ordered training-batch hashes, expected measured-step counts and finite final losses. This does not assert identical floating-point updates.
- Native CUDA/BF16 preflight passed. Captured test result: **168 passed in 8.05s**. Full returned evidence has no missing files, errors or truncated logs.
- GPU: NVIDIA L40S, reported total memory **47,695,921,152 bytes (44.42 GiB)**. PyTorch 2.11.0; one GPU.

![Throughput and peak allocated CUDA memory](throughput-memory.png)

## Measurements

All eight attempted cases are retained below. GiB means 2^30 bytes. Empty metrics indicate no valid completed measurement; they are not zero throughput or zero memory.

| Attention | Checkpoint | Microbatch / accumulation | Status | Median step (ms) | p90 step (ms) | Tokens/s | Allocated (GiB) | Reserved (GiB) | Validation loss |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDPA | none | 8 / 1 | completed | 280.94 | 282.19 | 29,160 | 16.92 | 17.83 | 4.630833 |
| SDPA | block | 8 / 1 | completed | 329.81 | 331.36 | 24,597 | 6.15 | 7.83 | 4.630326 |
| SDPA | none | 1 / 8 | completed | 751.18 | 774.09 | 10,864 | 7.04 | 7.54 | 4.631053 |
| EAGER | none | 1 / 8 | completed | 1070.90 | 1091.07 | 7,606 | 9.41 | 9.74 | 4.629509 |
| EAGER | none | 8 / 1 | completed | 1197.59 | 1198.31 | 6,841 | 35.88 | 37.12 | 4.628251 |
| SDPA | block | 1 / 8 | completed | 1426.94 | 1489.26 | 5,671 | 6.07 | 6.73 | 4.629031 |
| EAGER | block | 8 / 1 | completed | 1538.46 | 1539.54 | 5,324 | 8.16 | 8.99 | 4.628251 |
| EAGER | block | 1 / 8 | completed | 1678.88 | 1788.86 | 4,825 | 6.08 | 6.51 | 4.629509 |

The figure shows completed cases only (8/8); OOM outcomes are retained in the table and raw evidence.
Throughput is measured tokens divided by summed timed-step durations, not the reciprocal of the median.
Steps include sampling/transfers, forward/backward, accumulation, checkpoint recomputation, clipping, optimizer work and scalar diagnostics with CUDA synchronization. Imports/setup, evaluation and checkpoint-file saves are excluded from step timing. Memory peaks cover training and evaluation after model setup, including persistent state; allocated memory is not total GPU-process VRAM.

## Observations

1. **SDPA without checkpointing at microbatch 8 is fastest:** 29,160 tokens/s at 16.92 GiB. It is 4.263x the matched eager microbatch-8 throughput.
2. The fastest observed eager policy is instead microbatch 1 / accumulation 8 (7,606 tokens/s, 9.41 GiB). The independently selected best SDPA/eager ratio is 3.834x, with different memory use. Comparing only against eager microbatch 8 would overstate the best-policy throughput gap.
3. **Checkpointed SDPA at microbatch 8 is a useful memory candidate:** 24,597 tokens/s at 6.15 GiB, saving 63.7% memory for a 17.4% median-step increase. It also measured higher throughput and lower allocated memory than every eager case in this invocation.

Completed-case validation losses span 4.628251–4.631053 (absolute spread 0.00280225). These are numerical sanity checks on a small validation sample. Thirty updates and one seed do not establish convergence, long-run stability, quality equivalence or time to target. Do not compare or pool these losses across model sizes or contexts. The 25 timed steps are not 25 independent experiment repetitions. SDPA uses automatic dispatch; no specific FlashAttention kernel has been verified.

## Time and reproducibility

Remote function time was **365.012 seconds**. Attempted-case subprocess durations sum to **347.629 seconds**. Function timing includes preflight, tests and orchestration, but excludes client upload/image builds; these durations are not a provider bill or proof of dollar savings.

`launch.json`, `launch-summary.json`, `aggregate.json` and `evidence.json` are byte-for-byte copies of `runs/study-20261101-development/324m-1k-screen/`. `input-sha256.json` records their SHA-256 hashes. Evidence includes per-step metrics, case execution results and log tails, resolved configuration, source/package/hardware metadata, preflight and test logs. Model checkpoint binaries remain on Modal Volume `systems-for-llms-results`, under `study-20261101T132912Z-324m-1k-screen/`.

Reproduce the plot with matplotlib installed:

```sh
python scripts/plot_pilot.py results/2026-11-01-324m-1k \
  --title "324M model / 1K context / NVIDIA L40S"
```

See the [six-workload overview](../2026-11-01-scaling-screen/README.md) for scope and candidate-selection limits.
