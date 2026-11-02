# Six-workload L40S screening results

November 1, 2026. Three model sizes × two context lengths × eight policies: **48 attempted configurations, 46 completed and two CUDA OOMs; no timeouts.** Each case is one 30-update run at seed 1337. These are screening measurements for choosing confirmation experiments, not repeated-run estimates or convergence results.

The clearest observed pattern is an increasing eager/SDPA attention gap at 4K context, with explicit eager attention becoming infeasible without checkpointing at 324M/4K. Checkpointed SDPA retains useful throughput while reducing allocated memory. This measures established techniques in this custom implementation; it is not a novelty claim.

## Fastest observed feasible policies

`none` and `block` describe activation checkpointing; `mb/acc` gives microbatch/accumulation. Both backends may select different policies. Ratios therefore describe the best observed choices in this finite screen, rather than an isolated backend intervention. Allocated memory is the SDPA policy's peak; inspect each report for all eight rows and eager memory.

| Workload | Completed | Fastest SDPA policy | SDPA tokens/s | SDPA GiB | Fastest eager policy | Eager tokens/s | SDPA/eager throughput |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| [22M / 1K](../2026-11-01-22m-1k/README.md) | 8/8 | none, mb8/acc1 | 255,753 | 2.37 | none, mb8/acc1 | 75,871 | 3.371x |
| [22M / 4K](../2026-11-01-22m-4k/README.md) | 8/8 | none, mb2/acc1 | 247,938 | 2.38 | none, mb2/acc1 | 24,519 | 10.112x |
| [100M / 1K](../2026-11-01-100m-1k/README.md) | 8/8 | none, mb8/acc1 | 83,015 | 6.75 | none, mb1/acc8 | 25,416 | 3.266x |
| [100M / 4K](../2026-11-01-100m-4k/README.md) | 8/8 | none, mb1/acc2 | 77,191 | 4.47 | none, mb2/acc1 | 5,594 | 13.798x |
| [324M / 1K](../2026-11-01-324m-1k/README.md) | 8/8 | none, mb8/acc1 | 29,160 | 16.92 | none, mb1/acc8 | 7,606 | 3.834x |
| [324M / 4K](../2026-11-01-324m-4k/README.md) | 6/8 | none, mb2/acc1 | 29,706 | 16.96 | block, mb2/acc1 | 1,868 | 15.905x |

At 100M/4K, uncheckpointed SDPA microbatch 1 is only 2% faster than microbatch 2; at 324M/4K, their difference is 0.035%. Those rankings require repeat invocations. Both lower microbatches use less allocated memory. The 324M/4K best-policy ratio compares uncheckpointed SDPA with checkpointed eager; it must not be described as a matched no-checkpoint comparison.

## Same-policy attention comparisons

For every row below, checkpointing is disabled and microbatch is held equal within the pair. The largest tested microbatch processes 8,192 tokens in one forward/backward pass, with no accumulation: 8×1,024 or 2×4,096. All completed runs also use 8,192 tokens per optimizer update.

| Workload | Microbatch | Eager tokens/s | SDPA tokens/s | SDPA/eager throughput | Eager GiB | SDPA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22M / 1K | 8 | 75,871 | 255,753 | 3.371x | 3.91 | 2.37 |
| 22M / 4K | 2 | 24,519 | 247,938 | 10.112x | 11.85 | 2.38 |
| 100M / 1K | 8 | 18,452 | 83,015 | 4.499x | 14.03 | 6.75 |
| 100M / 4K | 2 | 5,594 | 75,694 | 13.531x | 39.54 | 6.77 |
| 324M / 1K | 8 | 6,841 | 29,160 | 4.263x | 35.88 | 16.92 |
| 324M / 4K | 2 | OOM | 29,706 | undefined (eager OOM) | — | 16.96 |

At 324M/4K **both** eager/no-checkpoint cases (microbatch 1 and 2) failed before completing their first update. Their throughput and peak-memory measurements are missing, not zero. No finite or infinite speedup is assigned against them. With **block checkpointing held equal at microbatch 2**, SDPA measured 24,725 versus 1,868 tokens/s (**13.238x**) at 6.17 versus 17.19 GiB allocated. Full-matrix comparability is insufficient for that workload; the six completed cases have matching dataset and ordered-batch hashes.

## Memory trade-offs and experiment priorities

- At microbatch 8 / 1K, checkpointed SDPA saves 37.8%, 59.3% and 63.7% allocated memory for 22M, 100M and 324M, with 16.3%, 16.7% and 17.4% longer median steps respectively.
- At microbatch 2 / 4K, the corresponding savings are 37.8%, 59.3% and 63.6%, with 18.9%, 18.2% and 19.6% longer median steps. These paired observations motivate comparing memory-constrained policies, not a universal checkpointing recommendation.
- Timing confirmation repeats all eight policies at 100M/1K and 100M/4K, including the fastest observed eager, SDPA and checkpointed choices. Quality confirmation keeps the three predeclared SDPA policies: no checkpoint at microbatch 1 or 8, and block checkpointing at microbatch 8. Freeze its validation target using the separate development run before fresh-seed quality confirmation. Screening losses do not set a common target across workloads.
- Repeated 4K comparisons and a separate profiler trace can test the large long-context gap and identify actual dispatch. Keep profiler timing separate from throughput measurements. Multi-GPU DDP/sharding experiments belong to the subsequent stage; this screen provides no tensor-, pipeline- or distributed-scaling measurements.

## Scope and reproducibility limits

The model family has 22,696,448 / 100,313,856 / 324,092,928 parameters. Layer counts, width, head counts and feed-forward widths change together, so differences across sizes are not an isolated causal effect of parameter count. All use an untied 10K vocabulary/head, head dimension 64, MHA and RoPE theta 10,000. TinyStories token arrays and tokenizer hashes match. BF16 autocast retains FP32 parameters, optimizer state and loss.

Each completed case trained on 245,760 tokens. The first five updates are timing warmup; the following 25 are timed observations, not independent trials. Validation uses 4,096 tokens within each workload. Ordered sample hashes match across completed policies within a workload; changing context changes sequence grouping and sampled windows. **Do not pool validation losses across model sizes or context lengths.** The short runs do not establish quality equivalence, convergence, long-run stability or dollars to a quality target.

**Exposed GPU memory differs across jobs.** Five jobs reported 47,695,921,152 bytes (44.42 GiB); 324M/4K reported 50,897,879,040 bytes (47.40 GiB). All report NVIDIA L40S. The cause is not established. Same-invocation policy pairs share hardware, while cross-invocation capacity boundaries must retain this caveat. Separate cloud invocations can also differ in performance; no hardware-matched repeat estimate is available here.

Throughput covers complete synchronized optimizer steps: sampling/transfers, forward/backward, accumulation/recomputation, clipping, optimizer work and scalar diagnostics. It excludes setup, evaluation and checkpoint-file saves. Memory covers training and evaluation after model setup, including persistent state; allocated peaks do not equal total process VRAM. SDPA uses automatic backend selection and the actual kernel has not been verified. No component-level bottleneck attribution has been established.

The three newly archived jobs each captured **168 passed** GPU tests and complete returned per-step/configuration/source/hardware/test evidence. The first three archives have successful preflight/test exit gates but do not yet contain the full test or per-step logs. All raw archives retain source JSON unchanged with SHA-256 manifests. The overview's `input-sha256.json` identifies the twelve aggregate/launch-summary inputs; `derived-summary.json` records the selected case IDs and exact ratios.

The six remote function durations sum to **1419.513 seconds (23.66 minutes)**. This includes preflight/tests/orchestration and excludes client upload/image builds. Actual provider charges have not been retrieved; this sum is not a bill.
