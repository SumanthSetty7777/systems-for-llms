# Distributed training: protocol and scope

The study compares training throughput and memory at fixed global batch across
1/2/4/8 GPUs. The [screen](../results/2026-11-01-distributed/README.md) contains
38 scale cases and a separate 10-case smoke stage. A
[three-seed confirmation](../results/2026-11-01-distributed-confirmation/README.md)
repeats seven selected policies. This document defines the controls and
measurement scopes used by both reports.

## Research questions

1. How much does delayed gradient synchronization improve accumulation, and how
   does that effect change as the per-rank batch shrinks?
2. Where does strong scaling stop helping this model, and what do communication
   traces and per-rank timing suggest is responsible?
3. How much optimizer memory does explicit state sharding save, and what is the
   cost of broadcasting updated parameters?
4. How do activation checkpointing, FSDP and tensor partitioning change memory
   and step time at 1K versus 4K context?

The project contributes a controlled implementation and empirical comparison.
DDP, state sharding and tensor parallelism are established techniques; using them
does not by itself constitute a new algorithm or research novelty.

## Workload and controls

The main configuration is the existing 324M-parameter dense decoder: width 1024,
24 layers, 16 attention heads, SwiGLU width 2752 and a 10K-token vocabulary. The
attention and feed-forward dimensions admit tensor-parallel degrees 2, 4 and 8.
Use the existing TinyStories token arrays and tokenizer, recorded source hashes,
SDPA attention, FP32 parameters/AdamW states and BF16 autocast.

For the main strong-scaling sweep, hold the global sequence batch at 32 and
context length at 1024. Each optimizer update therefore consumes 32,768 target
tokens. Hold the initialization, ordered global batch stream, optimizer settings,
learning-rate schedule, update count and evaluation data fixed. Match these
through recorded configuration and batch fingerprints, not just the seed label.

For data-parallel modes, divide each globally sampled batch among ranks. With
world size `N`, microbatch `m` and accumulation `a`, `N × m × a = 32`.
Pure tensor parallelism gives the same batch to every rank: `m × a = 32`.
Its token count must not be multiplied by the tensor-parallel degree.

A 4K-context comparison is a separate workload. Keep its global batch fixed
across its own techniques; report the resulting tokens per update explicitly.
If memory requires a smaller batch, do not silently combine that case with the
1K strong-scaling curve. Changing microbatch size and accumulation at fixed global
batch is an intentional systems variable and must be visible in each row.

All primary scaling cases used one eight-GPU allocation, including the baseline
with one active GPU. H100:8 was requested; recorded hardware was eight H200s.
Modal documents [H100-to-H200 upgrades at the requested GPU price](https://modal.com/docs/guide/gpu).
Hardware comparisons use observed identities; cost estimates use the request.
Record GPU UUIDs, software versions, peer access and `nvidia-smi topo -m`.
The completed run's topology query was unavailable; peer-access capability does
not establish link type or bandwidth.
This is single-node scaling and does not measure inter-node network performance.

## Experiment families

The emitted plan fixes exact cases and limits before execution. These families
describe the controls used in the completed screen. Causal communication claims
still require diagnostic traces or a controlled intervention.

| Family | Controlled comparison | Evidence required |
| --- | --- | --- |
| Strong scaling | Single-GPU reference and DDP at 2/4/8 ranks, fixed global batch | Step time, throughput, speedup, efficiency, rank timing and loss |
| Gradient communication | Manual all-reduce after backward versus bucketed DDP | Numerical agreement and separate communication/computation trace |
| Accumulation synchronization | Synchronize every microstep versus DDP `no_sync` until the last | Equal batch/update, number of microsteps, measured timing and trace |
| Bucket size | DDP bucket capacities with other settings fixed | Time and overlap differences; no assumption that smaller or larger is always better |
| Optimizer state | Replicated AdamW versus deterministic owner-based state sharding | Per-rank and aggregate moment-state bytes, update equivalence, broadcast cost |
| Activation memory | No checkpointing versus block recomputation | Allocated/reserved peaks and step time at each context length |
| Full sharding | FSDP2 versus replicated training, when CUDA gate passes | Global clipping/update equivalence and memory/communication measurements |
| Tensor parallelism | Attention-head/FFN partitioning at supported degrees | Reconstructed output/gradient/update agreement, global norm, memory and layer collectives |

The custom optimizer mode keeps full model parameters and gradients on every
rank. It partitions AdamW states by whole parameter, updates each parameter on
one owner, then broadcasts that parameter. Imbalance is possible because whole
tensors cannot always be divided equally; report the maximum rank and aggregate
state memory. This is a transparent Stage-1-style experiment, not ZeRO-2/3.

The tensor-parallel mode partitions Q/K/V output rows, attention-output input
columns, SwiGLU expansion rows and contraction columns. Embeddings, RMSNorm and
the vocabulary head stay replicated. Collectives have explicit backward rules;
gradient norm calculation must count replicated parameters once and combine
sharded contributions. This is pure tensor parallelism, without a simultaneous
data-parallel mesh.

Both TP and FSDP currently create the full FP32 model on each GPU before slicing
or sharding. A reduction in training-time allocated memory therefore does not
establish that this implementation can initialize a model larger than one GPU's
capacity. Record setup separately: reserved allocator memory may retain setup
allocations even after the full tensors are released.

## Numerical gates before performance claims

Start with small models and deterministic FP32 data on CPU, then run the CUDA
checks on two GPUs with the pinned runtime. Compare against the unpartitioned
reference on identical global batches. The CUDA gate must cover each admitted
distributed path and its relevant gradient clipping behavior.

Check loss, gradients and updated parameters with declared tolerances, multiple
optimizer updates where applicable, replicated rank agreement, optimizer state
ownership and tensor reconstruction. Test checkpoint recomputation with the
parallel path that uses it. Record the tolerance, dtype, step count and maximum
observed error. Exact byte hashes establish replica consistency when appropriate;
they do not replace tolerance-based comparisons between different arithmetic
orders. BF16 equivalence needs its own tolerance and training evidence.

If a path fails, retain its error and exclude its performance from accepted
comparisons until repaired and rerun. A passed CPU test does not imply CUDA/NCCL
correctness, and a passed two-GPU check does not establish eight-GPU stability.
The larger stage also checks its own execution and returned evidence.

## Measurement definitions

The trainer synchronizes devices around a complete optimizer update and records
the maximum elapsed time across ranks. The timed region includes batch handling,
forward/backward, the mode's collectives, clipping and the optimizer update.
Warmup updates train normally but are excluded from measured throughput. The
pre-step barrier, timing-max reduction, evaluation, process setup and metric
recording sit outside this timed region. This is synchronized-step throughput,
not uninterrupted end-to-end training throughput. Include those excluded costs
when reporting whole-job time and requested-resource estimates.

| Quantity | Definition and interpretation |
| --- | --- |
| Throughput | Total useful target tokens in measured updates divided by their summed maximum-rank times |
| Median/p90 step time | Distribution of complete-update times after warmup; describes this short run, not independent-run uncertainty |
| Strong-scaling speedup | `T1 / TN`, using the same summary statistic and matched workload for both times |
| Scaling efficiency | `T1 / (N × TN)`; report the one-GPU reference policy and microbatch/accumulation settings |
| Active GPU-seconds | Active rank count times the stated elapsed interval; describes the devices participating in a case |
| Reserved GPU-seconds | Allocated GPU count times the allocated elapsed interval; on the primary host this count is always eight |
| Peak memory | Maximum allocated and reserved CUDA memory on each rank, plus the maximum across ranks; state whether evaluation is included |
| State-memory breakdown | Local parameter, gradient and optimizer tensor bytes; separate these from allocator peaks and temporary activations |
| Validation loss | Fixed held-out tokens and evaluation configuration; short-run stability evidence only |
| Job cost | Provider resource rates times the actual allocation's wall time, plus applicable extra charges; reconcile with account usage |

Collect profiler traces in separate cases. Overlapping CUDA operations mean that
summing kernel or NCCL durations does not directly yield critical-path overhead.
Use traces to inspect where communication is exposed and describe conclusions as
inferences unless a controlled intervention isolates the cause.

Repeat central baseline/winner comparisons with independent launches or repeat
IDs; summarize between-run variation separately from within-run step p90. A
single screening pass can choose candidates but cannot establish robust speedup.
Include OOM, timeout, numerical failure and slower configurations in the report.

## Runs and resource accounting

The [campaign](../distributed/README.md#run-the-campaign) freezes source and plans
before launching a two-GPU correctness stage followed by the eight-GPU screen.
Each stage requires zero skipped tests and stops on failure. Cases run
sequentially in separate subprocesses on the same allocation.

The confirmation uses seeds 2027/2028/2029, 1K context, global batch 32,
microbatch 4, 60 updates and five timing warmups. The seven policies are single
GPU, DDP8, owner-sharded AdamW8, FSDP8 and TP8 without checkpointing, plus DDP8
and FSDP8 with block checkpointing. The 4K and communication ablations are not
repeated. Paired ratios are summarized by seed; optimizer steps are not treated
as independent experimental repeats. The three seeds share one allocation.

Recorded requested-resource rates and function times give $17.93 for smoke and
screen, and $10.32 for confirmation. These estimates exclude image builds,
startup, excess CPU/RAM, storage/egress, other apps and the interrupted earlier
confirmation. They are not invoices. An eight-GPU allocation incurs eight-GPU
cost even when a case uses one active rank. Run manifests preserve the request,
observed devices and rate snapshot used by each estimate.

## References

- [Stanford CS336](https://cs336.stanford.edu/spring2025/) and
  [Umar Jamil's DDP tutorial](https://github.com/hkproj/pytorch-transformer-distributed):
  learning references for the training and distributed experiments.
- [PyTorch DDP](https://arxiv.org/abs/2006.15704): gradient buckets,
  communication overlap and accumulation synchronization.
- [ZeRO](https://arxiv.org/abs/1910.02054): optimizer-state partitioning. The
  custom implementation here shards AdamW states by parameter owner; it is not
  a complete ZeRO implementation.
- [PyTorch FSDP](https://arxiv.org/abs/2304.11277): parameter, gradient and
  optimizer-state sharding. This project uses the framework's FSDP2 API.
- [Megatron-LM](https://arxiv.org/abs/1909.08053): tensor partitioning across
  attention heads and feed-forward layers.

## Limitations and follow-up

The runs measure short-horizon, single-host performance. Final validation loss
is an early-training check; distributed time-to-quality needs longer runs with
a separately fixed evaluation target. The 4K and communication ablations have
one screening seed. Physical interconnect topology is unavailable.

TP and FSDP initialize the full FP32 model on each GPU before sharding. Training
memory savings therefore do not establish initialization beyond one GPU's
capacity. TP leaves embeddings, normalization and the vocabulary head replicated.

Profiler traces are needed to investigate checkpoint timing stalls and TP's
slowdown. Multi-node scaling, hybrid meshes, distributed checkpoint/resume and
custom attention kernels are separate extensions.
