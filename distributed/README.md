# Distributed training study

This extension studies how communication, optimizer state, activation memory and
tensor partitioning change complete Transformer training steps on 1, 2, 4 and 8
GPUs. The primary workload is the existing 324M-parameter decoder at 1K context,
with selected 4K-context comparisons. The global batch stays fixed within each
scaling comparison.

**Completed: 10/10 smoke cases, 38/38 scale cases and 21/21 confirmation cases.**
The smoke host had two H100s; the scale and confirmation hosts each had eight
H200s. The initial stages passed 333 tests each, and confirmation passed
**335 tests with zero failures, errors or skips**, on PyTorch 2.11.0+cu130 /
CUDA 13.0. The [screen](../results/2026-11-01-distributed/README.md) and
[confirmation](../results/2026-11-01-distributed-confirmation/README.md) reports
preserve figures, per-rank metrics, test logs and source/data hashes.

At 1K, confirmation DDP throughput averaged 397,738 ± 3,531 tokens/s (sample SD).
The median paired eight-GPU speedup was **5.90×** (range 5.88–6.03×), or 73.8%
efficiency. Checkpointed FSDP8 used 2.071 GiB peak allocated memory versus 7.326 GiB
for checkpointed DDP8 (71.7% less), with median 29.2% lower throughput.
Uncheckpointed TP8 was median 17.2% slower than the single-GPU reference.
These synchronized-step results cover three fresh seeds and 60 updates per case,
with five warmup updates excluded; they do not establish convergence equivalence.

The [study protocol](../docs/distributed-study.md) defines the controls, metrics,
references and limitations.

## Run the campaign

The completed campaign is `runs/distributed-20261101-retry1`. To repeat the full
screen from the repository root, choose a fresh output directory:

```bash
uv run --no-project --python 3.12 --with modal==1.5.1 \
  python distributed/run_campaign.py \
  --output-dir runs/distributed-new \
  --execute
```

Omit `--execute` to inspect the frozen plans, source/data payload, allocations and
cost estimate without creating GPU work. Use a fresh output directory for each
execution. The campaign consists of:

| Stage | Requested allocation | Cases | Remote job limit |
| --- | --- | --- | --- |
| Correctness gate | 2 H100 GPUs | 10 small-model cases | 900 seconds |
| Scaling study | 8 H100 GPUs | 38 cases spanning 1/2/4/8 active ranks | 2,400 seconds |

The completed scale stage received H200s despite its H100:8 request. Modal
documents [H100-to-H200 upgrades at the requested GPU price](https://modal.com/docs/guide/gpu).
Record actual devices for hardware comparisons and the request for resource-cost
estimates. The topology query was unavailable; peer access does not identify
interconnect type or bandwidth.

Each stage requires the mandatory test suite to pass with zero skipped tests.
A failed test, numerical gate or case stops execution; subsequent work remains
unrun. There are no automatic retries or confirmation runs. The source snapshot
and plans are checked before each stage.

Allow roughly 30–75 minutes for the initial campaign, including queueing,
allocation, builds and tests, if correctness checks pass. Availability or debugging
can extend that estimate. The initial screen and selected confirmation are complete.

At the current requested resources and timeouts, the two stages total about
**$24.04**. This excludes other application usage, image builds/startup, excess
CPU/RAM and storage/egress; it is not an account billing cap. Resource accounting is described in the [study protocol](../docs/distributed-study.md). The completed
retry's reported function times give a $17.93 requested-resource estimate, not an
invoice or a complete account-spending total.

## Three-seed confirmation

Run `distributed-20261101T230037Z-e05d5479` completed all **21 cases** on
H200:8 in **988.25 seconds (16.47 minutes)**. The frozen plan repeats seven 1K
policies on seeds 2027, 2028 and 2029: single GPU, DDP8, owner-sharded AdamW8,
FSDP8 and TP8 without checkpointing, plus DDP8 and FSDP8 with block checkpointing.
Global batch 32, microbatch 4, 60 updates and five timing warmup updates remain
fixed. The 4K and communication ablations remain screen-only findings.

The [confirmation report](../results/2026-11-01-distributed-confirmation/README.md)
contains all paired seed ratios, their median/range, and policy means with sample
standard deviations. Optimizer steps are not treated as independent experimental
repeats. Both block policies are present, but there is no checkpointed single-GPU
confirmation baseline. The three seeds share one allocated host; host-to-host
variation was not measured.

To inspect or repeat the confirmation, choose a fresh output directory:

```bash
uv run --no-project --python 3.12 --with modal==1.5.1 \
  python distributed/run_modal.py \
  --stage confirmation \
  --config configs/scale-324m-s1024.json \
  --gpu H200:8 \
  --job-timeout-seconds 1800 \
  --output-dir runs/distributed-confirmation-new
```

Append `--execute` to launch. The completed run used
`runs/distributed-confirmation-20261101-retry1`. Its requested GPU/CPU/RAM
function-time estimate is **$10.3171**; the 1,800-second timeout estimate is
$18.7914. Both exclude startup/builds, excess resources and other usage and are
not provider invoices or account caps.

The earlier confirmation launch, `distributed-20261101T190503Z-bfcd75ec`, was
interrupted by a provider workspace-disablement error after entering the study
stage. Its local launch manifest and error remain preserved separately. Its
completed-case count and cost are unknown; it contributes no measurements here.

## Implementation

| Component | Location | Scope |
| --- | --- | --- |
| Training and evaluation | [llmdist/train.py](llmdist/train.py) | Single GPU, manual gradient reduction, PyTorch DDP, optimizer sharding, FSDP2 and pure tensor parallelism; common batch stream and measurements |
| Optimizer state sharding | [llmdist/optim_shard.py](llmdist/optim_shard.py) | Parameter owners hold AdamW moments and broadcast updated weights; parameters and gradients remain replicated |
| Tensor parallelism | [llmdist/tensor_parallel.py](llmdist/tensor_parallel.py) | Partition attention heads and SwiGLU hidden units; explicit autograd collectives and global gradient clipping |
| Experiment driver | [run_experiments.py](run_experiments.py) | Planned cases, bounded subprocesses, numerical gates and aggregate evidence |
| Modal launcher | [run_modal.py](run_modal.py) | Explicit source/data payload, topology capture, GPU tests, bounded execution and result retrieval |
| Campaign | [run_campaign.py](run_campaign.py) | Freeze source and plans; run the two-GPU gate before the eight-GPU study; stop on failure |
| Analysis | [analyze.py](analyze.py) | Recheck raw evidence, match baselines, report throughput/memory/cost and create portable tables and figures |

These are new project modules. The original `src/llmlab`, single-GPU runner and
frozen single-GPU study inputs are reused without changing their implementation.
The optimizer experiment is Stage-1-style state sharding, not a complete ZeRO
implementation. Tensor parallelism currently leaves embeddings, RMSNorm and the
vocabulary projection replicated. FSDP2 is admitted only after its CUDA numerical
gate passes; this gate passed on both completed hosts. CPU checks alone cannot
validate that path.

TP and FSDP currently initialize a full FP32 model on each GPU before partitioning
it. Their measurements can demonstrate memory savings during training, but do not
demonstrate initialization of a model too large for one GPU. Reserved allocator
memory may retain cached allocations from that setup.

## Individual stage commands

Use these when intentionally running a standalone stage rather than repeating
the campaign. The launcher defaults to a dry run and prints the case plan, upload
manifest, output path, allocation and requested-resource cost estimate.

```bash
python distributed/run_modal.py \
  --stage smoke \
  --config configs/scale-324m-s1024.json \
  --gpu H100:2 \
  --job-timeout-seconds 900
```

The standalone correctness stage uses two GPUs:

```bash
uv run --no-project --python 3.12 --with modal==1.5.1 \
  python distributed/run_modal.py \
  --stage smoke \
  --config configs/scale-324m-s1024.json \
  --gpu H100:2 \
  --job-timeout-seconds 900 \
  --execute
```

The scale stage is the 1/2/4/8-GPU study on an eight-GPU host. Launch it after the
correctness stage has completed and its numerical gates have passed:

```bash
uv run --no-project --python 3.12 --with modal==1.5.1 \
  python distributed/run_modal.py \
  --stage scale \
  --config configs/scale-324m-s1024.json \
  --gpu H100:8 \
  --job-timeout-seconds 2400 \
  --execute
```

Cases within a scaling stage run sequentially on the same host. This keeps GPU
type and topology fixed and avoids one case competing with another for bandwidth.
The plan, rather than this README, is the source of truth for exact case counts,
steps and parameters. Independent jobs may run concurrently when their resource
allocations and budget are accounted for separately.

For a local GPU host, the experiment-driver interface is:

```bash
PYTHONPATH=src:distributed python distributed/run_experiments.py \
  --plan /path/to/plan.json \
  --config configs/scale-324m-s1024.json \
  --output-dir runs/distributed-local \
  --execute
```

The launcher pins Python 3.12, PyTorch 2.11.0 and Modal 1.5.1; the printed payload
records the remaining packages and SHA-256 source snapshot.

A clean checkout additionally needs the prepared TinyStories token arrays and
tokenizer artifacts in the [root input instructions](../README.md#data-and-experiments).
They are not bundled in Git. `uv sync --group dev` installs the baseline package;
distributed imports and both test trees need explicit paths:

```bash
PYTHONPATH=src:distributed uv run pytest tests distributed/tests -q
# Full CUDA/NCCL checks require at least two native-BF16 CUDA GPUs:
PYTHONPATH=src:distributed LLMDIST_TEST_DEVICE=cuda \
  uv run pytest tests distributed/tests -q
```

CPU test skips are not a substitute for the mandatory zero-skip cloud gate.

## Read the evidence

The campaign saves its frozen definition and stage outcomes in `campaign.json`.
Each stage returns `launch-summary.json`, `aggregate.json` when available, and an
`evidence.json` bundle. The bundle preserves the submitted configuration and plan,
GPU identities and topology, tests, per-rank metrics, case summaries and bounded
logs. Full remote evidence is retained in the results Volume. Inspect a failed or
partial bundle before rerunning; a timeout or failed numerical gate is a result
to explain, not a completed comparison.

Topology and identity management queries are optional diagnostics. When
`nvidia-smi` queries fail, their exit status/stdout/stderr are retained and the
report marks unavailable topology explicitly. CUDA device-count/native-BF16
checks and the CUDA/NCCL numerical suite remain mandatory. Peer-access capability
does not establish interconnect type or bandwidth.

Reports can be regenerated from their checked-in input evidence into a fresh
output directory:

```bash
python3 distributed/analyze.py \
  --run-dir results/2026-11-01-distributed/inputs \
  --output-dir results/distributed-screen-regenerated

python3 distributed/analyze.py \
  --run-dir results/2026-11-01-distributed-confirmation/inputs/confirmation \
  --output-dir results/distributed-confirmation-regenerated
```

Install the `report` dependency group for PNG/SVG figures. `--no-plots` uses only
the standard library. The output directory must be new. The report copies its raw
inputs with SHA-256 checksums and leaves missing or unmatched comparisons
unmeasured. The checked-in reports contain those inputs, so the original cloud
account is not required to inspect or reproduce the numerical analysis.

Report throughput alongside per-rank peak allocated/reserved memory, optimizer
state bytes, validation loss, source/data identifiers and numerical gate results.
Profiler runs are diagnostic and must be excluded from headline throughput.
Short-run validation loss is a stability check; it does not establish a
time-to-quality advantage. Throughput uses synchronized maximum-rank optimizer
step times and excludes setup, evaluation and inter-step barriers. It is not
uninterrupted end-to-end training throughput.

**An eight-GPU allocation is charged for eight GPUs even during its one-GPU
baseline.** Active GPU-seconds describe computational use; reserved GPU-seconds
describe the GPU allocation. Use the launch wall time and allocation to estimate
cost, and reconcile spending against the provider's account usage. The timeout
estimate is not an invoice cap and excludes other applications.

## Limitations

These are short, single-host training comparisons. Profiler traces are needed
before attributing TP slowdown or checkpoint timing stalls to a particular
operation. Multi-node scaling, hybrid meshes and distributed checkpoint/resume
are not implemented. Fixed-quality training comparisons remain separate from
these synchronized-step measurements.
