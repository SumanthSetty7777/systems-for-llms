# Efficient LLM Training

A decoder Transformer training stack and experiments on attention, activation
memory and distributed training. The model uses RoPE, RMSNorm and SwiGLU, with
FP32 training or BF16 autocast, gradient accumulation and block checkpointing.
Distributed implementations include DDP, owner-sharded AdamW, FSDP2 and tensor
parallel attention/MLP layers.

## Results

The main workload is a 324M-parameter model trained from random initialization.
A 38-case screen covers 1/2/4/8 GPUs, followed by 21 confirmation cases across
three fresh seeds on eight H200s. The confirmation passed 335 GPU tests with
zero skips.

| Comparison at 1K context | Result |
| --- | --- |
| DDP, one to eight GPUs | **5.90× median paired speedup** (5.88–6.03×); mean 397,738 tokens/s |
| Owner-sharded AdamW vs DDP, eight GPUs | **87.4% less optimizer-state memory**; median 11.0% lower throughput |
| FSDP vs DDP, both block checkpointed, eight GPUs | **71.7% less peak allocated memory** (7.326→2.071 GiB); median 29.2% lower throughput |
| Tensor parallelism, eight GPUs vs single GPU | **62.8% less peak allocated memory**; median 17.2% lower throughput |

These are synchronized optimizer-step measurements from 60-update runs, with
five warmup updates excluded. They do not establish long-run convergence or
end-to-end training speed. All observed timing stalls remain in the results.

- [Distributed confirmation](results/2026-11-01-distributed-confirmation/README.md): per-seed results, figures, correctness logs and raw evidence.
- [Distributed screen](results/2026-11-01-distributed/README.md): GPU-count sweeps, 4K context, synchronization and bucket-size ablations.
- [Single-GPU screen](results/2026-11-01-scaling-screen/README.md): 48 attempts across 22.7M/100.3M/324.1M models at 1K/4K; 46 completed and two OOMs.
- [Quality development](results/2026-11-01-100m-quality-dev/README.md): three 1,200-update runs; checkpointing saved 59.3% allocated memory with 19.2% longer time to the selected loss target. This comparison has one development seed.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used below. The synthetic
smoke configuration needs no external data:

```bash
uv sync --frozen --python 3.12 --group dev
uv run --frozen python -m llmlab.train \
  --config configs/smoke.json --output-dir runs/smoke

PYTHONPATH=src:distributed uv run --frozen pytest tests distributed/tests -q
```

Each output directory must be new. Training writes metadata, per-step metrics,
a summary and a checkpoint. `runs/`, datasets and model checkpoints are ignored
by Git.

The test command selects both test directories; plain `pytest` selects only the
single-device suite. CUDA checks skip on CPU. The full GPU suite requires at
least two native-BF16 CUDA GPUs:

```bash
PYTHONPATH=src:distributed LLMDIST_TEST_DEVICE=cuda \
  uv run --frozen pytest tests distributed/tests -q
```

## Data and experiments

The TinyStories configurations use prepared token arrays and tokenizer files
that are not included in the repository. Their default paths point to a sibling
`assignment1-basics/` directory. Supply those files or update the paths in a copy
of the configuration. [Input requirements and code provenance](docs/provenance.md)
describe the formats and recorded identities.

- [Single-GPU runner and pilot](docs/gpu-pilot.md)
- [Model-size and context comparisons](docs/scaling-study.md)
- [Validation-target protocol](docs/quality-study.md)
- [Distributed commands and implementation](distributed/README.md)
- [Distributed measurement protocol](docs/distributed-study.md)

## Reproduce the analysis

The checked-in reports include the aggregate results, full metric streams,
bounded logs and input checksums. Numerical analysis needs only Python's standard
library; the optional `report` dependency group adds figures.

```bash
uv sync --frozen --group dev --group report
uv run --frozen python distributed/analyze.py \
  --run-dir results/2026-11-01-distributed-confirmation/inputs/confirmation \
  --output-dir runs/confirmation-report
```

Use `--no-plots` to generate tables and JSON without Matplotlib. See the
[distributed README](distributed/README.md#read-the-evidence) for screen analysis
and cloud launch commands.

## Limits

The distributed study uses one host per stage. Physical interconnect topology
was unavailable; peer-access checks do not identify link type or bandwidth.
TP and FSDP currently initialize the full model before sharding, so training
memory savings do not demonstrate initialization beyond one GPU's capacity.
Embeddings, normalization and the vocabulary head remain replicated in TP.

Single-device resume restores optimizer and sampler/RNG state; exact resumed
agreement is tested on CPU. Distributed resume, multi-node scaling and hybrid
parallelism are not implemented. Profiling the checkpoint stalls and extending
the fixed-quality comparisons are follow-up work.
