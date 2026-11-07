# CUDA split-KV grouped-query decode

This extension implements exact single-token attention with a contiguous KV
cache. It explores whether splitting a long cache across independent thread
blocks improves GPU occupancy for small decoding batches. It is a forward-only
CUDA C++ implementation, with FP16/BF16 inputs and FP32 online-softmax state.
The [completed H100 study](../results/h100-extensions-20261105/analysis/report.md)
passes **582/582 correctness checks**. Tile-level softmax reduced latency by
**16.1% geometrically** relative to the token-level version across 180 matched
split/shape/seed pairs (140 wins). Flash SDPA remained faster in all 30 workload
cases, including against each case's best observed custom configuration. The
optimization is an internal-kernel improvement, not a claim of state-of-the-art
decode performance.

## API

```python
import torch
from attention_lab.cuda_decode import prepare

q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, 16384, 128, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
lengths = torch.tensor([16000], device="cuda", dtype=torch.int32)

operation = prepare(q, k, v, lengths, splits=16, variant="tile")
output = operation()  # [batch, query_heads, head_dim]
```

`prepare` validates inputs, JIT-compiles or loads the CUDA extension, and
allocates reusable output and scratch storage. Repeated calls reuse that
storage, and the returned tensor is overwritten by the next call. The helper
`decode(q, k, v, lengths=None, splits=1, scale=None, variant="token")` prepares
and executes once. `variant="token"` preserves the original implementation;
`variant="tile"` selects the tile-level softmax optimization.
CUDA development headers, a host C++ compiler, and Ninja are needed for JIT
compilation. The prepared operation uses the caller's current CUDA stream and
supports CUDA graph capture after preparation.

| Input | Contract |
| --- | --- |
| Q | Contiguous `[B, Hq, D]` CUDA FP16 or BF16 |
| K, V | Contiguous `[B, Hkv, N, D]`, same device and dtype as Q |
| Head dimension | 64 or 128 |
| Grouped-query ratio | `Hq / Hkv` in 1, 2, 4, 8 |
| Lengths | Contiguous CUDA int32 `[B]`, each value in `[0, N]` |
| Splits | Integer from 1 through 128 |
| Scale | Positive finite number, default `D ** -0.5` |

Each query attends to all valid cache entries for its request; there is no
additional causal or document mask. An empty request returns zeros. The valid
lengths can be changed in place between calls, including graph replays, but must
remain in the documented range. Python validates their values during preparation;
the kernel defensively clamps them. Calls sharing one prepared operation must
not execute concurrently because its scratch and output storage are mutable.
No backward pass or autograd support is provided.

## Kernel design

One thread block owns a request, a KV head, and a cache partition. A query-head
warp distributes the head dimension across 32 lanes. Query warps in the same
GQA group reuse a shared-memory tile of 32 K/V tokens, avoiding physical KV-head
repetition. Dot products use warp reductions. Each warp accumulates a running
maximum, exponential sum, and weighted-value numerator in FP32.

The token variant rescales its state after every token. The tile variant keeps
one score per lane, reduces the maximum for the tile, evaluates its 32
exponentials in parallel, and rescales the old numerator once per tile. Both
use the same shared KV tiling, split partitioning, merge, and FP32 arithmetic.
Paired timings at the same split count isolate the effect of this optimization.

For partition `s`, the kernel produces `(m_s, l_s, u_s)`. The merge computes
`m = max_s(m_s)`, `w_s = exp(m_s - m)`, and
`output = sum_s(w_s * u_s) / sum_s(w_s * l_s)`. Empty partitions contribute zero.
This preserves global softmax normalization across independently computed
partitions. The split-one path writes the output directly and skips the merge.

The experiments vary partition count to expose the tradeoff between additional
parallel work and the scratch traffic plus reduction launch. Partition boundaries
are based on cache capacity; highly uneven valid lengths can leave many empty
partitions. There is no adaptive dispatcher or claim of a universally best
partition count.

## Validation and experiments

From the repository root:

```bash
python -m unittest attention_lab.cuda_decode.test_decode -v
python -m attention_lab.cuda_decode.study --stage smoke --output-dir results/cuda-smoke --time-budget-seconds 600 --execute
python -m attention_lab.cuda_decode.study --stage study --output-dir results/cuda-study --time-budget-seconds 900 --execute
```

Without `--execute`, the entry point only prints its plan. It never provisions
GPUs. The smoke suite checks both dtypes, both head dimensions, every supported
GQA ratio, lengths around tile boundaries, ragged and empty requests, split
counts 1/3/8, the maximum 128-way reduction with a custom scale, non-default
stream execution, and graph replay after lengths change. Its independent FP32
reference computes token-level attention without using the CUDA partitioning.
Both variants run the full suite (582 checks). The study reruns this preflight
before measuring performance.

The study covers batches 1/8/32, cache lengths 1K/4K/16K/64K, and three input
seeds. Each variant compares split counts 1/2/4/8/16/32 with forced Flash SDPA,
compiled FlexDecoding (`BACKEND="TRITON_DECODE"`), and standard compiled
FlexAttention (`BACKEND="TRITON"`) on equivalent unmasked, full-length inputs.
These are explicitly separate backends in PyTorch 2.11; selecting `TRITON`
does not allow automatic dispatch to FlexDecoding. Unsupported baselines are
explicitly recorded. Static Flex
compilation is reset between shapes to avoid a recompilation-limit fallback;
filesystem caches are retained.

Timing uses warmed CUDA graphs: 12 raw event samples, each averaging 10 calls.
It includes all device work, including the split merge and scratch traffic.
It excludes Python submission, validation, allocation, and compilation.
Synchronized setup/first-call wall times are recorded separately, with cache
state explicitly distinguished from a fresh compiler run. These are device
execution measurements, not server latency or tokens per second. Repeated seeds
are input replicates, not independent machines or cold-cache trials.

`environment.json`, `plan.json`, `correctness.json`, individual case files,
`benchmark.json`, and `summary.json` record configuration, errors, raw samples,
and completion status where applicable. Cases are saved incrementally. A time
budget can stop the sweep between cases; only completed cases are reported.
The best observed split is selected on the same measured case, so its speedup
is an observed upper bound for a future dispatcher, not a held-out result.
Results also preserve every matched-split tile-versus-token comparison and
compare each variant's best observed split separately.
Incomplete sweeps exit with status 2. Summaries include the number of completed
pairs for each baseline so missing comparisons cannot be hidden by an average.
Reported effective KV bandwidth divides the logical cache bytes by latency;
repeated graph calls can reuse the GPU cache, so it is not a measurement of
physical HBM traffic or a sustained DRAM bandwidth claim.

The immutable run `cohere-cuda-study-20261105T182625Z-06dfba` used the raw name
`flex_decode` for a `TRITON` baseline. Its results compare standard FlexAttention,
not FlexDecoding. Treat those external-baseline results as diagnostic only;
subsequent runs record both explicit backend names and this correction in
their environment metadata. The custom CUDA variants themselves are unchanged.

## Limits

This is a research prototype using ordinary CUDA arithmetic and warp reductions,
not Tensor Core instructions. Strong vendor or compiler-generated kernels may
be faster. The kernel does not implement a paged cache, quantized KV storage,
prefix sharing, speculative multi-token decoding, tensor-parallel communication,
or persistent scheduling. Inputs must be finite. Shared KV loads are part of
the design, but an isolated ablation would be required to attribute a measured
improvement specifically to that reuse.
