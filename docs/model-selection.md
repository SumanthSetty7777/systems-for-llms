# Model and workload

The experiments use one custom decoder architecture at three sizes. Training
starts from random weights. The model implementation was adapted from the local
CS336 learning code; [provenance](provenance.md) records the reused modules and
integration checks.

## Architecture

All six scale configurations use causal multi-head attention, pre-norm RMSNorm,
RoPE, SwiGLU, a 10,000-token vocabulary and separate token-embedding/output-head
weights. Head dimension is 64 and RoPE theta is 10,000.

| Config family | Parameters | Width | Layers | Attention heads | Feed-forward width |
| --- | ---: | ---: | ---: | ---: | ---: |
| [22M](../configs/scale-22m-s1024.json) | 22,696,448 | 512 | 4 | 8 | 1,344 |
| [100M](../configs/scale-100m-s1024.json) | 100,313,856 | 768 | 12 | 12 | 2,048 |
| [324M](../configs/scale-324m-s1024.json) | 324,092,928 | 1,024 | 24 | 16 | 2,752 |

Each family has 1,024- and 4,096-token context configurations. The width and depth
increase together, so the comparison does not isolate parameter count from
architecture shape. The earlier 22.7M integration pilot used 16 attention heads;
it is a separate configuration from the scale family.

The small model provides inexpensive integration checks. The 100M model supports
longer development runs; the 324M model provides the larger single-GPU and
distributed workload. The 324M dimensions support tensor-parallel degrees 2, 4
and 8 for the attention heads and feed-forward units.

These are MHA models with untied embeddings, not SmolLM2 or Qwen checkpoints.
The repository does not load those models' pretrained weights or implement their
GQA/tokenizer conventions.

## Data and evaluation

The measured workloads use prepared TinyStories training and validation token
arrays with the existing 10K BPE vocabulary and merges. Input paths are listed
in the configurations; the [setup instructions](../README.md#data-and-experiments)
describe the required files. They are not included in Git, and their preparation
pipeline is not bundled here.

Dataset and tokenizer file hashes identify the inputs used in each run. They do
not prove document-level train/validation separation or establish that a given
tokenizer produced the token array. Comparisons use the same files and fixed
sampled validation windows; this is not a separate downstream evaluation suite.

The [single-GPU screen](scaling-study.md) holds tokens per update at 8,192 by
using eight sequences at 1K context and two at 4K. The main
[distributed study](distributed-study.md) uses the 324M model with a fixed global
batch of 32 at 1K; its selected 4K comparisons use global batch eight. Contexts
remain separate workloads because their conditioning and validation windows
differ.

## What this workload supports

The completed experiments compare attention, recomputation, microbatching,
gradient communication and state/model partitioning under controlled inputs.
They measure the memory and throughput tradeoffs of these implementations on the
recorded GPUs. The model family and data stay fixed within each comparison.

Short screens do not establish convergence, scaling laws or downstream model
quality. Three longer 100M development runs selected a validation target of 2.40;
the [quality protocol](quality-study.md) preserves the target rule and deferred
confirmation. A separate corpus or pretrained architecture would require its own
reference checks and evaluation protocol.
