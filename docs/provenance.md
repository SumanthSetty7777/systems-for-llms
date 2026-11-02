# Model provenance and compatibility

The initial model and loss implementation was copied on October 30, 2026 from
the local `../llm-systems/src/llmlab` working tree into this repository:

- `model/layers.py`: linear layers, embeddings, RMSNorm, and SwiGLU.
- `model/attention.py`: rotary position embeddings and causal multi-head attention.
- `model/transformer.py`: transformer blocks and the language model.
- `training/loss.py`: cross-entropy and perplexity.

These source modules were untracked in the sibling repository at integration,
so its Git HEAD alone does not identify their contents. Run metadata records
content hashes of the integrated source. The original repositories, datasets
and checkpoints were left unmodified.

The model retains multi-head attention (MHA), pre-norm RMSNorm, RoPE, SwiGLU,
and separate, untied token-embedding and output-head weights. Layer and loss
arithmetic, parameter initialization, module construction order, and parameter
names were preserved. Default `attention_backend="eager"` uses the original
explicit causal attention calculation. A same-seed CPU comparison against the
source model confirmed identical initial parameters and eager outputs.

Initial changes added dimension/configuration and sequence-length checks,
plus an optional `attention_backend="sdpa"` path through PyTorch's
`scaled_dot_product_attention` with `is_causal=True` and `dropout_p=0.0`.
Both backends retain the same parameters and `state_dict` keys. CPU tests check
output and gradient agreement within tolerance, causal-prefix invariance,
cross-entropy against PyTorch, and the original key layout. They do not establish
GPU speed or numerical behavior.

The October 31 extension adds optional block activation checkpointing without
changing parameter names or initialization. For reduced-precision eager attention,
softmax normalization is explicitly computed in FP32. The runner adds accumulation
and CUDA BF16 autocast with FP32 model parameters and loss calculation. CPU tests
cover checkpointed gradients and accumulated updates. The subsequent Modal
studies exercised these paths on CUDA; the distributed confirmation passed
335 GPU tests with zero skips. See the
[confirmation evidence](../results/2026-11-01-distributed-confirmation/README.md).

The training runner, configuration/data handling, metrics, and resume protocol
are new in this repository. The runner uses `torch.optim.AdamW`; it does not
reuse the sibling project's optimizer implementation. Its checkpoint format
stores model and optimizer state alongside sampling/RNG state and
configuration, source, and data identities. A compatible legacy model
`state_dict` can populate matching model dimensions, but an old training
checkpoint cannot be passed directly to this runner's `--resume`. Continuing
an old training trajectory would require an explicit conversion and validation.

This is the local custom MHA model. It does not yet implement grouped-query
attention or pretrained SmolLM2/Qwen configuration, tokenizer, or weight-loading
compatibility. Matching a parameter count or some architectural components does
not make its checkpoints interchangeable with those model families.

## Input files

The TinyStories and scale configurations expect the following prepared files in
a sibling `assignment1-basics/` directory, or equivalent paths in a copied config:

- `data/TinyStories-train-tokens.npy` and `data/TinyStories-valid-tokens.npy`:
  one-dimensional integer token arrays using the 10K-token vocabulary.
- `outputs/tokenizers/tinystories_10k_mp/vocab.pkl` and `merges.pkl`:
  the corresponding tokenizer artifacts.

Paths resolve relative to the configuration file. These files and their
regeneration pipeline are not bundled here. Metadata and result reports retain
file checksums; matching them is required to reproduce the measured workload.
Checksums establish file identity, not document-level train/evaluation separation.
`configs/smoke.json` uses synthetic data and needs none of these inputs.
