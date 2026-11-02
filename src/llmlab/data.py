import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(config: TrainConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    if config.data.kind == "synthetic":
        rng = np.random.default_rng(config.seed)
        arrays = [rng.integers(0, config.model.vocab_size,
                              size=config.data.synthetic_tokens, dtype=np.int64)
                  for _ in range(2)]
        identities = [{"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                       "tokens": len(a), "dtype": str(a.dtype)} for a in arrays]
    else:
        paths = [config.data.train_path, config.data.valid_path]
        arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
        identities = [{"sha256": file_sha256(p), "tokens": len(a),
                       "dtype": str(a.dtype)} for p, a in zip(paths, arrays)]
    for array in arrays:
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("token data must be a one-dimensional integer array")
        if len(array) <= config.model.context_length:
            raise ValueError("token data must exceed the context length")
        if array.min() < 0 or array.max() >= config.model.vocab_size:
            raise ValueError("token ID outside model vocabulary")
    tokenizer = [{"name": Path(p).name, "sha256": file_sha256(p)}
                 for p in config.data.tokenizer_files]
    identity = {"kind": config.data.kind, "train": identities[0],
                "valid": identities[1], "tokenizer_files": tokenizer}
    return arrays[0], arrays[1], identity


def batch_at(data: np.ndarray, starts: list[int], context_length: int,
             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = np.stack([data[i:i + context_length + 1] for i in starts]).astype(np.int64)
    tokens = torch.from_numpy(tokens).to(device)
    return tokens[:, :-1], tokens[:, 1:]


def sample_batch(data: np.ndarray, config: TrainConfig, generator: torch.Generator,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - config.model.context_length,
                           (config.batch_size,), generator=generator).tolist()
    return batch_at(data, starts, config.model.context_length, device)
