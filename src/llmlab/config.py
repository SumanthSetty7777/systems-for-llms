import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    context_length: int = 8
    d_model: int = 16
    num_layers: int = 1
    num_heads: int = 2
    d_ff: int = 32
    rope_theta: float = 10000.0
    attention_backend: str = "eager"
    activation_checkpointing: str = "none"


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic"
    train_path: str | None = None
    valid_path: str | None = None
    tokenizer_files: tuple[str, ...] = ()
    synthetic_tokens: int = 4096


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    seed: int = 1337
    device: str = "cpu"
    num_threads: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 1
    precision: str = "fp32"
    max_steps: int = 20
    learning_rate: float = 0.003
    min_learning_rate: float = 0.0003
    warmup_steps: int = 2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 5
    eval_batches: int = 4
    eval_batch_size: int = 2
    checkpoint_interval: int = 10

    def validate(self) -> None:
        for name in ("num_threads", "batch_size", "grad_accum_steps", "max_steps", "eval_interval",
                     "eval_batches", "eval_batch_size", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            if type(getattr(self.model, name)) is not int or getattr(self.model, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1)")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.model.rope_theta) or self.model.rope_theta <= 0:
            raise ValueError("model.rope_theta must be finite and positive")
        if not 0 <= self.min_learning_rate <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("require 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be nonnegative and grad_clip positive")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps, or auto")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.model.attention_backend not in {"eager", "sdpa"}:
            raise ValueError("model.attention_backend must be eager or sdpa")
        if self.model.activation_checkpointing not in {"none", "block"}:
            raise ValueError("model.activation_checkpointing must be none or block")
        if self.data.kind not in {"synthetic", "npy"}:
            raise ValueError("data.kind must be synthetic or npy")
        if self.data.kind == "npy" and not (self.data.train_path and self.data.valid_path):
            raise ValueError("npy data requires train_path and valid_path")
        if self.data.kind == "synthetic" and (type(self.data.synthetic_tokens) is not int
                                              or self.data.synthetic_tokens <= self.model.context_length):
            raise ValueError("synthetic_tokens must be an integer exceeding context_length")

    def to_dict(self) -> dict:
        # Paths accepted by the Python API become portable checkpoint primitives.
        return json.loads(json.dumps(asdict(self), default=str))


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    model = ModelConfig(**raw.pop("model"))
    data = raw.pop("data")
    for key in ("train_path", "valid_path"):
        if data.get(key):
            data[key] = str((path.parent / data[key]).resolve())
    data["tokenizer_files"] = tuple(
        str((path.parent / p).resolve()) for p in data.get("tokenizer_files", [])
    )
    config = TrainConfig(model=model, data=DataConfig(**data), **raw)
    config.validate()
    return config
