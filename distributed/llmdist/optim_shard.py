"""Parameter-granularity AdamW state sharding for replicated data parallelism.

Parameters and gradients remain replicated. The caller must synchronize gradients
and clip the replicated gradients before step(). Each parameter has one optimizer
owner; only that rank holds its moments, and updated parameters are broadcast.
This is a simple synchronous stage-1 experiment, not FSDP or a fused ZeRO kernel.
"""

import hashlib
import json
from collections.abc import Iterable

import torch
import torch.distributed as dist


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class ShardedAdamW:
    """AdamW wrapper that assigns complete parameters to deterministic owners.

    ``param_groups`` contains the local optimizer groups (including empty groups)
    so changing their learning rates has the ordinary AdamW meaning. ``zero_grad``
    clears every replicated parameter, including parameters owned by other ranks.
    State dictionaries are local shards and cannot independently resume a model.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, betas=(0.9, 0.95),
                 weight_decay: float = 0.1, process_group=None, **kwargs):
        supplied = list(params)
        if not supplied:
            raise ValueError("ShardedAdamW requires at least one parameter")
        if isinstance(supplied[0], dict):
            if not all(isinstance(group, dict) for group in supplied):
                raise TypeError("Do not mix parameters and parameter groups")
            groups = [dict(group, params=list(group["params"])) for group in supplied]
        else:
            groups = [{"params": supplied}]
        self._parameters = [parameter for group in groups for parameter in group["params"]]
        if not self._parameters:
            raise ValueError("ShardedAdamW requires at least one parameter")
        if any(not isinstance(parameter, torch.Tensor) for parameter in self._parameters):
            raise TypeError("Optimizer parameters must be tensors")
        if len({id(parameter) for parameter in self._parameters}) != len(self._parameters):
            raise ValueError("Duplicate parameters cannot have separate optimizer owners")
        if process_group is not None and not dist.is_initialized():
            raise ValueError("A process group requires initialized distributed training")
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self._owner_numel = [0] * self.world_size
        self._owners = [0] * len(self._parameters)
        # Large parameters first, with stable parameter-index/rank tie breaking.
        for index in sorted(range(len(self._parameters)),
                            key=lambda i: (-self._parameters[i].numel(), i)):
            owner = min(range(self.world_size), key=lambda rank: (self._owner_numel[rank], rank))
            self._owners[index] = owner
            self._owner_numel[owner] += self._parameters[index].numel()
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, **kwargs}
        descriptor = {
            "parameters": [{"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                            "requires_grad": parameter.requires_grad}
                           for parameter in self._parameters],
            "group_sizes": [len(group["params"]) for group in groups],
            "group_options": [{key: value for key, value in group.items() if key != "params"}
                              for group in groups],
            "defaults": defaults, "owners": self._owners,
        }
        self._layout_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=str).encode()).hexdigest()
        if self.world_size > 1:
            hashes = [None] * self.world_size
            dist.all_gather_object(hashes, self._layout_hash, group=process_group)
            if len(set(hashes)) != 1:
                raise ValueError("Optimizer parameter layout or settings differ across ranks")
        index = 0
        local_groups = []
        for group in groups:
            local = dict(group, params=[])
            for parameter in group["params"]:
                if self._owners[index] == self.rank:
                    local["params"].append(parameter)
                index += 1
            local_groups.append(local)
        # AdamW accepts an empty params list inside a nonempty list of groups.
        # Keeping these groups lets empty-owner ranks accept identical LR changes.
        self._optimizer = torch.optim.AdamW(local_groups, **defaults)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def state(self):
        return self._optimizer.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                # DDP can expose gradients as views into communication buckets;
                # detach_ is illegal on those views. Ordinary gradients already
                # do not require grad, so preserve their bucket-storage alias.
                if parameter.grad.grad_fn is not None:
                    parameter.grad = parameter.grad.detach()
                with torch.no_grad():
                    parameter.grad.zero_()

    def step(self, closure=None):
        loss = self._optimizer.step(closure)
        if self.world_size > 1:
            with torch.no_grad():
                for parameter, owner in zip(self._parameters, self._owners, strict=True):
                    source = owner if self.process_group is None else dist.get_global_rank(self.process_group, owner)
                    dist.broadcast(parameter, src=source, group=self.process_group)
        return loss

    def local_state_bytes(self) -> int:
        """Tensor bytes in this rank's AdamW state, including step counters."""
        return _tensor_bytes(self._optimizer.state)

    def ownership_metadata(self) -> dict:
        return {"algorithm": "greedy whole-parameter optimizer-state sharding",
                "rank": self.rank, "world_size": self.world_size,
                "owners": list(self._owners), "parameter_numel_per_rank": list(self._owner_numel),
                "owned_parameter_indices": [i for i, owner in enumerate(self._owners) if owner == self.rank],
                "layout_sha256": self._layout_hash, "local_optimizer_state_bytes": self.local_state_bytes(),
                "parameter_scope": "replicated", "gradient_scope": "replicated; caller synchronizes",
                "state_scope": "AdamW moments and step counters exist only on the owning rank",
                "communication": "synchronous parameter broadcasts after the optimizer step"}

    def state_dict(self) -> dict:
        return {"format_version": 1, "sharding": self.ownership_metadata(),
                "optimizer": self._optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        metadata = state_dict["sharding"]
        if (state_dict.get("format_version") != 1 or metadata["rank"] != self.rank
                or metadata["world_size"] != self.world_size
                or metadata["layout_sha256"] != self._layout_hash):
            raise ValueError("Optimizer shard does not match this rank, world size or parameter layout")
        self._optimizer.load_state_dict(state_dict["optimizer"])
