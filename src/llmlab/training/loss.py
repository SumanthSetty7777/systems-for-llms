import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits

    log_sum_exp = torch.log(
        torch.sum(torch.exp(shifted_logits), dim=-1)
    )

    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits
    return torch.mean(losses)


def perplexity(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(cross_entropy(inputs, targets))
