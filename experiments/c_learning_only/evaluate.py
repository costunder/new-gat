"""Frozen validation and paired C=1 intervention; no test-set selection."""

import hashlib

import torch
from torch.nn import functional as F


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, batches, intervention=False):
    model.eval()
    identity = state_digest(model)
    correct = ones_correct = total = changed = 0
    sums = torch.zeros(5, dtype=torch.float64, device=next(model.parameters()).device)
    layer_error = torch.zeros(model.depth, 2, dtype=torch.float64, device=sums.device)
    for batch in batches:
        messages = {}

        def before(index, op, state, c, message, *unused, messages=messages):
            messages[index] = message.detach().cpu()

        def after(index, op, state, c, message, *unused, messages=messages):
            old = messages.pop(index).to(message.device)
            layer_error[index, 0] += (message.float() - old.float()).double().square().sum()
            layer_error[index, 1] += old.double().square().sum()

        logits = model(batch.graph, observer=before if intervention else None)
        ids = batch.selected_indices
        target = batch.graph.y[ids].reshape(-1)
        prediction = logits[ids]
        correct += int((prediction.argmax(-1) == target).sum())
        total += target.numel()
        sums[0] += F.cross_entropy(prediction.float(), target, reduction="sum")
        if intervention:
            with model.c_ones():
                replaced = model(batch.graph, observer=after)[ids]
            ones_correct += int((replaced.argmax(-1) == target).sum())
            changed += int((prediction.argmax(-1) != replaced.argmax(-1)).sum())
            sums[1] += F.cross_entropy(replaced.float(), target, reduction="sum")
            sums[2] += (prediction.float() - replaced.float()).double().square().sum()
            sums[3] += prediction.double().square().sum()
            sums[4] += prediction.numel()
    if total == 0:
        raise ValueError("evaluation received no supervised nodes")
    if identity != state_digest(model):
        raise RuntimeError("frozen evaluation mutated model parameters")
    result = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "cross_entropy": float(sums[0] / total),
        "parameter_sha256": identity,
    }
    if intervention:
        result["c_ones_intervention"] = {
            "correct": ones_correct,
            "accuracy": ones_correct / total,
            "cross_entropy": float(sums[1] / total),
            "changed_predictions": changed,
            "logit_change_rms": float((sums[2] / sums[4]).sqrt()),
            "logit_change_relative_l2": float((sums[2] / sums[3].clamp_min(1e-30)).sqrt()),
            "layer_message_change_relative_l2": (
                layer_error[:, 0] / layer_error[:, 1].clamp_min(1e-30)
            )
            .sqrt()
            .cpu()
            .tolist(),
            "sampling_correction": "identical; receiver normalization recomputed",
        }
    return result
