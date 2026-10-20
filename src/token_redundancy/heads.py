"""Attention-head gating and physical pruning for BERT.

transformers 5 removed `model.prune_heads()` and the `head_mask` argument, so both are
reimplemented here. Head ids always refer to the ORIGINAL numbering (0..11 in each layer);
`config.remaining_heads` records which original heads each layer still has.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.pytorch_utils import prune_linear_layer


def remaining_heads(model) -> dict[int, list[int]]:
    saved = getattr(model.config, "remaining_heads", None)
    if saved is not None:
        return {int(l): list(h) for l, h in saved.items()}
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    return {l: list(range(n_heads)) for l in range(n_layers)}


def prune_heads(model, heads_to_prune: dict[int, list[int]]) -> None:
    """Physically remove heads in place. `heads_to_prune` maps layer index -> original head ids."""
    remaining = remaining_heads(model)
    for layer_idx, drop in heads_to_prune.items():
        drop = set(drop)
        current = remaining[layer_idx]
        keep = [pos for pos, head in enumerate(current) if head not in drop]
        if not keep:
            raise ValueError(f"Cannot remove every head in layer {layer_idx}")
        if len(keep) == len(current):
            continue

        attn = model.bert.encoder.layer[layer_idx].attention
        size = attn.self.attention_head_size
        index = torch.cat([torch.arange(p * size, (p + 1) * size) for p in keep])
        attn.self.query = prune_linear_layer(attn.self.query, index)
        attn.self.key = prune_linear_layer(attn.self.key, index)
        attn.self.value = prune_linear_layer(attn.self.value, index)
        attn.output.dense = prune_linear_layer(attn.output.dense, index, dim=1)
        attn.self.num_attention_heads = len(keep)
        attn.self.all_head_size = len(keep) * size
        remaining[layer_idx] = [current[p] for p in keep]

    model.config.remaining_heads = {str(l): h for l, h in remaining.items()}


@contextmanager
def head_gates(model, gates: list[torch.Tensor]):
    """Multiply each head's output by a gate while active.

    `gates[l]` has one entry per head currently in layer l. Pass gates with
    requires_grad=True to get Michel et al. (2019) importance as |dL/dgate|.
    """
    hooks = []
    for layer, gate in zip(model.bert.encoder.layer, gates):
        size = layer.attention.self.attention_head_size

        def hook(module, args, gate=gate, size=size):
            context, residual = args
            b, t, _ = context.shape
            gated = (context.view(b, t, -1, size) * gate.view(1, 1, -1, 1)).view(b, t, -1)
            return gated, residual

        hooks.append(layer.attention.output.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def save_pruned(model, path: str | Path) -> None:
    # Keep current parameter names (v5 otherwise renames LayerNorm weight/bias to gamma/beta on save),
    # because load_model loads the pruned weights directly rather than through from_pretrained.
    model.save_pretrained(path, save_original_format=False)


def load_model(path: str | Path, **kwargs):
    """Load a fine-tuned checkpoint, pruned or not, with eager attention (needed for attention weights)."""
    config = AutoConfig.from_pretrained(path)
    if getattr(config, "remaining_heads", None) is None:
        return AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="eager", **kwargs)

    # from_pretrained would build full-size layers, so build, prune, then load the weights.
    remaining = {int(l): set(h) for l, h in config.remaining_heads.items()}
    del config.remaining_heads
    model = AutoModelForSequenceClassification.from_config(config, attn_implementation="eager")
    all_heads = range(config.num_attention_heads)
    prune_heads(model, {l: [h for h in all_heads if h not in keep] for l, keep in remaining.items()})
    model.load_state_dict(load_file(Path(path) / "model.safetensors"))
    return model


def heads_summary(model) -> str:
    return json.dumps({l: len(h) for l, h in remaining_heads(model).items()})
