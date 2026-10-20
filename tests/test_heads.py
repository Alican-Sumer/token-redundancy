import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from token_redundancy.config import MODEL_NAME
from token_redundancy.heads import head_gates, load_model, prune_heads, remaining_heads, save_pruned

PRUNE = {0: [0, 3], 5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11: [7]}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def inputs(tokenizer):
    return tokenizer(["a cat sat on the mat", "hi"], ["the cat sat", "hello there friend"],
                     padding=True, return_tensors="pt")


def fresh_model():
    torch.manual_seed(0)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation="eager").eval()


def zero_gates(model, heads_to_zero):
    return [torch.tensor([0.0 if h in heads_to_zero.get(l, []) else 1.0 for h in heads])
            for l, heads in remaining_heads(model).items()]


@torch.no_grad()
def test_pruning_matches_zeroed_heads(inputs):
    model = fresh_model()
    with head_gates(model, zero_gates(model, PRUNE)):
        gated = model(**inputs).logits
    prune_heads(model, PRUNE)
    assert model.bert.encoder.layer[5].attention.self.num_attention_heads == 1
    torch.testing.assert_close(model(**inputs).logits, gated, atol=1e-5, rtol=0)


@torch.no_grad()
def test_pruning_twice_uses_original_head_ids(inputs):
    once, twice = fresh_model(), fresh_model()
    prune_heads(once, {0: [0, 3]})
    prune_heads(once, {0: [5]})
    prune_heads(twice, {0: [0, 3, 5]})
    assert remaining_heads(once)[0] == remaining_heads(twice)[0] == [1, 2, 4, 6, 7, 8, 9, 10, 11]
    torch.testing.assert_close(once(**inputs).logits, twice(**inputs).logits)


def test_cannot_empty_a_layer():
    with pytest.raises(ValueError):
        prune_heads(fresh_model(), {2: list(range(12))})


@torch.no_grad()
def test_pruned_save_load_roundtrip(tmp_path, inputs):
    model = fresh_model()
    prune_heads(model, PRUNE)
    save_pruned(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    assert remaining_heads(loaded) == remaining_heads(model)
    assert torch.equal(loaded(**inputs).logits, model(**inputs).logits)


def test_gates_give_head_importance_gradients(inputs):
    model = fresh_model()
    gates = [torch.ones(12, requires_grad=True) for _ in range(12)]
    with head_gates(model, gates):
        model(**inputs, labels=torch.tensor([1, 0])).loss.backward()
    assert all(g.grad is not None and g.grad.abs().sum() > 0 for g in gates)


@torch.no_grad()
@pytest.mark.parametrize("pruned", [False, True])
def test_manual_layer_loop_reproduces_forward(inputs, pruned):
    """The removal code (plan Section 5) relies on this exact equality."""
    model = fresh_model()
    if pruned:
        prune_heads(model, PRUNE)
    h = model.bert.embeddings(input_ids=inputs.input_ids, token_type_ids=inputs.token_type_ids)
    mask = (1.0 - inputs.attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
    for layer in model.bert.encoder.layer:
        h = layer(h, mask)
    assert torch.equal(model.classifier(model.bert.pooler(h)), model(**inputs).logits)
