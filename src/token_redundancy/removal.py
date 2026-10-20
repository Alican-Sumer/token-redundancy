"""Person B: delete one token after layer l and finish the forward pass (plan Section 5).

Manual pass, verified to match model(**inputs).logits exactly on transformers 5.20.2:
    h = model.bert.embeddings(input_ids=..., token_type_ids=...)
    mask = (1 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
    for layer in model.bert.encoder.layer:
        h = layer(h, mask)
    logits = model.classifier(model.bert.pooler(h))
Cache h after layer l, and drop token i from both h and attention_mask before layer l+1.
"""
