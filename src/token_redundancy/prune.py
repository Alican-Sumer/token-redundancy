"""Person A: build M25 / M50 from M0 (plan Section 4).

1. Head importance on the train split (Michel et al. 2019): wrap the forward pass in
   heads.head_gates(model, gates) with gates of ones and requires_grad=True, accumulate
   |dL/dgate| over batches, then normalize per layer.
2. Remove the lowest-scoring heads globally, keeping at least one head per layer
   (heads.prune_heads).
3. Recovery fine-tune 1-2 epochs, save with save_pretrained, reload with heads.load_model.
"""
