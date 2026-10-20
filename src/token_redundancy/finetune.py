"""Person A: fine-tune M0 for each seed (plan Sections 3-4).

lr 2e-5, batch 32, 3-5 epochs, AdamW, linear warmup 10%, early stopping on the val split.
Save to config.CHECKPOINT_DIR / f"M0_seed{seed}".
"""
