"""Person B: per-token signals at layer l (plan Section 6).

attention received (mean over remaining heads), 1 - cos(h_{l-1}, h_l), ||h_l||_2, random.
Needs attn_implementation="eager" so output_attentions returns weights.
"""
