"""Person B: leave-one-out |dp| for each (example, token, drop layer) and model (plan Section 5).

Candidates: every token except [CLS] and [SEP]. Write results to config.RESULTS_DIR in chunks
so a Colab disconnect only loses the current chunk, and resume from the last chunk written.
"""
