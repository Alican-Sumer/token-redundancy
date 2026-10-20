"""Person A: accuracy/F1, FLOPs, latency, parameter count (plan Section 7, Table 1).

Latency: batch 32, length 128, warmup, mean over 100 runs, torch.cuda.synchronize() around
timers. Record torch.cuda.get_device_name() and take all Table 1 timings in one Colab session.
"""
