"""
Model profiling utilities: parameter counts, memory footprint, and inference latency.
"""
import time
import torch
import torch.nn as nn
import numpy as np

def count_parameters(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    size_mb = (total_params * 4) / (1024 * 1024) # 4 bytes per float32 parameter

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': round(size_mb, 3)
    }

def benchmark_inference_latency(model: nn.Module, device: torch.device, seq_len: int = 50, batch_size: int = 64, num_runs: int = 50) -> dict:
    model.eval()
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    lengths = torch.tensor([seq_len] * batch_size, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input, lengths)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input, lengths)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # in milliseconds

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    ms_per_sample = mean_ms / batch_size

    return {
        'batch_latency_mean_ms': round(mean_ms, 2),
        'batch_latency_std_ms': round(std_ms, 2),
        'per_sample_latency_ms': round(ms_per_sample, 3)
    }
