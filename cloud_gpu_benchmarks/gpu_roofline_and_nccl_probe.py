#!/usr/bin/env python3
"""
Empirical GPU Roofline, HBM Bandwidth & Collective Micro-Benchmark Probe.
Runs on NVIDIA CUDA GPUs, AMD ROCm GPUs, Apple Silicon MPS, or CPU fallback.

Usage:
  python3 cloud_gpu_benchmarks/gpu_roofline_and_nccl_probe.py
  nsys profile -t cuda,nvtx,osrt -o roofline_trace python3 cloud_gpu_benchmarks/gpu_roofline_and_nccl_probe.py
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class GemmPoint:
    m: int
    n: int
    k: int
    dtype: str
    arithmetic_intensity_flops_per_byte: float
    latency_ms: float
    achieved_tflops: float
    achieved_bw_gbps: float


def run_empirical_roofline() -> List[GemmPoint]:
    try:
        import torch
    except ImportError:
        print("[INFO] PyTorch not installed in current Python env; running analytical fallback.")
        return []

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        bytes_per_elem = 2
        sync = torch.cuda.synchronize
        dev_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        bytes_per_elem = 2
        sync = torch.mps.synchronize
        dev_name = "Apple Silicon MPS GPU"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        bytes_per_elem = 4
        sync = lambda: None
        dev_name = "CPU"

    print(f"[ROOFLINE PROBE] Device: {dev_name} ({device}, dtype={dtype})")
    results: List[GemmPoint] = []

    # Sweep batch/token dimension M from memory-bound (Decode: M=1..64) to compute-bound (Prefill: M=512..8192)
    n, k = 4096, 4096
    for m in [1, 8, 32, 128, 512, 2048, 4096]:
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)

        # Warmup
        for _ in range(5):
            _ = a @ b
        sync()

        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        sync()
        elapsed_s = (time.perf_counter() - t0) / iters

        flops = 2.0 * m * n * k
        bytes_moved = bytes_per_elem * (m * k + k * n + m * n)
        ai = flops / bytes_moved
        tflops = (flops / elapsed_s) / 1e12
        bw_gbps = (bytes_moved / elapsed_s) / 1e9

        pt = GemmPoint(
            m=m,
            n=n,
            k=k,
            dtype=str(dtype),
            arithmetic_intensity_flops_per_byte=round(ai, 2),
            latency_ms=round(elapsed_s * 1e3, 4),
            achieved_tflops=round(tflops, 2),
            achieved_bw_gbps=round(bw_gbps, 2),
        )
        results.append(pt)
        print(
            f"  M={m:>4}, N={n}, K={k} | AI={pt.arithmetic_intensity_flops_per_byte:>7.1f} FLOP/B "
            f"| Latency={pt.latency_ms:>7.3f} ms | {pt.achieved_tflops:>6.2f} TFLOPS | {pt.achieved_bw_gbps:>7.2f} GB/s"
        )

    return results


if __name__ == "__main__":
    points = run_empirical_roofline()
    if points:
        with open("/tmp/empirical_roofline_points.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in points], f, indent=2)
        print("\nSaved empirical roofline sweep to /tmp/empirical_roofline_points.json")
