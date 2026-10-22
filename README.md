# `rackscale-perf`: AI Infrastructure Rack Topology, Collective Communication & Roofline Performance Suite

`rackscale-perf` is a systems & performance engineering toolkit written in **Rust** and **Python** for modeling, scheduling, profiling, and diagnosing **AI infrastructure racks** (**NVIDIA GB200 NVL72**, **NVIDIA HGX H100/H200**, and **AMD Instinct MI300X**).

It bridges the gap between **physical rack architecture** (NVSwitch copper backplanes, NUMA/PCIe trees, and rail-optimized InfiniBand/RoCEv2 fabrics), **collective communication** (`AllReduce`, `ReduceScatter`, `AllGather`, `AllToAll` over Ring, DoubleBinaryTree, and NVLS SHARP), **GPU Roofline / MFU analysis**, and **DCGM/RDMA straggler telemetry isolation**.

---

## Architecture & Workspace Crates

| Crate / Module | Responsibility | Key Concepts Demonstrated |
|---|---|---|
| `crates/rack-topology` | Models multi-rack GPU clusters (`NvidiaGb200Nvl72`, `NvidiaHgxH100Rack64`, `AmdMi300xRack64`) and schedules 4D parallelism (`TP`, `EP`, `PP`, `DP/FSDP`) onto physical trays and NIC rails. | Scale-up vs. scale-out bandwidth hierarchy, NVSwitch domain packing, rail-aligned RDMA placement, link degradation fault injection, `Criterion` sub-microsecond scheduling benchmarks. |
| `crates/collective-roofline` | Simulates NCCL/RCCL collectives using the $\alpha$-$\beta$ cost model (`algbw` vs. `busbw`) and computes end-to-end training step Roofline & **Model FLOPs Utilization (MFU)**. | `Ring` vs. `DoubleBinaryTree` vs. `NvlsSharp`, Arithmetic Intensity ($\text{FLOPs/Byte}$), HBM-bound vs. Tensor-Core-bound vs. Exposed-Communication-bound regimes. |
| `crates/rack-telemetry` | Ingests per-GPU DCGM & RDMA hardware counters to isolate thermal stragglers, NVLink CRC degradation, and RoCEv2 PFC pause storms, exporting Prometheus metrics. | `DCGM_FI_DEV_*` metrics, Median Absolute Deviation (MAD) straggler detection, rack power/cooling transient monitoring, automated remediation actions. |
| `crates/rackperf-cli` | End-to-end CLI benchmark runner comparing placement policies, collective algorithms, MFU impact, and live fault diagnosis. | Reproducible rack-scale performance reports. |
| `cloud_gpu_benchmarks/` | Empirical PyTorch/CUDA/ROCm/MPS GEMM Roofline & HBM bandwidth sweep (`nsys` / `ncu` ready). | Empirical validation across decode (memory-bound) and prefill/training (compute-bound) batch sizes. |

---

## Quickstart

```bash
# Run all unit tests across topology, collective-roofline, and telemetry crates
cargo test

# Run the end-to-end rack performance analysis CLI
cargo run --release --bin rackperf

# Run Criterion microbenchmarks for 288-GPU GB200 NVL72 topology-aware placement
cargo bench -p rack-topology

# Run empirical GPU Roofline sweep (CUDA / ROCm / Apple MPS)
python3 cloud_gpu_benchmarks/gpu_roofline_and_nccl_probe.py
```

---

## Sample Output (`cargo run --bin rackperf`)

```text
[1] 4D Parallelism Placement Benchmark (128x H100 across 2 HGX Racks, TP=8, PP=2, DP=8)
  - Naive Strided Placement        : TP Bottleneck BW =   47.5 GB/s | TP Domain Spills = 16
  - Topology-Aware Rack Placement  : TP Bottleneck BW =  450.0 GB/s | TP Domain Spills = 0

[2] Collective Communication Simulation (72-GPU NVL72 Domain, 512 MB AllReduce)
  - Ring               : Duration =  1474.0 us | AlgBW =  364.2 GB/s | BusBW =   718.3 GB/s | Latency-Bound = 11.3%
  - DoubleBinaryTree   : Duration =  1456.7 us | AlgBW =  368.5 GB/s | BusBW =   726.8 GB/s | Latency-Bound =  1.5%
  - NvlsSharp          : Duration =   735.6 us | AlgBW =  729.9 GB/s | BusBW =  1439.4 GB/s | Latency-Bound =  0.8%

[3] Roofline & Model FLOPs Utilization (MFU) Analysis
  * Llama-3-70B (Naive Placement on 128x H100)              
    Ridge Point: 295.2 FLOP/B | Workload AI: 8176.0 FLOP/B | Regime: InterconnectCommunicationBound
    Compute: 19327.3 ms | Exposed Comm: 42220.4 ms | Step Time: 61547.7 ms | Achieved:  223.6 TFLOPS/GPU (MFU:  22.6%)

  * Llama-3-70B (Topology-Aware on 128x H100)               
    Ridge Point: 295.2 FLOP/B | Workload AI: 8176.0 FLOP/B | Regime: TensorCoreComputeBound
    Compute: 19327.3 ms | Exposed Comm: 2821.6 ms | Step Time: 22148.9 ms | Achieved:  621.4 TFLOPS/GPU (MFU:  62.8%)

  * Frontier-MoE-140B-Active (2x GB200 NVL72 Racks, 144 GPUs)
    Ridge Point: 281.2 FLOP/B | Workload AI: 12252.1 FLOP/B | Regime: TensorCoreComputeBound
    Compute: 16990.8 ms | Exposed Comm: 4602.9 ms | Step Time: 21593.7 ms | Achieved: 1274.7 TFLOPS/GPU (MFU:  56.7%)
```
