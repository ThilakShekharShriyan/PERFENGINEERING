//! `collective-roofline`: Analytical & topology-aware NCCL/RCCL collective communication
//! simulator (`AllReduce`, `ReduceScatter`, `AllGather`, `AllToAll`) + GPU Roofline &
//! Model FLOPs Utilization (MFU) analyzer.

use rack_topology::{ClusterTopology, ParallelismConfig, PlacementPlan};
use serde::{Deserialize, Serialize};

/// Collective communication primitives in distributed LLM training and inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectiveOp {
    /// Used in Tensor Parallelism and Data Parallel gradient synchronization.
    AllReduce,
    /// First half of ZeRO/FSDP gradient sharding or TP+SP activation reduction.
    ReduceScatter,
    /// Parameter gathering in FSDP/ZeRO-3 and TP+SP forward pass.
    AllGather,
    /// Token-to-expert routing in Mixture-of-Experts (MoE) Expert Parallelism.
    AllToAll,
}

/// NCCL / RCCL collective algorithm choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectiveAlgorithm {
    /// Bandwidth-optimal for large messages; latency scales linearly with `2 * (N - 1)` steps.
    Ring,
    /// Logarithmic latency (`2 * log2(N)` steps); preferred for small/medium messages across nodes.
    DoubleBinaryTree,
    /// NVLink SHARP (NVLS): hardware in-network multicast + reduction inside NVSwitch (Hopper/Blackwell).
    NvlsSharp,
}

/// Output of a collective simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveSimulationResult {
    pub op: CollectiveOp,
    pub algorithm: CollectiveAlgorithm,
    pub num_ranks: usize,
    pub message_size_bytes: usize,
    /// Total execution time in microseconds.
    pub duration_us: f64,
    /// Algorithm bandwidth (`message_size / duration`) in GB/s.
    pub algbw_gbps: f64,
    /// Bus bandwidth (normalized hardware link utilization, matching `nccl-tests` formula) in GB/s.
    pub busbw_gbps: f64,
    /// Fraction of time spent waiting on latency/synchronization vs bandwidth transfer.
    pub latency_bound_fraction: f64,
}

/// Simulates a collective operation using the $\alpha$-$\beta$ cost model combined with
/// physical bottleneck link bandwidth from the rack topology.
pub fn simulate_collective(
    op: CollectiveOp,
    algorithm: CollectiveAlgorithm,
    num_ranks: usize,
    message_size_bytes: usize,
    bottleneck_unidir_bw_gbps: f64,
    hop_latency_us: f64,
) -> CollectiveSimulationResult {
    let n = num_ranks.max(1) as f64;
    let size_gb = (message_size_bytes as f64) / 1e9;

    if num_ranks <= 1 || message_size_bytes == 0 {
        return CollectiveSimulationResult {
            op,
            algorithm,
            num_ranks,
            message_size_bytes,
            duration_us: 0.05,
            algbw_gbps: 0.0,
            busbw_gbps: 0.0,
            latency_bound_fraction: 1.0,
        };
    }

    // Kernel launch + NCCL proxy thread setup overhead in microseconds
    let base_launch_us = 3.5;

    // Compute latency term (alpha) and bandwidth term (beta) by op and algorithm
    let (latency_us, bw_time_us) = match (op, algorithm) {
        (CollectiveOp::AllReduce, CollectiveAlgorithm::Ring) => {
            let steps = 2.0 * (n - 1.0);
            let alpha_us = base_launch_us + steps * hop_latency_us;
            let volume_gb = 2.0 * ((n - 1.0) / n) * size_gb;
            let beta_us = (volume_gb / (bottleneck_unidir_bw_gbps * 0.90)) * 1e6;
            (alpha_us, beta_us)
        }
        (CollectiveOp::AllReduce, CollectiveAlgorithm::DoubleBinaryTree) => {
            let steps = 2.0 * n.log2().ceil();
            let alpha_us = base_launch_us + steps * (hop_latency_us * 1.15);
            let volume_gb = 2.0 * ((n - 1.0) / n) * size_gb;
            // Tree has slightly lower peak bus efficiency on asymmetric fabrics than Ring
            let beta_us = (volume_gb / (bottleneck_unidir_bw_gbps * 0.82)) * 1e6;
            (alpha_us, beta_us)
        }
        (CollectiveOp::AllReduce, CollectiveAlgorithm::NvlsSharp) => {
            // NVLS uses NVSwitch in-switch reduction: O(1) switch hops, ~1.35x effective algbw
            let alpha_us = base_launch_us + 2.0 * hop_latency_us;
            let volume_gb = 1.15 * size_gb;
            let beta_us = (volume_gb / (bottleneck_unidir_bw_gbps * 0.94)) * 1e6;
            (alpha_us, beta_us)
        }
        (CollectiveOp::ReduceScatter | CollectiveOp::AllGather, _) => {
            let steps = match algorithm {
                CollectiveAlgorithm::Ring => n - 1.0,
                CollectiveAlgorithm::DoubleBinaryTree => n.log2().ceil(),
                CollectiveAlgorithm::NvlsSharp => 1.5,
            };
            let alpha_us = base_launch_us + steps * hop_latency_us;
            let volume_gb = ((n - 1.0) / n) * size_gb;
            let eff = if algorithm == CollectiveAlgorithm::NvlsSharp {
                0.94
            } else {
                0.90
            };
            let beta_us = (volume_gb / (bottleneck_unidir_bw_gbps * eff)) * 1e6;
            (alpha_us, beta_us)
        }
        (CollectiveOp::AllToAll, _) => {
            // MoE AllToAll sends (N - 1)/N of tokens across bisection links simultaneously;
            // susceptible to dynamic routing skew and queue pair contention.
            let alpha_us = base_launch_us + (n - 1.0).min(16.0) * hop_latency_us * 0.8;
            let volume_gb = ((n - 1.0) / n) * size_gb;
            let routing_skew_efficiency = 0.78;
            let beta_us =
                (volume_gb / (bottleneck_unidir_bw_gbps * routing_skew_efficiency)) * 1e6;
            (alpha_us, beta_us)
        }
    };

    let duration_us = latency_us + bw_time_us;
    let duration_s = duration_us / 1e6;
    let algbw_gbps = size_gb / duration_s;

    // Standard `nccl-tests` BusBW correction factor relative to AlgBW
    let busbw_factor = match op {
        CollectiveOp::AllReduce => 2.0 * (n - 1.0) / n,
        CollectiveOp::ReduceScatter | CollectiveOp::AllGather | CollectiveOp::AllToAll => {
            (n - 1.0) / n
        }
    };
    let busbw_gbps = algbw_gbps * busbw_factor;

    CollectiveSimulationResult {
        op,
        algorithm,
        num_ranks,
        message_size_bytes,
        duration_us,
        algbw_gbps,
        busbw_gbps,
        latency_bound_fraction: latency_us / duration_us,
    }
}

/// Classification of an operator or workload stage on the hardware Roofline chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckRegime {
    /// Arithmetic Intensity is below the machine ridge point ($\text{FLOPs/Byte} < \text{Peak FLOPs} / \text{HBM BW}$).
    HbmBandwidthBound,
    /// Arithmetic Intensity exceeds the ridge point, saturating Tensor Cores.
    TensorCoreComputeBound,
    /// Exposed collective communication time exceeds compute/memory time.
    InterconnectCommunicationBound,
}

/// Roofline & end-to-end training/inference step analysis for an LLM workload on a given placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadRooflineReport {
    pub workload_name: String,
    /// Ridge point of the GPU in FLOPs/Byte ($\text{Peak TFLOPS} \times 1000 / \text{HBM GB/s}$).
    pub machine_ridge_point_flops_per_byte: f64,
    /// Operational arithmetic intensity of the main compute phase in FLOPs/Byte.
    pub workload_arithmetic_intensity: f64,
    pub regime: BottleneckRegime,
    /// Compute + HBM time per step in milliseconds.
    pub compute_time_ms: f64,
    /// Exposed collective communication time per step in milliseconds.
    pub exposed_comm_time_ms: f64,
    /// Total step time in milliseconds.
    pub step_time_ms: f64,
    /// Achieved TFLOPS per GPU.
    pub achieved_tflops_per_gpu: f64,
    /// Achieved Model FLOPs Utilization (MFU) in `[0.0, 1.0]`.
    pub mfu: f64,
}

/// Evaluates the Roofline model and achieved MFU for a distributed LLM training step
/// (e.g., Llama-3-70B or DeepSeek-style MoE) under a specific `PlacementPlan`.
pub fn evaluate_training_step_roofline(
    workload_name: &str,
    cluster: &ClusterTopology,
    cfg: &ParallelismConfig,
    placement: &PlacementPlan,
    active_params_billions: f64,
    global_tokens_per_step: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> WorkloadRooflineReport {
    let world_size = cfg.world_size() as f64;
    let ridge_point =
        (cluster.peak_bf16_tflops_per_gpu * 1000.0) / cluster.hbm_bw_gbps_per_gpu;

    // 6 * P * T total training FLOPs (forward + backward + activation recomputation factor)
    let total_flops = 6.0 * (active_params_billions * 1e9) * (global_tokens_per_step as f64);
    let flops_per_gpu = total_flops / world_size;

    // Effective tokens per micro-step per TP group determines GEMM arithmetic intensity
    let tokens_per_dp_rank = (global_tokens_per_step as f64) / (cfg.dp_degree as f64);
    let workload_ai = (2.0 * tokens_per_dp_rank * (hidden_dim as f64))
        / (2.0 * (tokens_per_dp_rank + (hidden_dim as f64) / (cfg.tp_degree as f64)));

    // Tensor Core hardware efficiency when compute-bound (~72% due to non-GEMM ops, SwiGLU, norms, softmax)
    let hardware_compute_eff = 0.72;
    let roofline_attained_tflops = if workload_ai < ridge_point {
        (workload_ai * cluster.hbm_bw_gbps_per_gpu / 1000.0) * hardware_compute_eff
    } else {
        cluster.peak_bf16_tflops_per_gpu * hardware_compute_eff
    };

    let compute_time_ms = (flops_per_gpu / (roofline_attained_tflops * 1e12)) * 1e3;

    // 1. Tensor Parallel communication (cannot be fully overlapped with compute; 4 collectives per layer in fwd+bwd)
    let tp_msg_bytes =
        ((tokens_per_dp_rank as usize) / cfg.pp_degree.max(1)) * hidden_dim * 2; // BF16
    let tp_algo = if placement.min_tp_bottleneck_bw_gbps >= 400.0 {
        CollectiveAlgorithm::NvlsSharp
    } else {
        CollectiveAlgorithm::Ring
    };
    let tp_comm = simulate_collective(
        CollectiveOp::AllReduce,
        tp_algo,
        cfg.tp_degree,
        tp_msg_bytes,
        placement.min_tp_bottleneck_bw_gbps,
        1.0,
    );
    // ~75% of TP communication is on the critical path (25% overlapped via async TP / GEMM pipelining)
    let total_tp_exposed_ms =
        (tp_comm.duration_us * 4.0 * (num_layers as f64) * 0.75) / 1000.0;

    // 2. Expert Parallel MoE AllToAll communication (if ep_degree > 1)
    let total_ep_exposed_ms = if cfg.ep_degree > 1 {
        let ep_comm = simulate_collective(
            CollectiveOp::AllToAll,
            CollectiveAlgorithm::Ring,
            cfg.ep_degree,
            tp_msg_bytes,
            placement.min_ep_bottleneck_bw_gbps,
            1.5,
        );
        // 4 AllToAll collectives per MoE layer (dispatch + combine in fwd & bwd), ~60% exposed
        (ep_comm.duration_us * 4.0 * (num_layers as f64) * 0.60) / 1000.0
    } else {
        0.0
    };

    // 3. FSDP / Data Parallel gradient ReduceScatter + AllGather (mostly overlapped with backward pass,
    // except tail bucket + any bandwidth stall when RDMA is bottlenecked)
    let dp_shard_bytes =
        ((active_params_billions * 1e9 * 2.0) / (cfg.tp_degree * cfg.pp_degree) as f64) as usize;
    let dp_rs = simulate_collective(
        CollectiveOp::ReduceScatter,
        CollectiveAlgorithm::Ring,
        cfg.dp_degree,
        dp_shard_bytes,
        placement.min_dp_bottleneck_bw_gbps,
        2.5,
    );
    let dp_ag = simulate_collective(
        CollectiveOp::AllGather,
        CollectiveAlgorithm::Ring,
        cfg.dp_degree,
        dp_shard_bytes,
        placement.min_dp_bottleneck_bw_gbps,
        2.5,
    );
    let total_dp_comm_ms = (dp_rs.duration_us + dp_ag.duration_us) / 1000.0;
    // Backward pass can overlap up to `0.65 * compute_time_ms` of DP communication
    let exposed_dp_ms = (total_dp_comm_ms - 0.65 * compute_time_ms).max(0.05 * total_dp_comm_ms);

    let exposed_comm_time_ms = total_tp_exposed_ms + total_ep_exposed_ms + exposed_dp_ms;
    let step_time_ms = compute_time_ms + exposed_comm_time_ms;

    let achieved_tflops_per_gpu = (flops_per_gpu / (step_time_ms / 1000.0)) / 1e12;
    let mfu = achieved_tflops_per_gpu / cluster.peak_bf16_tflops_per_gpu;

    let regime = if exposed_comm_time_ms > compute_time_ms * 0.35 {
        BottleneckRegime::InterconnectCommunicationBound
    } else if workload_ai < ridge_point {
        BottleneckRegime::HbmBandwidthBound
    } else {
        BottleneckRegime::TensorCoreComputeBound
    };

    WorkloadRooflineReport {
        workload_name: workload_name.to_string(),
        machine_ridge_point_flops_per_byte: ridge_point,
        workload_arithmetic_intensity: workload_ai,
        regime,
        compute_time_ms,
        exposed_comm_time_ms,
        step_time_ms,
        achieved_tflops_per_gpu,
        mfu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rack_topology::{schedule_workload, PlacementPolicy, RackArchitecture};

    #[test]
    fn test_nvls_sharp_outperforms_ring_at_scale() {
        let ring = simulate_collective(
            CollectiveOp::AllReduce,
            CollectiveAlgorithm::Ring,
            72,
            256 * 1024 * 1024, // 256 MB
            900.0,
            1.0,
        );
        let nvls = simulate_collective(
            CollectiveOp::AllReduce,
            CollectiveAlgorithm::NvlsSharp,
            72,
            256 * 1024 * 1024,
            900.0,
            1.0,
        );
        assert!(nvls.duration_us < ring.duration_us);
        assert!(nvls.algbw_gbps > ring.algbw_gbps);
    }

    #[test]
    fn test_mfu_improvement_from_topology_aware_placement() {
        let cluster = ClusterTopology::new(RackArchitecture::NvidiaHgxH100Rack64, 2);
        let cfg = ParallelismConfig {
            tp_degree: 8,
            ep_degree: 8,
            pp_degree: 1,
            dp_degree: 16,
        };
        let naive_plan =
            schedule_workload(&cluster, &cfg, PlacementPolicy::NaiveStrided).unwrap();
        let aware_plan =
            schedule_workload(&cluster, &cfg, PlacementPolicy::TopologyAwareHierarchical).unwrap();

        let naive_rep = evaluate_training_step_roofline(
            "Llama-3-70B-Naive",
            &cluster,
            &cfg,
            &naive_plan,
            70.0,
            4_194_304,
            8192,
            80,
        );
        let aware_rep = evaluate_training_step_roofline(
            "Llama-3-70B-TopologyAware",
            &cluster,
            &cfg,
            &aware_plan,
            70.0,
            4_194_304,
            8192,
            80,
        );

        assert!(aware_rep.mfu > naive_rep.mfu * 1.35);
    }
}
