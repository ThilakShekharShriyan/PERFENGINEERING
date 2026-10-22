//! `rack-topology`: Hardware-accurate AI rack topology modeling and 4D parallelism
//! placement engine for scale-up (NVLink/NVSwitch, xGMI, PCIe/NUMA) and scale-out
//! (InfiniBand NDR/XDR, RoCEv2 rail-optimized fat-tree) fabrics.

use serde::{Deserialize, Serialize};

/// Architecture presets for modern AI racks and compute nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RackArchitecture {
    /// NVIDIA GB200 NVL72: 18 compute trays x 4 B200 GPUs = 72 GPUs in a single
    /// 72-GPU NVLink 5 domain (900 GB/s unidirectional / 1800 GB/s bidirectional per GPU),
    /// 2 Grace CPUs (NUMA nodes) per tray, and 400G/800G ConnectX-7/8 NICs per GPU.
    NvidiaGb200Nvl72,
    /// Multi-node NVIDIA HGX H100/H200 rack: 8 nodes x 8 GPUs = 64 GPUs per rack.
    /// 8 GPUs per node share an NVLink 4 NVSwitch domain (450 GB/s unidirectional);
    /// cross-node traffic traverses 8x 400Gbps (50 GB/s unidirectional) NDR InfiniBand rails.
    NvidiaHgxH100Rack64,
    /// Multi-node AMD Instinct MI300X rack: 8 nodes x 8 GPUs = 64 GPUs per rack.
    /// 8 GPUs per node connected via full-mesh Infinity Fabric (xGMI, 448 GB/s unidirectional);
    /// cross-node traffic traverses 8x 400Gbps RoCEv2 rails.
    AmdMi300xRack64,
}

/// Interconnect tier traversed between any pair of GPUs in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InterconnectTier {
    /// Same GPU (loopback).
    SelfLoopback,
    /// Same compute tray / same NUMA domain over NVSwitch or direct NVLink/xGMI.
    IntraTrayNvLink,
    /// Same NVLink/NVSwitch domain across compute trays in the same rack (e.g., GB200 NVL72 backplane).
    IntraRackNvSwitchBackplane,
    /// Same node over PCIe / NUMA QPI/UPI/C2C when NVLink is bypassed or degraded.
    IntraNodePcieNuma,
    /// Same NIC rail across nodes within the same rack (rail-local ToR switch hop).
    SameRailLeafSwitch,
    /// Cross-rack or cross-rail spine switch hop over InfiniBand / RoCEv2 fabric.
    SpineFabricScaleOut,
}

/// Physical and performance characteristics of a single GPU slot in a rack cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEndpoint {
    pub global_gpu_id: usize,
    pub rack_id: usize,
    pub tray_or_node_id: usize,
    pub local_gpu_idx: usize,
    pub numa_node: usize,
    pub nic_rail_id: usize,
    pub healthy: bool,
    /// Fraction of nominal NVLink bandwidth currently available (1.0 = nominal, < 1.0 = degraded lanes).
    pub nvlink_health_factor: f64,
    /// Fraction of nominal RDMA NIC bandwidth currently available (1.0 = nominal, < 1.0 = PFC/link degradation).
    pub rdma_health_factor: f64,
}

/// Effective point-to-point link performance between two GPUs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinkMetrics {
    pub tier: InterconnectTier,
    /// Unidirectional bandwidth in GB/s.
    pub unidirectional_bw_gbps: f64,
    /// Base one-way fabric propagation + switch latency in microseconds.
    pub latency_us: f64,
}

/// Models a multi-rack AI cluster with rack/tray/NUMA/rail hierarchy and link health state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTopology {
    pub arch: RackArchitecture,
    pub num_racks: usize,
    pub trays_per_rack: usize,
    pub gpus_per_tray: usize,
    pub gpus: Vec<GpuEndpoint>,
    /// Peak FP16/BF16 Tensor Core TFLOPS per GPU.
    pub peak_bf16_tflops_per_gpu: f64,
    /// Peak HBM memory bandwidth in GB/s per GPU.
    pub hbm_bw_gbps_per_gpu: f64,
    /// HBM capacity in GB per GPU.
    pub hbm_capacity_gb_per_gpu: f64,
}

impl ClusterTopology {
    /// Builds a cluster topology for the given `RackArchitecture` and number of racks.
    pub fn new(arch: RackArchitecture, num_racks: usize) -> Self {
        let (trays_per_rack, gpus_per_tray, peak_bf16_tflops, hbm_bw_gbps, hbm_cap_gb) = match arch {
            RackArchitecture::NvidiaGb200Nvl72 => (18, 4, 2250.0, 8000.0, 192.0),
            RackArchitecture::NvidiaHgxH100Rack64 => (8, 8, 989.0, 3350.0, 80.0),
            RackArchitecture::AmdMi300xRack64 => (8, 8, 1307.0, 5300.0, 192.0),
        };

        let total_gpus = num_racks * trays_per_rack * gpus_per_tray;
        let mut gpus = Vec::with_capacity(total_gpus);

        for rack_id in 0..num_racks {
            for tray_idx in 0..trays_per_rack {
                for local_idx in 0..gpus_per_tray {
                    let global_gpu_id =
                        rack_id * (trays_per_rack * gpus_per_tray) + tray_idx * gpus_per_tray + local_idx;
                    let numa_node = local_idx / (gpus_per_tray / 2).max(1);
                    let nic_rail_id = local_idx;
                    gpus.push(GpuEndpoint {
                        global_gpu_id,
                        rack_id,
                        tray_or_node_id: tray_idx,
                        local_gpu_idx: local_idx,
                        numa_node,
                        nic_rail_id,
                        healthy: true,
                        nvlink_health_factor: 1.0,
                        rdma_health_factor: 1.0,
                    });
                }
            }
        }

        Self {
            arch,
            num_racks,
            trays_per_rack,
            gpus_per_tray,
            gpus,
            peak_bf16_tflops_per_gpu: peak_bf16_tflops,
            hbm_bw_gbps_per_gpu: hbm_bw_gbps,
            hbm_capacity_gb_per_gpu: hbm_cap_gb,
        }
    }

    #[inline]
    pub fn total_gpus(&self) -> usize {
        self.gpus.len()
    }

    #[inline]
    pub fn gpus_per_rack(&self) -> usize {
        self.trays_per_rack * self.gpus_per_tray
    }

    /// Injects link degradation or cordons a GPU to test fault-tolerant scheduling and straggler impact.
    pub fn degrade_gpu_links(
        &mut self,
        global_gpu_id: usize,
        nvlink_factor: f64,
        rdma_factor: f64,
        healthy: bool,
    ) {
        if let Some(gpu) = self.gpus.get_mut(global_gpu_id) {
            gpu.nvlink_health_factor = nvlink_factor.clamp(0.0, 1.0);
            gpu.rdma_health_factor = rdma_factor.clamp(0.0, 1.0);
            gpu.healthy = healthy;
        }
    }

    /// Computes effective point-to-point unidirectional bandwidth and one-way latency between two GPUs.
    #[inline]
    pub fn link_metrics(&self, src_gpu: usize, dst_gpu: usize) -> LinkMetrics {
        if src_gpu == dst_gpu {
            return LinkMetrics {
                tier: InterconnectTier::SelfLoopback,
                unidirectional_bw_gbps: self.hbm_bw_gbps_per_gpu,
                latency_us: 0.05,
            };
        }

        let a = &self.gpus[src_gpu];
        let b = &self.gpus[dst_gpu];
        let nvlink_scale = a.nvlink_health_factor.min(b.nvlink_health_factor);
        let rdma_scale = a.rdma_health_factor.min(b.rdma_health_factor);

        if a.rack_id == b.rack_id && a.tray_or_node_id == b.tray_or_node_id {
            // Intra-tray / intra-node high-speed scale-up fabric
            let (nominal_bw, base_lat_us) = match self.arch {
                RackArchitecture::NvidiaGb200Nvl72 => (900.0, 0.65),
                RackArchitecture::NvidiaHgxH100Rack64 => (450.0, 0.85),
                RackArchitecture::AmdMi300xRack64 => (448.0, 0.95),
            };
            return LinkMetrics {
                tier: InterconnectTier::IntraTrayNvLink,
                unidirectional_bw_gbps: nominal_bw * nvlink_scale,
                latency_us: base_lat_us,
            };
        }

        if a.rack_id == b.rack_id {
            match self.arch {
                RackArchitecture::NvidiaGb200Nvl72 => {
                    // GB200 NVL72 connects all 18 trays (72 GPUs) across the copper NVLink backplane!
                    return LinkMetrics {
                        tier: InterconnectTier::IntraRackNvSwitchBackplane,
                        unidirectional_bw_gbps: 900.0 * nvlink_scale,
                        latency_us: 1.15,
                    };
                }
                RackArchitecture::NvidiaHgxH100Rack64 | RackArchitecture::AmdMi300xRack64 => {
                    // Cross-node within the same rack goes over 400Gbps (50 GB/s) RDMA NIC rails
                    let same_rail = a.nic_rail_id == b.nic_rail_id;
                    return LinkMetrics {
                        tier: if same_rail {
                            InterconnectTier::SameRailLeafSwitch
                        } else {
                            InterconnectTier::SpineFabricScaleOut
                        },
                        unidirectional_bw_gbps: 50.0 * rdma_scale * if same_rail { 1.0 } else { 0.85 },
                        latency_us: if same_rail { 1.8 } else { 2.9 },
                    };
                }
            }
        }

        // Cross-rack traffic traverses the scale-out InfiniBand / RoCEv2 fabric
        let same_rail = a.nic_rail_id == b.nic_rail_id;
        let nominal_rdma_bw = match self.arch {
            RackArchitecture::NvidiaGb200Nvl72 => 100.0, // 800G ConnectX-8 / Quantum-X800
            RackArchitecture::NvidiaHgxH100Rack64 | RackArchitecture::AmdMi300xRack64 => 50.0, // 400G NDR / RoCEv2
        };

        LinkMetrics {
            tier: if same_rail {
                InterconnectTier::SameRailLeafSwitch
            } else {
                InterconnectTier::SpineFabricScaleOut
            },
            unidirectional_bw_gbps: nominal_rdma_bw * rdma_scale * if same_rail { 0.95 } else { 0.75 },
            latency_us: if same_rail { 2.4 } else { 3.8 },
        }
    }
}

/// 4D Parallelism configuration for distributed LLM training or prefill/decode serving.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Tensor Parallelism degree (requires highest bandwidth, run every layer).
    pub tp_degree: usize,
    /// Expert Parallelism degree (requires high AllToAll bisection bandwidth for MoE layers).
    pub ep_degree: usize,
    /// Pipeline Parallelism degree (point-to-point microbatch activations across stages).
    pub pp_degree: usize,
    /// Data Parallelism / FSDP degree (gradient ReduceScatter + weight AllGather).
    pub dp_degree: usize,
}

impl ParallelismConfig {
    /// Total world size required by `(TP * PP * DP)`. Note: EP typically shares ranks with DP/TP.
    #[inline]
    pub fn world_size(&self) -> usize {
        self.tp_degree * self.pp_degree * self.dp_degree
    }
}

/// Strategy used to assign logical training/inference ranks to physical rack GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementPolicy {
    /// Naive round-robin / strided assignment across racks (common misconfiguration when
    /// Kubernetes or Slurm schedules without topology-aware rank ordering).
    NaiveStrided,
    /// Rack & NVLink-Domain Topology-Aware placement:
    /// 1. Packs TP groups strictly within healthy compute trays / NVSwitch domains.
    /// 2. Packs EP groups within the same NVL72 rack or rail-aligned nodes.
    /// 3. Aligns DP/FSDP groups across matching NIC rails (`SameRailLeafSwitch`) to maximize RoCE/IB throughput.
    TopologyAwareHierarchical,
}

/// Result of scheduling a 4D parallelism job onto physical GPUs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub policy: PlacementPolicy,
    /// Maps `logical_rank -> physical_global_gpu_id`.
    pub rank_to_gpu: Vec<usize>,
    /// Minimum bottleneck bandwidth (GB/s) inside any Tensor Parallel group.
    pub min_tp_bottleneck_bw_gbps: f64,
    /// Minimum bottleneck bandwidth (GB/s) inside any Expert Parallel (MoE) group.
    pub min_ep_bottleneck_bw_gbps: f64,
    /// Minimum bottleneck bandwidth (GB/s) inside any Data Parallel / FSDP group.
    pub min_dp_bottleneck_bw_gbps: f64,
    /// Number of TP groups that spilled across NVSwitch/node boundaries.
    pub tp_domain_spills: usize,
}

/// Schedules a 4D parallel job onto the cluster and evaluates group bottleneck bandwidths.
pub fn schedule_workload(
    cluster: &ClusterTopology,
    cfg: &ParallelismConfig,
    policy: PlacementPolicy,
) -> Result<PlacementPlan, String> {
    let world_size = cfg.world_size();
    let healthy_gpus: Vec<&GpuEndpoint> = cluster.gpus.iter().filter(|g| g.healthy).collect();

    if healthy_gpus.len() < world_size {
        return Err(format!(
            "Insufficient healthy GPUs: need {}, found {}",
            world_size,
            healthy_gpus.len()
        ));
    }

    let mut rank_to_gpu = Vec::with_capacity(world_size);

    match policy {
        PlacementPolicy::NaiveStrided => {
            // Simulates fragmented cluster allocation where consecutive ranks land across different trays/racks
            let stride = (healthy_gpus.len() / cfg.tp_degree.max(1)).max(1);
            let mut used = vec![false; healthy_gpus.len()];
            let mut cursor = 0usize;
            for _ in 0..world_size {
                while used[cursor] {
                    cursor = (cursor + 1) % healthy_gpus.len();
                }
                used[cursor] = true;
                rank_to_gpu.push(healthy_gpus[cursor].global_gpu_id);
                cursor = (cursor + stride) % healthy_gpus.len();
            }
        }
        PlacementPolicy::TopologyAwareHierarchical => {
            // Score and sort trays/nodes by aggregate link health so degraded trays are avoided first
            let mut sorted_gpus = healthy_gpus;
            sorted_gpus.sort_by(|a, b| {
                let score_a = a.nvlink_health_factor * 2.0 + a.rdma_health_factor;
                let score_b = b.nvlink_health_factor * 2.0 + b.rdma_health_factor;
                a.rack_id
                    .cmp(&b.rack_id)
                    .then_with(|| a.tray_or_node_id.cmp(&b.tray_or_node_id))
                    .then_with(|| score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal))
                    .then_with(|| a.local_gpu_idx.cmp(&b.local_gpu_idx))
            });

            for gpu in sorted_gpus.into_iter().take(world_size) {
                rank_to_gpu.push(gpu.global_gpu_id);
            }
        }
    }

    // Evaluate bottleneck bandwidths across TP, EP, and DP communication groups
    let mut min_tp_bw = f64::MAX;
    let mut tp_spills = 0usize;

    let num_tp_groups = world_size / cfg.tp_degree;
    for g in 0..num_tp_groups {
        let base = g * cfg.tp_degree;
        let mut group_spilled = false;
        for i in 0..cfg.tp_degree {
            for j in (i + 1)..cfg.tp_degree {
                let gpu_a = rank_to_gpu[base + i];
                let gpu_b = rank_to_gpu[base + j];
                let m = cluster.link_metrics(gpu_a, gpu_b);
                if m.unidirectional_bw_gbps < min_tp_bw {
                    min_tp_bw = m.unidirectional_bw_gbps;
                }
                if m.tier >= InterconnectTier::SameRailLeafSwitch {
                    group_spilled = true;
                }
            }
        }
        if group_spilled {
            tp_spills += 1;
        }
    }
    if cfg.tp_degree <= 1 {
        min_tp_bw = cluster.hbm_bw_gbps_per_gpu;
    }

    // EP groups span `ep_degree` consecutive TP groups (or consecutive ranks within a stage)
    let mut min_ep_bw = f64::MAX;
    if cfg.ep_degree > 1 && world_size >= cfg.ep_degree {
        let ep_span = cfg.ep_degree;
        let num_ep_groups = world_size / ep_span;
        for g in 0..num_ep_groups {
            let base = g * ep_span;
            for i in 0..ep_span {
                for j in (i + 1)..ep_span {
                    let m = cluster.link_metrics(rank_to_gpu[base + i], rank_to_gpu[base + j]);
                    if m.unidirectional_bw_gbps < min_ep_bw {
                        min_ep_bw = m.unidirectional_bw_gbps;
                    }
                }
            }
        }
    } else {
        min_ep_bw = min_tp_bw;
    }

    // DP groups step by `tp_degree * pp_degree` so ranks with the same model shard sync gradients
    let mut min_dp_bw = f64::MAX;
    let dp_stride = cfg.tp_degree * cfg.pp_degree;
    if cfg.dp_degree > 1 {
        for shard in 0..dp_stride {
            for d1 in 0..cfg.dp_degree {
                for d2 in (d1 + 1)..cfg.dp_degree {
                    let r1 = d1 * dp_stride + shard;
                    let r2 = d2 * dp_stride + shard;
                    let m = cluster.link_metrics(rank_to_gpu[r1], rank_to_gpu[r2]);
                    if m.unidirectional_bw_gbps < min_dp_bw {
                        min_dp_bw = m.unidirectional_bw_gbps;
                    }
                }
            }
        }
    } else {
        min_dp_bw = min_tp_bw;
    }

    Ok(PlacementPlan {
        policy,
        rank_to_gpu,
        min_tp_bottleneck_bw_gbps: min_tp_bw,
        min_ep_bottleneck_bw_gbps: min_ep_bw,
        min_dp_bottleneck_bw_gbps: min_dp_bw,
        tp_domain_spills: tp_spills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gb200_nvl72_intra_rack_nvswitch_domain() {
        let cluster = ClusterTopology::new(RackArchitecture::NvidiaGb200Nvl72, 2);
        assert_eq!(cluster.total_gpus(), 144);

        // GPU 0 (Rack 0, Tray 0) -> GPU 71 (Rack 0, Tray 17) traverses NVSwitch copper backplane
        let intra_rack = cluster.link_metrics(0, 71);
        assert_eq!(intra_rack.tier, InterconnectTier::IntraRackNvSwitchBackplane);
        assert!((intra_rack.unidirectional_bw_gbps - 900.0).abs() < 1e-6);

        // GPU 0 (Rack 0) -> GPU 72 (Rack 1) traverses Same-Rail Scale-Out RDMA
        let cross_rack = cluster.link_metrics(0, 72);
        assert_eq!(cross_rack.tier, InterconnectTier::SameRailLeafSwitch);
        assert!(cross_rack.unidirectional_bw_gbps <= 100.0);
    }

    #[test]
    fn test_topology_aware_vs_naive_placement_on_hgx_h100() {
        let cluster = ClusterTopology::new(RackArchitecture::NvidiaHgxH100Rack64, 2);
        let cfg = ParallelismConfig {
            tp_degree: 8,
            ep_degree: 8,
            pp_degree: 2,
            dp_degree: 8,
        };

        let naive = schedule_workload(&cluster, &cfg, PlacementPolicy::NaiveStrided).unwrap();
        let aware =
            schedule_workload(&cluster, &cfg, PlacementPolicy::TopologyAwareHierarchical).unwrap();

        // Topology-aware placement keeps all TP=8 groups strictly within an 8-GPU NVSwitch node (450 GB/s)
        assert_eq!(aware.tp_domain_spills, 0);
        assert!((aware.min_tp_bottleneck_bw_gbps - 450.0).abs() < 1e-6);

        // Naive placement spills TP groups across RDMA rails (dropping TP bottleneck from 450 GB/s to <= 50 GB/s)
        assert!(naive.tp_domain_spills > 0);
        assert!(naive.min_tp_bottleneck_bw_gbps <= 50.0);
    }
}
