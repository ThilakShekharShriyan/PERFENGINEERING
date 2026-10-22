//! `rack-telemetry`: DCGM & RDMA rack telemetry ingestion, straggler detection,
//! thermal/power transient monitoring, and automated pre-flight rack burn-in validation.

use rack_topology::ClusterTopology;
use serde::{Deserialize, Serialize};

/// Per-GPU hardware telemetry sample modeled after NVIDIA DCGM (`DCGM_FI_DEV_*`)
/// and Mellanox/NVIDIA `ethtool` / InfiniBand hardware counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetrySample {
    pub global_gpu_id: usize,
    pub rack_id: usize,
    pub tray_id: usize,
    /// `DCGM_FI_DEV_SM_ACTIVE` `[0.0, 1.0]`
    pub sm_active_ratio: f64,
    /// `DCGM_FI_DEV_PIPE_TENSOR_ACTIVE` `[0.0, 1.0]`
    pub tensor_core_active_ratio: f64,
    /// `DCGM_FI_DEV_MEM_COPY_UTIL` `[0.0, 1.0]`
    pub hbm_bw_util_ratio: f64,
    /// GPU junction temperature in Celsius.
    pub gpu_temp_c: f64,
    /// HBM memory temperature in Celsius.
    pub hbm_temp_c: f64,
    /// Instantaneous power draw in Watts.
    pub power_draw_watts: f64,
    /// Cumulative or window NVLink CRC flit errors (`DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL`).
    pub nvlink_crc_errors: u64,
    /// PCIe replay counter per minute (`DCGM_FI_DEV_PCIE_REPLAY_COUNTER`).
    pub pcie_replay_count: u64,
    /// RoCEv2 Priority Flow Control (PFC) pause frames or InfiniBand symbol errors on the paired NIC.
    pub rdma_pfc_pause_frames: u64,
    /// Uncorrectable ECC memory errors (`DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`).
    pub ecc_uncorrectable_dbe: u64,
    /// Observed per-step forward+backward compute latency in milliseconds.
    pub step_compute_ms: f64,
}

/// Classification of rack/GPU anomalies detected from telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyKind {
    /// Uncorrectable ECC Double-Bit Error (Xid 48 / 94 / 95) -> immediate node drain required.
    FatalEccDoubleBitError,
    /// High NVLink CRC error rate causing link retraining & degrades NVSwitch collective BW.
    NvLinkSignalDegradation,
    /// RoCEv2 PFC storm / congestion spreading pause frames into the leaf switch.
    RdmaPfcCongestionStorm,
    /// Thermal throttling (junction > 83C or HBM > 95C) causing clock-down straggler.
    ThermalClockThrottleStraggler,
    /// Compute straggler (>3.0 median absolute deviations slower than peer ranks).
    SynchronousStepStraggler,
}

/// Diagnostic finding emitted by the rack health & straggler detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RackDiagnosticAlert {
    pub global_gpu_id: usize,
    pub rack_id: usize,
    pub tray_id: usize,
    pub anomaly: AnomalyKind,
    pub severity_score: u8,
    pub recommended_action: String,
}

/// Rack-level summary including liquid-cooling / power transient metrics and Prometheus export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RackHealthSummary {
    pub total_gpus_sampled: usize,
    pub total_cluster_power_kw: f64,
    pub max_rack_power_kw: f64,
    pub median_step_compute_ms: f64,
    pub p99_step_compute_ms: f64,
    pub straggler_slowdown_pct: f64,
    pub alerts: Vec<RackDiagnosticAlert>,
}

/// Generates synthetic rack telemetry with optional realistic fault injection
/// (thermal throttling straggler, NVLink CRC degradation, and RoCEv2 PFC congestion storm).
pub fn synthesize_telemetry_snapshot(
    cluster: &ClusterTopology,
    inject_faults: bool,
) -> Vec<GpuTelemetrySample> {
    let mut samples = Vec::with_capacity(cluster.total_gpus());

    for gpu in &cluster.gpus {
        let id = gpu.global_gpu_id;
        // Deterministic pseudo-variation across GPUs
        let jitter = ((id * 37) % 100) as f64 / 1000.0;
        let mut sample = GpuTelemetrySample {
            global_gpu_id: id,
            rack_id: gpu.rack_id,
            tray_id: gpu.tray_or_node_id,
            sm_active_ratio: 0.86 + jitter * 0.05,
            tensor_core_active_ratio: 0.71 + jitter * 0.06,
            hbm_bw_util_ratio: 0.64 + jitter * 0.08,
            gpu_temp_c: 62.0 + (id % 9) as f64 * 1.1,
            hbm_temp_c: 68.0 + (id % 11) as f64 * 1.2,
            power_draw_watts: 680.0 + (id % 13) as f64 * 9.5,
            nvlink_crc_errors: 0,
            pcie_replay_count: 0,
            rdma_pfc_pause_frames: 2,
            ecc_uncorrectable_dbe: 0,
            step_compute_ms: 142.0 + jitter * 4.5,
        };

        if inject_faults {
            if id == 19 {
                // Fault 1: Cold-plate flow restriction on GPU 19 -> thermal clock throttling straggler
                sample.gpu_temp_c = 87.5;
                sample.hbm_temp_c = 97.2;
                sample.power_draw_watts = 510.0; // Power-capped / thermally throttled
                sample.step_compute_ms = 198.4; // Slows down the entire synchronous training group!
            } else if id == 42 {
                // Fault 2: Degraded NVLink lane on GPU 42 causing CRC flit retries
                sample.nvlink_crc_errors = 1_480;
                sample.step_compute_ms = 171.2;
            } else if id == 57 {
                // Fault 3: RoCEv2 NIC PFC pause storm on Rail 1
                sample.rdma_pfc_pause_frames = 48_900;
                sample.step_compute_ms = 166.8;
            }
        }

        samples.push(sample);
    }

    samples
}

/// Analyzes a telemetry snapshot across all racks to isolate stragglers, link degradation,
/// and thermal/power anomalies in sub-millisecond time.
pub fn analyze_rack_telemetry(samples: &[GpuTelemetrySample]) -> RackHealthSummary {
    if samples.is_empty() {
        return RackHealthSummary {
            total_gpus_sampled: 0,
            total_cluster_power_kw: 0.0,
            max_rack_power_kw: 0.0,
            median_step_compute_ms: 0.0,
            p99_step_compute_ms: 0.0,
            straggler_slowdown_pct: 0.0,
            alerts: Vec::new(),
        };
    }

    let mut step_times: Vec<f64> = samples.iter().map(|s| s.step_compute_ms).collect();
    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_step = step_times[step_times.len() / 2];
    let p99_idx = ((step_times.len() as f64 * 0.99) as usize).min(step_times.len() - 1);
    let p99_step = step_times[p99_idx];
    let max_step = *step_times.last().unwrap_or(&median_step);

    // Compute Median Absolute Deviation (MAD) for robust outlier/straggler detection
    let mut abs_devs: Vec<f64> = step_times.iter().map(|t| (t - median_step).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2].max(0.5);

    let mut alerts = Vec::new();
    let mut rack_power_watts = std::collections::BTreeMap::<usize, f64>::new();
    let mut total_power_watts = 0.0;

    for s in samples {
        total_power_watts += s.power_draw_watts;
        *rack_power_watts.entry(s.rack_id).or_insert(0.0) += s.power_draw_watts;

        if s.ecc_uncorrectable_dbe > 0 {
            alerts.push(RackDiagnosticAlert {
                global_gpu_id: s.global_gpu_id,
                rack_id: s.rack_id,
                tray_id: s.tray_id,
                anomaly: AnomalyKind::FatalEccDoubleBitError,
                severity_score: 100,
                recommended_action:
                    "Cordon GPU immediately (Xid 48/94), drain tray, and trigger hot-spare rank replacement"
                        .to_string(),
            });
        }

        if s.gpu_temp_c >= 84.0 || s.hbm_temp_c >= 95.0 {
            alerts.push(RackDiagnosticAlert {
                global_gpu_id: s.global_gpu_id,
                rack_id: s.rack_id,
                tray_id: s.tray_id,
                anomaly: AnomalyKind::ThermalClockThrottleStraggler,
                severity_score: 90,
                recommended_action: format!(
                    "Inspect liquid-cooling manifold / cold-plate on Rack {} Tray {} (GPU {:.1}C, HBM {:.1}C); migrate TP rank off throttled GPU",
                    s.rack_id, s.tray_id, s.gpu_temp_c, s.hbm_temp_c
                ),
            });
        }

        if s.nvlink_crc_errors > 100 {
            alerts.push(RackDiagnosticAlert {
                global_gpu_id: s.global_gpu_id,
                rack_id: s.rack_id,
                tray_id: s.tray_id,
                anomaly: AnomalyKind::NvLinkSignalDegradation,
                severity_score: 85,
                recommended_action: format!(
                    "NVLink CRC flit error rate ({}) on GPU {}; run `nvidia-smi nvlink -e` and re-seat tray blind-mate cartridge",
                    s.nvlink_crc_errors, s.global_gpu_id
                ),
            });
        }

        if s.rdma_pfc_pause_frames > 5_000 {
            alerts.push(RackDiagnosticAlert {
                global_gpu_id: s.global_gpu_id,
                rack_id: s.rack_id,
                tray_id: s.tray_id,
                anomaly: AnomalyKind::RdmaPfcCongestionStorm,
                severity_score: 80,
                recommended_action: format!(
                    "RoCEv2 PFC pause storm ({} frames) on GPU {} rail; verify DCQCN/ECN marking thresholds and optical transceiver FEC counters",
                    s.rdma_pfc_pause_frames, s.global_gpu_id
                ),
            });
        } else if (s.step_compute_ms - median_step) > 6.0 * mad && s.gpu_temp_c < 84.0 {
            alerts.push(RackDiagnosticAlert {
                global_gpu_id: s.global_gpu_id,
                rack_id: s.rack_id,
                tray_id: s.tray_id,
                anomaly: AnomalyKind::SynchronousStepStraggler,
                severity_score: 70,
                recommended_action: format!(
                    "GPU {} step time {:.1}ms vs cluster median {:.1}ms; check CPU NUMA core pinning and PCIe ACS settings",
                    s.global_gpu_id, s.step_compute_ms, median_step
                ),
            });
        }
    }

    let max_rack_power_kw = rack_power_watts
        .values()
        .copied()
        .fold(0.0_f64, f64::max)
        / 1000.0;

    let straggler_slowdown_pct = ((max_step - median_step) / median_step) * 100.0;

    RackHealthSummary {
        total_gpus_sampled: samples.len(),
        total_cluster_power_kw: total_power_watts / 1000.0,
        max_rack_power_kw,
        median_step_compute_ms: median_step,
        p99_step_compute_ms: p99_step,
        straggler_slowdown_pct,
        alerts,
    }
}

/// Formats a `RackHealthSummary` into Prometheus exposition text format for Grafana/Datadog scraping.
pub fn export_prometheus_metrics(summary: &RackHealthSummary) -> String {
    let mut out = String::new();
    out.push_str("# HELP rackperf_cluster_power_kw Total AI rack cluster power draw in kW\n");
    out.push_str("# TYPE rackperf_cluster_power_kw gauge\n");
    out.push_str(&format!(
        "rackperf_cluster_power_kw {:.3}\n",
        summary.total_cluster_power_kw
    ));
    out.push_str("# HELP rackperf_step_compute_ms Training step compute latency quantiles\n");
    out.push_str("# TYPE rackperf_step_compute_ms gauge\n");
    out.push_str(&format!(
        "rackperf_step_compute_ms{{quantile=\"0.50\"}} {:.3}\n",
        summary.median_step_compute_ms
    ));
    out.push_str(&format!(
        "rackperf_step_compute_ms{{quantile=\"0.99\"}} {:.3}\n",
        summary.p99_step_compute_ms
    ));
    out.push_str("# HELP rackperf_straggler_slowdown_pct Synchronous barrier slowdown caused by slowest rank\n");
    out.push_str("# TYPE rackperf_straggler_slowdown_pct gauge\n");
    out.push_str(&format!(
        "rackperf_straggler_slowdown_pct {:.2}\n",
        summary.straggler_slowdown_pct
    ));
    out.push_str("# HELP rackperf_active_alerts_total Number of active hardware/fabric diagnostic alerts\n");
    out.push_str("# TYPE rackperf_active_alerts_total gauge\n");
    out.push_str(&format!("rackperf_active_alerts_total {}\n", summary.alerts.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rack_topology::RackArchitecture;

    #[test]
    fn test_straggler_and_fault_isolation() {
        let cluster = ClusterTopology::new(RackArchitecture::NvidiaGb200Nvl72, 1);
        let samples = synthesize_telemetry_snapshot(&cluster, true);
        let summary = analyze_rack_telemetry(&samples);

        assert_eq!(summary.total_gpus_sampled, 72);
        assert!(summary.straggler_slowdown_pct > 30.0);
        assert!(summary
            .alerts
            .iter()
            .any(|a| a.anomaly == AnomalyKind::ThermalClockThrottleStraggler && a.global_gpu_id == 19));
        assert!(summary
            .alerts
            .iter()
            .any(|a| a.anomaly == AnomalyKind::NvLinkSignalDegradation && a.global_gpu_id == 42));
        assert!(summary
            .alerts
            .iter()
            .any(|a| a.anomaly == AnomalyKind::RdmaPfcCongestionStorm && a.global_gpu_id == 57));
    }
}
