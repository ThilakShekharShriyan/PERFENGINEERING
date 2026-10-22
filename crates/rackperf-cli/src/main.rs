use collective_roofline::{
    evaluate_training_step_roofline, simulate_collective, CollectiveAlgorithm, CollectiveOp,
};
use rack_telemetry::{
    analyze_rack_telemetry, export_prometheus_metrics, synthesize_telemetry_snapshot,
};
use rack_topology::{
    schedule_workload, ClusterTopology, ParallelismConfig, PlacementPolicy, RackArchitecture,
};

fn main() {
    println!("================================================================================");
    println!("  RACKSCALE-PERF: AI Infrastructure Rack Topology, Collective & Roofline Suite  ");
    println!("================================================================================\n");

    // 1. Compare Rack Architectures & Topology-Aware 4D Parallelism Placement
    let hgx_cluster = ClusterTopology::new(RackArchitecture::NvidiaHgxH100Rack64, 2); // 128 H100 GPUs across 2 racks
    let gb200_cluster = ClusterTopology::new(RackArchitecture::NvidiaGb200Nvl72, 2); // 144 B200 GPUs across 2 NVL72 racks

    let hgx_cfg = ParallelismConfig {
        tp_degree: 8,
        ep_degree: 1,
        pp_degree: 2,
        dp_degree: 8,
    };

    let naive_plan =
        schedule_workload(&hgx_cluster, &hgx_cfg, PlacementPolicy::NaiveStrided).unwrap();
    let aware_plan = schedule_workload(
        &hgx_cluster,
        &hgx_cfg,
        PlacementPolicy::TopologyAwareHierarchical,
    )
    .unwrap();

    println!("[1] 4D Parallelism Placement Benchmark (128x H100 across 2 HGX Racks, TP=8, PP=2, DP=8)");
    println!(
        "  - Naive Strided Placement        : TP Bottleneck BW = {:>6.1} GB/s | TP Domain Spills = {}",
        naive_plan.min_tp_bottleneck_bw_gbps, naive_plan.tp_domain_spills
    );
    println!(
        "  - Topology-Aware Rack Placement  : TP Bottleneck BW = {:>6.1} GB/s | TP Domain Spills = {}\n",
        aware_plan.min_tp_bottleneck_bw_gbps, aware_plan.tp_domain_spills
    );

    // 2. Collective Communication Sweep (Ring vs DoubleBinaryTree vs NVLS SHARP on GB200 NVL72)
    println!("[2] Collective Communication Simulation (72-GPU NVL72 Domain, 512 MB AllReduce)");
    for algo in [
        CollectiveAlgorithm::Ring,
        CollectiveAlgorithm::DoubleBinaryTree,
        CollectiveAlgorithm::NvlsSharp,
    ] {
        let res = simulate_collective(
            CollectiveOp::AllReduce,
            algo,
            72,
            512 * 1024 * 1024,
            900.0,
            1.15,
        );
        println!(
            "  - {:<18} : Duration = {:>7.1} us | AlgBW = {:>6.1} GB/s | BusBW = {:>7.1} GB/s | Latency-Bound = {:>4.1}%",
            format!("{:?}", algo),
            res.duration_us,
            res.algbw_gbps,
            res.busbw_gbps,
            res.latency_bound_fraction * 100.0
        );
    }
    println!();

    // 3. Roofline & Achieved MFU Analysis
    let r_naive = evaluate_training_step_roofline(
        "Llama-3-70B (Naive Placement on 128x H100)",
        &hgx_cluster,
        &hgx_cfg,
        &naive_plan,
        70.0,
        4_194_304,
        8192,
        80,
    );
    let r_aware = evaluate_training_step_roofline(
        "Llama-3-70B (Topology-Aware on 128x H100)",
        &hgx_cluster,
        &hgx_cfg,
        &aware_plan,
        70.0,
        4_194_304,
        8192,
        80,
    );

    let gb200_cfg = ParallelismConfig {
        tp_degree: 8,
        ep_degree: 36,
        pp_degree: 2,
        dp_degree: 9,
    };
    let gb200_plan = schedule_workload(
        &gb200_cluster,
        &gb200_cfg,
        PlacementPolicy::TopologyAwareHierarchical,
    )
    .unwrap();
    let r_gb200 = evaluate_training_step_roofline(
        "Frontier-MoE-140B-Active (2x GB200 NVL72 Racks, 144 GPUs)",
        &gb200_cluster,
        &gb200_cfg,
        &gb200_plan,
        140.0,
        4_718_592,
        12288,
        96,
    );

    println!("[3] Roofline & Model FLOPs Utilization (MFU) Analysis");
    for rep in [&r_naive, &r_aware, &r_gb200] {
        println!(
            "  * {:<56}\n    Ridge Point: {:>5.1} FLOP/B | Workload AI: {:>6.1} FLOP/B | Regime: {:?}",
            rep.workload_name,
            rep.machine_ridge_point_flops_per_byte,
            rep.workload_arithmetic_intensity,
            rep.regime
        );
        println!(
            "    Compute: {:>6.1} ms | Exposed Comm: {:>6.1} ms | Step Time: {:>6.1} ms | Achieved: {:>6.1} TFLOPS/GPU (MFU: {:>5.1}%)\n",
            rep.compute_time_ms,
            rep.exposed_comm_time_ms,
            rep.step_time_ms,
            rep.achieved_tflops_per_gpu,
            rep.mfu * 100.0
        );
    }

    // 4. Rack Telemetry Straggler, Thermal & RoCEv2 PFC Fault Isolation
    println!("[4] Live Rack Telemetry Diagnostic & Straggler Isolation (GB200 NVL72 Rack 0)");
    let telemetry = synthesize_telemetry_snapshot(&gb200_cluster, true);
    let summary = analyze_rack_telemetry(&telemetry);
    println!(
        "  - Cluster Power Draw : {:.1} kW (Max Per-Rack: {:.1} kW)",
        summary.total_cluster_power_kw, summary.max_rack_power_kw
    );
    println!(
        "  - Step Compute Time  : Median = {:.1} ms | p99 = {:.1} ms | Straggler Barrier Tax = +{:.1}%",
        summary.median_step_compute_ms,
        summary.p99_step_compute_ms,
        summary.straggler_slowdown_pct
    );
    println!("  - Hardware & Fabric Alerts Isolated ({}):", summary.alerts.len());
    for alert in &summary.alerts {
        println!(
            "    [SEV {:>2}] GPU {:>3} (Rack {}, Tray {:>2}) {:?}\n             -> {}",
            alert.severity_score,
            alert.global_gpu_id,
            alert.rack_id,
            alert.tray_id,
            alert.anomaly,
            alert.recommended_action
        );
    }

    println!("\n[5] Prometheus Exporter Sample:");
    for line in export_prometheus_metrics(&summary).lines().take(8) {
        println!("  {}", line);
    }
}
