use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rack_topology::{
    schedule_workload, ClusterTopology, ParallelismConfig, PlacementPolicy, RackArchitecture,
};

fn bench_rack_placement(c: &mut Criterion) {
    let cluster = ClusterTopology::new(RackArchitecture::NvidiaGb200Nvl72, 4); // 288 GPUs
    let cfg = ParallelismConfig {
        tp_degree: 8,
        ep_degree: 16,
        pp_degree: 2,
        dp_degree: 18,
    };

    c.bench_function("gb200_nvl72_288gpu_topology_aware_schedule", |b| {
        b.iter(|| {
            schedule_workload(
                black_box(&cluster),
                black_box(&cfg),
                black_box(PlacementPolicy::TopologyAwareHierarchical),
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_rack_placement);
criterion_main!(benches);
