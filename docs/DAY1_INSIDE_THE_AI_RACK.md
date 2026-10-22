# Day 1 — Inside the AI Rack (0 → Hero)

**Goal tonight:** you can walk to a whiteboard and explain, without notes:

1. Why an 8-GPU HGX H100 node is a *different computer* than a 72-GPU GB200 NVL72 rack.
2. Why Tensor Parallelism that spills off NVSwitch onto a 400G NIC can drop Llama-70B MFU from ~63% to ~23%.
3. How to read `nvidia-smi topo -m` and decide whether GPUDirect RDMA will hit line rate.
4. How `crates/rack-topology/src/lib.rs` encodes those facts.

This day is **physical architecture only**. NCCL math, Roofline, and 4D parallelism mapping are Days 2–4. Do not skip the unit conversion section — half of interview mistakes are Gbps vs GB/s.

Estimated time: **8–10 hours** if you watch everything, **5–6 hours** if you skip the long GTC keynote and only watch the marked clips.

---

## 0. Units you must never mix up

| What people say | What it actually is | Convert |
|---|---|---|
| 400 Gbps NIC (ConnectX-7 NDR / 400G Ethernet) | **50 GB/s unidirectional** | divide bits by 8 |
| 800 Gbps NIC (ConnectX-8) | **100 GB/s unidirectional** | divide bits by 8 |
| H100 NVLink 4 “900 GB/s” | **900 GB/s bidirectional**, **450 GB/s unidirectional** | marketing is usually bidirectional |
| B200 NVLink 5 “1.8 TB/s” | **1800 GB/s bidirectional**, **900 GB/s unidirectional** | same rule |
| PCIe Gen5 x16 | **64 GB/s bidirectional**, **32 GB/s unidirectional** | ~14× slower than H100 NVLink |
| PCIe Gen6 x16 | **128 GB/s bidirectional**, **64 GB/s unidirectional** | still slower than NVLink |

Rule: **network people quote bits (Gbps). GPU people quote bytes (GB/s). Always convert.**

Our Rust model uses **unidirectional GB/s** everywhere (`unidirectional_bw_gbps` is a slightly sloppy field name — it is GB/s, not Gbps).

```text
H100 intra-node NVSwitch : 450 GB/s
H100 cross-node 400G NIC :  50 GB/s   ← 9× drop
B200 intra-rack NVSwitch : 900 GB/s
B200 cross-rack 800G NIC : 100 GB/s   ← 9× drop again
```

That 9× cliff is the entire Day 1 plot.

---

## 1. The one idea that organizes everything: scale-up vs scale-out

An AI cluster has **two completely different fabrics** that look like “the network” to a PyTorch programmer:

| | Scale-up (inside the domain) | Scale-out (between domains) |
|---|---|---|
| NVIDIA name | NVLink + NVSwitch | InfiniBand or RoCEv2 (ConnectX) |
| AMD name | Infinity Fabric / xGMI | RoCEv2 or InfiniBand |
| Domain size (Hopper) | **8 GPUs** (one HGX node) | thousands of GPUs |
| Domain size (Blackwell NVL72) | **72 GPUs** (one rack) | thousands of GPUs |
| Bandwidth | 450–900 GB/s per GPU | 50–100 GB/s per GPU |
| Latency | ~0.6–1.2 µs | ~1.8–4 µs + congestion |
| Used for | Tensor Parallel, Expert Parallel AllToAll | Data Parallel / FSDP, Pipeline, cross-rack EP |
| Failure mode | NVLink CRC, blind-mate reseat | PFC pause storm, optics flap, rail imbalance |

Plus two more networks you will see on every real rack (SemiAnalysis and NVIDIA both call these out):

3. **Frontend / north-south Ethernet** — Kubernetes, Slurm, checkpoints, data loading, storage. BlueField-3 on GB200 trays.
4. **Out-of-band / BMC / IPMI** — lights-out management. If this is down you cannot even power-cycle a tray.

If someone says “the network is slow,” your first question is: **which of the four?**

---

## 2. Watch / read order (do this in sequence)

Do not bounce around. Each block unlocks the next.

### Block A — Mental model (90 min)

**Watch (pick one long + one short):**

1. **Jensen Huang, GTC March 2024 keynote** — skip the concert intro, watch the Blackwell / GB200 NVL72 section.
   - Full: [https://www.youtube.com/watch?v=Y2F8yisiS6E](https://www.youtube.com/watch?v=Y2F8yisiS6E)
   - Alternate upload: [https://www.youtube.com/watch?v=f8DKD78BrQA](https://www.youtube.com/watch?v=f8DKD78BrQA)
   - What to extract: “one giant GPU,” copper cable cartridge, 120 kW liquid-cooled rack, 72 GPUs in one NVLink domain.

2. **10-minute recap if you are impatient:** [https://www.youtube.com/watch?v=qK2eCeRLh9k](https://www.youtube.com/watch?v=qK2eCeRLh9k)

**Read (required, ~25 min):**

3. NVIDIA Technical Blog — *GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference*
   - [https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/)
   - Memorize: compute tray = 2 Grace + 4 Blackwell; 18 compute trays + 9 NVLink switch trays; NVLink-C2C 900 GB/s; 1.8 TB/s bidirectional per GPU.

4. NVIDIA product page (skim the diagrams):
   - [https://www.nvidia.com/en-us/data-center/gb200-nvl72/](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

**Interactive (do not skip — this is the best physical intuition on the internet):**

5. **nvl72.dev Anatomy of the Rack** — drag the 3D rack, explode it, click the copper spine, busbar, CDU, switch band.
   - [https://nvl72.dev/anatomy](https://nvl72.dev/anatomy)
   - Takeaway: switch trays sit in the *middle* of the rack because copper reach at NVLink rates is short. The vertical layout is a cable-length optimization.

### Block B — What an 8-GPU node actually is (75 min)

**Read (required):**

6. NVIDIA Technical Blog — *Introducing NVIDIA HGX H100*
   - [https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/)
   - 8× H100 + **4 NVSwitch chips on the baseboard**. Any GPU talks to any other GPU at full 900 GB/s bidirectional. Non-blocking.

7. NVIDIA Technical Blog — *NVLink and NVSwitch Supercharge LLM Inference*
   - [https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/)
   - Why NVSwitch matters: it is not a mesh of pairwise links. It is a **crossbar**. Peak rate does not depend on how many peers are talking.

8. ServeTheHome — *This is the NVIDIA DGX GB200 NVL72* (photo teardown)
   - [https://www.servethehome.com/this-is-the-nvidia-dgx-gb200-nvl72/](https://www.servethehome.com/this-is-the-nvidia-dgx-gb200-nvl72/)
   - Look at: compute tray front (CX7 + BlueField-3 + E1.S), 9 NVSwitch shelves in the middle, rear busbar + liquid QD + copper cartridge.

9. ServeTheHome — *NVIDIA Blackwell Platform at Hot Chips 2024*
   - [https://www.servethehome.com/nvidia-blackwell-platform-at-hot-chips-2024/](https://www.servethehome.com/nvidia-blackwell-platform-at-hot-chips-2024/)

**Watch (physical hardware):**

10. ServeTheHome YouTube channel — browse recent GB200 / HGX H100 walkthroughs:
    - [https://www.youtube.com/@ServeTheHomeVideo](https://www.youtube.com/@ServeTheHomeVideo)
11. Supermicro HGX H100 breakdown (node-level, useful contrast to the rack):
    - [https://www.youtube.com/watch?v=_mG3Grqf2io](https://www.youtube.com/watch?v=_mG3Grqf2io)

**Official hardware docs (skim, do not memorize every table):**

12. DGX GB Rack Scale Systems User Guide — Hardware
    - [https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html)
13. SuperPOD GB200 Reference Architecture — Network Fabrics
    - [https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html)
14. GB200 NVL Multi-Node Tuning Guide — *Understanding Your Grace-Blackwell Systems*
    - [https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html)
    - This is the one that names the six networks: External-net, Internal-net, IPMI-net, Fabric-net, OS-net, NVLink-net.

### Block C — NUMA, PCIe, GPUDirect (the “first command you type” block) (90 min)

**Watch:**

15. NUMA explained (20 min, clean intro):
    - [https://www.youtube.com/watch?v=8BcFLLRX1z4](https://www.youtube.com/watch?v=8BcFLLRX1z4)
16. Optional systems-performance flavor (Brendan Gregg, LISA 2021):
    - [https://www.youtube.com/watch?v=5nN1wjA_S30](https://www.youtube.com/watch?v=5nN1wjA_S30)

**Read (required — this is interview gold):**

17. NVIDIA GPUDirect Storage benchmarking guide — the `nvidia-smi topo -mp` legend
    - [https://docs.nvidia.com/gpudirect-storage/configuration-guide/](https://docs.nvidia.com/gpudirect-storage/configuration-guide/)
    - Memorize the matrix codes in Section 3 below.
18. NVIDIA Technical Blog — *Benchmarking GPUDirect RDMA on Modern Server Platforms*
    - [https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/](https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/)
19. NVIDIA Technical Blog — *Accelerating IO in the Modern Data Center: Network IO*
    - [https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-network-io/](https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-network-io/)
20. GPUDirect family landing page:
    - [https://developer.nvidia.com/gpudirect](https://developer.nvidia.com/gpudirect)
21. Official GPUDirect RDMA overview (how BAR1 P2P actually works):
    - [https://docs.nvidia.com/cuda/gpudirect-rdma/](https://docs.nvidia.com/cuda/gpudirect-rdma/)
22. Dell / PowerScale PCIe topology write-up (clear PIX pairing table):
    - [https://infohub.delltechnologies.com/en-us/l/dell-powerscale-and-nvidia-gpudirect-performance-report/pcie-topology/](https://infohub.delltechnologies.com/en-us/l/dell-powerscale-and-nvidia-gpudirect-performance-report/pcie-topology/)
23. NVIDIA AI Enterprise — system topology + ACS/ATS notes:
    - [https://docs.nvidia.com/ai-enterprise/planning-resource/optimizing-vm-configuration-ai-inference/latest/system-setup-and-install.html](https://docs.nvidia.com/ai-enterprise/planning-resource/optimizing-vm-configuration-ai-inference/latest/system-setup-and-install.html)
24. GPUDirect Storage troubleshooting — ACS disable:
    - [https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)

### Block D — AMD so you are not NVIDIA-only (45 min)

25. AMD ROCm blog — *Understanding RCCL Bandwidth and xGMI Performance on MI300X*
    - [https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)
    - Critical difference vs NVSwitch: MI300X is a **full mesh of 7 point-to-point xGMI links**, not a non-blocking crossbar. One peer gets ~one link (~64 GB/s unidirectional spec, ~45–48 GB/s measured). Aggregate per GPU is ~448 GB/s unidirectional across all 7 peers.
26. AMD Instinct MI300 Series Cluster Reference Architecture (PDF)
    - [https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/other/instinct-mi300-series-cluster-reference-guide.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/other/instinct-mi300-series-cluster-reference-guide.pdf)
27. Hot Chips 2024 MI300X slides
    - [https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith%28MI300X%29.v1.Final.20240817.pdf](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith%28MI300X%29.v1.Final.20240817.pdf)

### Block E — Industry-grade rack analysis (60 min, paid is optional)

28. SemiAnalysis — *GB200 Hardware Architecture and Component Supply Chain* (best rack BOM / NVL36 vs NVL72 / 120 kW vs 40 kW H100 discussion in the industry)
    - [https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component](https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component)
29. SemiAnalysis — *Nvidia’s Optical Boogeyman* (why NVL72 uses copper, and why scale-out still needs 800G/1.6T optics)
    - [https://newsletter.semianalysis.com/p/nvidias-optical-boogeyman-nvl72-infiniband](https://newsletter.semianalysis.com/p/nvidias-optical-boogeyman-nvl72-infiniband)

Free substitute if you do not subscribe: NVIDIA SuperPOD RA + ServeTheHome teardown + nvl72.dev cover 80% of the same facts.

### Block F — First-principles performance mindset (30 min, sets up Day 3)

30. Horace He — *Making Deep Learning Go Brrrr From First Principles*
    - [https://horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)
    - Compute vs memory vs overhead. Day 1 adds a fourth regime: **interconnect-bound**. That is what naive TP placement does.

### Extra credit (watch later, not tonight)

- NVIDIA Hopper Architecture In-Depth: [https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Fabric Manager (how the host programs the NVSwitch fabric): [https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html)
- CUDA Programming Guide — multi-GPU systems: [https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- Hot Chips 2025 tutorial *Datacenter Racks* (NVIDIA GB200 NVL72 + Meta Catalina case studies): [https://hc2025.hotchips.org/](https://hc2025.hotchips.org/)
- NVIDIA marketing short on NVLink/NVSwitch: [https://www.youtube.com/watch?v=kstuoIB7rpE](https://www.youtube.com/watch?v=kstuoIB7rpE)

---

## 3. The hardware, layer by layer

### 3.1 A single GPU is already a NUMA machine

You do not need to know SM counts tonight. You do need three numbers per GPU:

| GPU | Peak BF16 Tensor | HBM BW | HBM capacity | Scale-up BW (uni) |
|---|---|---|---|---|
| H100 SXM | 989 TFLOPS | 3.35 TB/s | 80 GB | 450 GB/s NVLink 4 |
| B200 | 2250 TFLOPS | 8.0 TB/s | 192 GB | 900 GB/s NVLink 5 |
| MI300X | 1307 TFLOPS | 5.3 TB/s | 192 GB | ~64 GB/s per xGMI peer (448 GB/s aggregate) |

These are the constants in `ClusterTopology::new`.

### 3.2 The HGX H100 node (the thing most clusters still are)

```
                    ┌─────────────────────────────────────┐
                    │         HGX H100 baseboard          │
                    │                                     │
                    │   GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
                    │      \   |   |   |   |   |   |   /
                    │       \  |   |   |   |   |   |  /
                    │        NVSwitch × 4  (non-blocking)
                    │                                     │
                    │   CX7-0 CX7-1 ... CX7-7  (one NIC/GPU rail)
                    └─────────────────────────────────────┘
                              │
                    8× 400Gbps = 50 GB/s each, to ToR rails
```

Facts you will be asked:

- GPUs do **not** have a complete pairwise NVLink mesh on Hopper 8-GPU. They all go **into NVSwitch**. That is why any-to-any is full rate.
- The NVLink domain **dies at the node lid**. GPU0 on node A to GPU0 on node B is a 50 GB/s RDMA hop, even if they sit in the same 42U rack, one RU apart.
- Each GPU is paired with a NIC on the **same PCIe switch** (`PIX`). That pairing *is* the rail. NCCL’s topology file pins GPU `i` to NIC `i`.

This is `RackArchitecture::NvidiaHgxH100Rack64` with `trays_per_rack = 8`, `gpus_per_tray = 8`.

### 3.3 The GB200 NVL72 rack (the thing every 2026–2027 job description means)

```
 Front of rack (1U trays)
 ┌──────────────────────────┐
 │  compute trays 0..9      │  10 × (2 Grace + 4 B200)
 │  ── NVSwitch trays 0..8 ─│  9 × (2 NVLink Switch chips)   ← middle on purpose
 │  compute trays 10..17    │  8 × (2 Grace + 4 B200)
 └──────────────────────────┘
 Rear: copper NVLink cartridge + busbar + liquid manifolds (blind-mate)
 Scale-out: 4× CX7 400G (or CX8 800G) per compute tray = 1 NIC per GPU
 Frontend: 2× BlueField-3 per tray
```

Facts you will be asked:

- **GB200 Superchip** = 1 Grace + 2 Blackwell, joined by **NVLink-C2C** (900 GB/s bidirectional, coherent). The CPU can see GPU HBM. That is why they can oversubscribe CPU DRAM less aggressively and why unified memory stories show up in NVIDIA blogs.
- **Compute tray** = 2 Superchips = 2 Grace + 4 B200.
- **NVL72** = 18 trays × 4 GPUs = 72 GPUs, **one NVLink domain**.
- Each B200 has **18 NVLink 5 links**, one to each of the **18 switch chips** (9 trays × 2 chips). That is a one-hop all-to-all: GPU → its dedicated link → switch chip → destination GPU.
- Aggregate rack GPU-to-GPU bandwidth NVIDIA quotes: **130 TB/s**.
- Power: **~120 kW** (SemiAnalysis / ServeTheHome). A typical air-cooled H100 rack is ~40 kW. This is why the rack is 100% direct-to-chip liquid and why a clogged cold plate on GPU 19 becomes a cluster-level incident (your telemetry crate already models this).
- NVL36 / NVL36×2 exist because many colo rooms still cap at ~60–80 kW.

This is `RackArchitecture::NvidiaGb200Nvl72` with `trays_per_rack = 18`, `gpus_per_tray = 4`. Look at `link_metrics`: two GPUs in the **same rack, different trays** still get `IntraRackNvSwitchBackplane` at 900 GB/s. On H100 that same “same rack, different trays” path is `SameRailLeafSwitch` at 50 GB/s. That is the generation jump.

### 3.4 AMD MI300X node (know this so you do not sound NVIDIA-captive)

- 8 GPUs, **full mesh** Infinity Fabric (xGMI): each GPU has 7 bidirectional links.
- 192 GB HBM3 per GPU — inference people care more about this than peak FLOPS.
- There is **no NVSwitch equivalent** in the node. All-to-all NCCL/RCCL traffic is a mesh, so incast and last-hop contention look different than Hopper.
- Scale-out is still 400G RoCE/IB rails, same 9× cliff.

Our model currently treats intra-node MI300X like a 448 GB/s fabric (aggregate). Interview-correct statement: “448 GB/s is the sum of seven links; a single peer is one link.”

---

## 4. PCIe / NUMA / GPUDirect — the first command you type

On any strange node:

```bash
nvidia-smi topo -m
nvidia-smi topo -mp          # include NICs
nvidia-smi nvlink -s         # link status
numactl -H
lstopo --of ascii            # from hwloc
ibdev2netdev                 # mlx5_N → netdev
```

### 4.1 The topology matrix (memorize)

From NVIDIA’s GDS configuration guide:

| Code | Path | Verdict for GPU↔NIC or GPU↔GPU over PCIe |
|---|---|---|
| `PIX` | at most one PCIe bridge (same switch) | **best** — P2P stays in the switch |
| `PXB` | multiple PCIe bridges, no host bridge | good |
| `PHB` | through a PCIe Host Bridge (the CPU) | OK, some tax |
| `NODE` | between host bridges inside one NUMA node | worse |
| `SYS` | **cross-socket** UPI / QPI / Infinity Fabric | **bad** — 30–50% of line rate, sometimes P2P disabled |
| `NV#` | NVLink, `#` = hop count | this is the scale-up path, not PCIe |

On a proper HGX H100:

- GPU0–GPU7 to each other: `NV#` (via NVSwitch), not `PIX`.
- GPU0 to `mlx5_0`: `PIX`.
- GPU0 to `mlx5_7`: `SYS` or `NODE`. **Never pin NCCL to that NIC.**

If a cloud vendor gives you 8 GPUs and 1 NIC, or GPU0/`mlx5_0` is `SYS`, GPUDirect will not hit 400G. That is a bring-up bug, not an NCCL bug.

### 4.2 The three GPUDirects (do not conflate them)

| Name | What moves | Path | Day-1 failure mode |
|---|---|---|---|
| **GPUDirect P2P** | GPU HBM ↔ GPU HBM | NVLink (fast) or PCIe (slow) | ACS, wrong NUMA, IOMMU |
| **GPUDirect RDMA** | NIC ↔ GPU HBM | PCIe BAR1, `nvidia_peermem` | ACS redirect, `SYS` pairing, module not loaded |
| **GPUDirect Storage** | NVMe ↔ GPU HBM | PCIe P2P or NIC as “storage controller” | same ACS/topology issues |

Without GPUDirect RDMA, every NCCL/IB send does:

```
GPU HBM → cudaMemcpy → CPU DRAM → NIC DMA → wire
wire → NIC DMA → CPU DRAM → cudaMemcpy → GPU HBM
```

That is two extra copies, extra latency (~8–17 µs vs ~2 µs on the NVIDIA blog numbers), and it eats CPU memory bandwidth that the Grace/x86 sockets need for data loading.

### 4.3 ACS (Access Control Services)

ACS is a PCIe feature that **redirects peer-to-peer packets up to the root complex** so an IOMMU can snoop them. Great for multi-tenant isolation. Fatal for GPUDirect.

Symptoms: `ib_write_bw --use_cuda` is ~half of `ib_write_bw` host-to-host, or P2P is “enabled” but bandwidth is a joke.

Fixes you will see in runbooks (do not apply blindly on a shared university box):

- BIOS: disable ACS on the GPU/NIC switches.
- Kernel: `pci=disable_acs_redir=...` or `config_acs=...` (see GDS troubleshooting guide and `rdma_topo` in rdma-core).
- Confirm with `setpci` / `rdma_topo check` that ACS is `0x0000` on the relevant bridges.

You will not disable ACS on your Mac. Know the story for interviews.

### 4.4 Grace is a different NUMA picture

On GB200, NVLink-C2C makes Grace↔GPU coherent at 900 GB/s bidirectional. CPU DRAM is still slower than HBM. The win is: the NIC and GPU are designed to sit under the same PCIe switch on the Bianca board, so `PIX` is a board-level guarantee, not an OEM lottery. The tuning guide still tells you to verify it during burn-in.

---

## 5. Map this onto `rack-topology` (tonight’s lab, 45 min)

Open `crates/rack-topology/src/lib.rs` and read it top to bottom with this checklist.

1. `RackArchitecture` variants are the three machines in Sections 3.2–3.4.
2. `InterconnectTier` is the **ordered** bandwidth hierarchy. `PartialOrd` is deliberate: `IntraTrayNvLink < IntraRackNvSwitchBackplane < SameRailLeafSwitch < SpineFabricScaleOut`. A TP group that uses any tier `>= SameRailLeafSwitch` is a **spill**. That is `tp_domain_spills`.
3. `link_metrics`:
   - same tray → NVLink/xGMI (450 / 900 / 448).
   - same rack, GB200, different tray → **still 900 GB/s** (`IntraRackNvSwitchBackplane`).
   - same rack, H100, different node → **50 GB/s** RDMA, better if `nic_rail_id` matches.
   - different rack → 50 or 100 GB/s, rail-aligned is less bad than spine.
4. `PlacementPolicy::TopologyAwareHierarchical` sorts by rack, then tray, then link health. That *is* “pack TP inside the NVSwitch domain.”
5. `PlacementPolicy::NaiveStrided` walks with a stride so consecutive TP ranks land on different trays. That is the Kubernetes/Slurm “I got 8 GPUs somewhere” failure mode.
6. Run and stare at the numbers:

```bash
cargo test -p rack-topology -- --nocapture
cargo run --bin rackperf
```

You already measured:

```
Naive Strided        TP bottleneck =  47.5 GB/s   spills = 16
Topology-Aware       TP bottleneck = 450.0 GB/s   spills = 0
```

47.5 is 50 GB/s × 0.95 rail efficiency. 450 is H100 NVSwitch unidirectional. Write those two numbers on a sticky note.

Optional stretch: in a scratch `#[test]`, print `link_metrics(0, 71)` vs `link_metrics(0, 72)` on a 2-rack GB200. You will see backplane vs same-rail scale-out. That test already exists as `test_gb200_nvl72_intra_rack_nvswitch_domain`.

---

## 6. End-of-day self-test (close the laptop and answer out loud)

If any answer is fuzzy, re-read the matching block. Do not go to Day 2 until these are clean.

1. Convert 400 Gbps and 800 Gbps to GB/s unidirectional.
2. Why is H100 NVLink quoted as 900 GB/s but our code uses 450?
3. What is on one GB200 compute tray? What is on one NVLink switch tray? Why are the switch trays in the middle of the rack?
4. How many NVLink 5 links does one B200 have, and what is each one connected to?
5. Why can TP=8 live entirely on NVLink on HGX H100, and why does TP=16 on the same cluster become an RDMA problem?
6. Why can TP=16 or EP=36 live on NVLink on a single NVL72 rack?
7. What does `PIX` vs `SYS` mean, and which one must a GPU and its NCCL NIC be?
8. What does ACS do to GPUDirect, in one sentence?
9. Name the four networks in a GB200 rack and what each is for.
10. What is the difference between NVSwitch (crossbar) and MI300X xGMI (mesh) for an 8-GPU all-to-all?
11. Why does one thermally throttled GPU stall an entire training job even if 71 others are fine? (preview of Day 6; answer: synchronous collectives move at the slowest rank.)
12. Point at the line in `link_metrics` that encodes “NVL72 is a rack-scale NVLink domain.”

---

## 7. Tomorrow morning (Day 2 preview)

You will take this topology and put **NCCL collectives** on it: Ring vs Double Binary Tree vs NVLS, `algbw` vs `busbw`, InfiniBand credit-based lossless vs RoCEv2 PFC/DCQCN. The 9× cliff you learned today becomes the input `bottleneck_unidir_bw_gbps` to `simulate_collective`.
