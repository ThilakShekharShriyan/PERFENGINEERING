# 7-Day AI Infrastructure & Rack Performance Engineering Master Plan
**From Basics to Master + GitHub Codebase + LinkedIn Authority + Resume Transformation**

---

## Part 1: The Complete Technical Crash Course (Basics to Master in 7 Days)

### Day 1: Inside the AI Rack — Physical Architecture & Scale-Up Fabrics
* **What is an AI Rack?**
  * **Legacy HGX H100/H200 Rack**: 4 to 8 nodes (8 GPUs per node = 32–64 GPUs/rack), air or hybrid liquid-cooled (~40–60 kW/rack). The NVLink domain stops at the **8-GPU node boundary** (900 GB/s bidirectional NVLink 4). Crossing nodes inside the same rack drops bandwidth 9x to 400Gbps (50 GB/s unidirectional) InfiniBand/RoCEv2 NICs!
  * **NVIDIA GB200 / VR200 NVL72 Rack**: 18 compute trays (each with 2 Grace CPUs + 4 Blackwell GPUs = 72 GPUs) + 9 NVLink switch trays connected via a **5,000+ coaxial copper cable blind-mate backplane**, operating at **120–132 kW per rack** with 100% direct-to-chip liquid cooling (CDU manifolds). All **72 GPUs act as a single 130 TB/s NVLink 5 domain** (1.8 TB/s bidirectional / 900 GB/s unidirectional per GPU).
  * **AMD MI300X / MI350X Rack**: 8x GPUs per node connected via full-mesh **Infinity Fabric (xGMI)** with 192 GB–288 GB HBM3/HBM3e per GPU (massive memory capacity advantage for inference).
* **NUMA, PCIe & GPUDirect**:
  * Why `nvidia-smi topo -m` is the first command every perf engineer runs: verifying whether a GPU and its paired ConnectX-7/8 RDMA NIC sit under the **same PCIe root complex / PCIe switch (`PIX` or `PXB`)** vs crossing the CPU socket interconnects (`SYS` / UPI / QPI).
  * **GPUDirect RDMA (GDR)**: Allows the NIC to DMA directly to/from GPU HBM via PCIe BAR1 Peer-to-Peer without bouncing through host CPU DRAM (which ruins latency and saturates CPU memory channels).
  * **PCIe ACS (Access Control Services)**: If ACS is enabled in BIOS/IOMMU, PCIe P2P packets get redirected up to the CPU Root Complex, collapsing GPUDirect RDMA bandwidth by 50–70%.

### Day 2: Scale-Out Networking & Collective Communication (`NCCL` / `RCCL`)
* **Rail-Optimized Fat-Tree Topologies**:
  * Every GPU $i \in \{0..7\}$ in a node has a dedicated 400G/800G NIC connected to **Rail $i$ Leaf (ToR) switch**. Rank communication between GPU 0 on Node A and GPU 0 on Node B takes **1 switch hop** (`SameRailLeafSwitch`), whereas GPU 0 to GPU 3 crosses the Spine switches (`SpineFabricScaleOut`).
* **InfiniBand vs. RoCEv2 (RDMA over Converged Ethernet)**:
  * InfiniBand uses credit-based flow control at the link layer (lossless by design) and **SHARP** (in-network reductions).
  * RoCEv2 runs over UDP/IP/Ethernet and relies on **PFC (Priority Flow Control)** to prevent packet drops + **ECN (Explicit Congestion Notification)** and **DCQCN** to throttle senders. A bad NIC or receiver stall can trigger a **PFC Pause Storm** that freezes the entire leaf switch rail.
* **The Math of Collectives (`nccl-tests`)**:
  * **Algorithm Bandwidth (`algbw`)**: $\text{algbw} = \frac{S}{t}$, where $S$ is tensor size in bytes and $t$ is time.
  * **Bus Bandwidth (`busbw`)**: Normalizes `algbw` by the hardware traffic multiplier so you can compare directly against the physical NVLink/NIC spec:
    * `AllReduce`: $\text{busbw} = \text{algbw} \cdot \frac{2(N-1)}{N}$
    * `ReduceScatter` / `AllGather` / `AllToAll`: $\text{busbw} = \text{algbw} \cdot \frac{N-1}{N}$
  * **Ring vs. Tree vs. NVLS (NVLink SHARP)**:
    * **Ring**: $2(N-1)$ latency steps; optimal for huge tensors on small/medium rings, terrible latency at high $N$.
    * **Double Binary Tree**: $2\log_2(N)$ latency steps; scales much better across multi-rack clusters for medium messages.
    * **NVLS**: Uses NVSwitch hardware multicast + in-switch ALU reduction; reduces hop count to $O(1)$ inside the rack domain.

### Day 3: GPU Microarchitecture, Roofline Model & MFU
* **The Roofline Equations**:
  * Every GPU has Peak Compute $\Peak_{\text{FLOPS}}$ (e.g., H100 SXM = $989\text{ TFLOPS}$ BF16; B200 = $2250\text{ TFLOPS}$ BF16) and Peak HBM Bandwidth $\Peak_{\text{BW}}$ (H100 = $3.35\text{ TB/s}$; B200 = $8.0\text{ TB/s}$).
  * **Machine Ridge Point**:
    $$\text{AI}_{\text{ridge}} = \frac{\Peak_{\text{FLOPS}}}{\Peak_{\text{BW}}} \quad (\approx 295\text{ FLOPs/Byte on H100}, \approx 281\text{ FLOPs/Byte on B200})$$
  * For a matrix multiplication $C_{M \times N} = A_{M \times K} B_{K \times N}$ in BF16 (2 bytes/element):
    $$\text{Arithmetic Intensity (AI)} = \frac{2MNK}{2(MK + KN + MN)}$$
  * When $M$ is small (e.g., LLM autoregressive decode batch size $M \ll 295$), $\text{AI} \approx M$, so the kernel is strictly **HBM Bandwidth-Bound**. When $M$ is large (training or prefill $M \ge 512$), $\text{AI} > \text{AI}_{\text{ridge}}$, so the kernel is **Tensor Core Compute-Bound**.
* **Model FLOPs Utilization (MFU)**:
  $$\text{MFU} = \frac{6 \cdot P_{\text{active}} \cdot \text{Tokens/sec}}{\text{Num\_GPUs} \cdot \Peak_{\text{FLOPS/GPU}}}$$
  * World-class clusters achieve **45%–62% MFU** on dense models and **35%–58% MFU** on large MoEs.

### Day 4: Mapping 4D Parallelism to Physical Racks
* **Tensor Parallelism (TP)**: Splits weight matrices within each layer; requires `AllReduce` (or `ReduceScatter` + `AllGather` with Sequence Parallelism) **4 times per Transformer block** (forward + backward) on the critical path. **Rule**: TP must NEVER cross an NVSwitch boundary (TP $\le 8$ on H100/MI300X; up to TP $= 16..72$ on GB200 NVL72).
* **Expert Parallelism (EP)**: Places different MoE experts on different GPUs; requires `AllToAll` dispatch and combine every layer. Because GB200 NVL72 has 72 GPUs in one NVLink domain, you can run **EP=72 inside a single rack at 900 GB/s per GPU** instead of bottlenecking on 50 GB/s RoCE/IB!
* **Pipeline Parallelism (PP)** & **Context Parallelism (CP)**: Point-to-point activation transfers (`P2P Send/Recv`) or Ring Attention across racks.
* **FSDP / ZeRO-3 (DP)**: Shards parameters, gradients, and optimizer states across racks; overlaps `AllGather` (pre-fetching next layer's weights) and `ReduceScatter` (sharding gradients) with backward compute streams.

### Day 5: Inference Rack Disaggregation (Prefill vs. Decode & KV-Cache RDMA)
* Why modern serving engines (**vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo**) disaggregate **Prefill racks** from **Decode racks**:
  * **Prefill** is compute-bound (high TTFT sensitivity, saturates Tensor Cores, causes sharp power spikes).
  * **Decode** is HBM-bandwidth-bound and interconnect-latency-bound (high TPOT sensitivity, benefits from large EP/TP across NVL72 or high-HBM MI300X racks).
  * **KV-Cache Transfer (NIXL / Mooncake)**: Streams KV blocks from Prefill GPU HBM to Decode GPU HBM over GPUDirect RDMA asynchronously.

### Day 6: Rack Bring-Up, Telemetry, Stragglers & Power/Cooling Transients
* **Why One Slow GPU Stalls 10,000 GPUs**: Synchronous collectives (`AllReduce`, `AllToAll`) move at the speed of the **slowest rank** in the communicator.
* **The Top 4 Rack Failure Modes**:
  1. **Thermal Clock Throttling**: Air bubble or debris in a liquid cooling cold-plate manifold causes one GPU's HBM temp to cross $95^\circ\text{C}$, dropping SM clock from 1980 MHz to 1100 MHz (+39% step time barrier tax).
  2. **NVLink CRC Flit Errors**: Blind-mate backplane connector dust/misalignment causes link retraining and silently drops NVLink lanes.
  3. **RoCEv2 PFC Storms**: Optical transceiver flaps or ECMP hash collisions cause PFC pause frames to backpressure the leaf switch.
  4. **Power Synchronous Transients ($dI/dt$)**: When 72 GPUs enter a GEMM phase simultaneously and drop to idle during an exposed `AllReduce`, rack power swings by 60+ kW in milliseconds.

### Day 7: Profiling Workflow (`nsys`, `ncu`, PyTorch Kineto, DCGM) & Interview Mastery
* Hands-on trace collection on a cloud GPU (RunPod / Lambda $1.50/hr H100 or A100 instance) using `cloud_gpu_benchmarks/gpu_roofline_and_nccl_probe.py`.

---

## Part 2: Exact Resume Upgrade (Replacing `MinuteFlow`/`ContextBridge` with `RackScale-Perf`)

Use this drop-in project block and updated AWS/Skills framing on your resume for **AI Performance / GPU Systems / Rack Infrastructure** applications:

### Updated Project Block (Drop-In Replacement for Project #1)
> **RackScale-Perf | Rust, Python, PyTorch, CUDA/NCCL, Prometheus**
> * Built a rack-scale AI topology scheduler and collective simulator for **NVIDIA GB200 NVL72** and **HGX H100** clusters, mapping 4D parallelism (**TP, EP, PP, FSDP**) onto NVSwitch domains and rail-optimized **InfiniBand/RoCEv2** fabrics to eliminate cross-node TP spills and raise simulated **Llama-3-70B MFU from 22.6% to 62.8%**.
> * Modeled **NCCL/RCCL** `AllReduce`, `ReduceScatter`, `AllGather`, and MoE `AllToAll` collectives across **Ring**, **Double Binary Tree**, and **NVLS SHARP** algorithms (`algbw`/`busbw`), pairing analytical Roofline models with empirical **PyTorch/CUDA** GEMM sweeps across HBM-bound decode and Tensor-Core-bound prefill regimes.
> * Implemented a sub-millisecond **DCGM & RDMA** rack telemetry analyzer in Rust using Median Absolute Deviation (MAD) outlier isolation and **Prometheus** metrics to detect liquid-cooling thermal throttling stragglers, NVLink CRC flit degradation, and **RoCEv2 PFC** pause storms before synchronous barriers stall training.

### Subtle Reframing of Your AWS Bullet #1 & #3 (100% Truthful to Your Work, Tuned for AI Infra)
* Keep your AWS bullets intact, and emphasize in interviews: *"At AWS, I built a Rust session-placement engine doing bin-packing, spread, and fault-domain affinity scheduling with Criterion and flamegraphs at sub-microsecond latency—which is the exact same combinatorial optimization problem as mapping TP/EP/DP ranks onto NVLink domains and InfiniBand rails."*

### Updated Technical Skills Line for AI Infra Roles
* **Systems & AI Infrastructure**: Rust, C++20, Go, Python, CUDA/PyTorch Profiler, NCCL/RCCL Collectives, Roofline & MFU Modeling, NVLink/NVSwitch, InfiniBand/RoCEv2 RDMA, NUMA/PCIe Topology, DCGM, Prometheus, Grafana, Linux/Kubernetes (EKS)

---

## Part 3: 7-Day LinkedIn "Build-in-Public" Content Plan (High-Signal Engineering Posts)

Hiring managers and Principal Engineers at **NVIDIA, Meta, AWS Annapurna, AMD, CoreWeave, xAI, Together AI, and Modular** actively read technical benchmark posts on LinkedIn. Do **not** post generic AI hype—post **numbers, diagrams, terminal outputs, and hardware mechanics**.

### Post 1 (Day 1–2): Why Naive GPU Scheduling Destroys LLM Training MFU (62.8% vs 22.6%)
* **Hook**:
  > What happens when a single Tensor Parallel (TP=8) group spills across an 8-GPU NVSwitch boundary onto a 400Gbps RDMA NIC rail?
  >
  > I built `rackscale-perf` in Rust to benchmark 4D parallelism placement (`TP`, `EP`, `PP`, `FSDP`) across multi-rack NVIDIA HGX H100 and GB200 NVL72 topologies.
* **Body**:
  * Share the exact terminal output from `cargo run --bin rackperf`:
  * Show how dropping TP bottleneck bandwidth from **450 GB/s (NVSwitch)** to **47.5 GB/s (RoCEv2/IB rail)** increases exposed communication per step from **2.8s to 42.2s**, collapsing **Llama-3-70B MFU from 62.8% to 22.6%**.
  * Explain how GB200 NVL72 changes the game for Mixture-of-Experts (`AllToAll` EP=36/72) by extending the 900 GB/s unidirectional NVLink 5 domain across all 18 compute trays in the rack.
* **Call to Action**: Link to your GitHub repo (`github.com/ThilakShekharShriyan/rackscale-perf` or `PERFENGINEERING`) and mention you are wrapping up your MS CS at SJSU after 2.5+ years in distributed systems (Couchbase + AWS) and exploring AI Infrastructure & Performance Engineering roles.

### Post 2 (Day 3–4): Ring vs. Double Binary Tree vs. NVLink SHARP (NVLS) Collectives & Roofline Math
* **Hook**:
  > Why does `nccl-tests` report both `algbw` and `busbw`—and why does NVLink SHARP (NVLS) cut 72-GPU AllReduce latency in half?
* **Body**:
  * Walk through the $\frac{2(N-1)}{N}$ bus bandwidth derivation and show your simulator + empirical PyTorch GEMM Roofline sweep showing the exact transition from **HBM Bandwidth-Bound** ($M < 295\text{ FLOPs/Byte}$) during autoregressive Decode to **Tensor-Core Compute-Bound** during Prefill.

### Post 3 (Day 5–6): How One Thermally Throttled GPU Stalls a 72-GPU Liquid-Cooled Rack (+39.5% Barrier Tax)
* **Hook**:
  > In synchronous distributed training, your 100kW+ GPU rack runs at the speed of its slowest rank.
* **Body**:
  * Walk through how you built the `rack-telemetry` crate in Rust to ingest DCGM & RDMA counters (`SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `NVLINK_CRC_FLIT_ERROR`, `RoCEv2 PFC pause frames`) and use Median Absolute Deviation (MAD) to isolate cold-plate thermal throttling ($97.2^\circ\text{C}$ HBM), degraded NVLink lanes, and PFC congestion storms in sub-millisecond time.

### High-ROI Outbound Strategy After Posting
1. Search LinkedIn for:
   * `"AI Performance Engineer" OR "GPU Systems Engineer" OR "System Software Engineer" OR "Scale-Up" OR "NCCL" OR "NVL72"` at **NVIDIA (Santa Clara), AMD (San Jose), Meta (MPK/Sunnyvale), AWS Annapurna Labs (Cupertino/Santa Clara), CoreWeave, xAI (Palo Alto), Cerebras (Sunnyvale)**.
2. Send a concise connection note referencing your repo:
   > *"Hi [Name], I'm an ex-AWS (Rust compute placement) and Couchbase (2.5 yrs distributed systems) engineer finishing my MS at SJSU. I've been building a Rust toolkit modeling GB200 NVL72 topology placement, NCCL/NVLS collectives, and DCGM straggler telemetry ([GitHub link]). Would love to connect and follow your work on [Team/Area]."*

---

## Part 4: LinkedIn Headline + About (Paste-Ready)

### Headline (update this too — it is searched more than About)
`AI Infra & Performance Engineer | Rust · CUDA · Linux | GPU/Rack Systems, KV-Cache, Collectives | SJSU MS CS · AWS · Couchbase`

### About (2,348 characters — under LinkedIn’s 2,600 limit)

I work on AI infrastructure performance: rack-scale GPU systems, CUDA/Rust systems software, and the bare-metal path from NUMA/PCIe and NVLink up through NCCL collectives, scheduling, and KV-cache movement across heterogeneous hardware.

I’m a Computer Science master’s student at San José State University and a software engineer focused on performance engineering for cloud and AI infrastructure — distributed compute, databases, and GPU/rack systems.

At AWS, I build Rust infrastructure for distributed compute fleets: low-latency session placement, async execution, zero-copy telemetry, and reliable placement state. The same problems show up on AI racks — packing ranks onto NVLink domains, isolating fault domains, and keeping the hot path under a microsecond. Previously at Couchbase, I shipped cloud infrastructure on AWS, Kubernetes/EKS, Docker, and Linux, and owned Prometheus, Grafana, and Datadog observability for capacity and bottleneck analysis on live traffic.

I’m going deep on performance from the metal up: Linux NUMA pinning, PCIe topology and GPUDirect RDMA, HBM vs Tensor Core Roofline, NCCL/RCCL AllReduce and MoE AllToAll, DCGM/RDMA telemetry, and topology-aware mapping of TP/EP/PP/FSDP onto trays and NIC rails. I contribute CUDA and Rust systems code on host and device paths — kernels, async runtimes, and telemetry — and I work on KV-cache paging, reuse, and RDMA transfer across heterogeneous devices (NVIDIA GPUs, AMD accelerators, CPU DRAM/HBM, and ASIC runtimes) so prefill and decode can be disaggregated without stalling TTFT or TPOT.

I care about the failures that only appear at rack scale: a thermally throttled GPU stalling a synchronous collective, NVLink CRC degrading a scale-up domain, RoCEv2 PFC storms on a rail, and power/thermal transients on liquid-cooled NVL72-class racks.

Projects include RackScale-Perf (Rust topology scheduler, collective/Roofline simulator, DCGM-style straggler isolation), a hybrid-cloud HPC control plane for transformer/ASIC workflows, and AURORA, a CPU AI runtime optimizer driven by Linux perf/PMU telemetry.

Core stack: Rust, CUDA, C/C++, Python, Go, Linux, NCCL, NVLink/PCIe/RDMA, AWS, Kubernetes, Prometheus, and databases.

Open to connecting with people building AI infrastructure, GPU systems, performance engineering, cloud platforms, and distributed systems.
