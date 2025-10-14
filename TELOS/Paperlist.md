## FineMem: Breaking the Allocation Overhead vs. Memory Waste Dilemma in Fine-Grained Disaggregated Memory Management
**Problem:** 
Remote Memory Allocation can't achieve both high performance and fine-grained. 

**Design:** 
1. Use Memory Windows(MWs) instead of Memory Regions(MRs) to reduce the latency of registration. We pre-register the memory pool as a huge memory region, and divide it into lots of memory windows, then we could manage them easily with rkeys for safety.
2. Two-level bitmaps to allocate the memory(seems like multi-levels page table.)
3. A more CAS operation to achieve crash consistency(Like the transaction in Database).

**Evaluation:**
Metrics：
1. Latency(Contention Control contributes the most 44%).
2. Relative Runtime(performance).
3. Utilization
4. Throughput

Systems：
- User-kernel
- KV store
- Swap

**Conclusion:**
Based on one-sided RDMA to allocate the remote memory, FineMem reduces latency, providing the protection with its designs.

**Learned:**
- YCSB: Yahoo! Cloud Serving Benchmark, 分为ABCD 4种，每一种benchmark具有不同的特征。 
- 将这种要保证consistency的操作可以和数据库中transaction的思想蕴含在一起，做到异步恢复。

## Deterministic Client: Enforcing Determinism on Untrusted Machine Code
**Problem:**
Determinism in sandbox with low startup overhead is essential.

**Design:**
Deploy DeCl in both x86-64 and arm arch. 
1. Machine Code: use align-bundles with binary code insertion, splitting blocks of code.
2. Metering: Branch-based and Timer-based. Branch need gas detection and some optimization. Timer only need preempt based on gas count.
3. Integration of LFI: the need of position-oblivious code.

**Evaluation:**
The most important figure is Figure 7 and Figure 8 but with small size, which represent the overhead of time and throughput of DeCl.

**Conclusion**
DeCl is a software sandboxing system, enforcing deterministic and metered in x86 and arm64, with high performance and feasibility. ==The first time to use SFI in Deterministic rather than memory isolation.==

**Learned:**
- Present the important figure in a eye-catching area.
- Before read the paper, ensure you have some prerequisite or to read the book about this field.

## Extending Applications Safely and Efficiently
**Problem:**
**Design:**
**Evaluation:**
**Conclusion:**
**Learned:**
