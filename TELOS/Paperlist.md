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

**Design:**

**Evaluation:**

**Conclusion**

**Learned:**