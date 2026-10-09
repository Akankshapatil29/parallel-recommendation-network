# LLM Usage Log & AI-Assisted Engineering Record

**Project Title:** Parallel Recommendation Network on Bipartite Graphs  
**Problem Assigned:** Q14. Recommendation Network  
**Language & Runtime:** C (C99/C11) + OpenMP Multi-Threading  

---

## 1. LLM Prompt Documentation Table

The table below records the primary prompts used during project development, how the AI outputs were utilized, and the engineering verification steps performed by the team.

| Sr. No. | Purpose | Prompt Used | How the Response Was Used | Modification / Verification |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Algorithm Understanding** | *"Explain how Item-Based Collaborative Filtering works on bipartite user-item graph interaction matrices, including cosine similarity formulas and top-K predicted rating equations."* | Used the mathematical formulation to design the 3-stage pipeline (L2-norm precomputation, item-item similarity calculation, and top-K predicted score ranking). | Verified against standard literature on Recommender Systems (Sarwar et al., 2001) and manual numerical calculations on a small 3x3 test matrix. |
| **2** | **Data Structure Design** | *"What is the most cache-efficient C data structure for sparse bipartite graph interactions with non-uniform user ratings?"* | Adopted the Compressed Sparse Row (CSR) structure for both User-Item interactions and inverted Item-User index. | Implemented custom `CSRMatrix` in pure C (`row_ptr`, `col_idx`, `values`), removing all dynamic reallocation overhead during traversal. |
| **3** | **Parallelization Strategy** | *"How to parallelize item similarity computation and top-K recommendation generation in OpenMP without causing race conditions or false sharing?"* | Guided loop chunking and OpenMP directives: `#pragma omp parallel for schedule(dynamic, 16)` for similarity and `#pragma omp parallel for schedule(guided, 8)` for user ranking. | Created pre-allocated thread-private scratchpad structs (`ThreadScratchpad`) to completely avoid mutex lock contention. |
| **4** | **Debugging & Correctness** | *"Why would OpenMP loop produce different Top-K recommendations compared to sequential execution when updating score accumulators?"* | Identified race condition caused by concurrent writes to shared score arrays when multiple threads process user recommendations simultaneously. | Resolved by isolating score accumulators into thread-local buffers. Verified with automated numerical test (`verify.c`) achieving 100% Top-K overlap and 0.00 MAE. |
| **5** | **Optimization & Scaling** | *"How can we avoid O(N^2) memory footprint and lock contention when calculating symmetric item similarity matrices in OpenMP?"* | Implemented upper-triangular computation with a lock-free symmetrization pass, plus power-of-two memory alignment. | Benchmark comparisons showed speedup increased from 4.2x to **9.92x on 16 threads**. |

---

## 2. Critical Evaluation of LLM Output

> **Mandatory Assignment Requirement:** Document at least one instance where an LLM-generated suggestion was modified, rejected, or found unsuitable.

### Case Study: Rejection of Shared Global Accumulator & Mutex Locking

#### 1. What the LLM Suggested
During initial parallelization of Stage 3 (User Top-K Recommendation scoring), an LLM initially proposed using a single shared global array for score accumulation protected by `#pragma omp critical` or `#pragma omp atomic`:

```c
// Flawed LLM Initial Suggestion:
#pragma omp parallel for
for (int u = 0; u < num_users; u++) {
    // ...
    for (int j = 0; j < num_items; j++) {
        #pragma omp atomic
        global_score_acc[j] += sim * r_ui;
    }
}
```

#### 2. Why the Suggestion Was Evaluated
The suggestion aimed to conserve RAM by reusing a single accumulator array rather than allocating separate arrays for each thread.

#### 3. Why It Was Rejected / Modified
- **Massive Performance Degradation:** When 16 threads concurrently attempted to update `global_score_acc` using atomic operations or critical sections, severe thread serialization occurred. The execution was **3x slower than sequential execution** due to lock contention and cache line bouncing (false sharing).
- **Correctness Hazard:** Resetting `global_score_acc` between users created race conditions where one thread would wipe the accumulated values of an active thread.

#### 4. How the Final Decision Was Implemented & Verified
- **Our Fix:** We designed a `ThreadScratchpad` structure allocated **once** per thread outside the parallel region:
  ```c
  typedef struct {
      float *dot_acc;
      int *common_acc;
      int *touched_items;
      float *score_acc;
      float *sim_sum_acc;
      bool *rated_mask;
      int *candidate_items;
  } ThreadScratchpad;
  ```
- Each thread accesses only `pads[omp_get_thread_num()]`, ensuring **zero locking, zero atomic overhead, and zero false sharing**.
- **Verification:** Benchmarking confirmed an immediate speedup jump to **9.92x on 16 threads** with **100.00% exact numerical match** against the sequential baseline.
