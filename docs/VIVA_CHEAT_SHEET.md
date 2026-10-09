# Viva & Presentation Defense Cheat Sheet

This guide prepares your team to answer questions confidently during your oral presentation and viva.

---

### Q1: What problem did your team solve, and which algorithm did you use?
**Answer:**  
We solved **Problem Q14: Recommendation Network**. We implemented an **Item-Based Collaborative Filtering Recommender on Bipartite Graphs** in pure C and parallelized it with OpenMP. The system precomputes item L2-norms, calculates an Item-Item Cosine Similarity Matrix, and predicts Top-$K$ recommendations for every user based on items they previously rated.

---

### Q2: Why did you choose Item-Based Collaborative Filtering over User-Based?
**Answer:**  
1. **Stability & Invariance:** In real-world systems, items change much less frequently than user behaviors, making item similarity matrices more reusable.
2. **Computational Suitability:** Item similarity computations are embarrassingly parallel across item rows.
3. **Graph Bipartite Structure:** Allows using an inverted Item-User CSR index to compute sparse dot products in $O(\text{co-rated users})$ time instead of scanning unrated pairs.

---

### Q3: What data structures did you use, and why?
**Answer:**  
We used **Compressed Sparse Row (CSR)** matrices for two key representations:
1. `user_item_csr`: Efficiently queries all items rated by user $u$.
2. `item_user_csr` (Inverted Index): Efficiently queries all users who rated item $i$.  
**Why:** Real recommendation datasets are $>95\%$ sparse. Dense 2D arrays ($U \times I$) would waste gigabytes of RAM and suffer massive cache misses. CSR reduces space to $O(\text{Non-Zero Interactions})$ and ensures contiguous cache-line memory access.

---

### Q4: How did you parallelize the algorithm with OpenMP?
**Answer:**  
We parallelized two compute-intensive stages:
1. **Stage 2 (Item Similarity Matrix):** Distributed item rows across threads using `#pragma omp for schedule(dynamic, 16)`. Each thread computes upper-triangular similarity for its assigned item, followed by a parallel symmetrization step.
2. **Stage 3 (User Top-K Recommendation):** Distributed users across threads using `#pragma omp for schedule(guided, 8)` to predict scores and extract top-$K$ items per user.

---

### Q5: Why did you use `schedule(dynamic)` and `schedule(guided)` instead of `schedule(static)`?
**Answer:**  
Because user activity and item popularity in bipartite graphs follow a **Power-Law (Zipf-like) distribution** — a few popular items/users have thousands of ratings while others have very few. `schedule(static)` creates severe **workload imbalance** where threads assigned popular items take much longer. `schedule(dynamic)` and `guided` dynamically assign chunks of work as threads become free, maximizing CPU utilization.

---

### Q6: How did you prevent race conditions and false sharing without slow mutex locks?
**Answer:**  
We allocated a dedicated **`ThreadScratchpad` structure per thread** outside the parallel region. Each thread writes exclusively to its private accumulator arrays (`dot_acc`, `score_acc`, `rated_mask`). Because no two threads write to the same memory addresses or cache lines, we achieved **100% lock-free execution with zero atomic overhead and zero false sharing**.

---

### Q7: What performance results did you achieve?
**Answer:**  
On a 16-core CPU:
- **Sequential Execution Time:** 0.377 s (for 3,000 users, 1,000 items)
- **16-Thread OpenMP Time:** 0.038 s
- **Speedup Achieved:** **9.92x**
- **Parallel Efficiency:** **98.2% on 2 cores**, scaling to **62.0% on 16 cores**.

---

### Q8: Why does Parallel Efficiency decrease at 16 threads (from 98% to 62%)?
**Answer:**  
This is governed by **Memory Bandwidth Saturation** and **Amdahl's Law**. At 16 threads, all cores concurrently stream large arrays through the shared L3 cache and memory bus, causing memory channel contention. Additionally, thread synchronization barriers at loop ends introduce small constant overheads.

---

### Q9: How did you verify the correctness of the parallel program?
**Answer:**  
We built an automated verification module (`verify.c`) that compares the sequential and parallel outputs:
- **Top-$K$ Recommendation Overlap:** **100.00% exact match**
- **Mean Absolute Error (MAE):** **0.00e+00**
- **Root Mean Squared Error (RMSE):** **0.00e+00**

---

### Q10: How did you use AI / LLM during the project, and can you give an example of an AI suggestion you modified or rejected?
**Answer:**  
We used the LLM as an engineering assistant for algorithm exploration, OpenMP syntax, and debugging.  
**Critical Evaluation Example:** The LLM initially suggested using a shared global score accumulator array protected by `#pragma omp atomic` or `#pragma omp critical`. We evaluated this suggestion and found that heavy atomic contention caused thread serialization, making it **3x slower than sequential code**. We rejected the suggestion and replaced it with **pre-allocated thread-private scratchpads**, which resolved the bottleneck and unlocked our **9.92x speedup**.
