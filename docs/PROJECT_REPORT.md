# Parallel Recommendation Network on Bipartite Graphs
## High-Performance Collaborative Filtering Using C and OpenMP

---

### Executive Summary
This project designs, implements, and evaluates a high-performance computational solution for generating Top-$K$ recommendations from large-scale user-item interaction datasets (bipartite graphs). We develop both an optimized **Sequential baseline in C** and a scalable **Parallel implementation using OpenMP multi-threading**. Using a lock-free parallelization strategy with thread-private memory scratchpads and Compressed Sparse Row (CSR) graph representations, our parallel implementation achieves a peak speedup of **9.92x on 16 CPU cores** with **100.00% exact numerical match** against the sequential baseline.

---

## 1. Problem Definition & Mathematical Formulation

### 1.1 Problem Statement (Q14)
Given a bipartite interaction graph $G = (U, I, E)$, where:
- $U = \{u_1, u_2, \dots, u_N\}$ is the set of $N$ users,
- $I = \{i_1, i_2, \dots, i_M\}$ is the set of $M$ items,
- $E = \{(u, i, r_{ui})\}$ is the set of observed interactions with explicit rating weight $r_{ui} \in [1.0, 5.0]$.

The computational objective is to predict missing interaction scores $\hat{r}_{uj}$ for all unobserved pairs $(u, j) \notin E$ and extract the **Top-$K$ recommended items** per user with highest predicted affinity.

### 1.2 Algorithmic Formulation (Item-Based Collaborative Filtering)
The pipeline is structured into three discrete stages:

#### Stage 1: Item L2-Norm Precomputation
For each item $i \in I$:
$$\|i\|_2 = \sqrt{\sum_{u \in \text{Users}(i)} r_{ui}^2}$$

#### Stage 2: Item-Item Cosine Similarity Matrix Computation
The similarity between items $i$ and $j$ is calculated across co-rated users:
$$\text{sim}(i, j) = \begin{cases} \frac{\sum_{u \in U_i \cap U_j} r_{ui} \cdot r_{uj}}{\|i\|_2 \cdot \|j\|_2} & \text{if } |U_i \cap U_j| \ge \tau \\ 0 & \text{otherwise} \end{cases}$$
where $\tau$ is the minimum common user threshold to suppress spurious high similarities from single isolated co-ratings.

#### Stage 3: Top-K Score Prediction & Ranking
For an active user $u$ and unrated candidate item $j$:
$$\hat{r}_{uj} = \frac{\sum_{i \in \text{Rated}(u)} \text{sim}(i, j) \cdot r_{ui}}{\sum_{i \in \text{Rated}(u)} |\text{sim}(i, j)|}$$
The top $K$ items with maximum predicted rating $\hat{r}_{uj}$ are sorted and output.

---

## 2. System Architecture & Data Structures

```
 +-------------------------------------------------------------+
 |                   Dataset Loader / Generator                |
 |  (COO Tuples -> User-Item CSR + Inverted Item-User CSR)     |
 +-------------------------------------------------------------+
                               |
                               v
 +-------------------------------------------------------------+
 |       Stage 1: Parallel Item L2-Norm Precomputation        |
 |             #pragma omp parallel for schedule(static)       |
 +-------------------------------------------------------------+
                               |
                               v
 +-------------------------------------------------------------+
 |     Stage 2: Parallel Item-Item Cosine Similarity Matrix    |
 |     - Lock-Free Row Distribution                            |
 |     - Dynamic Scheduling #pragma omp for schedule(dynamic)  |
 |     - Parallel Symmetrization Pass                          |
 +-------------------------------------------------------------+
                               |
                               v
 +-------------------------------------------------------------+
 |     Stage 3: Parallel Top-K Recommendation Generation       |
 |     - Thread-Private Accumulator Scratchpads                |
 |     - Guided Scheduling #pragma omp for schedule(guided)    |
 |     - In-Place Min-Heap Top-K Extraction                    |
 +-------------------------------------------------------------+
                               |
                               v
 +-------------------------------------------------------------+
 |        Verification Engine & Performance Analyzer           |
 |  (MAE = 0.00, Overlap = 100%, Speedup & Efficiency Tables)  |
 +-------------------------------------------------------------+
```

### 2.1 Compressed Sparse Row (CSR) Representation
To prevent $O(N \cdot M)$ memory wastage on sparse interaction matrices ($>95\%$ sparsity), interactions are stored as:
1. **User-Item CSR:** Rapidly traverses items rated by user $u$.
2. **Item-User Inverted CSR:** Rapidly traverses users who rated item $i$.

```c
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int *row_ptr;      // Size: num_rows + 1
    int *col_idx;      // Size: nnz
    float *values;     // Size: nnz
} CSRMatrix;
```

---

## 3. Parallelization Strategy (OpenMP)

### 3.1 Lock-Free Design & Private Scratchpads
To eliminate thread locking and false sharing, each worker thread is provisioned with a private `ThreadScratchpad`:
- `dot_acc`: Accumulates inner products for active item row.
- `score_acc` & `sim_sum_acc`: Accumulates predicted ratings for unrated candidate items.
- `rated_mask`: $O(1)$ lookup bitmask to filter previously rated items.

### 3.2 Dynamic & Guided Loop Scheduling
- **Stage 2 (Similarity Computation):** Item popularity in real-world graphs follows a power-law distribution. Using `schedule(dynamic, 16)` prevents thread stalling on densely connected items.
- **Stage 3 (User Recommendations):** Uses `schedule(guided, 8)` to balance work dynamically as remaining users decrease.

---

## 4. Experimental Results & Performance Evaluation

### 4.1 Benchmarking Environment
- **CPU:** 16-Core Multi-Threaded Processor
- **Compiler:** GCC 16.2.0 (WinLibs 64-bit UCRT) with `-O3 -fopenmp -static`
- **Dataset:** 3,000 Users, 1,000 Items, 120,501 Ratings (Sparsity: 95.98%)

### 4.2 Multi-Thread Scaling Results

| Configuration | Threads ($P$) | Similarity Time (s) | Top-10 Recs Time (s) | Total Time ($T_P$) | Speedup ($S$) | Parallel Efficiency ($E$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sequential** | 1 | 0.0230 s | 0.3540 s | 0.3770 s | 1.00x | 100.00 % |
| **OpenMP** | 1 | 0.0140 s | 0.3480 s | 0.3630 s | 1.04x | 103.86 % |
| **OpenMP** | 2 | 0.0180 s | 0.1740 s | 0.1920 s | 1.96x | 98.18 % |
| **OpenMP** | 4 | 0.0060 s | 0.0900 s | 0.0970 s | **3.89x** | **97.16 %** |
| **OpenMP** | 8 | 0.0040 s | 0.0490 s | 0.0540 s | **6.98x** | **87.27 %** |
| **OpenMP** | 12 | 0.0030 s | 0.0380 s | 0.0420 s | **8.98x** | **74.80 %** |
| **OpenMP** | 16 | 0.0040 s | 0.0340 s | 0.0380 s | **9.92x** | **62.01 %** |

### 4.3 Analysis of Bottlenecks
1. **Compute vs. Memory Bandwidth:** At 1 to 4 threads, speedup is near-linear ($E \approx 98\%$), indicating the problem is compute-bound. Beyond 8 threads, memory bus contention on shared L3 cache and DRAM channels introduces sub-linear scaling ($E = 62.0\%$).
2. **Amdahl's Law:** Non-parallelized sequential components (file I/O, scratchpad allocation) represent $<1\%$ of execution time, establishing a theoretical maximum speedup of $\sim 14.5\text{x}$ on 16 cores.

---

## 5. Correctness Verification

The parallel implementation was verified against the sequential baseline across all users:
- **Top-$K$ Overlap Accuracy:** **100.00%**
- **Mean Absolute Error (MAE):** **0.00e+00**
- **Max Absolute Discrepancy:** **0.00e+00**
- **Verification Status:** **PASS (Bit-exact match)**

---

## 6. Individual Team Contributions

| Member Name | Specific Contribution & Responsibilities |
| :--- | :--- |
| **Member 1** | Problem formulation, CSR data structure architecture, and sequential baseline implementation. |
| **Member 2** | OpenMP parallelization of item similarity and recommendation loops, thread-private scratchpad design. |
| **Member 3** | Correctness verification engine (`verify.c`), synthetic graph generator, and experimental benchmarking suite. |
| **Member 4** | Performance analysis, speedup/efficiency evaluation, LLM usage documentation, and project reporting. |
