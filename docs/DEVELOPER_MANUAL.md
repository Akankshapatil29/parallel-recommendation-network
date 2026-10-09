# Developer Manual: Architecture & Implementation Guide

**Project:** High-Performance Parallel Recommendation Engine (OpenMP)  
**Language:** Pure C (C99 / C11 Standard)  
**Parallel Model:** Shared Memory Multi-Processing (OpenMP 4.5+)  
**Target Audience:** Software Engineers, Researchers, and Evaluators  

---

## 1. Architectural Overview

The recommendation engine is an **Item-Based Collaborative Filtering (IBCF)** system optimized for sparse user-item bipartite graphs. It computes pairwise item cosine similarities and generates personalized Top-$K$ recommendations for $N$ users over $M$ items.

```
+--------------------------------------------------------------------------------+
|                                DATA INGESTION                                  |
|  - Parse CSV / MovieLens format into COO (Coordinate List)                    |
|  - Sort & Compress into Dual CSR Formats (User-Item & Item-User Transpose)    |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                        STAGE 1: ITEM NORM PRECOMPUTATION                       |
|  - Compute ||v_i|| = sqrt(sum(r^2)) for all items i in [0, M-1]                |
|  - OpenMP parallelized across item rows                                       |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                   STAGE 2: PAIRWISE ITEM SIMILARITY MATRIX                     |
|  - Dense Upper-Triangular computation using Item-User Inverted Index          |
|  - Dot product: sum_{u in U_i cap U_j} (r_{u,i} * r_{u,j})                     |
|  - Sim(i, j) = DotProduct / (||v_i|| * ||v_j||)                               |
|  - OpenMP parallelized across outer row loop (i)                               |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                  STAGE 3: TOP-K USER RECOMMENDATION & RANKING                  |
|  - Thread-private scratchpads to eliminate race conditions & false sharing     |
|  - Predict score: P_{u,j} = sum_{i in Rated(u)} (r_{u,i} * Sim(i,j))           |
|  - Extract Top-K items using Min-Heap / Selection sort                         |
|  - OpenMP parallelized across users (u)                                       |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                 STAGE 4: VERIFICATION & STATISTICAL METRICS                    |
|  - Top-K Set Overlap (%), MAE, RMSE, Max Discrepancy                           |
|  - Execution Time, Speedup (S_p = T_1 / T_p), Parallel Efficiency (E_p)        |
+--------------------------------------------------------------------------------+
```

---

## 2. Directory and File Structure

```
recommendation_network/
├── include/
│   ├── dataset.h          # CSR, COO, and Dataset structs & prototypes
│   ├── recommender.h      # Sequential & OpenMP solver prototypes & structs
│   └── timer.h            # High-precision OpenMP wall-clock timer wrapper
├── src/
│   ├── dataset.c          # Dataset loaders, synthetic graph generator, CSR builders
│   ├── sequential.c       # Single-threaded baseline reference solver
│   ├── parallel_omp.c     # Multi-threaded OpenMP solver (thread scratchpads)
│   ├── verify.c           # Correctness verification engine (Overlap, MAE, RMSE)
│   └── main.c             # Interactive CLI wizard & benchmark suite
├── data/
│   ├── ml-1m/             # MovieLens 1M dataset (ratings.dat, movies.dat)
│   └── sample_ratings.csv # 120k ratings benchmark CSV
├── benchmark/
│   ├── visualizer.html    # Interactive Chart.js scaling dashboard
│   └── run_experiments.bat# Batch scaling harness
├── docs/
│   ├── USER_MANUAL.md     # Step-by-step user guide
│   ├── DEVELOPER_MANUAL.md# Technical architecture guide (this document)
│   ├── PROJECT_REPORT.md  # Comprehensive academic project report
│   ├── LLM_USAGE_LOG.md   # Prompt engineering log & design decisions
│   └── VIVA_CHEAT_SHEET.md# Oral defense Q&A reference
├── build.bat              # One-click Windows build script
├── Makefile               # Cross-platform GNU Make configuration
└── README.md              # Project landing page
```

---

## 3. Data Structures & Memory Layout

### 3.1 Compressed Sparse Row (CSR) Representation
To minimize memory overhead for sparse rating matrices ($>95\%$ sparsity), interactions are represented as CSR:

```c
typedef struct {
    int num_rows;     // Number of rows (e.g., N users)
    int num_cols;     // Number of columns (e.g., M items)
    int nnz;          // Total number of non-zero entries (E ratings)
    int *row_ptr;     // Prefix sum offsets (size: num_rows + 1)
    int *col_idx;     // Column indices (size: nnz)
    float *values;    // Rating values (size: nnz)
} CSRMatrix;
```

### 3.2 Dual CSR Representation (`Dataset`)
The system builds two complementary CSR structures at startup:
1. `user_item_csr`: Maps $\text{User} \rightarrow \text{Rated Items}$. Used in Stage 3 for fast user history lookup.
2. `item_user_csr`: Transposed index mapping $\text{Item} \rightarrow \text{Users who rated it}$. Used in Stage 2 for fast sparse intersection dot-products.

```c
typedef struct {
    int num_users;
    int num_items;
    int num_ratings;
    CSRMatrix user_item_csr;
    CSRMatrix item_user_csr;
    char **item_names;  // Optional title strings for items
} Dataset;
```

---

## 4. Parallelization Strategy & Optimization Techniques

### 4.1 Stage 1: Item Norm Precomputation
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < num_items; i++) {
    int start = item_csr->row_ptr[i];
    int end = item_csr->row_ptr[i + 1];
    float sum_sq = 0.0f;
    for (int idx = start; idx < end; idx++) {
        sum_sq += item_csr->values[idx] * item_csr->values[idx];
    }
    norms[i] = sqrtf(sum_sq);
}
```
* **Thread Safety:** Each iteration writes to a distinct `norms[i]` index. Zero race conditions.
* **Scheduling:** `schedule(static)` distributes balanced chunks with zero runtime scheduling overhead.

---

### 4.2 Stage 2: Pairwise Cosine Similarity Computation
```c
#pragma omp parallel for schedule(dynamic, 16)
for (int i = 0; i < num_items; i++) {
    for (int j = i + 1; j < num_items; j++) {
        // Two-pointer intersection of sorted user lists
        float dot = compute_dot_product(i, j, item_csr);
        float sim = (norms[i] > 0 && norms[j] > 0) ? (dot / (norms[i] * norms[j])) : 0.0f;
        similarity_matrix[i * num_items + j] = sim;
        similarity_matrix[j * num_items + i] = sim; // Symmetric
    }
}
```
* **Scheduling:** `schedule(dynamic, 16)` balances the triangular loop workload ($j = i+1 \dots M-1$), preventing early thread starvation.
* **Symmetry Exploitation:** Only upper triangle $(i < j)$ is calculated, halving required operations.

---

### 4.3 Stage 3: Top-K Recommendation via Thread-Private Scratchpads

#### Problem: Race Conditions & False Sharing in Shared Accumulators
A naive approach would allocate a global `predicted_scores[num_items]` array updated by multiple threads, requiring expensive atomic locks and causing cache line bouncing (false sharing).

#### Solution: Thread-Private Scratchpads
Each OpenMP thread allocates its own private scratchpad before entering the loop:

```c
typedef struct {
    float *scores;      // Size: num_items
    bool *rated_mask;   // Size: num_items
    int *rated_items;   // Active buffer
} ThreadScratchpad;
```

```c
#pragma omp parallel
{
    ThreadScratchpad pad;
    allocate_scratchpad(&pad, num_items);

    #pragma omp for schedule(guided)
    for (int u = 0; u < num_users; u++) {
        // 1. Mark user's rated items in pad.rated_mask
        // 2. Accumulate scores into pad.scores
        // 3. Extract Top-K into results->recs[u * top_k]
        // 4. Reset pad for next user iteration
    }

    free_scratchpad(&pad);
}
```
* **Benefits:** 100% lock-free, zero synchronization overhead, zero false sharing, linear cache efficiency.

---

## 5. Computational Complexity Analysis

| Stage | Sequential Time Complexity | Parallel Time Complexity ($P$ Threads) | Space Complexity |
| :--- | :--- | :--- | :--- |
| **CSR Construction** | $O(E \log E)$ | $O(E \log E)$ (Preprocessing) | $O(N + M + E)$ |
| **Stage 1 (Norms)** | $O(E)$ | $O(E / P)$ | $O(M)$ |
| **Stage 2 (Similarity)** | $O(M^2 \cdot \bar{D}_u)$ | $O(\frac{M^2 \cdot \bar{D}_u}{P})$ | $O(M^2)$ |
| **Stage 3 (Top-K Ranking)**| $O(N \cdot (\bar{D}_i \cdot \bar{K}_{\text{sim}} + M \log K))$ | $O(\frac{N \cdot (\bar{D}_i \cdot \bar{K}_{\text{sim}} + M \log K)}{P})$ | $O(P \cdot M + N \cdot K)$ |

*(Where $N = \text{Users}, M = \text{Items}, E = \text{Ratings}, P = \text{Threads}, \bar{D} = \text{Average Degree}, K = \text{Top-K Depth}$).*

---

## 6. Verification and Error Analysis Protocol

The verification engine in `src/verify.c` calculates:
1. **Top-K Jaccard Overlap:**
   $$\text{Overlap} = \frac{\sum_{u=0}^{N-1} |R_{\text{seq}}(u) \cap R_{\text{par}}(u)|}{N \cdot K} \times 100\%$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{|R|} \sum_{(u,i) \in R} |S_{\text{seq}}(u,i) - S_{\text{par}}(u,i)|$$
3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{|R|} \sum_{(u,i) \in R} (S_{\text{seq}}(u,i) - S_{\text{par}}(u,i))^2}$$
4. **Max Discrepancy:**
   $$\text{MaxDiff} = \max_{(u,i) \in R} |S_{\text{seq}}(u,i) - S_{\text{par}}(u,i)|$$

A test is accepted if $\text{Overlap} \ge 99.0\%$ and $\text{MaxDiff} \le 10^{-4}$.

---

## 7. Compilation Flags and GCC Optimization

When building `recommender.exe`, the following flags are used:
* `-O3`: Highest optimization level (vectorization, loop unrolling, inline functions).
* `-fopenmp`: Enables OpenMP multi-threading pragmas and runtime linking.
* `-static`: Embeds MinGW runtime and OpenMP DLLs into a standalone executable.
* `-lm`: Links the standard C math library (`sqrt`, `fabs`).
