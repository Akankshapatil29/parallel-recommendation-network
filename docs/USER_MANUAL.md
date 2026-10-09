# User Manual: Parallel Recommendation Network Engine

**Course / Project:** Parallel Programming & High-Performance Computing (OpenMP)  
**Target Audience:** Students, Evaluators, Professors, and End-Users  

---

## 1. Introduction

The **Parallel Recommendation Network Engine** is a high-performance C program accelerated with **OpenMP multi-threading**. It performs Item-Based Collaborative Filtering (Cosine Similarity and Sparse Neighbor Ranking) on large bipartite user-item interaction networks (such as MovieLens 1M).

---

## 2. System Requirements

* **Operating System:** Windows 10/11 (64-bit), Linux (Ubuntu/Debian/RHEL), or macOS.
* **Compiler:** GCC (MinGW-w64 on Windows) supporting C99/C11 and OpenMP (`-fopenmp`).
* **RAM:** Minimum 512 MB (MovieLens 1M requires only ~76 MB).
* **CPU:** Multi-core processor (Intel / AMD with 2 to 16+ hardware threads).

---

## 3. How to Build / Compile

### Quick Build on Windows:
Double-click `build.bat` or run:
```cmd
build.bat
```

### Manual Compilation using GCC:
```cmd
gcc -O3 -fopenmp -static -Iinclude src/main.c src/dataset.c src/sequential.c src/parallel_omp.c src/verify.c -o recommender.exe -lm
```

### Compilation using Makefile (Linux / Mac / Windows):
```bash
make clean
make
```

This will produce the standalone executable `recommender.exe` (or `./recommender`).

---

## 4. How to Run the Program

You can run the program in **Interactive Wizard Mode** (recommended) or via **Command-Line Arguments**.

### Method A: Interactive Wizard Mode (Double-click or run without arguments)
```cmd
recommender.exe
```

The interactive menu will guide you step-by-step through 4 stages:

#### Step 1: Select Dataset
```
[1] Standard MovieLens 1M Dataset (6,040 Users, 3,952 Movies, 1,000,209 Ratings)
[2] Sample Ratings Dataset (3,000 Users, 1,000 Items, 120,501 Ratings)
[3] Create Custom User-Movie Graph (Enter your own custom numbers)
```
* **Option 1 (MovieLens 1M):** Loads real-world film titles and 1 million ratings.
* **Option 2 (Sample Dataset):** Quick validation on a medium-sized CSV dataset.
* **Option 3 (Custom Synthetic Graph):** Allows you to test any number of Users (100–20,000), Movies (50–5,000), and Average ratings (5–200).

#### Step 2: Recommendation Depth (Top-K)
```
How many Top recommendations do you want per user? (Valid Range: 1 to 50) [Default: 10]:
```
* Enter how many movie suggestions should be generated for each user (e.g., 5, 10, 20).

#### Step 3: Execution and Benchmark Mode
```
[1] Single Multi-Core Test Run (Compare 1 Core vs Custom Thread Count & Verify)
[2] Full Multi-Core Scaling Benchmark (Test 1, 2, 4, 8, 12, 16 Cores)
[3] Benchmark Dataset Size Scaling (Fixed 4 Cores across Small, Medium, Large, Huge Graphs)
```
* **Option 1:** Compares Sequential (1 core) against OpenMP (e.g., 4 or 8 cores) and prints speedup, parallel efficiency, and bit-exact correctness verification.
* **Option 2:** Runs an automated multi-threaded benchmark across all core counts and exports `thread_scaling_results.csv`.
* **Option 3:** Measures how the algorithm scales as dataset size increases from 500 to 10,000 users.

#### Step 4: Sample Recommendation Preview
```
How many sample users would you like to preview recommendations for? (Valid Range: 1 to 20) [Default: 3]:
```
* Choose how many sample user recommendation lists you want to inspect on screen.

---

## 5. Understanding the Screen Output

### Example Output Section:
```text
[1/3] Running Sequential Baseline Algorithm (1 Core)...
      -> Movie Similarity Time   :   0.5580 seconds
      -> Top-10 Recs Ranking Time:   8.8710 seconds
      -> Total Sequential Time   :   9.4310 seconds

[2/3] Running OpenMP Parallel Algorithm (4 Threads/Cores)...
      -> Movie Similarity Time   :   0.1580 seconds
      -> Top-10 Recs Ranking Time:   2.3970 seconds
      -> Total OpenMP Time       :   2.5570 seconds
      -> Speedup Achieved        :     3.69x Faster!
      -> Parallel Efficiency     :    92.21 %
```

### Metrics Explained:
1. **Movie Similarity Time:** Time spent calculating pairwise cosine similarity across all movies.
2. **Top-K Recs Ranking Time:** Time spent predicting unrated movie scores and sorting top recommendations.
3. **Speedup Achieved:** $\text{Speedup} = \frac{\text{Sequential Execution Time}}{\text{Parallel Execution Time}}$. (e.g., $3.69\times$ means $369\%$ faster).
4. **Parallel Efficiency:** $\text{Efficiency} = \frac{\text{Speedup}}{\text{Number of Threads}} \times 100\%$. (e.g., $92.21\%$ indicates high core utilization).

---

## 6. Correctness Verification Report

The engine automatically verifies OpenMP parallel results against the single-threaded sequential baseline:
```text
===================================================================
                  CORRECTNESS VERIFICATION REPORT
===================================================================
  Status                  : [PASS] Results Match Exactly!
  Top-K Recommendation Set: 100.00 % Overlap
  Mean Absolute Error     : 0.00e+00
  Root Mean Squared Error : 0.00e+00
  Max Score Discrepancy   : 0.00e+00
===================================================================
```
* **Top-K Overlap = 100.00%:** Sequential and Parallel recommend the identical set of movies for all users.
* **MAE / RMSE = 0.00:** Exact score match with zero floating-point divergence.

---

## 7. Command-Line (Non-Interactive) Options

For automated scripts and grading, the executable accepts command-line flags:

```cmd
# Run on MovieLens 1M with 8 threads and Top-10 recommendations
recommender.exe --dataset data/ml-1m/ratings.dat --movies data/ml-1m/movies.dat --threads 8 --topk 10

# Run full core-scaling benchmark
recommender.exe --benchmark-threads --dataset data/ml-1m/ratings.dat

# Run dataset-size scaling benchmark
recommender.exe --benchmark-sizes --threads 4

# Run on synthetic graph (5000 users, 1000 items, 40 ratings/user)
recommender.exe --synthetic --users 5000 --items 1000 --density 40 --threads 4
```

---

## 8. Interactive HTML Visualizer Dashboard

To view interactive scaling charts in your web browser:
1. Navigate to the `benchmark/` folder.
2. Double-click `visualizer.html` in any web browser (Chrome, Edge, Firefox).
3. Inspect interactive charts for:
   * **Execution Time vs Number of Cores**
   * **Speedup Curve vs Ideal Linear Speedup**
   * **Parallel Efficiency (%)**
   * **Dataset Size vs Execution Time**
