# Parallel Recommendation Network (OpenMP + C)

High-Performance Graph-Based Recommendation Engine using Item-Based Collaborative Filtering and OpenMP Multi-Threading.

---

## 📁 Repository Structure

```
recommendation_network/
├── include/
│   ├── dataset.h             # Data structures (CSR matrix, COO, Dataset)
│   ├── recommender.h         # Sequential & OpenMP algorithm interfaces
│   └── timer.h               # High-resolution wall-clock timer
├── src/
│   ├── dataset.c             # Synthetic graph generator & MovieLens CSV parser
│   ├── sequential.c          # Sequential baseline recommender in pure C
│   ├── parallel_omp.c        # Lock-free OpenMP parallel recommender
│   ├── verify.c              # Verification module (MAE, RMSE, Overlap %)
│   └── main.c                # CLI driver and benchmark suite
├── data/
│   └── sample_ratings.csv    # Benchmark dataset (120,501 interactions)
├── benchmark/
│   ├── run_experiments.bat   # Automated multi-thread & dataset scaling runner
│   ├── thread_scaling.csv    # Exported performance benchmarks
│   └── visualizer.html       # Interactive performance dashboard (Chart.js)
├── docs/
│   ├── PROJECT_REPORT.md     # Academic project report
│   ├── LLM_USAGE_LOG.md      # Course-mandated LLM usage log & evaluation
│   └── VIVA_CHEAT_SHEET.md   # Oral defense Q&A preparation guide
├── build.bat                 # Windows one-click build script
├── Makefile                  # Cross-platform Makefile
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start & Compilation

### 1. One-Click Build on Windows
```bat
build.bat
```

### 2. Manual GCC Compilation
```bash
gcc -O3 -fopenmp -Wall -Wextra -Iinclude src/dataset.c src/sequential.c src/parallel_omp.c src/verify.c src/main.c -o recommender.exe -lm
```

---

## 🚀 Running the Engine

### Run on Generated Synthetic Graph
```bash
recommender.exe -u 3000 -i 1000 -d 40 -k 10
```

### Run on CSV Dataset
```bash
recommender.exe -f data\sample_ratings.csv -k 10
```

### Run Multi-Thread Scaling Benchmark (1, 2, 4, 8, 12, 16 Threads)
```bash
recommender.exe -f data\sample_ratings.csv -b -o benchmark\thread_scaling.csv
```

### Generate and Save Custom Graph Dataset
```bash
recommender.exe -u 5000 -i 2000 -d 50 -s data\custom_graph.csv
```

---

## 📊 Performance Summary (16-Core Machine)

| Threads | Total Time (s) | Speedup | Parallel Efficiency |
| :---: | :---: | :---: | :---: |
| **Sequential (1)** | 0.3770 s | 1.00x | 100.00 % |
| **2 Threads** | 0.1920 s | 1.96x | 98.18 % |
| **4 Threads** | 0.0970 s | 3.89x | 97.16 % |
| **8 Threads** | 0.0540 s | 6.98x | 87.27 % |
| **16 Threads** | 0.0380 s | **9.92x** | 62.01 % |

**Verification Status:** `[PASS]` 100.00% Top-K Overlap, 0.00 MAE Error.

---

## 📈 Viewing the Interactive Dashboard
Double-click [`benchmark/visualizer.html`](benchmark/visualizer.html) in any web browser to view interactive speedup and runtime charts.
