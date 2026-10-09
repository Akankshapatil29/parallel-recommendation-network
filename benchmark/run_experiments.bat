@echo off
setlocal enabledelayedexpansion

echo ===================================================================
echo   RUNNING FULL RECOMMENDATION NETWORK PERFORMANCE EXPERIMENTS
echo ===================================================================

set PATH=C:\mingw64\bin;C:\Windows\System32;%PATH%

if not exist recommender.exe (
    echo [INFO] Building recommender.exe first...
    call build.bat
)

echo.
echo [EXPERIMENT 1] Multi-Thread Scaling (Users: 4000, Items: 1500, Density: 40)
recommender.exe -u 4000 -i 1500 -d 40 -k 10 -b -o benchmark\thread_scaling.csv

echo.
echo [EXPERIMENT 2] Input Size Scaling (Dataset Size Increases with Fixed 16 Threads)
echo size_name,num_users,num_items,ratings,seq_time,omp_time,speedup > benchmark\dataset_scaling.csv

echo Running Small Dataset (1,000 Users, 500 Items)...
recommender.exe -u 1000 -i 500 -d 30 -k 10 -t 16

echo Running Medium Dataset (3,000 Users, 1,000 Items)...
recommender.exe -u 3000 -i 1000 -d 40 -k 10 -t 16

echo Running Large Dataset (6,000 Users, 2,000 Items)...
recommender.exe -u 6000 -i 2000 -d 40 -k 10 -t 16

echo.
echo ===================================================================
echo   EXPERIMENTS COMPLETED! CSV logs saved in benchmark\ directory.
echo ===================================================================
