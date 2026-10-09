@echo off
title Parallel Recommendation Network - Live Demo
color 0A

cd /d "C:\Users\Akanksha Patil\.gemini\antigravity\scratch\recommendation_network"
set PATH=C:\mingw64\bin;C:\Windows\System32;%PATH%

echo ===================================================================
echo     PARALLEL RECOMMENDATION NETWORK ENGINE (C + OpenMP)
echo ===================================================================
echo Current Directory: %CD%
echo.
echo [1/3] Building latest code with GCC -O3 -fopenmp...
call build.bat

echo.
echo [2/3] Running Recommendation Engine on Dataset (16 Threads)...
recommender.exe -f data\sample_ratings.csv -k 10

echo.
echo [3/3] Running Multi-Thread Scaling Benchmark (1 to 16 Threads)...
recommender.exe -f data\sample_ratings.csv -b

echo.
echo ===================================================================
echo   DEMO FINISHED! You can now run custom commands directly below:
echo   Example: recommender.exe -f data\sample_ratings.csv -t 4
echo ===================================================================
echo.
cmd /k
