@echo off
setlocal

set CC=C:\mingw64\bin\gcc.exe
if not exist "%CC%" (
    set CC=gcc
)

echo [BUILD] Compiling Recommendation Network Engine with OpenMP (-O3 -static)...
"%CC%" -O3 -fopenmp -static -static-libgcc -Wall -Wextra -Iinclude src/dataset.c src/sequential.c src/parallel_omp.c src/verify.c src/main.c -o recommender.exe -lm

if %ERRORLEVEL% equ 0 (
    echo [BUILD SUCCESS] Executable created: recommender.exe
) else (
    echo [BUILD ERROR] Compilation failed!
)
