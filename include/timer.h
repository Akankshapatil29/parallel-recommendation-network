/**
 * ============================================================================
 * File: include/timer.h
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: High-resolution wall-clock timer wrapper using OpenMP API.
 * ============================================================================
 */

#ifndef TIMER_H
#define TIMER_H

#include <omp.h>
#include <stdio.h>

/**
 * @brief Returns the current high-resolution wall-clock timestamp in seconds.
 * 
 * Uses omp_get_wtime() which provides sub-microsecond precision across
 * both Windows and POSIX operating systems.
 * 
 * @return Current timestamp in seconds (double precision).
 */
static inline double get_time_sec(void) {
    return omp_get_wtime();
}

#endif // TIMER_H
