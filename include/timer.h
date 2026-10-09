#ifndef TIMER_H
#define TIMER_H

#include <omp.h>
#include <stdio.h>

/**
 * @brief Get high-resolution timestamp in seconds using OpenMP wall-clock timer.
 */
static inline double get_time_sec(void) {
    return omp_get_wtime();
}

#endif // TIMER_H
