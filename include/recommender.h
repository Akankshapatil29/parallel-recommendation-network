/**
 * ============================================================================
 * File: include/recommender.h
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: Algorithm definitions, execution timing structures, and
 *              correctness verification metrics for Sequential and OpenMP.
 * ============================================================================
 */

#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include "dataset.h"
#include <stdbool.h>

/**
 * @struct RecommendationResults
 * @brief Encapsulates generated Top-K recommendations and execution stage timestamps.
 */
typedef struct {
    int num_users;                  /**< Number of users evaluated */
    int top_k;                      /**< Top-K recommendation depth */
    Recommendation *recs;           /**< Flat array of recommendations (Size: num_users * top_k) */
    double time_similarity_sec;     /**< Wall-clock time spent in Stage 2 (Similarity Matrix) */
    double time_recommendation_sec; /**< Wall-clock time spent in Stage 3 (Top-K Ranking) */
    double time_total_sec;          /**< Total computational execution time */
} RecommendationResults;

/**
 * @struct VerificationReport
 * @brief Statistical metrics comparing OpenMP parallel output against Sequential baseline.
 */
typedef struct {
    bool passed;                    /**< True if parallel output matches within tolerance */
    double top_k_accuracy;          /**< Percentage of identical top-K items (Target: 100%) */
    double mean_absolute_error;     /**< Mean Absolute Error (MAE) across predicted scores */
    double root_mean_squared_err;   /**< Root Mean Squared Error (RMSE) across predicted scores */
    double max_score_diff;          /**< Maximum absolute discrepancy detected in any score */
} VerificationReport;

/* ============================================================================
 * Memory Allocation and Deallocation Prototypes
 * ============================================================================ */

/**
 * @brief Allocates and initializes memory for RecommendationResults structure.
 * @param num_users Number of users.
 * @param top_k Top-K recommendation depth.
 * @return Allocated pointer or NULL on failure.
 */
RecommendationResults* allocate_results(int num_users, int top_k);

/**
 * @brief Safely frees memory held by RecommendationResults.
 * @param res Pointer to RecommendationResults structure.
 */
void free_results(RecommendationResults *res);

/* ============================================================================
 * Recommender Algorithm Prototypes
 * ============================================================================ */

/**
 * @brief Sequential baseline implementation of Item-Based Collaborative Filtering.
 * 
 * Executes purely on 1 CPU core to establish the baseline execution time T_1.
 * 
 * @param ds Input bipartite Dataset.
 * @param top_k Number of top recommendations to generate per user.
 * @param min_common_users Minimum co-raters required between items.
 * @return Pointer to computed RecommendationResults.
 */
RecommendationResults* run_sequential_recommender(const Dataset *ds, int top_k, int min_common_users);

/**
 * @brief OpenMP multi-threaded parallel implementation of Item-Based Collaborative Filtering.
 * 
 * Parallelized with thread-private scratchpads, static/dynamic/guided scheduling,
 * and lock-free similarity accumulation.
 * 
 * @param ds Input bipartite Dataset.
 * @param top_k Number of top recommendations to generate per user.
 * @param min_common_users Minimum co-raters required between items.
 * @param num_threads Number of OpenMP threads / hardware cores to allocate.
 * @return Pointer to computed RecommendationResults.
 */
RecommendationResults* run_parallel_omp_recommender(const Dataset *ds, int top_k, int min_common_users, int num_threads);

/* ============================================================================
 * Verification and Error Analysis Prototypes
 * ============================================================================ */

/**
 * @brief Compares Sequential vs OpenMP recommendation lists to ensure bit-exact numerical equality.
 * @param seq Sequential baseline results.
 * @param par OpenMP parallel results.
 * @param tolerance Allowed floating-point difference threshold (e.g. 1e-4).
 * @return Statistical VerificationReport struct.
 */
VerificationReport verify_results(const RecommendationResults *seq, const RecommendationResults *par, double tolerance);

/**
 * @brief Formats and prints the statistical verification report to standard output.
 * @param report Pointer to VerificationReport.
 */
void print_verification_report(const VerificationReport *report);

#endif // RECOMMENDER_H
