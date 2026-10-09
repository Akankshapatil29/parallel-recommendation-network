#ifndef RECOMMENDER_H
#define RECOMMENDER_H

#include "dataset.h"
#include <stdbool.h>

/**
 * @brief Holds Top-K recommendations and execution timings.
 */
typedef struct {
    int num_users;
    int top_k;
    Recommendation *recs;         // Flat array: size = num_users * top_k
    double time_similarity_sec;   // Time spent computing item-item similarities
    double time_recommendation_sec;// Time spent predicting & ranking top-K
    double time_total_sec;        // Total compute time
} RecommendationResults;

/* Memory management */
RecommendationResults* allocate_results(int num_users, int top_k);
void free_results(RecommendationResults *res);

/* Algorithms */
RecommendationResults* run_sequential_recommender(const Dataset *ds, int top_k, int min_common_users);
RecommendationResults* run_parallel_omp_recommender(const Dataset *ds, int top_k, int min_common_users, int num_threads);

/* Verification & Metrics */
typedef struct {
    bool passed;
    double top_k_accuracy;        // % of identical top-K item sets
    double mean_absolute_error;   // MAE across score values
    double root_mean_squared_err; // RMSE across score values
    double max_score_diff;        // Max difference in any predicted score
} VerificationReport;

VerificationReport verify_results(const RecommendationResults *seq, const RecommendationResults *par, double tolerance);
void print_verification_report(const VerificationReport *report);

#endif // RECOMMENDER_H
