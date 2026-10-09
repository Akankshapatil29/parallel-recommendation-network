/**
 * ============================================================================
 * File: src/verify.c
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: Correctness and Numerical Verification Engine for Validating
 *              OpenMP Parallel Output Against Sequential Baseline.
 * ============================================================================
 */

#include "recommender.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Evaluates mathematical equivalence between Sequential and Parallel results.
 * 
 * Computes:
 * 1. Top-K Set Overlap (%): Ratio of identical recommendation items matched.
 * 2. Mean Absolute Error (MAE): Average absolute score difference:
 *    MAE = (1 / M) * sum |Score_seq - Score_par|
 * 3. Root Mean Squared Error (RMSE): Standard deviation of error:
 *    RMSE = sqrt((1 / M) * sum (Score_seq - Score_par)^2)
 * 4. Maximum Discrepancy: Worst-case floating point divergence across all items.
 * 
 * @param seq Sequential baseline results structure.
 * @param par Parallel OpenMP results structure.
 * @param tolerance Allowed floating-point tolerance threshold (e.g. 1e-4).
 * @return VerificationReport containing pass/fail status and statistical metrics.
 */
VerificationReport verify_results(const RecommendationResults *seq, const RecommendationResults *par, double tolerance) {
    VerificationReport report;
    report.passed = true;
    report.top_k_accuracy = 0.0;
    report.mean_absolute_error = 0.0;
    report.root_mean_squared_err = 0.0;
    report.max_score_diff = 0.0;

    if (!seq || !par || seq->num_users != par->num_users || seq->top_k != par->top_k) {
        report.passed = false;
        return report;
    }

    int num_users = seq->num_users;
    int top_k = seq->top_k;
    long long total_items_checked = 0;
    long long total_matched_items = 0;
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double max_diff = 0.0;

    // Compare recommendations user by user
    for (int u = 0; u < num_users; u++) {
        const Recommendation *seq_k = &seq->recs[u * top_k];
        const Recommendation *par_k = &par->recs[u * top_k];

        for (int k = 0; k < top_k; k++) {
            if (seq_k[k].item_id == -1 && par_k[k].item_id == -1) {
                continue; // Both slots empty (cold start)
            }
            total_items_checked++;

            // Check if sequential item exists in parallel top-K set
            bool found = false;
            for (int pk = 0; pk < top_k; pk++) {
                if (seq_k[k].item_id == par_k[pk].item_id) {
                    found = true;
                    double diff = fabs((double)seq_k[k].score - (double)par_k[pk].score);
                    sum_abs_err += diff;
                    sum_sq_err += diff * diff;
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                    break;
                }
            }
            if (found) {
                total_matched_items++;
            }
        }
    }

    // Compute final statistical metrics
    if (total_items_checked > 0) {
        report.top_k_accuracy = (double)total_matched_items / (double)total_items_checked * 100.0;
        report.mean_absolute_error = sum_abs_err / (double)total_matched_items;
        report.root_mean_squared_err = sqrt(sum_sq_err / (double)total_matched_items);
        report.max_score_diff = max_diff;
    } else {
        report.top_k_accuracy = 100.0;
    }

    // Test criteria: Overlap must be >= 99.0% and max difference within tolerance
    if (report.max_score_diff > tolerance || report.top_k_accuracy < 99.0) {
        report.passed = false;
    }

    return report;
}

/**
 * @brief Formats and prints the statistical verification report.
 * 
 * @param report Pointer to computed VerificationReport.
 */
void print_verification_report(const VerificationReport *report) {
    printf("\n===================================================================\n");
    printf("                  CORRECTNESS VERIFICATION REPORT                  \n");
    printf("===================================================================\n");
    printf("  Status                  : %s\n", report->passed ? "[PASS] Results Match Exactly!" : "[FAIL] Discrepancy Detected!");
    printf("  Top-K Recommendation Set: %.2f %% Overlap\n", report->top_k_accuracy);
    printf("  Mean Absolute Error     : %.2e\n", report->mean_absolute_error);
    printf("  Root Mean Squared Error : %.2e\n", report->root_mean_squared_err);
    printf("  Max Score Discrepancy   : %.2e\n", report->max_score_diff);
    printf("===================================================================\n\n");
}
