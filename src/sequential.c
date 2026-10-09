#include "recommender.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

RecommendationResults* allocate_results(int num_users, int top_k) {
    RecommendationResults *res = (RecommendationResults *)malloc(sizeof(RecommendationResults));
    if (!res) return NULL;
    res->num_users = num_users;
    res->top_k = top_k;
    res->recs = (Recommendation *)malloc(num_users * top_k * sizeof(Recommendation));
    for (int i = 0; i < num_users * top_k; i++) {
        res->recs[i].item_id = -1;
        res->recs[i].score = 0.0f;
    }
    res->time_similarity_sec = 0.0;
    res->time_recommendation_sec = 0.0;
    res->time_total_sec = 0.0;
    return res;
}

void free_results(RecommendationResults *res) {
    if (!res) return;
    if (res->recs) free(res->recs);
    free(res);
}

// Min-heap or insertion sort for maintaining top-K items per user
static void insert_top_k(Recommendation *top_k_arr, int k, int item_id, float score) {
    if (score <= top_k_arr[k - 1].score) return;

    // Find insertion position
    int pos = k - 1;
    while (pos > 0 && top_k_arr[pos - 1].score < score) {
        top_k_arr[pos] = top_k_arr[pos - 1];
        pos--;
    }
    top_k_arr[pos].item_id = item_id;
    top_k_arr[pos].score = score;
}

RecommendationResults* run_sequential_recommender(const Dataset *ds, int top_k, int min_common_users) {
    int num_users = ds->num_users;
    int num_items = ds->num_items;
    const CSRMatrix *user_item = &ds->user_item_csr;
    const CSRMatrix *item_user = &ds->item_user_csr;

    RecommendationResults *results = allocate_results(num_users, top_k);
    if (!results) return NULL;

    double t_total_start = get_time_sec();

    /* =========================================================================
     * STAGE 1: Precompute Item L2-Norms
     * ========================================================================= */
    float *item_norms = (float *)calloc(num_items, sizeof(float));
    for (int item = 0; item < num_items; item++) {
        int start = item_user->row_ptr[item];
        int end = item_user->row_ptr[item + 1];
        double sum_sq = 0.0;
        for (int idx = start; idx < end; idx++) {
            float r = item_user->values[idx];
            sum_sq += (double)r * r;
        }
        item_norms[item] = (sum_sq > 1e-9) ? (float)sqrt(sum_sq) : 0.0f;
    }

    /* =========================================================================
     * STAGE 2: Compute Item-Item Cosine Similarity Matrix (Sequential)
     * ========================================================================= */
    double t_sim_start = get_time_sec();

    // Allocate flat symmetric similarity matrix
    float *sim_matrix = (float *)calloc((size_t)num_items * num_items, sizeof(float));
    float *dot_acc = (float *)calloc(num_items, sizeof(float));
    int *common_acc = (int *)calloc(num_items, sizeof(int));
    int *touched_items = (int *)malloc(num_items * sizeof(int));

    for (int i = 0; i < num_items; i++) {
        float norm_i = item_norms[i];
        if (norm_i <= 1e-6f) continue;

        int touched_count = 0;
        int u_start = item_user->row_ptr[i];
        int u_end = item_user->row_ptr[i + 1];

        // Traverse all users who rated item i
        for (int u_idx = u_start; u_idx < u_end; u_idx++) {
            int u = item_user->col_idx[u_idx];
            float r_ui = item_user->values[u_idx];

            // Traverse other items rated by user u
            int j_start = user_item->row_ptr[u];
            int j_end = user_item->row_ptr[u + 1];
            for (int j_idx = j_start; j_idx < j_end; j_idx++) {
                int j = user_item->col_idx[j_idx];
                if (j <= i) continue; // Only upper triangular for symmetry

                if (common_acc[j] == 0) {
                    touched_items[touched_count++] = j;
                }
                dot_acc[j] += r_ui * user_item->values[j_idx];
                common_acc[j]++;
            }
        }

        // Calculate cosine similarity for all touched items
        for (int idx = 0; idx < touched_count; idx++) {
            int j = touched_items[idx];
            if (common_acc[j] >= min_common_users) {
                float norm_j = item_norms[j];
                if (norm_j > 1e-6f) {
                    float sim = dot_acc[j] / (norm_i * norm_j);
                    sim_matrix[(size_t)i * num_items + j] = sim;
                    sim_matrix[(size_t)j * num_items + i] = sim;
                }
            }
            // Reset accumulators for next iteration
            dot_acc[j] = 0.0f;
            common_acc[j] = 0;
        }
    }

    free(dot_acc);
    free(common_acc);
    free(touched_items);

    double t_sim_end = get_time_sec();
    results->time_similarity_sec = t_sim_end - t_sim_start;

    /* =========================================================================
     * STAGE 3: Compute Top-K Recommendations for Every User (Sequential)
     * Optimized: Build compact sparse similarity adjacency for non-zero entries
     * ========================================================================= */
    double t_rec_start = get_time_sec();

    // 1. Build sparse row representations for fast neighbor lookup
    int *sim_deg = (int *)calloc(num_items, sizeof(int));
    for (int i = 0; i < num_items; i++) {
        const float *row = &sim_matrix[(size_t)i * num_items];
        int d = 0;
        for (int j = 0; j < num_items; j++) {
            if (row[j] > 1e-6f) d++;
        }
        sim_deg[i] = d;
    }

    int *sim_row_ptr = (int *)calloc(num_items + 1, sizeof(int));
    for (int i = 0; i < num_items; i++) {
        sim_row_ptr[i + 1] = sim_row_ptr[i] + sim_deg[i];
    }

    int total_sim_entries = sim_row_ptr[num_items];
    int *sim_col_idx = (int *)malloc((size_t)(total_sim_entries + 1) * sizeof(int));
    float *sim_val = (float *)malloc((size_t)(total_sim_entries + 1) * sizeof(float));

    int *write_cursor = (int *)calloc(num_items, sizeof(int));
    for (int i = 0; i < num_items; i++) {
        write_cursor[i] = sim_row_ptr[i];
    }

    for (int i = 0; i < num_items; i++) {
        const float *row = &sim_matrix[(size_t)i * num_items];
        for (int j = 0; j < num_items; j++) {
            float s = row[j];
            if (s > 1e-6f) {
                int pos = write_cursor[i]++;
                sim_col_idx[pos] = j;
                sim_val[pos] = s;
            }
        }
    }
    free(write_cursor);
    free(sim_deg);
    free(sim_matrix); // Free dense matrix early to conserve RAM

    float *score_acc = (float *)calloc(num_items, sizeof(float));
    float *sim_sum_acc = (float *)calloc(num_items, sizeof(float));
    bool *rated_mask = (bool *)calloc(num_items, sizeof(bool));
    int *candidate_items = (int *)malloc(num_items * sizeof(int));

    for (int u = 0; u < num_users; u++) {
        int u_start = user_item->row_ptr[u];
        int u_end = user_item->row_ptr[u + 1];

        // Mark items already rated by user u
        for (int idx = u_start; idx < u_end; idx++) {
            rated_mask[user_item->col_idx[idx]] = true;
        }

        int candidate_count = 0;

        // Traverse only non-zero similarity neighbors
        for (int idx = u_start; idx < u_end; idx++) {
            int i = user_item->col_idx[idx];
            float r_ui = user_item->values[idx];
            int n_start = sim_row_ptr[i];
            int n_end = sim_row_ptr[i + 1];

            for (int n_idx = n_start; n_idx < n_end; n_idx++) {
                int j = sim_col_idx[n_idx];
                float sim = sim_val[n_idx];

                if (!rated_mask[j]) {
                    if (sim_sum_acc[j] == 0.0f) {
                        candidate_items[candidate_count++] = j;
                    }
                    score_acc[j] += sim * r_ui;
                    sim_sum_acc[j] += (float)fabs(sim);
                }
            }
        }

        // Initialize top-K buffer for user u
        Recommendation *user_top_k = &results->recs[u * top_k];
        for (int k = 0; k < top_k; k++) {
            user_top_k[k].item_id = -1;
            user_top_k[k].score = -1.0f;
        }

        // Rank top-K candidates
        for (int c = 0; c < candidate_count; c++) {
            int j = candidate_items[c];
            if (sim_sum_acc[j] > 1e-6f) {
                float predicted_rating = score_acc[j] / sim_sum_acc[j];
                insert_top_k(user_top_k, top_k, j, predicted_rating);
            }
            score_acc[j] = 0.0f;
            sim_sum_acc[j] = 0.0f;
        }

        // Unmark rated mask
        for (int idx = u_start; idx < u_end; idx++) {
            rated_mask[user_item->col_idx[idx]] = false;
        }
    }

    free(score_acc);
    free(sim_sum_acc);
    free(rated_mask);
    free(candidate_items);
    free(sim_row_ptr);
    free(sim_col_idx);
    free(sim_val);
    free(item_norms);

    double t_rec_end = get_time_sec();
    results->time_recommendation_sec = t_rec_end - t_rec_start;
    results->time_total_sec = t_rec_end - t_total_start;

    return results;
}
