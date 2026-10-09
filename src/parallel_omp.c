#include "recommender.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/**
 * @brief Thread-local scratchpad to eliminate memory allocations and race conditions in parallel regions.
 */
typedef struct {
    float *dot_acc;
    int *common_acc;
    int *touched_items;
    float *score_acc;
    float *sim_sum_acc;
    bool *rated_mask;
    int *candidate_items;
} ThreadScratchpad;

static ThreadScratchpad* create_scratchpads(int num_threads, int num_items) {
    ThreadScratchpad *pads = (ThreadScratchpad *)malloc(num_threads * sizeof(ThreadScratchpad));
    for (int t = 0; t < num_threads; t++) {
        pads[t].dot_acc = (float *)calloc(num_items, sizeof(float));
        pads[t].common_acc = (int *)calloc(num_items, sizeof(int));
        pads[t].touched_items = (int *)malloc(num_items * sizeof(int));
        pads[t].score_acc = (float *)calloc(num_items, sizeof(float));
        pads[t].sim_sum_acc = (float *)calloc(num_items, sizeof(float));
        pads[t].rated_mask = (bool *)calloc(num_items, sizeof(bool));
        pads[t].candidate_items = (int *)malloc(num_items * sizeof(int));
    }
    return pads;
}

static void free_scratchpads(ThreadScratchpad *pads, int num_threads) {
    if (!pads) return;
    for (int t = 0; t < num_threads; t++) {
        free(pads[t].dot_acc);
        free(pads[t].common_acc);
        free(pads[t].touched_items);
        free(pads[t].score_acc);
        free(pads[t].sim_sum_acc);
        free(pads[t].rated_mask);
        free(pads[t].candidate_items);
    }
    free(pads);
}

// In-place top-K insertion
static inline void insert_top_k_parallel(Recommendation *top_k_arr, int k, int item_id, float score) {
    if (score <= top_k_arr[k - 1].score) return;

    int pos = k - 1;
    while (pos > 0 && top_k_arr[pos - 1].score < score) {
        top_k_arr[pos] = top_k_arr[pos - 1];
        pos--;
    }
    top_k_arr[pos].item_id = item_id;
    top_k_arr[pos].score = score;
}

RecommendationResults* run_parallel_omp_recommender(const Dataset *ds, int top_k, int min_common_users, int num_threads) {
    int num_users = ds->num_users;
    int num_items = ds->num_items;
    const CSRMatrix *user_item = &ds->user_item_csr;
    const CSRMatrix *item_user = &ds->item_user_csr;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    RecommendationResults *results = allocate_results(num_users, top_k);
    if (!results) return NULL;

    ThreadScratchpad *pads = create_scratchpads(num_threads, num_items);
    if (!pads) {
        free_results(results);
        return NULL;
    }

    double t_total_start = get_time_sec();

    /* =========================================================================
     * STAGE 1: Parallel Precomputation of Item L2-Norms (Static Scheduling)
     * ========================================================================= */
    float *item_norms = (float *)calloc(num_items, sizeof(float));

    #pragma omp parallel for schedule(static)
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
     * STAGE 2: Parallel Item-Item Cosine Similarity Matrix (Dynamic Scheduling)
     * Lock-free design: each thread writes strictly to its assigned row 'i'
     * ========================================================================= */
    double t_sim_start = get_time_sec();
    float *sim_matrix = (float *)calloc((size_t)num_items * num_items, sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadScratchpad *pad = &pads[tid];
        float *dot_acc = pad->dot_acc;
        int *common_acc = pad->common_acc;
        int *touched_items = pad->touched_items;

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < num_items; i++) {
            float norm_i = item_norms[i];
            if (norm_i <= 1e-6f) continue;

            int touched_count = 0;
            int u_start = item_user->row_ptr[i];
            int u_end = item_user->row_ptr[i + 1];

            for (int u_idx = u_start; u_idx < u_end; u_idx++) {
                int u = item_user->col_idx[u_idx];
                float r_ui = item_user->values[u_idx];

                int j_start = user_item->row_ptr[u];
                int j_end = user_item->row_ptr[u + 1];
                for (int j_idx = j_start; j_idx < j_end; j_idx++) {
                    int j = user_item->col_idx[j_idx];
                    if (j <= i) continue; // Upper triangular only

                    if (common_acc[j] == 0) {
                        touched_items[touched_count++] = j;
                    }
                    dot_acc[j] += r_ui * user_item->values[j_idx];
                    common_acc[j]++;
                }
            }

            // Compute cosine similarities and write to row 'i'
            for (int idx = 0; idx < touched_count; idx++) {
                int j = touched_items[idx];
                if (common_acc[j] >= min_common_users) {
                    float norm_j = item_norms[j];
                    if (norm_j > 1e-6f) {
                        float sim = dot_acc[j] / (norm_i * norm_j);
                        sim_matrix[(size_t)i * num_items + j] = sim;
                    }
                }
                dot_acc[j] = 0.0f;
                common_acc[j] = 0;
            }
        }
    }

    // Parallel symmetrize pass (zero locks, cache-blocked)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_items; i++) {
        for (int j = i + 1; j < num_items; j++) {
            float sim = sim_matrix[(size_t)i * num_items + j];
            if (sim > 0.0f) {
                sim_matrix[(size_t)j * num_items + i] = sim;
            }
        }
    }

    double t_sim_end = get_time_sec();
    results->time_similarity_sec = t_sim_end - t_sim_start;

    /* =========================================================================
     * STAGE 3: Parallel Top-K Recommendation Generation (Guided Scheduling)
     * Optimized: Build compact sparse similarity adjacency for non-zero entries
     * ========================================================================= */
    double t_rec_start = get_time_sec();

    // 1. Build sparse row representations in parallel
    int *sim_deg = (int *)calloc(num_items, sizeof(int));
    #pragma omp parallel for schedule(static)
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

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_items; i++) {
        const float *row = &sim_matrix[(size_t)i * num_items];
        int write_pos = sim_row_ptr[i];
        for (int j = 0; j < num_items; j++) {
            float s = row[j];
            if (s > 1e-6f) {
                sim_col_idx[write_pos] = j;
                sim_val[write_pos] = s;
                write_pos++;
            }
        }
    }
    free(sim_deg);
    free(sim_matrix); // Free dense matrix early to conserve RAM

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadScratchpad *pad = &pads[tid];
        float *score_acc = pad->score_acc;
        float *sim_sum_acc = pad->sim_sum_acc;
        bool *rated_mask = pad->rated_mask;
        int *candidate_items = pad->candidate_items;

        #pragma omp for schedule(guided, 8)
        for (int u = 0; u < num_users; u++) {
            int u_start = user_item->row_ptr[u];
            int u_end = user_item->row_ptr[u + 1];

            // Mark already rated items
            for (int idx = u_start; idx < u_end; idx++) {
                rated_mask[user_item->col_idx[idx]] = true;
            }

            int candidate_count = 0;

            // Score candidate items using sparse similarity neighbors
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

            // Top-K ranking
            Recommendation *user_top_k = &results->recs[u * top_k];
            for (int k = 0; k < top_k; k++) {
                user_top_k[k].item_id = -1;
                user_top_k[k].score = -1.0f;
            }

            for (int c = 0; c < candidate_count; c++) {
                int j = candidate_items[c];
                if (sim_sum_acc[j] > 1e-6f) {
                    float predicted_rating = score_acc[j] / sim_sum_acc[j];
                    insert_top_k_parallel(user_top_k, top_k, j, predicted_rating);
                }
                score_acc[j] = 0.0f;
                sim_sum_acc[j] = 0.0f;
            }

            for (int idx = u_start; idx < u_end; idx++) {
                rated_mask[user_item->col_idx[idx]] = false;
            }
        }
    }

    double t_rec_end = get_time_sec();
    results->time_recommendation_sec = t_rec_end - t_rec_start;
    results->time_total_sec = t_rec_end - t_total_start;

    free_scratchpads(pads, num_threads);
    free(sim_row_ptr);
    free(sim_col_idx);
    free(sim_val);
    free(item_norms);

    return results;
}
