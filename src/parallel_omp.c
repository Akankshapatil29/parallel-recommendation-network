/**
 * ============================================================================
 * File: src/parallel_omp.c
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: OpenMP Multi-Threaded Parallel Recommender Engine with Lock-Free
 *              Thread-Private Scratchpads, Dynamic/Guided Scheduling, and High-
 *              Performance Cache-Blocked Sparse Inverted Index Traversals.
 * ============================================================================
 */

#include "recommender.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/**
 * @struct ThreadScratchpad
 * @brief Private working memory allocated per OpenMP thread.
 * 
 * Design Rationale:
 * By providing each thread with its own dedicated accumulator arrays:
 * 1. False Sharing is completely eliminated (threads do not share cache lines).
 * 2. Critical sections / atomic locks are avoided (100% lock-free execution).
 * 3. Memory allocation overhead is moved outside parallel loops.
 */
typedef struct {
    float *dot_acc;         /**< Accumulates dot products for pairwise similarity */
    int *common_acc;        /**< Counts shared co-rating users between item pairs */
    int *touched_items;     /**< Sparse list of non-zero item indices touched */
    float *score_acc;       /**< Accumulates weighted predicted recommendation scores */
    float *sim_sum_acc;     /**< Accumulates normalization sum of absolute similarities */
    bool *rated_mask;       /**< Fast O(1) boolean lookup for items already rated by user */
    int *candidate_items;   /**< Sparse list of candidate item IDs evaluated for user */
} ThreadScratchpad;

/**
 * @brief Allocates an array of ThreadScratchpads (one per active OpenMP thread).
 * 
 * @param num_threads Number of active hardware threads.
 * @param num_items Number of items in the dataset.
 * @return Allocated array of ThreadScratchpad structures.
 */
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

/**
 * @brief Safely frees all memory allocated across all thread scratchpads.
 * 
 * @param pads Array of ThreadScratchpads.
 * @param num_threads Number of threads allocated.
 */
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

/**
 * @brief Thread-safe in-place Top-K insertion sort.
 * 
 * Operates purely on thread-private local memory to maintain top-K recommendations.
 * 
 * @param top_k_arr Target user recommendation buffer.
 * @param k Size of top-K buffer.
 * @param item_id Candidate item ID.
 * @param score Predicted rating score.
 */
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

/**
 * @brief OpenMP Multi-Threaded Parallel Implementation of Collaborative Filtering.
 * 
 * Multi-Core Optimization Highlights:
 * 1. Stage 1 (Norms): Static scheduling partitions item list evenly across cores.
 * 2. Stage 2 (Similarity): Dynamic scheduling balances triangular loop workloads.
 * 3. Stage 3 (Top-K Ranking): Guided scheduling dynamically adapts to power-law user degree variance.
 * 
 * @param ds Input bipartite dataset.
 * @param top_k Top-K depth.
 * @param min_common_users Minimum shared raters threshold.
 * @param num_threads Target thread count (0 = auto-detect max cores).
 * @return Computed RecommendationResults containing parallel timings.
 */
RecommendationResults* run_parallel_omp_recommender(const Dataset *ds, int top_k, int min_common_users, int num_threads) {
    int num_users = ds->num_users;
    int num_items = ds->num_items;
    const CSRMatrix *user_item = &ds->user_item_csr;
    const CSRMatrix *item_user = &ds->item_user_csr;

    // Detect hardware thread count if not specified
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    RecommendationResults *results = allocate_results(num_users, top_k);
    if (!results) return NULL;

    // Allocate thread-private scratchpads
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

        // Dynamic scheduling with chunk size 16 to handle triangular loop load imbalance
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < num_items; i++) {
            float norm_i = item_norms[i];
            if (norm_i <= 1e-6f) continue;

            int touched_count = 0;
            int u_start = item_user->row_ptr[i];
            int u_end = item_user->row_ptr[i + 1];

            // Inverted index traversal: iterate users who rated item i
            for (int u_idx = u_start; u_idx < u_end; u_idx++) {
                int u = item_user->col_idx[u_idx];
                float r_ui = item_user->values[u_idx];

                // Forward index traversal: iterate other items j rated by user u
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

            // Compute cosine similarities and write to row 'i' in parallel
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

    // Parallel symmetrization pass (zero locks)
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
    free(sim_matrix); // Free large dense matrix early to conserve RAM

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadScratchpad *pad = &pads[tid];
        float *score_acc = pad->score_acc;
        float *sim_sum_acc = pad->sim_sum_acc;
        bool *rated_mask = pad->rated_mask;
        int *candidate_items = pad->candidate_items;

        // Guided scheduling adapts to skewed user rating counts (power-law distribution)
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

            // Top-K ranking buffer initialization
            Recommendation *user_top_k = &results->recs[u * top_k];
            for (int k = 0; k < top_k; k++) {
                user_top_k[k].item_id = -1;
                user_top_k[k].score = -1.0f;
            }

            // Rank candidates into user top-K buffer
            for (int c = 0; c < candidate_count; c++) {
                int j = candidate_items[c];
                if (sim_sum_acc[j] > 1e-6f) {
                    float predicted_rating = score_acc[j] / sim_sum_acc[j];
                    insert_top_k_parallel(user_top_k, top_k, j, predicted_rating);
                }
                score_acc[j] = 0.0f;
                sim_sum_acc[j] = 0.0f;
            }

            // Unmark rated mask for user u
            for (int idx = u_start; idx < u_end; idx++) {
                rated_mask[user_item->col_idx[idx]] = false;
            }
        }
    }

    double t_rec_end = get_time_sec();
    results->time_recommendation_sec = t_rec_end - t_rec_start;
    results->time_total_sec = t_rec_end - t_total_start;

    // Cleanup resources
    free_scratchpads(pads, num_threads);
    free(sim_row_ptr);
    free(sim_col_idx);
    free(sim_val);
    free(item_norms);

    return results;
}
