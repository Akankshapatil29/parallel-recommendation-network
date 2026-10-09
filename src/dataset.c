/**
 * ============================================================================
 * File: src/dataset.c
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: Implementation of Sparse Matrix Construction (CSR), MovieLens
 *              Data Ingestion, and Synthetic Power-Law Bipartite Graph Generation.
 * ============================================================================
 */

#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Safely deallocates internal arrays of a Compressed Sparse Row (CSR) matrix.
 * 
 * Frees row_ptr, col_idx, and values arrays and resets matrix metadata to 0.
 * 
 * @param mat Pointer to CSRMatrix to free.
 */
void free_csr(CSRMatrix *mat) {
    if (!mat) return;
    if (mat->row_ptr) free(mat->row_ptr);
    if (mat->col_idx) free(mat->col_idx);
    if (mat->values) free(mat->values);
    mat->row_ptr = NULL;
    mat->col_idx = NULL;
    mat->values = NULL;
    mat->num_rows = mat->num_cols = mat->nnz = 0;
}

/**
 * @brief Deallocates all memory associated with a Dataset struct.
 * 
 * Releases both forward (User-Item) and inverted (Item-User) CSR matrices,
 * as well as all dynamically allocated movie title strings.
 * 
 * @param ds Pointer to Dataset structure to free.
 */
void free_dataset(Dataset *ds) {
    if (!ds) return;
    free_csr(&ds->user_item_csr);
    free_csr(&ds->item_user_csr);
    if (ds->item_names) {
        for (int i = 0; i < ds->num_items; i++) {
            if (ds->item_names[i]) free(ds->item_names[i]);
        }
        free(ds->item_names);
        ds->item_names = NULL;
    }
    free(ds);
}

/**
 * @brief Comparator for qsort: Orders interactions primarily by user_id, secondarily by item_id.
 * 
 * Required for building the standard User -> Item Compressed Sparse Row (CSR) matrix.
 */
static int compare_coo_user_item(const void *a, const void *b) {
    const COOInteraction *ia = (const COOInteraction *)a;
    const COOInteraction *ib = (const COOInteraction *)b;
    if (ia->user_id != ib->user_id) {
        return ia->user_id - ib->user_id;
    }
    return ia->item_id - ib->item_id;
}

/**
 * @brief Comparator for qsort: Orders interactions primarily by item_id, secondarily by user_id.
 * 
 * Required for building the inverted Item -> User Transposed CSR matrix.
 */
static int compare_coo_item_user(const void *a, const void *b) {
    const COOInteraction *ia = (const COOInteraction *)a;
    const COOInteraction *ib = (const COOInteraction *)b;
    if (ia->item_id != ib->item_id) {
        return ia->item_id - ib->item_id;
    }
    return ia->user_id - ib->user_id;
}

/**
 * @brief Converts Coordinate (COO) interaction list into Dual CSR representation.
 * 
 * Constructs:
 * 1. User-Item CSR: rows = users, cols = items (size: O(N + E))
 * 2. Item-User CSR: rows = items, cols = users (size: O(M + E))
 * 
 * Time Complexity: O(E log E) due to in-memory sorting.
 * Space Complexity: O(N + M + 2E).
 * 
 * @param num_users Total count of users.
 * @param num_items Total count of items.
 * @param num_ratings Total number of interaction records.
 * @param coo_list Array of raw COOInteraction coordinates.
 * @return Allocated and populated Dataset pointer.
 */
Dataset* create_dataset_from_coo(int num_users, int num_items, int num_ratings, COOInteraction *coo_list) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_users = num_users;
    ds->num_items = num_items;
    ds->num_ratings = num_ratings;
    ds->item_names = NULL;

    /* -------------------------------------------------------------------------
     * 1. Build Forward User-Item CSR Matrix
     * ------------------------------------------------------------------------- */
    qsort(coo_list, num_ratings, sizeof(COOInteraction), compare_coo_user_item);

    ds->user_item_csr.num_rows = num_users;
    ds->user_item_csr.num_cols = num_items;
    ds->user_item_csr.nnz = num_ratings;
    ds->user_item_csr.row_ptr = (int *)calloc(num_users + 1, sizeof(int));
    ds->user_item_csr.col_idx = (int *)malloc(num_ratings * sizeof(int));
    ds->user_item_csr.values = (float *)malloc(num_ratings * sizeof(float));

    // Count frequency of interactions per user
    for (int i = 0; i < num_ratings; i++) {
        ds->user_item_csr.row_ptr[coo_list[i].user_id + 1]++;
        ds->user_item_csr.col_idx[i] = coo_list[i].item_id;
        ds->user_item_csr.values[i] = coo_list[i].rating;
    }
    // Prefix sum to compute row start offsets
    for (int u = 0; u < num_users; u++) {
        ds->user_item_csr.row_ptr[u + 1] += ds->user_item_csr.row_ptr[u];
    }

    /* -------------------------------------------------------------------------
     * 2. Build Inverted Item-User CSR Matrix (Transposed Index for Dot Products)
     * ------------------------------------------------------------------------- */
    qsort(coo_list, num_ratings, sizeof(COOInteraction), compare_coo_item_user);

    ds->item_user_csr.num_rows = num_items;
    ds->item_user_csr.num_cols = num_users;
    ds->item_user_csr.nnz = num_ratings;
    ds->item_user_csr.row_ptr = (int *)calloc(num_items + 1, sizeof(int));
    ds->item_user_csr.col_idx = (int *)malloc(num_ratings * sizeof(int));
    ds->item_user_csr.values = (float *)malloc(num_ratings * sizeof(float));

    // Count frequency of interactions per item
    for (int i = 0; i < num_ratings; i++) {
        ds->item_user_csr.row_ptr[coo_list[i].item_id + 1]++;
        ds->item_user_csr.col_idx[i] = coo_list[i].user_id;
        ds->item_user_csr.values[i] = coo_list[i].rating;
    }
    // Prefix sum to compute item start offsets
    for (int item = 0; item < num_items; item++) {
        ds->item_user_csr.row_ptr[item + 1] += ds->item_user_csr.row_ptr[item];
    }

    return ds;
}

/**
 * @brief Generates a realistic synthetic bipartite interaction graph.
 * 
 * Incorporates:
 * 1. Power-law / Zipf popularity skew (top 20% items receive ~70-80% of all ratings).
 * 2. User activity variance (each user rates between avg/2 and 1.5*avg items).
 * 3. Exact deduplication tracking to ensure each user rates an item at most once.
 * 
 * @param num_users Number of simulated user nodes.
 * @param num_items Number of simulated item nodes.
 * @param avg_ratings_per_user Target average ratings per user.
 * @param seed Random number generator seed.
 * @return Allocated Dataset pointer with synthetic interactions.
 */
Dataset* create_synthetic_dataset(int num_users, int num_items, int avg_ratings_per_user, unsigned int seed) {
    srand(seed);
    size_t capacity = (size_t)num_users * (size_t)avg_ratings_per_user * 2 + 1024;
    COOInteraction *coo_list = (COOInteraction *)malloc(capacity * sizeof(COOInteraction));
    if (!coo_list) return NULL;

    int actual_ratings = 0;
    // Track seen items per user using a fast boolean array
    bool *seen = (bool *)calloc(num_items, sizeof(bool));

    for (int u = 0; u < num_users; u++) {
        // Vary interactions per user (power-law distribution)
        int ratings_for_u = avg_ratings_per_user / 2 + (rand() % (avg_ratings_per_user + 1));
        if (ratings_for_u > num_items) ratings_for_u = num_items;

        int added = 0;
        int attempts = 0;
        while (added < ratings_for_u && attempts < ratings_for_u * 4) {
            attempts++;
            // Skew item popularity (Zipf-like distribution: top 20% items get majority of ratings)
            int item;
            if ((rand() % 100) < 70) {
                item = rand() % (num_items / 5 + 1);
            } else {
                item = rand() % num_items;
            }
            if (item >= num_items) item = num_items - 1;

            if (!seen[item]) {
                seen[item] = true;
                coo_list[actual_ratings].user_id = u;
                coo_list[actual_ratings].item_id = item;
                // Realistic rating scale: 1.0 to 5.0 in 0.5 increments
                float rating = 1.0f + 0.5f * (float)(rand() % 9);
                coo_list[actual_ratings].rating = rating;
                actual_ratings++;
                added++;
            }
        }
        // Reset seen mask for this user in O(added) operations
        int start_u_idx = actual_ratings - added;
        for (int i = start_u_idx; i < actual_ratings; i++) {
            seen[coo_list[i].item_id] = false;
        }
    }
    free(seen);

    Dataset *ds = create_dataset_from_coo(num_users, num_items, actual_ratings, coo_list);
    free(coo_list);
    return ds;
}

/**
 * @brief Parses interaction records from standard MovieLens .dat (::), CSV (,), or TSV files.
 * 
 * Auto-detects delimiter formats:
 * - MovieLens 1M: "UserID::MovieID::Rating::Timestamp"
 * - Standard CSV: "userId,movieId,rating"
 * - Tab-separated: "userId\tmovieId\trating"
 * 
 * Dynamically reallocates buffer capacity as needed.
 * 
 * @param filepath File path to dataset.
 * @return Allocated Dataset pointer, or NULL on file error.
 */
Dataset* load_csv_dataset(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "[Error] Could not open dataset file: %s\n", filepath);
        return NULL;
    }

    size_t capacity = 1000000;
    COOInteraction *coo_list = (COOInteraction *)malloc(capacity * sizeof(COOInteraction));
    if (!coo_list) {
        fclose(fp);
        return NULL;
    }

    char line[512];
    int count = 0;
    int max_user_id = -1;
    int max_item_id = -1;

    // Read and parse lines
    while (fgets(line, sizeof(line), fp)) {
        // Skip header lines
        if (strstr(line, "userId") || strstr(line, "user_id") || strstr(line, "user") || strstr(line, "User")) {
            continue;
        }
        int u = 0, item = 0;
        float r = 0.0f;
        
        // Multi-format parser support
        if (sscanf(line, "%d::%d::%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d,%d,%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d\t%d\t%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d %d %f", &u, &item, &r) == 3) {
            
            // Reallocate buffer if capacity is reached
            if (count >= (int)capacity) {
                capacity *= 2;
                COOInteraction *new_buf = (COOInteraction *)realloc(coo_list, capacity * sizeof(COOInteraction));
                if (!new_buf) {
                    free(coo_list);
                    fclose(fp);
                    return NULL;
                }
                coo_list = new_buf;
            }

            coo_list[count].user_id = u;
            coo_list[count].item_id = item;
            coo_list[count].rating = r;

            if (u > max_user_id) max_user_id = u;
            if (item > max_item_id) max_item_id = item;
            count++;
        }
    }
    fclose(fp);

    if (count == 0) {
        fprintf(stderr, "[Error] No interactions parsed from: %s\n", filepath);
        free(coo_list);
        return NULL;
    }

    int num_users = max_user_id + 1;
    int num_items = max_item_id + 1;

    Dataset *ds = create_dataset_from_coo(num_users, num_items, count, coo_list);
    free(coo_list);
    return ds;
}

/**
 * @brief Parses MovieLens metadata (movies.dat) to attach title names to item IDs.
 * 
 * Format expected: "MovieID::Title (Year)::Genres"
 * 
 * @param ds Dataset pointer to attach title strings to.
 * @param movies_filepath Path to movies.dat.
 */
void load_item_names(Dataset *ds, const char *movies_filepath) {
    if (!ds || !movies_filepath) return;
    FILE *fp = fopen(movies_filepath, "r");
    if (!fp) return;

    ds->item_names = (char **)calloc(ds->num_items, sizeof(char *));
    if (!ds->item_names) {
        fclose(fp);
        return;
    }

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        int id = 0;
        char title[256] = {0};
        char genres[256] = {0};
        // Parse MovieLens movies.dat: 1::Toy Story (1995)::Animation|Children's|Comedy
        if (sscanf(line, "%d::%255[^:]::%255[^\n]", &id, title, genres) >= 2) {
            if (id < ds->num_items) {
                ds->item_names[id] = strdup(title);
            }
        }
    }
    fclose(fp);
}

/**
 * @brief Exports current Dataset interactions into a standard CSV file.
 * 
 * CSV Header: "userId,itemId,rating"
 * 
 * @param filepath Target file path.
 * @param ds Source Dataset.
 * @return True on success, False on failure.
 */
bool save_dataset_to_csv(const char *filepath, const Dataset *ds) {
    if (!ds || !filepath) return false;
    FILE *fp = fopen(filepath, "w");
    if (!fp) return false;

    fprintf(fp, "userId,itemId,rating\n");
    const CSRMatrix *csr = &ds->user_item_csr;
    for (int u = 0; u < csr->num_rows; u++) {
        int start = csr->row_ptr[u];
        int end = csr->row_ptr[u + 1];
        for (int idx = start; idx < end; idx++) {
            fprintf(fp, "%d,%d,%.1f\n", u, csr->col_idx[idx], csr->values[idx]);
        }
    }
    fclose(fp);
    return true;
}
