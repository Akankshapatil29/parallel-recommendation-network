#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// Comparison function for sorting COO interactions by (user_id, item_id)
static int compare_coo_user_item(const void *a, const void *b) {
    const COOInteraction *ia = (const COOInteraction *)a;
    const COOInteraction *ib = (const COOInteraction *)b;
    if (ia->user_id != ib->user_id) {
        return ia->user_id - ib->user_id;
    }
    return ia->item_id - ib->item_id;
}

// Comparison function for sorting COO interactions by (item_id, user_id)
static int compare_coo_item_user(const void *a, const void *b) {
    const COOInteraction *ia = (const COOInteraction *)a;
    const COOInteraction *ib = (const COOInteraction *)b;
    if (ia->item_id != ib->item_id) {
        return ia->item_id - ib->item_id;
    }
    return ia->user_id - ib->user_id;
}

Dataset* create_dataset_from_coo(int num_users, int num_items, int num_ratings, COOInteraction *coo_list) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_users = num_users;
    ds->num_items = num_items;
    ds->num_ratings = num_ratings;
    ds->item_names = NULL;

    // 1. Build User-Item CSR
    qsort(coo_list, num_ratings, sizeof(COOInteraction), compare_coo_user_item);

    ds->user_item_csr.num_rows = num_users;
    ds->user_item_csr.num_cols = num_items;
    ds->user_item_csr.nnz = num_ratings;
    ds->user_item_csr.row_ptr = (int *)calloc(num_users + 1, sizeof(int));
    ds->user_item_csr.col_idx = (int *)malloc(num_ratings * sizeof(int));
    ds->user_item_csr.values = (float *)malloc(num_ratings * sizeof(float));

    for (int i = 0; i < num_ratings; i++) {
        ds->user_item_csr.row_ptr[coo_list[i].user_id + 1]++;
        ds->user_item_csr.col_idx[i] = coo_list[i].item_id;
        ds->user_item_csr.values[i] = coo_list[i].rating;
    }
    for (int u = 0; u < num_users; u++) {
        ds->user_item_csr.row_ptr[u + 1] += ds->user_item_csr.row_ptr[u];
    }

    // 2. Build Item-User CSR (Inverted Index for Fast Dot-Products)
    qsort(coo_list, num_ratings, sizeof(COOInteraction), compare_coo_item_user);

    ds->item_user_csr.num_rows = num_items;
    ds->item_user_csr.num_cols = num_users;
    ds->item_user_csr.nnz = num_ratings;
    ds->item_user_csr.row_ptr = (int *)calloc(num_items + 1, sizeof(int));
    ds->item_user_csr.col_idx = (int *)malloc(num_ratings * sizeof(int));
    ds->item_user_csr.values = (float *)malloc(num_ratings * sizeof(float));

    for (int i = 0; i < num_ratings; i++) {
        ds->item_user_csr.row_ptr[coo_list[i].item_id + 1]++;
        ds->item_user_csr.col_idx[i] = coo_list[i].user_id;
        ds->item_user_csr.values[i] = coo_list[i].rating;
    }
    for (int item = 0; item < num_items; item++) {
        ds->item_user_csr.row_ptr[item + 1] += ds->item_user_csr.row_ptr[item];
    }

    return ds;
}

Dataset* create_synthetic_dataset(int num_users, int num_items, int avg_ratings_per_user, unsigned int seed) {
    srand(seed);
    size_t capacity = (size_t)num_users * (size_t)avg_ratings_per_user * 2 + 1024;
    COOInteraction *coo_list = (COOInteraction *)malloc(capacity * sizeof(COOInteraction));
    if (!coo_list) return NULL;

    int actual_ratings = 0;
    // Track seen items per user using a boolean array
    bool *seen = (bool *)calloc(num_items, sizeof(bool));

    for (int u = 0; u < num_users; u++) {
        // Vary interactions per user (power-law / skewed distribution)
        int ratings_for_u = avg_ratings_per_user / 2 + (rand() % (avg_ratings_per_user + 1));
        if (ratings_for_u > num_items) ratings_for_u = num_items;

        int added = 0;
        int attempts = 0;
        while (added < ratings_for_u && attempts < ratings_for_u * 4) {
            attempts++;
            // Skew item popularity (Zipf-like distribution: top 20% items get 80% ratings)
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
                // Realistic rating: 1.0 to 5.0 in 0.5 increments
                float rating = 1.0f + 0.5f * (float)(rand() % 9);
                coo_list[actual_ratings].rating = rating;
                actual_ratings++;
                added++;
            }
        }
        // Reset seen items for this user efficiently
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

    // Read lines
    while (fgets(line, sizeof(line), fp)) {
        // Skip header lines
        if (strstr(line, "userId") || strstr(line, "user_id") || strstr(line, "user") || strstr(line, "User")) {
            continue;
        }
        int u = 0, item = 0;
        float r = 0.0f;
        
        // Support MovieLens format (1::1193::5::978300760), CSV (1,1193,5), TSV, or space-separated
        if (sscanf(line, "%d::%d::%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d,%d,%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d\t%d\t%f", &u, &item, &r) == 3 ||
            sscanf(line, "%d %d %f", &u, &item, &r) == 3) {
            
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

