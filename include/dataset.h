#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**
 * @brief Represents a single coordinate entry (User, Item, Rating).
 */
typedef struct {
    int user_id;
    int item_id;
    float rating;
} COOInteraction;

/**
 * @brief Compressed Sparse Row (CSR) representation for fast row-based traversal.
 */
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;           // Number of non-zero entries
    int *row_ptr;      // Size: num_rows + 1
    int *col_idx;      // Size: nnz
    float *values;     // Size: nnz
} CSRMatrix;

/**
 * @brief Single recommended item with its predicted rating score.
 */
typedef struct {
    int item_id;
    float score;
} Recommendation;

/**
 * @brief Full dataset containing both User-Item CSR and inverted Item-User CSR.
 */
typedef struct {
    int num_users;
    int num_items;
    int num_ratings;
    CSRMatrix user_item_csr; // Users as rows, Items as columns
    CSRMatrix item_user_csr; // Items as rows, Users as columns (transposed)
    char **item_names;       // Optional item/movie names (e.g. "Toy Story (1995)")
} Dataset;

/* Function Prototypes */
void free_csr(CSRMatrix *mat);
void free_dataset(Dataset *ds);

Dataset* create_dataset_from_coo(int num_users, int num_items, int num_ratings, COOInteraction *coo_list);
Dataset* create_synthetic_dataset(int num_users, int num_items, int avg_ratings_per_user, unsigned int seed);
Dataset* load_csv_dataset(const char *filepath);
void load_item_names(Dataset *ds, const char *movies_filepath);
bool save_dataset_to_csv(const char *filepath, const Dataset *ds);

#endif // DATASET_H
