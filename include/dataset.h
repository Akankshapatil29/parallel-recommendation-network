/**
 * ============================================================================
 * File: include/dataset.h
 * Project: High-Performance Parallel Recommendation Engine (OpenMP)
 * Description: Header definitions for Sparse Bipartite Interaction Graphs,
 *              Compressed Sparse Row (CSR) Matrices, and Dataset Loaders.
 * ============================================================================
 */

#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**
 * @struct COOInteraction
 * @brief Represents a single coordinate rating edge (User -> Item -> Rating).
 * 
 * Used during initial data ingestion from CSV or DAT files before sorting
 * and compressing into dual CSR formats.
 */
typedef struct {
    int user_id;     /**< Zero-based user index */
    int item_id;     /**< Zero-based item / movie index */
    float rating;    /**< Rating value (e.g. 1.0 to 5.0) */
} COOInteraction;

/**
 * @struct CSRMatrix
 * @brief Compressed Sparse Row (CSR) representation for ultra-fast sparse traversal.
 * 
 * Space Complexity: O(num_rows + nnz)
 * - row_ptr[r] stores the starting index in col_idx/values for row r.
 * - row_ptr[r+1] - row_ptr[r] gives the exact degree (number of non-zeros) of row r.
 */
typedef struct {
    int num_rows;    /**< Total number of rows in the matrix */
    int num_cols;    /**< Total number of columns in the matrix */
    int nnz;         /**< Number of Non-Zero (NNZ) entries */
    int *row_ptr;    /**< Array of row offset pointers (Size: num_rows + 1) */
    int *col_idx;    /**< Column indices for non-zero values (Size: nnz) */
    float *values;   /**< Non-zero floating-point ratings (Size: nnz) */
} CSRMatrix;

/**
 * @struct Recommendation
 * @brief Stores an individual recommended item ID and its predicted score.
 */
typedef struct {
    int item_id;     /**< ID of the recommended item (-1 if empty/unassigned) */
    float score;     /**< Predicted collaborative filtering score */
} Recommendation;

/**
 * @struct Dataset
 * @brief Comprehensive bipartite interaction graph containing dual CSR representations.
 * 
 * Contains:
 * 1. user_item_csr: Enables O(degree) lookup of all items rated by user u.
 * 2. item_user_csr: Inverted index enabling O(degree) dot-product intersections between items.
 * 3. item_names: Optional human-readable titles (e.g., "Toy Story (1995)").
 */
typedef struct {
    int num_users;           /**< Total count of users in the network */
    int num_items;           /**< Total count of movies / items in the network */
    int num_ratings;         /**< Total count of rating edges */
    CSRMatrix user_item_csr; /**< Forward index: User (rows) -> Items (cols) */
    CSRMatrix item_user_csr; /**< Inverted index: Item (rows) -> Users (cols) */
    char **item_names;       /**< Optional item title strings (Size: num_items) */
} Dataset;

/* ============================================================================
 * Memory Management Prototypes
 * ============================================================================ */

/**
 * @brief Safely frees all dynamically allocated memory within a CSR matrix.
 * @param mat Pointer to CSRMatrix structure.
 */
void free_csr(CSRMatrix *mat);

/**
 * @brief Safely frees all memory associated with a Dataset, including both CSRs and titles.
 * @param ds Pointer to Dataset structure.
 */
void free_dataset(Dataset *ds);

/* ============================================================================
 * Dataset Construction and Ingestion Prototypes
 * ============================================================================ */

/**
 * @brief Constructs dual CSR structures (User-Item and Item-User) from raw COO interactions.
 * @param num_users Total number of users.
 * @param num_items Total number of items.
 * @param num_ratings Total number of interaction records.
 * @param coo_list Array of raw COOInteraction coordinates.
 * @return Fully initialized Dataset pointer, or NULL on allocation failure.
 */
Dataset* create_dataset_from_coo(int num_users, int num_items, int num_ratings, COOInteraction *coo_list);

/**
 * @brief Generates a synthetic power-law bipartite graph for scalability benchmarking.
 * @param num_users Number of simulated users.
 * @param num_items Number of simulated items.
 * @param avg_ratings_per_user Average interactions per user node.
 * @param seed Pseudo-random generator seed for reproducible benchmarks.
 * @return Fully initialized synthetic Dataset pointer.
 */
Dataset* create_synthetic_dataset(int num_users, int num_items, int avg_ratings_per_user, unsigned int seed);

/**
 * @brief Parses CSV / DAT datasets (e.g. MovieLens 1M `ratings.dat` or comma-separated files).
 * @param filepath Path to ratings file on disk.
 * @return Fully initialized Dataset pointer.
 */
Dataset* load_csv_dataset(const char *filepath);

/**
 * @brief Reads movie metadata (e.g. MovieLens `movies.dat`) and attaches titles to Dataset.
 * @param ds Pointer to target Dataset.
 * @param movies_filepath Path to movies.dat.
 */
void load_item_names(Dataset *ds, const char *movies_filepath);

/**
 * @brief Exports current Dataset interactions into a standard CSV file.
 * @param filepath Destination file path.
 * @param ds Source Dataset pointer.
 * @return True if write succeeded, False otherwise.
 */
bool save_dataset_to_csv(const char *filepath, const Dataset *ds);

#endif // DATASET_H
