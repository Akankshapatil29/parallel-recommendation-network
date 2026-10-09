#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "dataset.h"
#include "recommender.h"
#include "timer.h"

static void print_usage(const char *prog) {
    printf("===================================================================\n");
    printf("   Parallel Graph Recommendation Engine (OpenMP & Sequential)      \n");
    printf("===================================================================\n");
    printf("Usage (Command-line Mode):\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -u <num>       Number of users (Synthetic mode, default: 3000)\n");
    printf("  -i <num>       Number of items / movies (default: 1000)\n");
    printf("  -d <num>       Average ratings per user (default: 40)\n");
    printf("  -k <num>       Top-K recommendations per user (default: 10)\n");
    printf("  -f <path>      Load dataset from CSV/DAT file\n");
    printf("  -t <threads>   Run OpenMP with specific thread count\n");
    printf("  -b             Run automated multi-thread scaling benchmark\n");
    printf("  -s <path>      Save generated dataset to CSV and exit\n");
    printf("  -o <path>      Export benchmark metrics to CSV for plotting\n");
    printf("  -h             Show this help message\n\n");
    printf("Note: Run without any options for the Interactive User Prompt Wizard!\n");
    printf("===================================================================\n");
}

static void print_single_user_recs(int u, const RecommendationResults *res, const Dataset *ds) {
    if (u < 0 || u >= res->num_users) {
        printf("[Notice] User ID %d is outside the dataset range (0 to %d).\n", u, res->num_users - 1);
        return;
    }
    printf("-------------------------------------------------------------------\n");
    printf("👤 User %d Top-%d Recommended Movies:\n", u, res->top_k);
    int valid_count = 0;
    for (int k = 0; k < res->top_k; k++) {
        Recommendation r = res->recs[u * res->top_k + k];
        if (r.item_id != -1) {
            valid_count++;
            if (ds && ds->item_names && r.item_id < ds->num_items && ds->item_names[r.item_id]) {
                printf("   [%2d] %-44s (Score: %.2f)\n", k + 1, ds->item_names[r.item_id], r.score);
            } else {
                printf("   [%2d] Movie ID %-4d                     (Score: %.2f)\n", k + 1, r.item_id, r.score);
            }
        }
    }
    if (valid_count == 0) {
        printf("   (No unrated items with sufficient similarity score / Cold-start user)\n");
    }
}

static void print_sample_recs(const RecommendationResults *res, const Dataset *ds, int num_users_to_print) {
    if (num_users_to_print <= 0) return;
    if (num_users_to_print > res->num_users) num_users_to_print = res->num_users;

    printf("\n===================================================================\n");
    printf("        SAMPLE PERSONALIZED RECOMMENDATIONS (First %d Users)        \n", num_users_to_print);
    printf("===================================================================\n");

    // In MovieLens, User IDs start at 1, so if User 0 is empty, start from User 1
    int start_user = 1;
    if (ds && ds->user_item_csr.row_ptr[1] > 0) {
        // User 0 has ratings, start from 0
        start_user = 0;
    }

    int printed = 0;
    for (int u = start_user; u < res->num_users && printed < num_users_to_print; u++) {
        print_single_user_recs(u, res, ds);
        printed++;
    }
    printf("===================================================================\n");
}

static void run_engine(Dataset *ds, int top_k, int min_common_users, int target_threads, bool run_benchmark_mode, const char *export_csv_file, int num_preview_users) {
    int max_system_threads = omp_get_max_threads();

    // Calculate actual active user count (excluding empty user 0 if present)
    int active_users = ds->num_users;
    if (ds->user_item_csr.row_ptr[1] == 0) {
        active_users = ds->num_users - 1;
    }

    printf("\n===================================================================\n");
    printf("                      DATASET GRAPH SUMMARY                        \n");
    printf("===================================================================\n");
    printf("  Total Users (Nodes)     : %d\n", active_users);
    printf("  Total Movies / Items    : %d\n", ds->num_items);
    printf("  Total Ratings (Edges)   : %d\n", ds->num_ratings);
    double sparsity = 100.0 * (1.0 - ((double)ds->num_ratings / ((double)ds->num_users * ds->num_items)));
    printf("  Graph Sparsity          : %.2f %%\n", sparsity);
    printf("  Available CPU Cores     : %d logical cores\n", max_system_threads);
    printf("===================================================================\n\n");

    /* ---------------------------------------------------------
     * 1. RUN SEQUENTIAL IMPLEMENTATION
     * --------------------------------------------------------- */
    printf("[1/3] Running Sequential Baseline Algorithm (1 Core)...\n");
    RecommendationResults *seq_res = run_sequential_recommender(ds, top_k, min_common_users);
    printf("      -> Movie Similarity Time   : %8.4f seconds\n", seq_res->time_similarity_sec);
    printf("      -> Top-%d Recs Ranking Time : %8.4f seconds\n", top_k, seq_res->time_recommendation_sec);
    printf("      -> Total Sequential Time   : %8.4f seconds\n\n", seq_res->time_total_sec);

    /* ---------------------------------------------------------
     * 2. RUN PARALLEL OPENMP IMPLEMENTATION & VERIFICATION
     * --------------------------------------------------------- */
    int omp_threads = (target_threads > 0) ? target_threads : max_system_threads;
    printf("[2/3] Running OpenMP Parallel Algorithm (%d Threads/Cores)...\n", omp_threads);
    RecommendationResults *omp_res = run_parallel_omp_recommender(ds, top_k, min_common_users, omp_threads);
    printf("      -> Movie Similarity Time   : %8.4f seconds\n", omp_res->time_similarity_sec);
    printf("      -> Top-%d Recs Ranking Time : %8.4f seconds\n", top_k, omp_res->time_recommendation_sec);
    printf("      -> Total OpenMP Time       : %8.4f seconds\n", omp_res->time_total_sec);
    
    double speedup = seq_res->time_total_sec / omp_res->time_total_sec;
    double efficiency = (speedup / (double)omp_threads) * 100.0;
    printf("      -> Speedup Achieved        : %8.2fx Faster!\n", speedup);
    printf("      -> Parallel Efficiency     : %8.2f %%\n\n", efficiency);

    // Verify correctness
    printf("[3/3] Verifying Parallel Output Correctness against Sequential...\n");
    VerificationReport v_report = verify_results(seq_res, omp_res, 1e-4);
    print_verification_report(&v_report);

    // Display sample recommendations
    if (num_preview_users > 0) {
        print_sample_recs(omp_res, ds, num_preview_users);
    }

    // Allow interactive user search if interactive
    char search_buf[64];
    while (1) {
        printf("\nWould you like to view recommendations for a specific User ID?\n(Enter a User ID from 1 to %d, or press Enter to continue): ", active_users);
        if (fgets(search_buf, sizeof(search_buf), stdin)) {
            if (search_buf[0] == '\n' || search_buf[0] == '\r') {
                break;
            }
            int target_u = atoi(search_buf);
            if (target_u > 0 && target_u < ds->num_users) {
                print_single_user_recs(target_u, omp_res, ds);
            } else {
                printf("[Invalid User ID] Please enter a valid number between 1 and %d.\n", active_users);
            }
        } else {
            break;
        }
    }

    /* ---------------------------------------------------------
     * 3. MULTI-THREAD BENCHMARK & CSV EXPORT
     * --------------------------------------------------------- */
    if (run_benchmark_mode) {
        printf("\n=========================================================================================\n");
        printf("                   MULTI-THREAD SCALING BENCHMARK RESULTS TABLE                          \n");
        printf("=========================================================================================\n");
        printf(" Threads | Similarity (s) | Recs Time (s) | Total Time (s) | Speedup (x) | Efficiency (%%) \n");
        printf("---------+----------------+---------------+----------------+-------------+---------------\n");
        printf(" Seq (1) | %14.4f | %13.4f | %14.4f | %11.2f | %13.2f \n",
               seq_res->time_similarity_sec, seq_res->time_recommendation_sec, seq_res->time_total_sec, 1.0, 100.0);

        FILE *exp_fp = NULL;
        if (export_csv_file) {
            exp_fp = fopen(export_csv_file, "w");
            if (exp_fp) {
                fprintf(exp_fp, "threads,sim_time,recs_time,total_time,speedup,efficiency\n");
                fprintf(exp_fp, "0,%.4f,%.4f,%.4f,1.0,100.0\n",
                        seq_res->time_similarity_sec, seq_res->time_recommendation_sec, seq_res->time_total_sec);
            }
        }

        int thread_configs[] = {1, 2, 4, 8, 12, 16};
        int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);

        for (int c = 0; c < num_configs; c++) {
            int t = thread_configs[c];
            if (t > max_system_threads) continue;

            RecommendationResults *bench_res = run_parallel_omp_recommender(ds, top_k, min_common_users, t);
            double sp = seq_res->time_total_sec / bench_res->time_total_sec;
            double eff = (sp / (double)t) * 100.0;

            printf(" %7d | %14.4f | %13.4f | %14.4f | %11.2fx| %12.2f%% \n",
                   t, bench_res->time_similarity_sec, bench_res->time_recommendation_sec,
                   bench_res->time_total_sec, sp, eff);

            if (exp_fp) {
                fprintf(exp_fp, "%d,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                        t, bench_res->time_similarity_sec, bench_res->time_recommendation_sec,
                        bench_res->time_total_sec, sp, eff);
            }

            free_results(bench_res);
        }
        if (exp_fp) {
            fclose(exp_fp);
            printf("  [Export] Benchmark metrics exported to: %s\n", export_csv_file);
        }
        printf("=========================================================================================\n");
    }

    free_results(seq_res);
    free_results(omp_res);
}

static void run_interactive_wizard(void) {
    char input_buf[128];
    int max_threads = omp_get_max_threads();

    printf("\n");
    printf("===================================================================\n");
    printf("    🎬 WELCOME TO THE PARALLEL RECOMMENDATION NETWORK ENGINE       \n");
    printf("===================================================================\n");
    printf(" Detected CPU Hardware: %d Parallel Cores / Threads Available\n", max_threads);
    printf("-------------------------------------------------------------------\n\n");

    while (1) {
        // STEP 1: CHOOSE DATASET
        int dataset_choice = 1;
        while (1) {
            printf(">>> STEP 1: CHOOSE DATASET SOURCE\n");
            printf("  [1] MovieLens 1 Million Dataset (1,000,209 Ratings, 6,040 Users, 3,952 Movies)\n");
            printf("  [2] Sample Ratings Dataset (120,501 Ratings, 3,000 Users, 1,000 Movies)\n");
            printf("  [3] Create Custom User-Movie Graph (Enter your own numbers)\n");
            printf("  [4] Exit Program\n");
            printf("Enter your choice (1 to 4) [Default: 1]: ");

            if (fgets(input_buf, sizeof(input_buf), stdin)) {
                if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                    dataset_choice = 1;
                    break;
                }
                int val = atoi(input_buf);
                if (val >= 1 && val <= 4) {
                    dataset_choice = val;
                    break;
                }
            }
            printf("\n[Invalid Input] Please enter a valid number between 1 and 4.\n\n");
        }

        if (dataset_choice == 4) {
            printf("\nExiting. Thank you!\n");
            break;
        }

        Dataset *ds = NULL;
        int num_users = 3000;
        int num_items = 1000;
        int avg_ratings = 40;

        if (dataset_choice == 1) {
            printf("\n>> Loading MovieLens 1 Million dataset (data\\ml-1m\\ratings.dat)...");
            fflush(stdout);
            ds = load_csv_dataset("data\\ml-1m\\ratings.dat");
            if (ds) {
                printf(" Loaded %d ratings!\n", ds->num_ratings);
                printf(">> Loading movie titles (data\\ml-1m\\movies.dat)...");
                fflush(stdout);
                load_item_names(ds, "data\\ml-1m\\movies.dat");
                printf(" Done!\n");
            } else {
                printf("\n[Error] Could not find data\\ml-1m\\ratings.dat!\n");
            }
        } else if (dataset_choice == 3) {
            printf("\n--- CUSTOM GRAPH DATASET CONFIGURATION ---\n");
            
            // Users prompt with validation
            while (1) {
                printf("Enter number of USERS (Range: 10 to 50000) [Default: 3000]: ");
                if (fgets(input_buf, sizeof(input_buf), stdin)) {
                    if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                        num_users = 3000;
                        break;
                    }
                    int val = atoi(input_buf);
                    if (val >= 10 && val <= 50000) {
                        num_users = val;
                        break;
                    }
                }
                printf("[Invalid Input] Please enter a number between 10 and 50000.\n");
            }

            // Items prompt with validation
            while (1) {
                printf("Enter number of MOVIES / ITEMS (Range: 5 to 5000) [Default: 1000]: ");
                if (fgets(input_buf, sizeof(input_buf), stdin)) {
                    if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                        num_items = 1000;
                        break;
                    }
                    int val = atoi(input_buf);
                    if (val >= 5 && val <= 5000) {
                        num_items = val;
                        break;
                    }
                }
                printf("[Invalid Input] Please enter a number between 5 and 5000.\n");
            }

            // Average ratings prompt with validation
            while (1) {
                printf("Enter average ratings per user (Range: 1 to 200) [Default: 40]: ");
                if (fgets(input_buf, sizeof(input_buf), stdin)) {
                    if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                        avg_ratings = 40;
                        break;
                    }
                    int val = atoi(input_buf);
                    if (val >= 1 && val <= 200) {
                        avg_ratings = val;
                        break;
                    }
                }
                printf("[Invalid Input] Please enter a number between 1 and 200.\n");
            }

            printf("\n>> Generating custom bipartite graph (%d Users, %d Movies, ~%d ratings/user)...",
                   num_users, num_items, avg_ratings);
            fflush(stdout);
            ds = create_synthetic_dataset(num_users, num_items, avg_ratings, 42);
            printf(" Done!\n");
        } else {
            printf("\n>> Loading dataset from data\\sample_ratings.csv...");
            fflush(stdout);
            ds = load_csv_dataset("data\\sample_ratings.csv");
            if (!ds) {
                printf("\n[Notice] Generating fresh sample_ratings.csv...");
                ds = create_synthetic_dataset(num_users, num_items, avg_ratings, 42);
                save_dataset_to_csv("data\\sample_ratings.csv", ds);
            }
            printf(" Done!\n");
        }

        if (!ds) {
            printf("[Error] Failed to load or generate dataset. Please try again.\n\n");
            continue;
        }

        // STEP 2: Top-K recommendations prompt with validation
        int top_k = 10;
        printf("\n>>> STEP 2: RECOMMENDATION DEPTH (TOP-K)\n");
        while (1) {
            printf("How many Top recommendations do you want per user? (Range: 1 to 50) [Default: 10]: ");
            if (fgets(input_buf, sizeof(input_buf), stdin)) {
                if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                    top_k = 10;
                    break;
                }
                int val = atoi(input_buf);
                if (val >= 1 && val <= 50) {
                    top_k = val;
                    break;
                }
            }
            printf("[Invalid Input] Please enter an integer between 1 and 50.\n");
        }

        // STEP 3: Execution Mode prompt with simpler friendly names
        int exec_mode = 1;
        printf("\n>>> STEP 3: HOW DO YOU WANT TO RUN THE TEST?\n");
        printf("  [1] Quick Test (Compare 1 Core vs %d Cores)\n", max_threads);
        printf("  [2] Full Speed Test (Test 1, 2, 4, 8, 12, %d Cores to see speedup)\n", max_threads);
        printf("  [3] Custom Test (Choose exact number of cores to use)\n");

        while (1) {
            printf("Enter your choice (1, 2, or 3) [Default: 1]: ");
            if (fgets(input_buf, sizeof(input_buf), stdin)) {
                if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                    exec_mode = 1;
                    break;
                }
                int val = atoi(input_buf);
                if (val >= 1 && val <= 3) {
                    exec_mode = val;
                    break;
                }
            }
            printf("[Invalid Input] Please enter 1, 2, or 3.\n");
        }

        int target_threads = max_threads;
        bool run_benchmark = false;

        if (exec_mode == 2) {
            run_benchmark = true;
        } else if (exec_mode == 3) {
            while (1) {
                printf("Enter number of threads to test (Valid Range: 1 to %d) [Default: %d]: ", max_threads, max_threads);
                if (fgets(input_buf, sizeof(input_buf), stdin)) {
                    if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                        target_threads = max_threads;
                        break;
                    }
                    int val = atoi(input_buf);
                    if (val >= 1 && val <= max_threads) {
                        target_threads = val;
                        break;
                    }
                }
                printf("[Invalid Input] Please enter an integer between 1 and %d.\n", max_threads);
            }
        }

        // STEP 4: Choose how many users to preview in recommendations
        int num_preview_users = 3;
        printf("\n>>> STEP 4: PREVIEW DISPLAY\n");
        while (1) {
            printf("How many sample users would you like to preview recommendations for? (Valid Range: 1 to 20) [Default: 3]: ");
            if (fgets(input_buf, sizeof(input_buf), stdin)) {
                if (input_buf[0] == '\n' || input_buf[0] == '\r') {
                    num_preview_users = 3;
                    break;
                }
                int val = atoi(input_buf);
                if (val >= 1 && val <= 20) {
                    num_preview_users = val;
                    break;
                }
            }
            printf("[Invalid Input] Please enter an integer between 1 and 20.\n");
        }

        // Run the engine
        run_engine(ds, top_k, 2, target_threads, run_benchmark, "benchmark\\thread_scaling.csv", num_preview_users);
        free_dataset(ds);

        printf("\n-------------------------------------------------------------------\n");
        printf("Would you like to run another experiment? (y/n) [Default: y]: ");
        if (fgets(input_buf, sizeof(input_buf), stdin)) {
            if (input_buf[0] == 'n' || input_buf[0] == 'N') {
                printf("\nExiting. Thank you!\n");
                break;
            }
        }
        printf("\n\n");
    }
}

int main(int argc, char *argv[]) {
    // If no arguments were passed, run the Interactive User Prompt Wizard!
    if (argc == 1) {
        run_interactive_wizard();
        return 0;
    }

    int num_users = 3000;
    int num_items = 1000;
    int avg_ratings = 40;
    int top_k = 10;
    int min_common_users = 2;
    int target_threads = 0;
    bool run_benchmark_mode = false;
    const char *csv_file = NULL;
    const char *save_csv_file = NULL;
    const char *export_csv_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            num_users = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            num_items = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            avg_ratings = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            target_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0) {
            run_benchmark_mode = true;
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            save_csv_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            export_csv_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (save_csv_file) {
        printf("  Generating and saving dataset (%d Users, %d Items) to: %s\n",
               num_users, num_items, save_csv_file);
        Dataset *gen_ds = create_synthetic_dataset(num_users, num_items, avg_ratings, 42);
        if (gen_ds) {
            save_dataset_to_csv(save_csv_file, gen_ds);
            printf("  [SUCCESS] Saved %d interactions to %s\n", gen_ds->num_ratings, save_csv_file);
            free_dataset(gen_ds);
        }
        return 0;
    }

    Dataset *ds = NULL;
    if (csv_file) {
        ds = load_csv_dataset(csv_file);
        if (strstr(csv_file, "ratings.dat")) {
            load_item_names(ds, "data\\ml-1m\\movies.dat");
        }
    } else {
        ds = create_synthetic_dataset(num_users, num_items, avg_ratings, 42);
    }

    if (!ds) {
        fprintf(stderr, "[Error] Failed to initialize dataset.\n");
        return 1;
    }

    run_engine(ds, top_k, min_common_users, target_threads, run_benchmark_mode, export_csv_file, 3);
    free_dataset(ds);

    return 0;
}
