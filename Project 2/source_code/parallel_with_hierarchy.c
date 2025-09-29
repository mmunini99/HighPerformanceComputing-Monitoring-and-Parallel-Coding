/*  Hierarchical MPI/OpenMP Fractal Generator

 *  Level 1: 2D Block decomposition across MPI processes

 *  Level 2: Cyclic distribution within OpenMP threads

 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>



static inline unsigned char compute_mandelbrot(double real_c, double imag_c, int iterations)

{

    double real_z = real_c, imag_z = imag_c;

    for (int iter = 0; iter < iterations; ++iter) {

        double real_sq = real_z * real_z, imag_sq = imag_z * imag_z;

        if (real_sq + imag_sq > 4.0) return (unsigned char)iter;

        imag_z = 2.0 * real_z * imag_z + imag_c;

        real_z = real_sq - imag_sq + real_c;

    }

    return (unsigned char)iterations;

}



// Function to find best 2D process grid factorization

void find_process_grid(int total_processes, int *px, int *py) {

    int best_px = 1;

    int min_diff = total_processes;

    

    // Find factors that minimize difference (closest to square)

    for (int i = 1; i <= (int)sqrt(total_processes) + 1; ++i) {

        if (total_processes % i == 0) {

            int j = total_processes / i;

            int diff = abs(i - j);

            if (diff < min_diff) {

                min_diff = diff;

                best_px = i;

            }

        }

    }

    

    *px = best_px;

    *py = total_processes / best_px;

}



int main(int argc, char *argv[])

{

    MPI_Init(&argc, &argv);

    int process_id, total_processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    

    /* ---------------- parameter validation ---------------- */

    if (argc != 9) {

        if (process_id == 0) fprintf(stderr, "Usage: %s width height x_min y_min x_max y_max max_iter threads\n", argv[0]);

        MPI_Finalize();

        return 1;

    }

    

    const int img_width      = atoi(argv[1]);

    const int img_height     = atoi(argv[2]);

    const double min_x       = atof(argv[3]);

    const double min_y       = atof(argv[4]);

    const double max_x       = atof(argv[5]);

    const double max_y       = atof(argv[6]);

    const int max_iterations = atoi(argv[7]);

    const int thread_count   = atoi(argv[8]);

    

    omp_set_num_threads(thread_count);

    

    /* =============== LEVEL 1: 2D MPI BLOCK DECOMPOSITION =============== */

    

    int px, py;  // Process grid dimensions

    find_process_grid(total_processes, &px, &py);

    

    // Find this process's position in the 2D grid

    int proc_x = process_id % px;

    int proc_y = process_id / px;

    

    // Calculate block dimensions for this process

    int base_block_width = img_width / px;

    int extra_width = img_width % px;

    int local_width = base_block_width + (proc_x < extra_width ? 1 : 0);

    int width_offset = proc_x * base_block_width + (proc_x < extra_width ? proc_x : extra_width);

    

    int base_block_height = img_height / py;

    int extra_height = img_height % py;

    int local_height = base_block_height + (proc_y < extra_height ? 1 : 0);

    int height_offset = proc_y * base_block_height + (proc_y < extra_height ? proc_y : extra_height);

    

    // Local buffer for this process's block

    unsigned char *local_buffer = (unsigned char *)malloc(local_width * local_height);

    if (!local_buffer) {

        fprintf(stderr, "Memory allocation failed on process %d\n", process_id);

        MPI_Finalize();

        return 1;

    }

    

    if (process_id == 0) {

        printf("Process grid: %dx%d\n", px, py);

        printf("Image: %dx%d, Block size: %dx%d (approx)\n", 

               img_width, img_height, base_block_width, base_block_height);

    }

    

    /* ============== COMPUTATION WITH HIERARCHICAL PARALLELIZATION ============== */

    

    double start_time = MPI_Wtime();

    

    /* =============== LEVEL 2: CYCLIC OPENMP WITHIN BLOCK =============== */

    int total_pixels = local_width * local_height;

    

    #pragma omp parallel

    {

        int thread_id = omp_get_thread_num();

        int num_threads = omp_get_num_threads();

        

        // Each thread processes pixels cyclically within the local block

        for (int pixel_idx = thread_id; pixel_idx < total_pixels; pixel_idx += num_threads) {

            

            // Convert linear index to 2D coordinates within local block

            int local_x = pixel_idx % local_width;

            int local_y = pixel_idx / local_width;

            

            // Convert to global image coordinates

            int global_x = width_offset + local_x;

            int global_y = height_offset + local_y;

            

            // Map to complex plane coordinates

            double coord_x = min_x + global_x * (max_x - min_x) / img_width;

            double coord_y = min_y + global_y * (max_y - min_y) / img_height;

            

            // Compute Mandelbrot value

            unsigned char value = compute_mandelbrot(coord_x, coord_y, max_iterations);

            

            // Store in row-major order within local block

            local_buffer[local_y * local_width + local_x] = value;

        }

    }

    

    double computation_time = MPI_Wtime() - start_time;

    

    /* =============== GATHER RESULTS TO MASTER PROCESS =============== */

    

    unsigned char *final_image = NULL;

    int *receive_sizes = NULL;

    int *displacements = NULL;

    

    if (process_id == 0) {

        final_image = (unsigned char *)malloc(img_width * img_height);

        receive_sizes = (int *)malloc(total_processes * sizeof(int));

        displacements = (int *)malloc(total_processes * sizeof(int));

        

        if (!final_image || !receive_sizes || !displacements) {

            fprintf(stderr, "Memory allocation failed on master process\n");

            MPI_Finalize();

            return 1;

        }

    }

    

    // Gather all local block sizes

    int local_size = local_width * local_height;

    MPI_Gather(&local_size, 1, MPI_INT, receive_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    

    if (process_id == 0) {

        displacements[0] = 0;

        for (int i = 1; i < total_processes; ++i) {

            displacements[i] = displacements[i-1] + receive_sizes[i-1];

        }

    }

    

    // Gather all local blocks

    MPI_Gatherv(local_buffer, local_size, MPI_UNSIGNED_CHAR,

                final_image, receive_sizes, displacements, MPI_UNSIGNED_CHAR,

                0, MPI_COMM_WORLD);

    

    /* =============== RECONSTRUCT FINAL IMAGE AND OUTPUT =============== */

    

    if (process_id == 0) {

        // Reconstruct image from 2D blocks

        unsigned char *reconstructed_image = (unsigned char *)malloc(img_width * img_height);

        

        int block_idx = 0;

        for (int proc = 0; proc < total_processes; ++proc) {

            int p_x = proc % px;

            int p_y = proc / px;

            

            // Calculate block dimensions for process proc

            int block_w = base_block_width + (p_x < extra_width ? 1 : 0);

            int block_h = base_block_height + (p_y < extra_height ? 1 : 0);

            int w_off = p_x * base_block_width + (p_x < extra_width ? p_x : extra_width);

            int h_off = p_y * base_block_height + (p_y < extra_height ? p_y : extra_height);

            

            // Copy block data to correct position in final image

            for (int by = 0; by < block_h; ++by) {

                for (int bx = 0; bx < block_w; ++bx) {

                    int global_pos = (h_off + by) * img_width + (w_off + bx);

                    int block_pos = by * block_w + bx;

                    reconstructed_image[global_pos] = final_image[displacements[proc] + block_pos];

                }

            }

        }

        

        // Write PGM file

        FILE *output_file = fopen("fractal_hierarchical.pgm", "wb");

        if (output_file) {

            fprintf(output_file, "P5\n%d %d\n255\n", img_width, img_height);

            fwrite(reconstructed_image, 1, img_width * img_height, output_file);

            fclose(output_file);

        }

        

        // Performance statistics

        printf("=== HIERARCHICAL PARALLELIZATION RESULTS ===\n");

        printf("Process grid: %dx%d (%d processes)\n", px, py, total_processes);

        printf("Threads per process: %d\n", thread_count);

        printf("Total parallel units: %d\n", total_processes * thread_count);

        printf("Image size: %dx%d = %d pixels\n", img_width, img_height, img_width * img_height);

        printf("Computation time: %.6f seconds\n", computation_time);

        printf("Pixels per second: %.2f million\n", (img_width * img_height) / (computation_time * 1e6));

        

        /* --------------- TIMING OUTPUT FOR EXPERIMENTS --------------- */

        printf("%.6f\n", computation_time);

        

        free(reconstructed_image);

        free(final_image);

        free(receive_sizes);

        free(displacements);

    }

    

    free(local_buffer);

    MPI_Finalize();

    return 0;

}

