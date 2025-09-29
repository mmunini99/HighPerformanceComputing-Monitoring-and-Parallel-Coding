/*  MPI/OpenMP Fractal Generator – vertical strip division  */
#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int process_id, total_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    
    /* ---------------- parameter validation ---------------- */
    if (argc != 9) {
        if (process_id == 0) fprintf(stderr, "incorrect arguments\n");
        MPI_Finalize();
        return 1;
    }
    
    const int img_width    = atoi(argv[1]);
    const int img_height   = atoi(argv[2]);
    const double min_x     = atof(argv[3]);
    const double min_y     = atof(argv[4]);
    const double max_x     = atof(argv[5]);
    const double max_y     = atof(argv[6]);
    const int max_iterations = atoi(argv[7]);
    const int thread_count   = atoi(argv[8]);
    
    omp_set_num_threads(thread_count);
    
    /* --------------- vertical strip division --------------- */
    int base_strips = img_width / total_processes;
    int extra_strips = img_width % total_processes;
    int local_strips = base_strips + (process_id < extra_strips ? 1 : 0);
    int strip_offset = process_id * base_strips + (process_id < extra_strips ? process_id : extra_strips);
    
    /* local storage is arranged strip-major */
    unsigned char *local_buffer = (unsigned char *)malloc(local_strips * img_height);
    
    double start_time = MPI_Wtime();
    
    #pragma omp parallel for schedule(dynamic)
    for (int strip = 0; strip < local_strips; ++strip) {
        int global_strip = strip_offset + strip;
        double coord_x = min_x + global_strip * (max_x - min_x) / img_width;
        for (int pixel = 0; pixel < img_height; ++pixel) {
            double coord_y = min_y + pixel * (max_y - min_y) / img_height;
            local_buffer[strip * img_height + pixel] = compute_mandelbrot(coord_x, coord_y, max_iterations);
        }
    }
    
    /* --------------- data collection at master --------------- */
    int *receive_sizes = NULL, *data_offsets = NULL;
    unsigned char *final_image = NULL;
    
    if (process_id == 0) {
        receive_sizes = (int *)malloc(total_processes * sizeof(int));
        data_offsets = (int *)malloc(total_processes * sizeof(int));
        final_image = (unsigned char *)malloc(img_width * img_height);
        
        for (int proc = 0, offset = 0; proc < total_processes; ++proc) {
            int strips = base_strips + (proc < extra_strips ? 1 : 0);
            receive_sizes[proc] = strips * img_height;
            data_offsets[proc] = offset;
            offset += strips * img_height;
        }
    }
    
    MPI_Gatherv(local_buffer, local_strips * img_height, MPI_UNSIGNED_CHAR,
                final_image, receive_sizes, data_offsets, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    
    double computation_time = MPI_Wtime() - start_time;
    
    if (process_id == 0) {
        /* rearrange to standard pixel-major for PGM */
        unsigned char *standard_layout = (unsigned char *)malloc(img_width * img_height);
        
        for (int proc = 0, strip_base = 0; proc < total_processes; ++proc) {
            int strips = base_strips + (proc < extra_strips ? 1 : 0);
            for (int strip = 0; strip < strips; ++strip)
                for (int pixel = 0; pixel < img_height; ++pixel)
                    standard_layout[pixel * img_width + (strip_base + strip)] =
                        final_image[data_offsets[proc] + strip * img_height + pixel];
            strip_base += strips;
        }
        
        FILE *output_file = fopen("fractal_column.pgm", "wb");
        if (output_file) {
            fprintf(output_file, "P5\n%d %d\n255\n", img_width, img_height);
            fwrite(standard_layout, 1, img_width * img_height, output_file);
            fclose(output_file);
        }
        
        free(standard_layout);
        free(final_image);
        free(receive_sizes);
        free(data_offsets);
        
        /* --------------- TIMING OUTPUT ONLY --------------- */
        printf("%.6f\n", computation_time);
    }
    
    free(local_buffer);
    MPI_Finalize();
    return 0;
}