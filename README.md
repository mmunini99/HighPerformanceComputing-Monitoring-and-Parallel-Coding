# HighPerformanceComputing-Monitoring-and-Parallel-Coding


In this repository there is all the necessary code for two HPC project:
- **PROJECT 1 :** *Comparison between OpenMPI algorithms for collective operation*
- **PROJECT 2 :** *Mandelbrot set implementation with a hybrid MPI+OpenMP code*.

## Initial Set up
These projects have been run in Orfeo cluster. So, the first the user needs to do once logged in into Orfeo is to:
* Download the GitHub repository
* Remove all output, photo and slurm-output folders
* Define the folder structure and download the requirements (maybe could be needed to give right permission to the file using   ```chmod +x file.sh```):

* For *Project 1* move into Project 1 directory and the into source code folder:
```bash
cd Project\ 1/source_code/

bash get_osu.sh # this will download the osu benchmarks library and create the folder structure
```

* For *Project 2* move into Project 2 directory and the into source code folder:
```bash
cd Project\ 2/source_code/

bash get_structured_folder.sh # this will create the folder structure
```

## How to run a job on Orfeo

To run a job and let it be scheduled by slurm, the user needs to run the following command:
```bash
sbatch file.sh
```


## Project 1: 
For this task it has been compared **Broadcast** and **Reduce** algorithms and specifically for each of one of these algorithms there is a version with fixed and variable size. For each of these algorithms, both for kind of size, three types of algorithm implementation have been tested
- **Broadcast**: 
  - Algo 1: Basic Linear (type 1 in osu library)
  - Algo 2: Pipeline (type 3 in osu library)  --> here the user will find two bash files, since for time constraint it was not possible to complete the job.
  - Algo 3: Binary Tree (type 5 in osu library)
- **Reduce**:
  - Algo 1: Linear (type 1 in osu library)
  - Algo 2: Chain (type 2 in osu library)
  - Algo 3: Binary (type 4 in osu library)


## Project 2: 
For developing a strategy for Mandelbrot Set Parallelization, an **Hybrid MPI** + **OpenMP** strategy has been defined and three version have been tried: by splitting the image on columns, then on rows and finally as an extra, a final trial tentative hierarchical using 2D block decomposition + cyclic pixel distribution (I was curious on how to (try) to improve the algorithms used before).


For all the projects there is an analysis done in Python and a report. In the presentation will be highlighted a problem found during the development.


## The structure of the folder:
```
C:.
¦   .gitignore
¦   README.md
¦   structure.txt
¦   
+---Project 1
¦   +---output
¦   ¦   ¦   latency.txt
¦   ¦   ¦   
¦   ¦   +---broadcast_fixed
¦   ¦   ¦       broadcast_algo1_fixed_core.csv
¦   ¦   ¦       broadcast_algo2_fixed_core.csv
¦   ¦   ¦       broadcast_algo2_fixed_core_secondpart.csv
¦   ¦   ¦       broadcast_algo3_fixed_core.csv
¦   ¦   ¦       
¦   ¦   +---broadcast_var
¦   ¦   ¦       broadcast_algo1_variable_core.csv
¦   ¦   ¦       broadcast_algo2_variable_core.csv
¦   ¦   ¦       broadcast_algo3_variable_core.csv
¦   ¦   ¦       
¦   ¦   +---reduce_fixed
¦   ¦   ¦       reduce_algo1_fixed_core.csv
¦   ¦   ¦       reduce_algo2_fixed_core.csv
¦   ¦   ¦       reduce_algo3_fixed_core.csv
¦   ¦   ¦       
¦   ¦   +---reduce_var
¦   ¦           reduce_algo1_variable_core.csv
¦   ¦           reduce_algo2_variable_core.csv
¦   ¦           reduce_algo3_variable_core.csv
¦   ¦           
¦   +---slurm-output

¦   ¦       
¦   +---source_code
¦           broadcast_size_fix_algo1.sh
¦           broadcast_size_fix_algo2.sh
¦           broadcast_size_fix_algo2_second_part.sh
¦           broadcast_size_fix_algo3.sh
¦           broadcast_variable_size_algo1.sh
¦           broadcast_variable_size_algo2.sh
¦           broadcast_variable_size_algo3.sh
¦           get_osu.sh
¦           latency_test.sh
¦           reduce_size_fix_algo1.sh
¦           reduce_size_fix_algo2.sh
¦           reduce_size_fix_algo3.sh
¦           reduce_variable_size_algo1.sh
¦           reduce_variable_size_algo2.sh
¦           reduce_variable_size_algo3.sh
¦           
+---Project 2
    ¦   photo.zip
    ¦   
    +---output
    ¦   +---strong_scaling_mpi
    ¦   ¦       mpi_strong_columns.csv
    ¦   ¦       mpi_strong_hiera.csv
    ¦   ¦       mpi_strong_rows.csv
    ¦   ¦       
    ¦   +---strong_scaling_omp
    ¦   ¦       omp_strong_columns.csv
    ¦   ¦       omp_strong_hiera.csv
    ¦   ¦       omp_strong_rows.csv
    ¦   ¦       
    ¦   +---weak_scaling_mpi
    ¦   ¦       mpi_weak_columns.csv
    ¦   ¦       mpi_weak_hiera.csv
    ¦   ¦       mpi_weak_rows.csv
    ¦   ¦       
    ¦   +---weak_scaling_omp
    ¦           omp_weak_columns.csv
    ¦           omp_weak_hiera.csv
    ¦           omp_weak_rows.csv
    ¦           
    +---photo
    ¦       fractal_column.pgm
    ¦       fractal_hierarchical.pgm
    ¦       fractal_row.pgm
    ¦       
    +---slurm-output

    ¦       
    +---source_code
            column_mpi_strong_experiment.sh
            column_mpi_weak_experiment.sh
            column_openmp_strong_experiment.sh
            column_openmp_weak_experiment.sh
            get_structured_folder.sh
            hiera_mpi_strong_experiment.sh
            hiera_mpi_weak_experiment.sh
            hiera_openmp_strong_experiment.sh
            hiera_openmp_weak_experiment.sh
            parallel_on_columns_code
            parallel_on_columns_code.c
            parallel_on_rows_code
            parallel_on_rows_code.c
            parallel_with_hierarchy
            parallel_with_hierarchy.c
            row_mpi_strong_experiment.sh
            row_mpi_weak_experiment.sh
            row_openmp_strong_experiment - Copia.sh
            row_openmp_strong_experiment.sh
            row_openmp_weak_experiment - Copia.sh
            row_openmp_weak_experiment.sh
            

```
