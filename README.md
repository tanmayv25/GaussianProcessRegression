## Problem 

Parallelize Gaussian Process Regression MxM double-precision floating point values and run on a single streaming multiprocessor of Nvidia K20 GPU.  

## Steps
The steps to compile and execute the code :

    1. module load intel/2017A CUDA
    2. nvcc gpr_gpu.cu -o gpr_all_gpu.exe
    3. ./gpr_all_gpu.exe 64 0.05 0.05
    4. nvcc gpr_gpu_single.cu -o gpr_single_gpu.exe
    5. ./gpr_single_gpu.exe 64 0.05 0.05
    6. bsub < gpr_gpu.job (the job file is attached in appendix)


## Analysis
As the problem statement required us to use only a single SM,  the inter-thread communication would not be an issue. The main performance bottlenecks were identified as the data dependency in solvers and cholesky decomposition. For example, the triangular solver function can not jump to next row until and unless the previous unknown has been computed. And for Cholesky, the entire column/row values of output triangular matrix depends upon diagnonal element and previous columns/rows for Lower Triangular/Upper Triangular respectively. Given these data dependencies, we are restricted to perform these calculations in-order. 

There are basically two components for parallelization, sum reductions and data independent computations. 

####Sum Reduction:
In my implementation, I  have used a shared memory array of type double and length equal to the number of threads to store the partial sums which each thread calculates in parallel. In the next step, each thread calls upon the routine to get_total_sum. In this routine, at every step, half as many threads  aggregates the partial some from other halves in a stride friendly manner for memory bandwidth optimization. The memory accesses to adjacent locations can be served in parallel giving higher bandwidth.  The shared memory access is faster as it is closer to the core.  Also, the logic computes the sum only from populated entries avoiding unnecessary calculations. 


####Cholesky Decomposition:
Though the code refers to L for a Lower triangular matrix, the matrix is populated and stored as Upper Triangular to improve memory bandwidth as accesses can be coalesced among different threads easily. It should be noted that  Cholesky factors are transpose of one another. Hence, we can either compute L or L’. As the memory accesses for L’ is better optimized for L’ , I have stored it as L’. 
This routine goes to each row in L one by one. All threads in a row compute the diagonal element using Sum Reductions. Once the diagonal element is available, the dependency for entire row is complete and all threads start calculating a corresponding element. Again shared memory array is used to store the working sum which improves the memory operations. After the completion of this  row, same steps are carried out for the next row as well.

**cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)**
The above configuration for shared memory bank size optimizes memory for double precision accesses and reduce conflicts.

<details>
    <summary> Output using a complete Streaming Processor(196 GPU cores) </summary>

Selected m value : 64
The required Rstar value : 2.000000, 2.000000
Input: N = 192, threads_per_block = 192
The predicted value of f at r_star : -0.149718
Elapsed time: Cholesky = 17478.107422 ms
Elapsed time: Solver = 128.109604 ms
Elapsed time: Kernel = 18134.296875 ms
Floating point operations Cholesky Factorization: 24534278144
Floating point operations per second (FLOPS) Cholesky : 1.307311 Gflops
Floating point operations Solver: 35135488
Floating point operations per second (FLOPS) Solver: 0.255426 Gflops


Hence, the FLOPS for Cholesky my implementation offers is 1.307311 Gflops for a matrix of dimension 4096x4096. For solver it comes to  0.255426 Gflops for the same dimension.

Peak Flops over SM = (2 x  0.71 x 196) Gflops  = 272.32 Gflops (The 2 is for FMA operations)

My Cholesky Implementation is 0.46% of the peak rate and solver is 0.09% of the peak rate. 

There are many synchronization calls and presence of divergent threads in my code which eats up most of the performance.
</details>

<details>
    <summary> Output using 1 GPU core </summary>
----
Selected m value : 64
The required Rstar value : 2.000000, 2.000000
Input: N = 1, threads_per_block = 1
The predicted value of f at r_star : -0.149718
Elapsed time: Cholesky = 2261328.000000 ms
Elapsed time: Solver = 7504.510742 ms
Elapsed time: Kernel = 2347107.500000 ms
Floating point operations Cholesky Factorization: 22931662848
Floating point operations per second (FLOPS) Cholesky : 0.009444 Gflops
Floating point operations Solver: 33570816
Floating point operations per second (FLOPS) Solver: 0.004166 Gflops
----
</details>

The speed-up/efficiency obtained on 196 cores over a single core :

          |Single_core|Full SM|Speed-up |Efficiency|
----------|-----------|-------|---------|----------|
Cholesky  |2261328	  |17478  |129.38	|66.01%    |
----------|-----------|-------|---------|----------|
Solver	  |7504	      |128	  |58.63	|29.91%    |
----------|-----------|-------|---------|----------|
Total	  |2347107	  |18134  |129.43	|66.04%    |
----------|-----------|-------|---------|----------|


