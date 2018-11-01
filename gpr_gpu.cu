#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NOISE_PARAMETER 0.01
#define GET_RAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05

__device__ struct XY {
    double x;
    double y;
} XY;

__device__ double d_f_pred; 
__device__ int n;
__device__ double sum;
__device__ int count;


// Code to get the number of cores in a SM
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };
    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

__device__ void get_total_sum(double *partial_sum, int dummy) {
    if(threadIdx.x == 0) {
        count = dummy;
        if(count %2 != 0) {
            count++;
            partial_sum[count-1] = 0;
        }
    }
    __syncthreads();
    for(int i = count/2; i > 0; i = i/2) {
        if(threadIdx.x < i)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
        __syncthreads();
        if(threadIdx.x == 0) {
            if(i%2 != 0 && i != 1) {
                partial_sum[0] += partial_sum[--i];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    return;
}

//-------------------------------------------------------------------------------
// Kernel function to compute the K' matrix
__global__ void gpr_get_K(int N, int m, double *K, struct XY *xy) 
{
    // xy[r] store the x and y coordinates of the rth point
    n = m * m;
    double d[2];
    // Allocate and initialze K
    for(int i = threadIdx.x; i < n; i += N) {
        for(int j = 0; j < n; j++) {
            d[0] = pow(xy[i].x - xy[j].x, 2);
            d[1] = pow(xy[i].y - xy[j].y, 2);
            if(i == j)
                K[ i*n + j] = exp(-1 * (d[0] + d[1])) + 0.01;
            else
                K[ i*n + j] = exp(-1 * (d[0] + d[1]));
               
        }
    }
    
}

// Kernel function to calculate the cholesfy factors
__global__ void gpr_cholesky(int N, double *K, double *L) {
    // LU factorization
    extern __shared__ double partial_sum[];
    for(int k = 0; k < n; k++) {
        partial_sum[threadIdx.x] = 0;
        for(int j = threadIdx.x; j < k; j += N) {
            partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + (L[j * n + k] * L[j * n +k]);
        }
        __syncthreads();
        get_total_sum(partial_sum, (N<k)?N:k);
        if(threadIdx.x == 0) {
            L[k * n + k] = sqrt(K[k * n + k] - partial_sum[0]);
        }
        __syncthreads();
        for(int i = k + threadIdx.x + 1; i < n; i+=N) { //Removing zeroing
            partial_sum[threadIdx.x] = 0;
            for(int j = 0; j < k; j++) {
                partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + L[j * n + i] * L[j * n + k];  
            }
            L[k * n + i] = (K[k * n + i] - partial_sum[threadIdx.x]) / L[k * n + k];
        }
        __syncthreads();

    }
}

// Kernel code to solve for z
__global__ void gpr_solver(int N, double *Y, double *z, double *L, double *f) 
{
    extern __shared__ double partial_sum[];
    // Solving K'z = f => LUz = F => LY = F
    // Solving for Y
    for(int i = 0; i < n; i++) {
        partial_sum[threadIdx.x] = 0;
        for(int j = threadIdx.x; j < i; j += N) {
            partial_sum[threadIdx.x] += (L[j * n + i] * Y[j]);
        }
        __syncthreads();
        get_total_sum(partial_sum, (N<i)?N:i);
        if(threadIdx.x == 0) {
            Y[i] = (f[i] - partial_sum[0]) / L[i * n + i];
        }
        __syncthreads();
    }
    __syncthreads();

    // Solving for z
    for(int i = n-1; i >= 0; i--) {
        partial_sum[threadIdx.x] = 0;
        for(int j = n-1-threadIdx.x; j > i; j -= N) {
            partial_sum[threadIdx.x] += (L[i * n + j] * z[j]); // U component is nothing but L'
        }
        __syncthreads();
        get_total_sum(partial_sum, (N < (n - 1 - i))?N:(n-1-i));
        if(threadIdx.x == 0) {
            z[i] = (Y[i] - partial_sum[0]) / L[i * n + i];
        }
        __syncthreads();
    }
    return;

}

//Kernel code to run the final prediction 
__global__ void gpr_predict(int N, int m, double a, double b, double *k, double *z, struct XY *xy) 
{   
    // Computing the f(predicted) value at rstar
    double rstar[2] = {a, b};
    extern __shared__ double partial_sum[];
    // Initializing k

    double d[2];
    for(int i = threadIdx.x; i < n; i += N) {
        d[0] = pow(xy[i].x - rstar[0], 2);
        d[1] = pow(xy[i].y - rstar[1], 2);
        k[i] = exp(-1 * (d[0] + d[1]));
    }
    
    partial_sum[threadIdx.x] = 0.0;
    for(int i = threadIdx.x; i < n; i += N) {
        partial_sum[threadIdx.x] += k[i] * z[i];
    }
    __syncthreads();

    get_total_sum(partial_sum, (N<n)?N:n);
    if(threadIdx.x == 0) {
        d_f_pred = partial_sum[0];
    }
    return;
}


//Main function to take in the parameters and call the GPU kernel to calculate the predicted values
int main(int argc, char* argv[]) {
    int m;
    int num_threads;
    double rstar[2];

    if(argc != 4) {
        printf("Aborting! Invalid number of input arguements. Please execute the binary as ./a.out m xstar ystar\n");
        return 0;
    } else {
        m = atoi(argv[1]);
        rstar[0] = atof(argv[2]);
        rstar[1] = atof(argv[3]);
        printf("Selected m value : %d \n", m);
        printf("The required Rstar value : %f, %f\n", rstar[0], rstar[1]);
    }
    


    /* Validate the input parameters */
    if(rstar[0] < 0 || rstar[0] >= m || rstar[1] < 0 || rstar[1] >= m ) {
        printf("Aborting! Rstar selected out of Bound! \n");
        return 0;
    }
   
    //Get the cores in a SM
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for ( device = 0; device < deviceCount; ++device ) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        int temp =  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        if(temp > num_threads)
            num_threads = temp;
    }

    //num_threads = 1;
    printf("Input: N = %d, threads_per_block = %d\n", num_threads, num_threads); 
   
    double *f, *k, *Y, *z, *K, *L;
    struct XY *xy;

    //Allocating data structures for GPU
    cudaMalloc(&f, (m * m * sizeof(double)));
    cudaMalloc(&k, (m * m * sizeof(double)));
    cudaMalloc(&Y, (m * m * sizeof(double)));
    cudaMalloc(&z, (m * m * sizeof(double)));
    cudaMalloc(&xy, (m * m * sizeof(struct XY)));

    int n = m*m;
    cudaMalloc(&K, (n * n * sizeof(double)));
    cudaMalloc(&L, (n * n * sizeof(double)));
    


    // Initializing the grid and f 
    // xy[r] store the x and y coordinates of the rth point
    n = m * m;
    struct XY *h_xy = (struct XY *) malloc( n * sizeof(struct XY));
    double h = 1.0 / (double)(m + 1);
    int idx = 0;
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < m; j++) {
            h_xy[idx].x = (i + 1) * h;
            h_xy[idx].y = (j +1) * h;
            idx++;
        }
    }
    //Exporting to the GPU
    cudaMemcpy(xy, h_xy, n*sizeof(struct XY), cudaMemcpyHostToDevice);
    

    // Allocate and initialize observed data vector f
    double* h_f = (double*) malloc(n * sizeof(double));
    for(idx = 0; idx < n; idx++) {
        h_f[idx] = 1 - (((h_xy[idx].x - 0.5) * (h_xy[idx].x - 0.5)) +
             ((h_xy[idx].y - 0.5) * (h_xy[idx].y - 0.5))) + GET_RAND;
    }
    // Exporting to GPU
    cudaMemcpy(f, h_f, n*sizeof(double), cudaMemcpyHostToDevice);

    // Initialize timing events
    cudaEvent_t start_kernel, stop_kernel, start_cholesky, stop_cholesky, start_solver, stop_solver;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_cholesky);
    cudaEventCreate(&stop_cholesky);
    cudaEventCreate(&start_solver);
    cudaEventCreate(&stop_solver);
  

    //Connfiguring the shared memory banks for double precision
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); 
    // Record timing event - start
    cudaEventRecord(start_kernel, 0);
    gpr_get_K<<<1,num_threads>>>(num_threads, m, K, xy);
    cudaEventRecord(start_cholesky, 0);
    gpr_cholesky<<<1,num_threads, num_threads * sizeof(double)>>>(num_threads, K, L);
    cudaEventRecord(stop_cholesky, 0);
    cudaEventSynchronize(stop_cholesky);
    cudaEventRecord(start_solver, 0);
    gpr_solver<<<1,num_threads, num_threads * sizeof(double)>>>(num_threads, Y, z, L, f);
    cudaEventRecord(stop_solver, 0);
    cudaEventSynchronize(stop_solver);
    gpr_predict<<<1,num_threads, num_threads * sizeof(double)>>>(num_threads, m, rstar[0], rstar[1], k, z, xy);
    
    // Record timing event - stop
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);

    typeof(d_f_pred) f_pred;
    cudaMemcpyFromSymbol(&f_pred, d_f_pred, sizeof(d_f_pred), 0, cudaMemcpyDeviceToHost);
    printf("The predicted value of f at r_star : %f\n", f_pred);

    // Compute elapsed time
    float elapsedTime_cholesky;
    cudaEventElapsedTime(&elapsedTime_cholesky, start_cholesky, stop_cholesky);
    printf("Elapsed time: Cholesky = %f ms\n", elapsedTime_cholesky);
    float elapsedTime_solver;
    cudaEventElapsedTime(&elapsedTime_solver, start_solver, stop_solver);
    printf("Elapsed time: Solver = %f ms\n", elapsedTime_solver);
    float elapsedTime_kernel;
    cudaEventElapsedTime(&elapsedTime_kernel, start_kernel, stop_kernel);
    printf("Elapsed time: Kernel = %f ms\n", elapsedTime_kernel);
    long flops_cholesky = 0;
    long flops_solver = 0;
    for(int i = 0; i < n; i++) {
        flops_solver += (2*i + num_threads + 2);
    }
    flops_solver *= 2;

    for(int i = 0; i < n; i++) {
        flops_cholesky += (2 * i + num_threads + 2) * (n - i);
    }
    printf("Floating point operations Cholesky Factorization: %ld\n", flops_cholesky); //Update needed
    printf("Floating point operations per second (FLOPS) Cholesky : %f Gflops\n", (flops_cholesky)/(elapsedTime_cholesky/1000.0)/(1024.0*1024*1024)); //Update Needed
    printf("Floating point operations Solver: %ld\n", flops_solver); //Update needed
    printf("Floating point operations per second (FLOPS) Solver: %f Gflops\n", (flops_solver)/(elapsedTime_solver/1000.0)/(1024.0*1024*1024)); //Update Needed
    
    //for(int i = 0; i < m*m; i++)
    //    printf("%f \n", h_f[i]);

    cudaFree(f);
    cudaFree(k);
    cudaFree(Y);
    cudaFree(z);
    cudaFree(xy);

    cudaFree(K);
    cudaFree(L);

    free(h_xy);
    free(h_f);



    // Delete timing events
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_cholesky);
    cudaEventDestroy(stop_cholesky);
    cudaEventDestroy(start_solver);
    cudaEventDestroy(stop_solver);
}
