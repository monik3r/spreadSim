#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define DSIZE 1024*1024
#define nTPB 256

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}

/*************************/
/* CURAND INITIALIZATION */
/*************************/
__global__ void initCurand(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

/**********************/
/*  CURAND GENERATION */
/**********************/
__global__ void genUniform(unsigned long seed, float *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    a[idx] = curand_uniform(&state);
}

__global__ void genNormal(unsigned long seed, float *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    a[idx] = curand_normal(&state);
}

/********/
/* MAIN */
/********/
int main() {
    curandState *devState;  gpuErrchk(cudaMalloc((void**)&devState, DSIZE*sizeof(curandState)));
    float *d_a;             gpuErrchk(cudaMalloc((void**)&d_a, DSIZE*sizeof(float)));

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    initCurand<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(devState, 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    genUniform<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(1, d_a);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Initialization time:  %3.1f ms \n", time);

}