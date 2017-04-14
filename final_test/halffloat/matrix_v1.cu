#define 	BLOCK_SIZE	64
#define   uint_t uint
#define MV_USE_SHFL 1
#define NS 320
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <fstream>
#include <sstream>
#include "fp16type.h"


#include <iostream>
#include <string>

//#define		RESTRICT __restrict__
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

void Print_matrix(float* mat, int n, int m, int r_c){
    //const char nmfile[] = "out.txt";
    float *data_host;
    data_host=(float*)malloc(n*m*sizeof(float));
    cudaErrCheck(cudaMemcpy(data_host, mat, n*m*sizeof(float), cudaMemcpyDeviceToHost));  // this won't work, will throw error
    if(r_c==0){
      for (int jj=0; jj<n; jj++)
      {
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+jj*m+ii);
          printf("%f ", *temp);
                  //if(jj==101) printf("%f ", *temp);
        }
        printf("\n");
      }
    }else{
      for (int jj=0; jj<n; jj++)
      {
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+ii*n+jj);
          printf("%f ", *temp);
                    //if(jj==101) printf("%f ", *temp);
        }
        printf("\n");
      }
    }
    free(data_host);
  }

__global__ void conv_32to16_kernel(const float*   d,  fp16* d16t, int size){
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	d16t[tid]=fp32_to_fp16_gpu(d[tid]);
}


//__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in);
//__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in);
__global__ void matvec_kernel(const float*   dA, const float*  dx,
    float*  dy, const uint_t nRows, const uint_t nx)
{
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float x_shared[BLOCK_SIZE];

  float y_val1 = 0.0;
  float y_val2 = 0.0;
  
  //y_val=fp16_to_fp32_gpu(fp32_to_fp16_gpu(y_val));
  #pragma unroll
  for (unsigned int m = 0; m < ((nx + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {

    if ((m * BLOCK_SIZE + threadIdx.x) < nx)
      x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
    else
      x_shared[threadIdx.x] = 0.f;

    __syncthreads();

    #pragma unroll
    for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
      doublefp32 temp=dfp16_to_dfp32_gpu(dA[tid + (e + BLOCK_SIZE * m) * nRows/2]);//dA[tid + (e + BLOCK_SIZE * m) * nRows)];
      
      //doublefp32 t = dfp16_to_dfp32_gpu(*(float *)(&temp));
//	register float t1 = fp16_to_fp32_gpu((fp16)((uint32_t)temp>>16));
      //register float t2 = fp16_to_fp32_gpu((fp16)(uint32_t)temp);
      y_val1 += temp.H * x_shared[e];
      y_val2 += temp.L * x_shared[e];
    }

    __syncthreads();
  }

  if (tid*2 < nRows){
    dy[tid*2] = y_val1;
    dy[tid*2+1] = y_val2;
  }
} /* End function matvec_kernel */

__global__ void matvec_kernel_16(const fp16*   dA, const float*  dx,
    float*  dy, const uint_t nRows, const uint_t nx)
{
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float x_shared[BLOCK_SIZE];

  float y_val = 0.0;
//  float y_val2 = 0.0;
  
  //y_val=fp16_to_fp32_gpu(fp32_to_fp16_gpu(y_val));
  #pragma unroll
  for (unsigned int m = 0; m < ((nx + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {

    if ((m * BLOCK_SIZE + threadIdx.x) < nx)
      x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
    else
      x_shared[threadIdx.x] = 0.f;

    __syncthreads();

    #pragma unroll
    for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
      float t = fp16_to_fp32_gpu(dA[tid + (e + BLOCK_SIZE * m) * nRows]);
      y_val += t * x_shared[e];
    }

    __syncthreads();
  }

  if (tid < nRows){
    dy[tid] = y_val;
  }
} /* End function matvec_kernel */


void matvec(const float* dA, const float* dx,
    float *dy, const uint_t nRows, const uint_t nx) {
  dim3 dim_grid((nRows/2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE);
 matvec_kernel<<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);
  
/*
  dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE);
  matvec_kernel_16<<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);*/
  

 /* dim3 dim_grid((nRows + 16 - 1) / 16);
  dim3 dim_block(16,16);
 matvec_kernel_blocked<<<dim_grid, dim_block>>>(dA, dx, dy);
*/
  
 /* dim3 dim_grid((nRows + 16 - 1) / 16,2);
  dim3 dim_block(16,16);
 matvec_kernel_blocked2<<<dim_grid, dim_block>>>(dA, dx, dy);
  */
  /*dim3 dim_grid((nRows + 32 - 1) / 32);
  dim3 dim_block(32,16);
 matvec_kernel_blocked3<<<dim_grid, dim_block>>>(dA, dx, dy);
 */
  /*
  dim3 dim_grid((nRows + 32 - 1) / 32);
  dim3 dim_block(32,8);
 matvec_kernel_blocked4<<<dim_grid, dim_block>>>(dA, dx, dy);*/
  /*dim3 dim_grid((nRows + 64 - 1) / 64);
  dim3 dim_block(64,8);
 matvec_kernel_blocked5<<<dim_grid, dim_block>>>(dA, dx, dy);
*/
}

void matvec16(const fp16* dA, const float* dx,
    float *dy, const uint_t nRows, const uint_t nx) {
  

  dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE);
  matvec_kernel_16<<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);

}
__global__ void conv_dfp16todfp32_kernel(const float*   dfp16,  float* dfp32, int size){
  	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	doublefp32 dfp32_t=dfp16_to_dfp32_gpu(dfp16[tid]);
	dfp32[tid*2]=dfp32_t.H;
	dfp32[tid*2+1]=dfp32_t.L;
	/*fp16* p16= (fp16*)dfp16;
	dfp32[tid*2]=fp16_to_fp32_gpu(p16[tid*2]);
	dfp32[tid*2+1]=fp16_to_fp32_gpu(p16[tid*2+1]);
*/
}

int main(int argc, char* argv[]) {
  float* T;
  float* i_data;
  float* res;
  fp16* T16t;
  float* T32t;
  fp16*	i_data16t;
  fp16* res16t;

  int m=4*320;
  int n=320;
  
  float mean = 0.0;
  float stdev = 1.0;
  float alpha = 1.f;
  float beta  = 0.f;
  float elapsedTime=0.f;

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  cudaMalloc((void**)&T, m * n* sizeof(float));
  cudaMalloc((void**)&i_data, n * sizeof(float));
  cudaMalloc((void**)&res,m * sizeof(float));
  cudaMalloc((void**)&T16t, m * n* sizeof(fp16));
  cudaMalloc((void**)&i_data16t, n * sizeof(fp16));
  cudaMalloc((void**)&res16t,m * sizeof(fp16));
  
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  curandGenerateNormal(rng, i_data, n,mean, stdev);
  curandGenerateNormal(rng, T, m*n, mean, stdev);
  
  dim3 dim_grid((m*n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE);

  conv_32to16_kernel<<<dim_grid,dim_block>>>(T,  T16t, m*n);


  //dim3 dim_grid1((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  //Print_matrix(i_data, n, 1, 1);
  //printf("\n");
  //conv_32to16_kernel<<<dim_grid1,dim_block>>>(i_data,  i_data16t, n);
  //conv_dfp16todfp32_kernel<<<dim_grid1,dim_block>>>((float*)(i_data16t),  i_data, n/2);
  //Print_matrix(i_data, n, 1, 1);
  //printf("\n");
  T32t=(float*)(T16t);
  curandDestroyGenerator(rng);
  // Make sure everything is done before we start the timers
  cudaDeviceSynchronize();
  cudaMemset(res, 0, m*sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  //matvec(T32t, i_data, res, m, n);
  matvec16(T16t,i_data, res, m, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("Runtime %fms\n", elapsedTime );
  //Print_matrix(T, m,n,1);
  //printf("\n");
  //Print_matrix(i_data, n,1,1);
  //printf("\n");
  //Print_matrix(res, m,1,1);

  elapsedTime=0.f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cublasSgemm(handle,
    transa, transb,
    m, //m, number of rows of matrix op(A) and C.
    1 , //n, number of cols of matrix op(B) and C.
    n,  //k, number of cols of matrix op(A) and rows of op(B).
    &alpha,
    T,
    m,  //leading dimension = number of rows (I use the number of col because I do the transpose with transa)
    i_data,
    n,
    &beta,
    res,
    m);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);


  printf("Runtime cublas %fms\n", elapsedTime );
  //Print_matrix(res, m,1,1);

  elapsedTime=0.f;
cudaMemset(res, 0, m*sizeof(float));
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cublasSgemv(handle,
    transa,
    m, //m, number of rows of matrix op(A) and C.
    n , //n, number of cols of matrix op(B) and C.
    &alpha,
    T,
    m,  //leading dimension = number of rows (I use the number of col because I do the transpose with transa)
    i_data,
    1,
    &beta,
    res,
    1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("Runtime cublas gemv %fms\n", elapsedTime );

  //Print_matrix(res, m,1,1);
cudaFree(T);
cudaFree(i_data);
cudaFree(res);

}
