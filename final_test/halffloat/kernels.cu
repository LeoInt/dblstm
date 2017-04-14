#include <float.h>
#include <math.h>
#include <iostream>

#include "kernels.h"
// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
  return 1.f / (1.f + expf(-in));  
}



__global__ void rand_generate(fp16 *y,  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fp32_to_fp16_gpu(sinf(i)/10);
}

__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += bias[i % nBias];
}

__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, float *a,  float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__global__ void pw_tanh(float *y, float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = sigmoidf(a[i]);
}
__global__ void add_pw_vecMul(float *y, float *a,  float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + a[i] * b[i];
}

__global__ void initKernel(float * devPtr, const int val, int nwords){
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(; tidx < nwords; tidx += stride)
    devPtr[tidx] = val;
}


__global__ void testfloat(float * devPtr, float * devPtr1){
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  devPtr[0]=devPtr[0]*devPtr1[0];
}

__global__ void elementWise_fp(int hiddenSize, float *tmp_h, float *tmp_i, float *bias, float *phole_i, float *phole_f, float *phole_o, float *h_out, float *i_out, float *c_in, float *c_out) {  
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (index >= hiddenSize) return;
   
   //int batch = index / hiddenSize;
   int gateIndex = (index % hiddenSize); //+ 4 * batch * hiddenSize;   
   
   float g[4];

   for (int i = 0; i < 4; i++) {
      g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
      g[i] += bias[i * hiddenSize + index % hiddenSize];
   }  

    g[1] += c_in[index]*phole_i[index];
    g[2] += c_in[index]*phole_f[index];
    //g[3] += c_in[index]*phole_o[index];
        
   
   float in_gate2    = tanhf(g[0]);   
   float in_gate     = sigmoidf(g[1]);
   float forget_gate = sigmoidf(g[2]);

   float out_gate    = g[3];
   
   float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
     
   c_out[index] = val;
   out_gate += val*phole_o[index];
   out_gate = sigmoidf(out_gate);
   val = out_gate * tanhf(val);                                   

   h_out[index] = val;
   i_out[index] = val;
}


// Fused forward kernel
__global__ void elementWise_fp_1(int hiddenSize,
                               float *tmp_h, 
                               float *tmp_i, 
                               float *bias,
                               float *phole_i,
                               float *phole_f,
                               float *c_in) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (index >= hiddenSize*4) return;
   
   int gateIndex = (index % hiddenSize);// + 4 * batch * hiddenSize;   
   
   float g;
   g = tmp_i[index] + tmp_h[index] + bias[index];
   
   if(index < hiddenSize ){
     //g += c_in[gateIndex]*phole_i[gateIndex];
     tmp_i[index]=tanhf(g); //ingate2
   } else if(index >=hiddenSize && index <2*hiddenSize ){
    g += c_in[gateIndex]*phole_i[gateIndex]; 
    tmp_i[index]=sigmoidf(g); //input
   }else if(index >=2*hiddenSize && index <3*hiddenSize ){
    g += c_in[gateIndex]*phole_f[gateIndex];
    tmp_i[index]=sigmoidf(g); //forget
   }else{
    //g += c_in[gateIndex]*phole_o[gateIndex];
    tmp_i[index]=g; //out
   }


}   
 __global__ void elementWise_fp_2(int hiddenSize,
                               float *tmp_i, 
                               float *c_in,
                               float *phole_o,
                               float *h_out,
                               float *i_out,
                               float *c_out) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
   float out_gate;
   if (index >= hiddenSize) return;
   
   float val = (tmp_i[2*hiddenSize + index] * c_in[index]) + (tmp_i[index] * tmp_i[index+hiddenSize]);
   
   c_out[index] = val;
   out_gate =tmp_i[index+3*hiddenSize]; 
   out_gate += val*phole_o[index];
   val = sigmoidf(out_gate) * tanhf(val);                                   

   h_out[index] = val;
   i_out[index] = val;
}

__global__ void add_vec_to_rows(float alpha, float* row, float beta, float* dst, int nrow, int ncol) { //46(outdim) col and 100 row(seqlength) in row major, 46 rrow and 100 col in col major 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j*ncol;
  if (i < ncol && j < nrow)
    dst[index] = alpha*row[i] + beta*dst[index];
}
//y = e^x_j/sum_j(e^x_j)
__global__ void _softmax_reduce(float*y, float*x, int nrow, int ncol) { //nrow = seqLength, ncol=output_dim.
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= nrow) return;

  __shared__ float aux[CU1DBLOCK];
  int steps = (ncol - 1) / THREADS + 1;

  //copy input to aux
  aux[threadIdx.x] = x[threadIdx.x+j*ncol]; //stride=ncol
  for(int i=1; i<steps; ++i) {
    if(threadIdx.x+i*THREADS < ncol && aux[threadIdx.x] < x[threadIdx.x+i*THREADS+j*ncol])
  aux[threadIdx.x] = x[threadIdx.x+i*THREADS+j*ncol];
  }

  //get the maximum value
  int nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads && aux[threadIdx.x] < aux[threadIdx.x+halfPoint])
        aux[threadIdx.x] = aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  float max = aux[0];
  __syncthreads();
  
   // subtract max, apply exp, sum up...
  y[threadIdx.x+j*ncol] = exp(x[threadIdx.x+j*ncol] - max);
  aux[threadIdx.x] = y[threadIdx.x+j*ncol];
  for(int i=1; i<steps; i++) {
    if(threadIdx.x+i*THREADS < ncol) {
      y[threadIdx.x+i*THREADS+j*ncol] = exp(x[threadIdx.x+i*THREADS+j*ncol] - max);
      aux[threadIdx.x] += y[threadIdx.x+i*THREADS+j*ncol];
    }
  }
  nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads)
        aux[threadIdx.x] += aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  float sum = aux[0];
  __syncthreads();

  //normalize by sum...
  for(int i=0; i<steps; i++) {
    if(threadIdx.x+i*THREADS < ncol) {
      y[threadIdx.x+i*THREADS+j*ncol] = y[threadIdx.x+i*THREADS+j*ncol] / sum;
    }
  }

}

__global__ void matvec_16(const fp16*   dA, const float*  dx,
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

__global__ void matmat_16(const fp16*  __restrict__ A, const float*  __restrict__ B,
    float*  __restrict__ C, const uint_t M, const uint_t K, const uint_t N)
{
    //#define SUB_M  64//subultiple pf M  //rule : SUB_M=SUB_K and SUB_N * SUB_K < 1024
    //#define SUB_N  16//submultiple of N
    //#define SUB_K  64//submultple of K

  const int tid = threadIdx.x; 
  const unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const uint_t K_sub_blocks = (K + SUB_K - 1) / SUB_K;
  float out[SUB_N];

  __shared__ float B_shared[SUB_N][SUB_K];
  
  #pragma unroll  
  for(int i=0; i<SUB_N; i++){
    out[i]=0;
  }

  register float y_val = 0.0;

  //#pragma unroll
  for (unsigned int m = 0; m < K_sub_blocks; ++m) {
    //#pragma unroll
    for(int a=0;a<SUB_N;a++){
        if(blockIdx.y*SUB_N+a<N)
            B_shared[a][tid]=B[blockIdx.y*K*SUB_N +a*K + m*SUB_K+tid];
        else
            B_shared[a][tid]=0;
    }
    __syncthreads();

    //#pragma unroll
    for (unsigned int e = 0; e < SUB_K; ++e) {
      //if((SUB_K * m + e)<K){
        float temp = fp16_to_fp32_gpu(A[tidx + (SUB_K * m + e)*M]); 
        //#pragma unroll

        for (unsigned int i = 0; i < SUB_N; i++ ){
                out[i] += temp * B_shared[i][e];
        }
      //}
    }

    __syncthreads();
  }
   #pragma unroll
  for(unsigned int i = 0; i < SUB_N; i++ ){
        if(blockIdx.y*SUB_N+i<N)
            C[blockIdx.y*M*SUB_N + i*M + tidx] = out[i];
  }

}
__global__ void elementWise_fp16(int hiddenSize, float *tmp_h, float *tmp_i, fp16 *bias, fp16 *phole_i, fp16 *phole_f, fp16 *phole_o, float *h_out, float *i_out, float *c_in, float *c_out) {  
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (index >= hiddenSize) return;
   
   //int batch = index / hiddenSize;
   int gateIndex = (index % hiddenSize); //+ 4 * batch * hiddenSize;   
   
   float g[4];

   for (int i = 0; i < 4; i++) {
      g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
      g[i] +=  fp16_to_fp32_gpu(bias[i * hiddenSize + index % hiddenSize]);
   }  

    g[1] += c_in[index]* fp16_to_fp32_gpu(phole_i[index]);
    g[2] += c_in[index]* fp16_to_fp32_gpu(phole_f[index]);
    //g[3] += c_in[index]*phole_o[index];
        
   
   float in_gate2    = tanhf(g[0]);   
   float in_gate     = sigmoidf(g[1]);
   float forget_gate = sigmoidf(g[2]);

   float out_gate    = g[3];
   
   float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
     
   c_out[index] = val;
   out_gate += val* fp16_to_fp32_gpu(phole_o[index]);
   out_gate = sigmoidf(out_gate);
   val = out_gate * tanhf(val);                                   

   h_out[index] = val;
   i_out[index] = val;
}

// Pointwise functions
void rand_generate_w( fp16 *y,  int n) {
  dim3 Gr;
  dim3 Bl;
  Bl.x = 128;
  Gr.x = (n + Bl.x - 1) / Bl.x;
    
  rand_generate<<<Gr, Bl>>>(y, n);

}
void pw_biasAdd_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *bias, int n, int nBias){
    pw_biasAdd<<<Gr, Bl, 0, stream>>>(y, bias, n, nBias);
  }

void pw_vecAdd_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n){
  pw_vecAdd<<<Gr, Bl, 0, stream>>>(y,a, b, n);
}

void pw_vecMul_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n){
  pw_vecMul<<<Gr, Bl, 0, stream>>>(y, a, b, n);
}

void pw_tanh_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a, int n){
  pw_tanh<<<Gr, Bl, 0, stream>>>(y, a, n); 
}

void pw_sigmoid_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a, int n){
  pw_sigmoid<<<Gr, Bl, 0, stream>>>(y, a, n);
}
void add_pw_vecMul_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n){
  add_pw_vecMul<<<Gr, Bl, 0, stream>>>(y, a, b, n); 
}

void initKernel_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float * devPtr, const int val, int nwords){
 initKernel<<<Gr, Bl, 0, stream>>>(devPtr, val, nwords);
}

void testfloat_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float * devPtr, float * devPtr1){

testfloat<<<Gr, Bl, 0, stream>>>(devPtr, devPtr1);
   
}

void elementWise_fp_w(dim3 Gr, dim3 Bl, cudaStream_t stream,int hiddenSize, float *tmp_h, float *tmp_i, float *bias, float *phole_i, float *phole_f, float *phole_o, float *h_out, float *i_out, float *c_in, float *c_out){
  elementWise_fp<<<Gr,Bl,0,stream>>>(hiddenSize,tmp_h, tmp_i, bias, phole_i, phole_f, phole_o, h_out, i_out, c_in, c_out);
}

void elementWise_fp_1_w(dim3 Gr, dim3 Bl, cudaStream_t stream, int hiddenSize,
                               float *tmp_h, 
                               float *tmp_i, 
                               float *bias,
                               float *phole_i,
                               float *phole_f,
                               float *c_in) {
  elementWise_fp_1<<<Gr,Bl,0,stream>>>(hiddenSize, tmp_h, tmp_i, bias, phole_i, phole_f, c_in);
}

void elementWise_fp_2_w(dim3 Gr, dim3 Bl, cudaStream_t stream, int hiddenSize,
                               float *tmp_i, 
                               float *c_in,
                               float *phole_o,
                               float *h_out,
                               float *i_out,
                               float *c_out){
  elementWise_fp_2<<<Gr,Bl,0,stream>>>(hiddenSize, tmp_i, c_in, phole_o, h_out, i_out, c_out);
} 

void add_vec_to_rows_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float alpha, float* row, float beta, float* dst, int nrow, int ncol) {
  add_vec_to_rows<<<Gr,Bl,0,stream>>>(alpha, row, beta, dst, nrow, ncol);
}

void softmax_reduce_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float*y, float*x, int nrow, int ncol){
_softmax_reduce<<<Gr,Bl,0,stream>>>(y, x, nrow, ncol);
}

void mygemv16( cudaStream_t stream, const fp16*   dA, const float*  dx,float*  dy, const uint_t nRows, const uint_t nx){
 dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE);
  matvec_16<<<dim_grid, dim_block,0,stream>>>(dA, dx, dy, nRows, nx);

}
void mygemm16(cudaStream_t stream, const fp16 *A, const float *B, float *C, int M, int N, int K){

  dim3 threadsPerBlock(SUB_M);
  dim3 blocksPerGrid((M + SUB_M - 1) / SUB_M, (N + SUB_N - 1) / SUB_N);
    
  matmat_16<<<blocksPerGrid,threadsPerBlock,0,stream>>>(A, B, C, M, K, N);
}
void elementWise_fp16_w(dim3 Gr, dim3 Bl, cudaStream_t stream,int hiddenSize, float *tmp_h, float *tmp_i, fp16 *bias, fp16 *phole_i, fp16 *phole_f, fp16 *phole_o, float *h_out, float *i_out, float *c_in, float *c_out){
  elementWise_fp16<<<Gr,Bl,0,stream>>>(hiddenSize,tmp_h, tmp_i, bias, phole_i, phole_f, phole_o, h_out, i_out, c_in, c_out);
}
