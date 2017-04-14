#ifndef KERNELS_H_
#define KERNELS_H_

#define   uint_t uint
#define CU1DBLOCK 256
#define BLOCK_SIZE 64  //for mygemv16
#include <sstream>
#include <cublas_v2.h>
#include <curand.h>
#include <fstream>
#include <stdio.h>

extern "C" {


// Device functions

// Pointwise functions
 void pw_biasAdd_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *bias, int n, int nBias);

 void pw_vecAdd_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n);

 void pw_vecMul_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n);

 void pw_tanh_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a, int n);

 void pw_sigmoid_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a, int n);
 void add_pw_vecMul_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float *y, float *a,  float *b, int n);

 void initKernel_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float * devPtr, const int val, int nwords);

 void testfloat_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float * devPtr, float * devPtr1);

 void elementWise_fp_w(dim3 Gr, dim3 Bl, cudaStream_t stream, int hiddenSize, float *tmp_h, float *tmp_i, float *bias, float *phole_i, float *phole_f, float *phole_o, float *h_out, float *i_out, float *c_in, float *c_out);

void elementWise_fp_1_w(dim3 Gr, dim3 Bl, cudaStream_t stream, int hiddenSize,
                               float *tmp_h, 
                               float *tmp_i, 
                               float *bias,
                               float *phole_i,
                               float *phole_f,
                               float *c_in);
void elementWise_fp_2_w(dim3 Gr, dim3 Bl, cudaStream_t stream, int hiddenSize,
                               float *tmp_i, 
                               float *c_in,
                               float *phole_o,
                               float *h_out,
                               float *i_out,
                               float *c_out);
void matvecfuse_w(dim3 Gr, dim3 Bl, cudaStream_t stream, const float*   dA, const float*  h_data, const float* x_in, const float *phole_i, const float *phole_f, const float *phole_o, float* c_in, float* c_out, 
float*  h_out, float*  i_out, const uint_t nRows, const uint_t nx);
void add_vec_to_rows_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float alpha, float* row, float beta, float* dst, int nrow, int ncol);
void softmax_reduce_w(dim3 Gr, dim3 Bl, cudaStream_t stream, float*y, float*x, int nrow, int ncol);
}
#endif