#include <float.h>
#include <math.h>
#include <iostream>

#include "kernels.h"
// Device functions
float sigmoidf(float in) {
  return 1.f / (1.f + expf(-in));  
}
// Pointwise functions
 
// Pointwise functions
void pw_biasAdd_w(float *y, float *bias, int n, int nBias){
   // pw_biasAdd<<<Gr, Bl, 0, stream>>>(y, bias, n, nBias);
  //  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=0;j<n;j++){
     y[j] += bias[j];   
    }
  }

void pw_vecAdd_w(float *y, float *a,  float *b, int n){
  for(int i=0;i<n;i++){
     y[i] = a[i] + b[i];
  }
}

void pw_vecMul_w(float *y, float *a,  float *b, int n){
  for(int i=0;i<n;i++){
     y[i] = a[i]*b[i];
  } 
}

void pw_tanh_w( float *y, float *a, int n){
  for(int i=0;i<n;i++){  
    y[i] = tanhf(a[i]);
  }
}

void pw_sigmoid_w(float *y, float *a, int n){
 
for(int i=0;i<n;i++){  
     y[i] = sigmoidf(a[i]);
  }
}

void add_pw_vecMul_w(float *y, float *a,  float *b, int n){
  for(int i=0;i<n;i++){
    y[i] = y[i] + a[i] * b[i];
  }
}


void generate_random_numbers_cu(float* numbers, int Np){
  
  for(int i=0;i<Np;i++){
    if((int)(clock())%2==0)
      numbers[i]=sin(i)/10;
    else
      numbers[i]=cos(i)/10;
  }

}

void elementWise_fp_w(int hiddenSize, float *tmp_h, float *tmp_i, float *bias, float *phole_i, float *phole_f, float *phole_o, float *h_out, float *i_out, float *c_in, float *c_out) {  
   
   //for(i=0;i<4*hiddenSize;i++)
   //   g[i] = tmp_i[i] + tmp_h[i]+bias[i];
    float g[4];
   for(int i=0;i<hiddenSize;i++){
   

    for (int j = 0; j < 4; j++) {
      g[i] = tmp_i[j * hiddenSize + (i % hiddenSize)] + tmp_h[j * hiddenSize + (i % hiddenSize)];
      //g[i] = tmp_h[i * hiddenSize + gateIndex];
      g[i] += bias[j * hiddenSize + (i % hiddenSize)];
    }  
   
    g[1] += c_in[i]*phole_i[i];
    g[2] += c_in[i]*phole_f[i];
    //g[3] += c_in[index]*phole_o[index];
   float in_gate2    = tanhf(g[0]);   
   float in_gate     = sigmoidf(g[1]);
   float forget_gate = sigmoidf(g[2]);

   float out_gate    = g[3];
   
   float val = (forget_gate * c_in[i]) + (in_gate * in_gate2);
     
   c_out[i] = val;
   out_gate += val*phole_o[i];
   out_gate = sigmoidf(out_gate);
   val = out_gate * tanhf(val);                                   

   h_out[i] = val;
   i_out[i] = val;
  }    
   /*
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
   i_out[index] = val;*/
}

void add_vec_to_rows_w(float alpha, float* row, float beta, float* dst, int nrow, int ncol) {

for (int i = 0; i < ncol; i++, i++) {
      for (int j = 0; j < nrow; j++)
        dst[j + i*nrow] = alpha * row[j] + beta*dst[j + i*nrow];
    }
}
