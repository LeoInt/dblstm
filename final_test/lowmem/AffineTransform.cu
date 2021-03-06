#include "AffineTransform.h"

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
static void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
static void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
static void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}


static  void Print_matrix(float* mat, int n, int m, int r_c){
    //const char nmfile[] = "out.txt";
    float *data_host;
    data_host=(float*)malloc(n*m*sizeof(float));
    cudaMemcpy(data_host, mat, n*m*sizeof(float), cudaMemcpyDeviceToHost);  // this won't work, will throw error
    if(r_c==0){
      for (int jj=0; jj<n; jj++)
      {
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+jj*m+ii);
          printf("%.10e ", *temp);
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
          printf("%.10e ", *temp);
                    //if(jj==101) printf("%f ", *temp);
        }
        printf("\n");
      }
    }
    free(data_host);
  }

 AffineTransform::AffineTransform(int input_dim, int output_dim) { 
      input_dim_= input_dim; // 640
      output_dim_= output_dim; //640

  cudaErrCheck(cudaMalloc((void**)&wei_affine_,  input_dim_  * output_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&bias_, output_dim_ * sizeof(float)));
  
    }
  AffineTransform::~AffineTransform(){
  
  cudaErrCheck(cudaFree(wei_affine_));
  cudaErrCheck(cudaFree(bias_));

  } 
  float AffineTransform::Propagate(cublasHandle_t handle, float* in, float* out, int seqLength){
      //continue
  cudaStream_t stream=NULL;
  float alpha = 1.f;
  float beta  = 0.f;
  int frame;
  cudaEvent_t start, stop;
  float elapsedTime=0.f;
  dim3 blockDim, gridDim;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;

  //x ncol outdim, y nrow seqLength
  blockDim.x = 32;
  gridDim.x = (output_dim_ + blockDim.x - 1) / blockDim.x;
  blockDim.y = 32;
  gridDim.y = (seqLength + blockDim.y - 1) / blockDim.y;

  add_vec_to_rows_w(gridDim, blockDim, stream, alpha, bias_, beta, out, seqLength, output_dim_);
  
  beta=1.f;
  
  cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        output_dim_, //m, number of rows of matrix op(A) and C.
                        seqLength, //n, number of cols of matrix op(B) and C.
                        input_dim_,  //k, number of cols of matrix op(A) and rows of op(B).
                        &alpha,
                        wei_affine_,
                        output_dim_,  //leading dimension = number of rows (I use the number of col because I do the transpose with transa)
                        in,
                        input_dim_,
                        &beta,
                        out,
                        output_dim_));

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime, start, stop);
return elapsedTime;  
}


  void AffineTransform::Init() {


  float mean = 0.0;
  float stdev = 1.0;
 
  curandGenerator_t rng;
  curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
  curandErrCheck(curandGenerateNormal(rng, wei_affine_, input_dim_  * output_dim_, mean, stdev));
  curandErrCheck(curandGenerateNormal(rng, bias_,  output_dim_, mean, stdev));
  curandErrCheck(curandDestroyGenerator(rng));

  // initialize layer
  }

