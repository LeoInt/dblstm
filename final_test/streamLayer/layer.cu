#include "layer.h"
#define impl 4
// 1 basic, 2 fused
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

 Layer::Layer(int input_dim, int output_dim, int cell_dim) { 
      input_dim_= input_dim; // 640
      output_dim_= output_dim; //640
      cell_dim_ = cell_dim; //320
        //cuda malloc
  /*float* wei_gifo_x;
  
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_x,  input_dim_  * 8 * cell_dim_ * sizeof(float)));
  wei_gifo_x_fw_=wei_gifo_x;
  wei_gifo_x_bw_=wei_gifo_x+input_dim_  * 4 * cell_dim_;
  //cudaErrCheck(cudaMalloc((void**)&wei_gifo_x_fw_,  input_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_m_fw_,  cell_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&bias_fw_, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_o_c_fw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_i_c_fw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_f_c_fw_, cell_dim_ * sizeof(float)));

  //cudaErrCheck(cudaMalloc((void**)&wei_gifo_x_bw_,  input_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_m_bw_,  cell_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&bias_bw_, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_o_c_bw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_i_c_bw_,cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_f_c_bw_, cell_dim_ * sizeof(float)));
  */
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_x_fw_,  input_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_m_fw_,  cell_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&bias_fw_, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_o_c_fw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_i_c_fw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_f_c_fw_, cell_dim_ * sizeof(float)));

  cudaErrCheck(cudaMalloc((void**)&wei_gifo_x_bw_,  input_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&wei_gifo_m_bw_,  cell_dim_  * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&bias_bw_, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_o_c_bw_, cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_i_c_bw_,cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&phole_f_c_bw_, cell_dim_ * sizeof(float)));
  

  }
  Layer::~Layer(){
  
  cudaErrCheck(cudaFree(wei_gifo_x_fw_));
  cudaErrCheck(cudaFree(wei_gifo_m_fw_));
  cudaErrCheck(cudaFree(wei_gifo_x_bw_));
  cudaErrCheck(cudaFree(wei_gifo_m_bw_));
  cudaErrCheck(cudaFree(bias_fw_));
  cudaErrCheck(cudaFree(bias_bw_));
  cudaErrCheck(cudaFree(phole_o_c_fw_));
  cudaErrCheck(cudaFree(phole_i_c_fw_));
  cudaErrCheck(cudaFree(phole_f_c_fw_));
  cudaErrCheck(cudaFree(phole_o_c_bw_));
  cudaErrCheck(cudaFree(phole_i_c_bw_));
  cudaErrCheck(cudaFree(phole_f_c_bw_));
  
  } 

#if impl==4
  float Layer::Propagate(cublasHandle_t handle, cudaStream_t stream, float* in, float* out, int seqLength, float* tmp_h_fw, float* tmp_i_fw, float* tmp_h_bw, float* tmp_i_bw, float* h_data, float* c_data, float* h_data_bw, float* c_data_bw){
      //continue
  //cudaStream_t stream=NULL;
  dim3 blockDim, gridDim;
  int frame, frame_bw;
  float alpha = 1.f;
  float beta  = 0.f;
  cudaEvent_t start, stop;
  float elapsedTime=0.f;
  
 /* cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
*/
  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;

  
   cublasSetStream(handle, stream);
 
  cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        4 * cell_dim_, //m, number of rows of matrix op(A) and C.
                        seqLength , //n, number of cols of matrix op(B) and C.
                        input_dim_,  //k, number of cols of matrix op(A) and rows of op(B).
                        &alpha,
                        wei_gifo_x_fw_,
                        transa == CUBLAS_OP_N ? 4 * cell_dim_ : input_dim_,  //leading dimension = number of rows (I use the number of col because I do the transpose with transa)
                        in,
                        input_dim_,
                        &beta,
                        tmp_i_fw,
                        4 * cell_dim_));
  cublasSetStream(handle, stream);
 
  cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        4 * cell_dim_, //m, number of rows of matrix op(A) and C.
                        seqLength , //n, number of cols of matrix op(B) and C.
                        input_dim_,  //k, number of cols of matrix op(A) and rows of op(B).
                        &alpha,
                        wei_gifo_x_bw_,
                        transa == CUBLAS_OP_N ? 4*cell_dim_ : input_dim_,  //leading dimension = number of rows (I use the number of col because I do the transpose with transa)
                        in,
                        input_dim_,
                        &beta,
                        tmp_i_bw,
                        4 * cell_dim_));
 for (frame=0, frame_bw=seqLength-1; frame < seqLength ; frame ++, frame_bw --){
         cublasSetStream(handle, stream);
   
         cublasErrCheck(cublasSgemv(handle, transa,
               4 * cell_dim_, //m, number of rows of matrix op(A) 
               cell_dim_, //n, number of cols of matrix op(A).
               &alpha,
               wei_gifo_m_fw_, 
               4 * cell_dim_,
               h_data + frame * cell_dim_, 
               1, //stride input array
               &beta,
               tmp_h_fw, 
               1)); //stide output array
    
    
    cublasSetStream(handle, stream);
           cublasErrCheck(cublasSgemv(handle, transa,
               4*cell_dim_, //m, number of rows of matrix op(A) 
               cell_dim_, //n, number of cols of matrix op(A).
               &alpha,
               wei_gifo_m_bw_, 
               4*cell_dim_,
               h_data_bw + (frame_bw+1) * cell_dim_, 
               1, //stride input array
               &beta,
               tmp_h_bw, 
               1)); //stide output array
    
   
    blockDim.x = 128;
    gridDim.x = (cell_dim_ + blockDim.x - 1) / blockDim.x;
    
    elementWise_fp_w( gridDim, blockDim, stream, cell_dim_,tmp_h_fw, 
                       tmp_i_fw + 4 * frame * cell_dim_, 
                       bias_fw_,
                       phole_i_c_fw_,
                       phole_f_c_fw_,
                       phole_o_c_fw_,
                       h_data + (frame + 1) * cell_dim_,
                       out + frame * 2* cell_dim_,
                       c_data + frame * cell_dim_,
                       c_data + (frame + 1) * cell_dim_);
       elementWise_fp_w( gridDim, blockDim, stream, cell_dim_,tmp_h_bw, 
                       tmp_i_bw + 4 * frame_bw * cell_dim_, 
                       bias_bw_,
                       phole_i_c_bw_,
                       phole_f_c_bw_,
                       phole_o_c_bw_,
                       h_data_bw + frame_bw * cell_dim_,
                       out + frame_bw * 2* cell_dim_ + cell_dim_,
                       c_data_bw + (frame_bw+1) * cell_dim_,
                       c_data_bw + frame_bw * cell_dim_);
 
    }//frame loop

/////////////////////////////frame loop
/*cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime, start, stop);*/
return elapsedTime;  
}
#endif  


  void Layer::Init() {


  //float mean = 0.0;
  //float stdev = 1.0;
  
  curandGenerator_t rng;
  curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
//malloc forward//////////////////////////////////////////////////////////////////////////////////

//curandErrCheck(curandGenerateNormal(rng, i_data_bw, (seqLength) * 2 * , mean, stdev));
//cudaMemset(i_data+hiddenSize, 0, hiddenSize*sizeof(float));
  //generate_random_numbers_cu( wei_gifo_x_fw_, cell_dim_  * 4 * cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, wei_gifo_x_fw_, input_dim_  * 4 * cell_dim_));
  generate_random_numbers_cu(wei_gifo_x_fw_, input_dim_  * 4 * cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, wei_gifo_m_fw_,  cell_dim_  * 4 * cell_dim_));
  generate_random_numbers_cu(wei_gifo_m_fw_,  cell_dim_  * 4 * cell_dim_);
  //cudaMemset(T+hiddenSize*hiddenSize, 0, hiddenSize*sizeof(float));
  curandErrCheck(curandGenerateUniform(rng, bias_fw_, 4 * cell_dim_));
  generate_random_numbers_cu(bias_fw_, 4 * cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_o_c_fw_, cell_dim_));
  generate_random_numbers_cu(phole_o_c_fw_, cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_f_c_fw_, cell_dim_));
  generate_random_numbers_cu(phole_f_c_fw_, cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_i_c_fw_, cell_dim_));
  generate_random_numbers_cu(phole_i_c_fw_, cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, wei_gifo_x_bw_, input_dim_  * 4 * cell_dim_));
  generate_random_numbers_cu(wei_gifo_x_bw_, input_dim_  * 4 * cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, wei_gifo_m_bw_,  cell_dim_  * 4 * cell_dim_));
  generate_random_numbers_cu(wei_gifo_m_bw_,  cell_dim_  * 4 * cell_dim_);
//cudaMemset(T+hiddenSize*hiddenSize, 0, hiddenSize*sizeof(float));
  curandErrCheck(curandGenerateUniform(rng, bias_bw_, 4 * cell_dim_));
  generate_random_numbers_cu(bias_bw_, 4 * cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_o_c_bw_, cell_dim_));
  generate_random_numbers_cu(phole_o_c_bw_, cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_f_c_bw_, cell_dim_));
  generate_random_numbers_cu(phole_f_c_bw_, cell_dim_);
  curandErrCheck(curandGenerateUniform(rng, phole_i_c_bw_, cell_dim_));
  generate_random_numbers_cu(phole_i_c_bw_, cell_dim_);
////////////////////////////////////////////////////////////////////////////////////////////////
  curandErrCheck(curandDestroyGenerator(rng));

  // initialize layer
  }

