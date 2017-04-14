#include "layer.h"
#define impl 2
// 1 basic, 2 fused

 Layer::Layer(int input_dim, int output_dim, int cell_dim) { 
      input_dim_= input_dim; // 640
      output_dim_= output_dim; //640
      cell_dim_ = cell_dim; //320
        //cuda malloc
  
  wei_gifo_x_fw_=(float*)malloc( input_dim_  * 4 * cell_dim_ * sizeof(float));
  wei_gifo_m_fw_=(float*)malloc(cell_dim_  * 4 * cell_dim_ * sizeof(float));
  bias_fw_=(float*)malloc(4 * cell_dim_ * sizeof(float));
  phole_o_c_fw_=(float*)malloc(cell_dim_ * sizeof(float));
  phole_i_c_fw_=(float*)malloc(cell_dim_ * sizeof(float));
  phole_f_c_fw_=(float*)malloc(cell_dim_ * sizeof(float));
  
  wei_gifo_x_bw_=(float*)malloc( input_dim_  * 4 * cell_dim_ * sizeof(float));
  wei_gifo_m_bw_=(float*)malloc(cell_dim_  * 4 * cell_dim_ * sizeof(float));
  bias_bw_=(float*)malloc(4 * cell_dim_ * sizeof(float));
  phole_o_c_bw_=(float*)malloc(cell_dim_ * sizeof(float));
  phole_i_c_bw_=(float*)malloc(cell_dim_ * sizeof(float));
  phole_f_c_bw_=(float*)malloc(cell_dim_ * sizeof(float));
  }
  
  Layer::~Layer(){
  
  free(wei_gifo_x_fw_);
  free(wei_gifo_m_fw_);
  free(wei_gifo_x_bw_);
  free(wei_gifo_m_bw_);
  free(bias_fw_);
  free(bias_bw_);
  free(phole_o_c_fw_);
  free(phole_i_c_fw_);
  free(phole_f_c_fw_);
  free(phole_o_c_bw_);
  free(phole_i_c_bw_);
  free(phole_f_c_bw_);
  
  } 

  double Layer::Propagate(float* in, float* out, int seqLength, float* tmp_h_fw, float* tmp_i_fw, float* tmp_h_bw, float* tmp_i_bw, float* h_data, float* c_data, float* h_data_bw, float* c_data_bw){
      //continue
  float alpha = 1.f;
  float beta  = 0.f;
  int frame;

  double elapsedTime=0.f;
    clock_t begin = clock();

  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,4 * cell_dim_,seqLength,input_dim_,alpha,wei_gifo_x_fw_,4 * cell_dim_ ,in,input_dim_,beta,tmp_i_fw,4 * cell_dim_);
  /*cublasErrCheck(cublasSgemm(handle,
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
                        4 * cell_dim_));*/
  for (frame=0; frame < seqLength ; frame ++){
          cblas_sgemv(CblasColMajor, CblasNoTrans, 4*cell_dim_, cell_dim_, alpha, wei_gifo_m_fw_, 4*cell_dim_, h_data + frame * cell_dim_, 1, beta, tmp_h_fw, 1);
        /* cublasErrCheck(cublasSgemv(handle, transa,
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
           //this procedure using matrix ve product also works but it is tricky to set parameters, the matrix dim is lda x n (instead of beinf lda x m) so m and n are inverted and the transpose is applied thanks to the flag
    */
    
    pw_vecAdd_w( tmp_i_fw  + 4 * frame * cell_dim_, tmp_i_fw  + 4 * frame * cell_dim_, tmp_h_fw  , 4*cell_dim_);
      
    pw_biasAdd_w( tmp_i_fw + 4 * frame * cell_dim_, bias_fw_  ,  4*cell_dim_, 4*cell_dim_);
    
    add_pw_vecMul_w( tmp_i_fw + 1 * cell_dim_ + 4 * frame * cell_dim_, c_data + frame * cell_dim_ , phole_i_c_fw_ , cell_dim_);
    add_pw_vecMul_w( tmp_i_fw + 2 * cell_dim_ + 4 * frame * cell_dim_, c_data + frame * cell_dim_ , phole_f_c_fw_ , cell_dim_);
    pw_tanh_w(tmp_i_fw + 0 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_fw + 0 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
    pw_sigmoid_w(tmp_i_fw + 1 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_fw + 1 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
    pw_sigmoid_w(tmp_i_fw + 2 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_fw + 2 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
  
    float *in_gate2    = tmp_i_fw + 0 * cell_dim_ + 4 * frame * cell_dim_;
    float *in_gate     = tmp_i_fw + 1 * cell_dim_ + 4 * frame * cell_dim_;
    float *forget_gate = tmp_i_fw + 2 * cell_dim_ + 4 * frame * cell_dim_;
    float *out_gate    = tmp_i_fw + 3 * cell_dim_ + 4 * frame * cell_dim_;
    float *c_in        = c_data + frame * cell_dim_;
    float* c_out;

    c_out       = c_data + (frame + 1) * cell_dim_ ;

    if (c_in == NULL) {
      pw_vecMul_w(in_gate, in_gate, in_gate2, cell_dim_);
    }
    else {              
      pw_vecMul_w(forget_gate, forget_gate, c_in, cell_dim_);
      pw_vecMul_w( in_gate, in_gate, in_gate2, cell_dim_);
      pw_vecAdd_w( in_gate, in_gate, forget_gate, cell_dim_); //c_out computed
    }
    add_pw_vecMul_w( out_gate, in_gate, phole_o_c_fw_ , cell_dim_); //out=out+c*p_hole
    pw_sigmoid_w(out_gate, out_gate, cell_dim_);
  
    
  if (c_out != NULL) {
    for(int i=0;i<cell_dim_;i++)
      c_out[i] = in_gate[i];
  }

    pw_tanh_w(in_gate, in_gate, cell_dim_);
    pw_vecMul_w( h_data + (frame+1) * cell_dim_ , out_gate, in_gate, cell_dim_);
    pw_vecMul_w( out + frame * 2* cell_dim_ , out_gate, in_gate, cell_dim_);
  }//frame loop





////backward//////////////////////////////////////////////////////////////////////
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,4 * cell_dim_,seqLength,input_dim_,alpha,wei_gifo_x_bw_,4 * cell_dim_ ,in,input_dim_,beta,tmp_i_bw,4 * cell_dim_);
  /*cublasErrCheck(cublasSgemm(handle,
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
                        4 * cell_dim_));*/
  for (frame=seqLength-1; frame >=0 ; frame --){
        cblas_sgemv(CblasColMajor, CblasNoTrans, 4*cell_dim_, cell_dim_, alpha, wei_gifo_m_bw_, 4*cell_dim_,h_data_bw + (frame+1) * cell_dim_, 1, beta, tmp_h_bw, 1);
        /* cublasErrCheck(cublasSgemv(handle, transa,
               4*cell_dim_, //m, number of rows of matrix op(A) 
               cell_dim_, //n, number of cols of matrix op(A).
               &alpha,
               wei_gifo_m_bw_, 
               4*cell_dim_,
               h_data_bw + (frame+1) * cell_dim_, 
               1, //stride input array
               &beta,
               tmp_h_bw, 
               1)); //stide output array
               */
  
  pw_vecAdd_w(tmp_i_bw + 4 * frame * cell_dim_, tmp_i_bw + 4 * frame * cell_dim_, tmp_h_bw  , 4*cell_dim_);
  pw_biasAdd_w(tmp_i_bw  + 4 * frame * cell_dim_, bias_bw_  , 4*cell_dim_, 4*cell_dim_);
   
  add_pw_vecMul_w( tmp_i_bw + 1 * cell_dim_ + 4 * frame * cell_dim_, c_data_bw + (frame+1) * cell_dim_ , phole_i_c_bw_ , cell_dim_);
  add_pw_vecMul_w( tmp_i_bw + 2 * cell_dim_ + 4 * frame * cell_dim_, c_data_bw + (frame+1) * cell_dim_ , phole_f_c_bw_ , cell_dim_);
  pw_tanh_w (tmp_i_bw + 0 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_bw + 0 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
            //in gate 2
  pw_sigmoid_w( tmp_i_bw + 1 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_bw + 1 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
            //input gate
  pw_sigmoid_w( tmp_i_bw + 2 * cell_dim_ + 4 * frame * cell_dim_, tmp_i_bw + 2 * cell_dim_ + 4 * frame * cell_dim_, cell_dim_);
            //forget gate

  float *in_gate2    = tmp_i_bw + 0 * cell_dim_ + 4 * frame * cell_dim_;
  float *in_gate     = tmp_i_bw + 1 * cell_dim_ + 4 * frame * cell_dim_;
  float *forget_gate = tmp_i_bw + 2 * cell_dim_ + 4 * frame * cell_dim_;
  float *out_gate    = tmp_i_bw + 3 * cell_dim_ + 4 * frame * cell_dim_;
  float *c_in        = c_data_bw + (frame+1) * cell_dim_;
  float* c_out;

  c_out       = c_data_bw + frame  * cell_dim_;

  if (c_in == NULL) {
    pw_vecMul_w( in_gate, in_gate, in_gate2, cell_dim_);
  }
  else {              
    pw_vecMul_w( forget_gate, forget_gate, c_in, cell_dim_);
    pw_vecMul_w(in_gate, in_gate, in_gate2, cell_dim_);
    pw_vecAdd_w(in_gate, in_gate, forget_gate, cell_dim_);
  }



  add_pw_vecMul_w( out_gate, in_gate, phole_o_c_bw_ , cell_dim_);
  pw_sigmoid_w(out_gate, out_gate, cell_dim_);
            //output gate

  if (c_out != NULL) {
    for(int i=0;i<cell_dim_;i++)
      c_out[i] = in_gate[i];
  }

  pw_tanh_w(in_gate, in_gate, cell_dim_);
  pw_vecMul_w( h_data_bw + (frame) * cell_dim_ , out_gate, in_gate, cell_dim_);
  pw_vecMul_w( out + cell_dim_ + frame * 2* cell_dim_, out_gate, in_gate, cell_dim_);
  }
    clock_t end = clock();
  elapsedTime=(double)(end - begin) / CLOCKS_PER_SEC;
//printf("\n %f s", elapsedTime);
return elapsedTime;  
}

/*
#if impl==2
  float Layer::Propagate(cublasHandle_t handle, float* in, float* out, int seqLength, float* tmp_h_fw, float* tmp_i_fw, float* tmp_h_bw, float* tmp_i_bw, float* h_data, float* c_data, float* h_data_bw, float* c_data_bw){
      //continue
  dim3 blockDim, gridDim;
  cudaStream_t stream=NULL;
  float alpha = 1.f;
  float beta  = 0.f;
  int frame;
  cudaEvent_t start, stop;
  float elapsedTime=0.f;
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;

  
  
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
  for (frame=0; frame < seqLength ; frame ++){
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
       
    
    stream=NULL;
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
    
    }//frame loop

////backward//////////////////////////////////////////////////////////////////////
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
  for (frame=seqLength-1; frame >=0 ; frame --){
         cublasErrCheck(cublasSgemv(handle, transa,
               4*cell_dim_, //m, number of rows of matrix op(A) 
               cell_dim_, //n, number of cols of matrix op(A).
               &alpha,
               wei_gifo_m_bw_, 
               4*cell_dim_,
               h_data_bw + (frame+1) * cell_dim_, 
               1, //stride input array
               &beta,
               tmp_h_bw, 
               1)); //stide output array
    cudaStream_t stream=NULL;
    blockDim.x = 128;
    gridDim.x = (cell_dim_ + blockDim.x - 1) / blockDim.x;
    
    elementWise_fp_w( gridDim, blockDim, stream, cell_dim_,tmp_h_bw, 
                       tmp_i_bw + 4 * frame * cell_dim_, 
                       bias_bw_,
                       phole_i_c_bw_,
                       phole_f_c_bw_,
                       phole_o_c_bw_,
                       h_data_bw + frame * cell_dim_,
                       out + frame * 2* cell_dim_ + cell_dim_,
                       c_data_bw + (frame+1) * cell_dim_,
                       c_data_bw + frame * cell_dim_);
  
  }

/////////////////////////////frame loop
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime, start, stop);
return elapsedTime;  
}
#endif  
*/
  void Layer::Init() {


  //float mean = 0.0;
  //float stdev = 1.0;
  
  generate_random_numbers_cu(wei_gifo_x_fw_, input_dim_  * 4 * cell_dim_);
  generate_random_numbers_cu(wei_gifo_m_fw_,  cell_dim_  * 4 * cell_dim_);
  //cudaMemset(T+hiddenSize*hiddenSize, 0, hiddenSize*sizeof(float));
  generate_random_numbers_cu(bias_fw_, 4 * cell_dim_);
  generate_random_numbers_cu(phole_o_c_fw_, cell_dim_);
  generate_random_numbers_cu(phole_f_c_fw_, cell_dim_);
  generate_random_numbers_cu(phole_i_c_fw_, cell_dim_);
  generate_random_numbers_cu(wei_gifo_x_bw_, input_dim_  * 4 * cell_dim_);
  generate_random_numbers_cu(wei_gifo_m_bw_,  cell_dim_  * 4 * cell_dim_);
//cudaMemset(T+hiddenSize*hiddenSize, 0, hiddenSize*sizeof(float));
  generate_random_numbers_cu(bias_bw_, 4 * cell_dim_);
  generate_random_numbers_cu(phole_o_c_bw_, cell_dim_);
  generate_random_numbers_cu(phole_f_c_bw_, cell_dim_);
  generate_random_numbers_cu(phole_i_c_bw_, cell_dim_);
////////////////////////////////////////////////////////////////////////////////////////////////
  
  // initialize layer
  }

