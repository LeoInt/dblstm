#include "AffineTransform.h"



 AffineTransform::AffineTransform(int input_dim, int output_dim) { 
      input_dim_= input_dim; // 640
      output_dim_= output_dim; //640

  wei_affine_=(float*)malloc( input_dim_  * output_dim_ * sizeof(float));
  bias_=(float*)malloc( output_dim_ * sizeof(float));
  
    }
  AffineTransform::~AffineTransform(){
  
  free(wei_affine_);
  free(bias_);

  } 
  double AffineTransform::Propagate( float* in, float* out, int seqLength){
      //continue
  float alpha = 1.f;
  float beta  = 0.f;
  int frame;
  double elapsedTime=0.f;
  clock_t begin = clock();
  
  add_vec_to_rows_w(alpha, bias_, beta, out, seqLength, output_dim_);
  
  beta=1.f;
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,output_dim_,seqLength,input_dim_,alpha,wei_affine_,output_dim_ ,in,input_dim_,beta,out,output_dim_);

  /*cublasSgemm(handle,
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
                        output_dim_);*/
   clock_t end = clock();
  elapsedTime=(double)(end - begin) / CLOCKS_PER_SEC;

return elapsedTime;  
}


  void AffineTransform::Init() {

  generate_random_numbers_cu(wei_affine_, input_dim_  * output_dim_);
  generate_random_numbers_cu(bias_,  output_dim_);
  // initialize layer
  }

