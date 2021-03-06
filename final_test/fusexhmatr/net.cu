#include "net.h"
//#include "layer.h"

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



Net::Net(int nLayers, int cell_dim) {
  // copy the layers
  //input_buf_dim_=input_buf_dim;
  cell_dim_=cell_dim;
  //layers_=NULL;

  //input_buf_ = NULL; ///< buffers for forward pass
  propagate_buf_ = NULL; ///< buffers for forward pass
  tmp_h_fw_n = NULL;
  tmp_i_fw_n = NULL;

    // back-propagation buffer
  tmp_h_bw_n = NULL;
  tmp_i_bw_n = NULL;

  h_data_n = NULL;
  c_data_n = NULL;
  
  h_data_bw_n = NULL;
  c_data_bw_n = NULL;

  /*for(int i=0; i<nLayers; i++) {
    Layer* L=NULL;
    layers_.push_back(L);
  }*/    
  // create empty buffers
  //propagate_buf_.resize(NumLayers()+1); 
}


Net::~Net() {
  //cudaErrCheck(cudaFree(input_buf_));
  cudaErrCheck(cudaFree(propagate_buf_)); //2 buffers
  //cudaErrCheck(cudaFree(data_));
  cudaErrCheck(cudaFree(h_data_n));
  cudaErrCheck(cudaFree(c_data_n));
  cudaErrCheck(cudaFree(c_data_bw_n));
  cudaErrCheck(cudaFree(h_data_bw_n));
  cudaErrCheck(cudaFree(tmp_h_fw_n));
  cudaErrCheck(cudaFree(tmp_i_fw_n));
  cudaErrCheck(cudaFree(tmp_h_bw_n));
  cudaErrCheck(cudaFree(tmp_i_bw_n));
  Destroy();
}

void Net::Resize(int seqLength){
  //cudaErrCheck(cudaFree(input_buf_));
  cudaErrCheck(cudaFree(propagate_buf_)); //2 buffers
  //cudaErrCheck(cudaFree(data_));
  cudaErrCheck(cudaFree(h_data_n));
  cudaErrCheck(cudaFree(c_data_n));
  cudaErrCheck(cudaFree(c_data_bw_n));
  cudaErrCheck(cudaFree(h_data_bw_n));
  cudaErrCheck(cudaFree(tmp_h_fw_n));
  cudaErrCheck(cudaFree(tmp_i_fw_n));
  cudaErrCheck(cudaFree(tmp_h_bw_n));
  cudaErrCheck(cudaFree(tmp_i_bw_n)); 
  //cudaErrCheck(cudaMalloc((void**)&input_buf_, seqLength * input_buf_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&propagate_buf_,2* seqLength * 2 * cell_dim_ * sizeof(float))); //2 buffers
  //cudaErrCheck(cudaMalloc((void**)&data_,3 * cell_dim_ * sizeof(float))); //pay attentio for cases with non cell_dim input data
  cudaErrCheck(cudaMalloc((void**)&h_data_n, 2 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_data_n, 2 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_data_bw_n, 2 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&h_data_bw_n, 2 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&tmp_h_fw_n, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&tmp_i_fw_n, (seqLength+1) * 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&tmp_h_bw_n, 4 * cell_dim_ * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&tmp_i_bw_n, (seqLength+1) * 4 * cell_dim_ * sizeof(float)));

  /*cudaMemset(h_data_n, 0, cell_dim_*sizeof(float));
  cudaMemset(c_data_n, 0, cell_dim_*sizeof(float));
  cudaMemset(h_data_bw_n + seqLength*cell_dim_, 0, cell_dim_*sizeof(float));
  cudaMemset(c_data_bw_n + seqLength*cell_dim_, 0, cell_dim_*sizeof(float));
*/
  }

void Net::Feedforward(cublasHandle_t handle, float* in, float* out, int seqLength) {
  // we need at least 2 input buffers
  // propagate by using exactly 2 auxiliary buffers
  int L = 0;
  float time=0.f;
  
  time+=layers_[L]->Propagate(handle, in, propagate_buf_ + (L%2)*seqLength * 2 * cell_dim_, seqLength, tmp_h_fw_n, tmp_i_fw_n, tmp_h_bw_n, tmp_i_bw_n, h_data_n, c_data_n, h_data_bw_n, c_data_bw_n);
  for(L++; L<NumLayers(); L++) {
    time+=layers_[L]->Propagate(handle, propagate_buf_ + ((L-1)%2)*seqLength*2*cell_dim_ ,propagate_buf_ + (L%2)*seqLength*2*cell_dim_, seqLength, tmp_h_fw_n, tmp_i_fw_n, tmp_h_bw_n, tmp_i_bw_n, h_data_n, c_data_n, h_data_bw_n, c_data_bw_n);
  }
  time+=Af_l_->Propagate(handle, propagate_buf_ + ((L-1)%2)*seqLength*2*cell_dim_, out, seqLength);
  //printf("timing precise = %f ms", time);
  //layers_[L]->Propagate(propagate_buf_[(L-1)%2], out); //not commented
  // release the buffers we don't need anymore
}


int Net::OutputDim() {
  return layers_.back()->OutputDim();
}

int Net::InputDim() {
  return layers_.front()->InputDim();
}


Layer* Net::GetLayer(int layer) {
  return layers_[layer];
}

void Net::SetLayer(int c, Layer *layer) {
  delete layers_[c];
  layers_[c] = layer;
}

void Net::AppendLayer(Layer* dynamically_allocated_layer) {
  // append,
  layers_.push_back(dynamically_allocated_layer);
 }

void Net::AppendAffineTransformLayer(AffineTransform *dynamically_allocated_AffineTransform){
  Af_l_=dynamically_allocated_AffineTransform;
}


void Net::Destroy() {
  for(int i=0; i<NumLayers(); i++) {
    delete layers_[i];
  }
  delete Af_l_;
  layers_.resize(0);
  }




