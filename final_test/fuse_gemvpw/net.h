#ifndef NET_H_
#define NET_H_

#include <vector>

#include "layer.h"
#include "AffineTransform.h"

// Define some error checking macros.

class Net {
 public:
  Net(int nLayers, int cell_dim);
  
  ~Net(); 

 public:
  /// Perform forward pass through the network
  //void Propagate(float *in, float *out); 
  void Resize(int seqLength);

  void Feedforward(cublasHandle_t handle, float* in, float* out, int seqLength); 

  /// Dimensionality on network input (input feature dim.)
  int InputDim(); 
  /// Dimensionality of network outputs (posteriors | bn-features | etc.)
  int OutputDim(); 

  /// Returns number of layers
  int NumLayers() { return layers_.size(); }

  Layer* GetLayer(int c);

  /// Sets the c'th layer to "layer", taking ownership of the pointer
  /// and deleting the corresponding one that we own.
  void SetLayer(int c, Layer *layer);
 

  /// Access to forward pass buffers
  float* PropagateBuffer()  { 
    return propagate_buf_; 
  }
 
  /// Appends this layer to the layers already in the neural net.
  void AppendLayer(Layer *dynamically_allocated_layer);

  void AppendAffineTransformLayer(AffineTransform *dynamically_allocated_AffineTransform);

  /// Relese the memory
  void Destroy();

 private:
  /// Vector which contains all the layers composing the neural network,
  /// the layers are for example: AffineTransform, Sigmoid, Softmax
  std::vector<Layer*> layers_; //array of Layer* 
  AffineTransform* Af_l_;
  int cell_dim_;
  //int input_buf_dim_;

  float *propagate_buf_; ///< buffers for forward pass
  float *tmp_h_fw_n;
  float *tmp_i_fw_n;

    // back-propagation buffer
  float *tmp_h_bw_n;
  float *tmp_i_bw_n;

  float *h_data_n;
  float *c_data_n;
  
  float *h_data_bw_n;
  float *c_data_bw_n;
};
  


#endif  // NET_H_

