
#ifndef AFFINETRANSFORM_H_
#define AFFINETRANSFORM_H_

//#include </usr/local/cuda-8.0/include/vector_types.h>
#include "kernels.h"



/**
 * class, building block of the network.
 * It is able to propagate (PropagateFnc: compute the output based on its input)
 * and backpropagate (BackpropagateFnc: i.e. transform loss derivative w.r.t. output to derivative w.r.t. the input)
 * the formulas are implemented in descendant classes (AffineTransform,Sigmoid,Softmax,...).
 */ 
class AffineTransform {

  
 /// General interface of a component  
 public:
  AffineTransform(int input_dim, int output_dim);
 ~AffineTransform();

 
  /// Get size of input vectors
  int InputDim() const { 
    return input_dim_; 
  }  
  /// Get size of output vectors 
  int OutputDim() const { 
    return output_dim_; 
  }

  void Init();
  // Perform forward pass propagation Input->Output
  float Propagate(cublasHandle_t handle, float* in, float* out, int seqLength); 
  // Perform backward pass propagation, out_diff -> in_diff
 
 float* wei_affine(){
  return wei_affine_;
 }
 float* bias(){
  return bias_;
 }
 /// Data members
 protected:
  int input_dim_;  ///< Size of input vectors
  int output_dim_; ///< Size of output vectors

    // parameters of the forward layer
  float* wei_affine_;
  float* bias_;
  
    // propagaton buffer
    
};



#endif
