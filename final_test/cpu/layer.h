
#ifndef LAYER_H_
#define LAYER_H_

//#include </usr/local/cuda-8.0/include/vector_types.h>
#include "kernels.h"
#include <stdlib.h>
extern "C"
{
   #include <cblas.h>
}

/**#include <stdlib.h>
 * class, building block of the network.
 * It is able to propagate (PropagateFnc: compute the output based on its input)
 * and backpropagate (BackpropagateFnc: i.e. transform loss derivative w.r.t. output to derivative w.r.t. the input)
 * the formulas are implemented in descendant classes (AffineTransform,Sigmoid,Softmax,...).
 */ 
class Layer {

  
 /// General interface of a component  
 public:
  Layer(int input_dim, int output_dim, int cell_dim);
 ~Layer();

 
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
  double Propagate(float* in, float* out, int seqLength, float* tmp_h_fw, float* tmp_i_fw, float* tmp_h_bw, float* tmp_i_bw, float* h_data, float* c_data, float* h_data_bw, float* c_data_bw); 
  // Perform backward pass propagation, out_diff -> in_diff
 
 float* wei_gifo_x_fw(){
  return wei_gifo_x_fw_;
 }
 float* wei_gifo_m_fw(){
  return wei_gifo_m_fw_;
 }

 float* bias_fw(){
   return bias_fw_;
 }
  float* phole_i_c_fw(){
    return phole_i_c_fw_;
  }
  float* phole_f_c_fw(){
    return phole_f_c_fw_;
  }
  float* phole_o_c_fw(){
    return phole_o_c_fw_;
  }
 float* wei_gifo_x_bw(){
  return wei_gifo_x_bw_;
 }
 float* wei_gifo_m_bw(){
  return wei_gifo_m_bw_;
 }
 float* bias_bw(){
   return bias_bw_;
 }
 float* phole_i_c_bw(){
    return phole_i_c_bw_;
  }
  float* phole_f_c_bw(){
    return phole_f_c_bw_;
  }
  float* phole_o_c_bw(){
    return phole_o_c_bw_;
  }
 /// Data members
 protected:
  int input_dim_;  ///< Size of input vectors
  int output_dim_; ///< Size of output vectors

  int cell_dim_;
    // parameters of the forward layer
  float* wei_gifo_x_fw_;
  float* wei_gifo_m_fw_;
  float* bias_fw_;
  float* phole_i_c_fw_;
  float* phole_f_c_fw_;
  float* phole_o_c_fw_;
    // parameters of the backward layer
  float* wei_gifo_x_bw_;
  float* wei_gifo_m_bw_;
  float* bias_bw_;
  float* phole_i_c_bw_;
  float* phole_f_c_bw_;
  float* phole_o_c_bw_;
    // propagaton buffer
    
};



#endif
