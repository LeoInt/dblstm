#include <cuda.h>

typedef float fp32;
typedef unsigned short fp16;

typedef struct 
{
    float H;
    float L;
} doublefp32;

__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in);
__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in);
__device__ inline doublefp32 dfp16_to_dfp32_gpu(const fp32 in);


__device__ inline doublefp32 dfp16_to_dfp32_gpu(const fp32 input){
	unsigned int t1;
	unsigned int t2;
	unsigned int t3;
	unsigned int t11,t12;
	unsigned int t21,t22;
	unsigned int t31,t32;

	unsigned int in=*((unsigned int*)&input);	
	//unsigned int ins=(unsigned int)input>>16;
	
	doublefp32 out;
	
	t1 = in & 0x7fff7fff;
	//t11 =in & 0x7fff;   
	
	t2 = in & 0x80008000;                       // Sign bit
	//t22 = ins & 0x8000;                       // Sign bit
	
	t3 = in & 0x7c007c00;                       // Exponent
       	//t32 = ins & 0x7c00;                       // Exponent
         
	//t1 = in & 0x7fff;                       // Non-sign bits 	
	//t2 = in & 0x8000;                       // Sign bit
	//t3 = in & 0x7c00;                       // Exponent
        //t3 = t3>>16;
	t12= t1 & 0xffff;	
	t11= t1>>16;

	t11 <<= 13;                              // Align mantissa on MSB
	t12 <<= 13;                              // Align mantissa on MSB
	
	t22= t2 & 0xffff;	
	t21= t2 & 0xffff0000; //already shifted 	
	t22 <<= 16;                              // Shift sign bit into position
	//t22 <<= 16;                              // Shift sign bit into position
	
	
	t11 += 0x38000000;                       // Adjust bias
	t12 += 0x38000000;                       // Adjust bias
	
	t31=t3>>16;
	t32=t3 & 0xffff;

	t11 = (t31 == 0 ? 0 : t11);                // Denormals-as-zero
	t12 = (t32 == 0 ? 0 : t12);                // Denormals-as-zero

	t11 |= t21;                               // Re-insert sign bit
	t12 |= t22;                               // Re-insert sign bit

	
	*((unsigned int*)(&(out.H))) = t12;
	*((unsigned int*)(&(out.L))) = t11;
	return out;

}

__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in){
	unsigned int t1;
	unsigned int t2;
	unsigned int t3;
	fp32 out;

	t1 = in & 0x7fff;                       // Non-sign bits
	t2 = in & 0x8000;                       // Sign bit
	t3 = in & 0x7c00;                       // Exponent
        
	t1 <<= 13;                              // Align mantissa on MSB
	t2 <<= 16;                              // Shift sign bit into position

	t1 += 0x38000000;                       // Adjust bias
	t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero
	t1 |= t2;                               // Re-insert sign bit

	*((unsigned int*)(&out)) = t1;
	return out;
}
__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in){
	unsigned int inu = *((unsigned int*)&in);
	unsigned int t1;
	unsigned int t2;
	unsigned int t3;
	fp16 out;

	t1 = inu & 0x7fffffff;                 // Non-sign bits
	t2 = inu & 0x80000000;                 // Sign bit
	t3 = inu & 0x7f800000;                 // Exponent
        
	t1 >>= 13;                             // Align mantissa on MSB
	t2 >>= 16;                             // Shift sign bit into position

	t1 -= 0x1c000;                         // Adjust bias

	t1 = (t3 < 0x38800000) ? 0 : t1;       // Flush-to-zero
	t1 = (t3 > 0x8e000000) ? 0x7bff : t1;  // Clamp-to-max
	t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

	t1 |= t2;                              // Re-insert sign bit

	*((unsigned short*)(&out)) = t1;
	return out;
}



