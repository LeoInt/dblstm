#include "layer.h"
#include "AffineTransform.h"
#include "net.h"
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <pthread.h>
#define DEBUG 0
struct arg_struct {
    Net* net;
    cublasHandle_t handle;
    float* in;
    float* out;
    int seq;
    cudaStream_t str;
};

void *launch_net(void *arguments)
{
	Net* Nx;	
	struct arg_struct *args = (struct arg_struct *)arguments;
   	Nx = args -> net;
	cublasHandle_t handle = args->handle;
	float* x_in_d= args->in;
	float* x_out_d= args->out;
	int seqLength = args->seq;
	cudaStream_t stream= args->str;
	Nx->Feedforward(handle, stream, x_in_d, x_out_d, seqLength);
   
    //cudaStreamSynchronize(0);

    return NULL;
}


static void Print_matrix_to_file1(const char nmfile[], float* mat, int n, int m, int r_c){
    //const char nmfile[] = "out.txt";
    std::ofstream outseis(nmfile); // output, normal file
    float *data_host;
    data_host=(float*)malloc(n*m*sizeof(float));
    cudaMemcpy(data_host, mat, n*m*sizeof(float), cudaMemcpyDeviceToHost);  // this won't work, will throw error
    if(r_c==0){
      for (int jj=0; jj<n; jj++)
      {
        std::stringstream buf;
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+jj*m+ii);
                    //printf("%f ", temp);
          buf  << *temp <<" ";
                  //if(jj==101) printf("%f ", *temp);
        }
        outseis << buf.str() << "\n";
            //printf("\n%d %d row, col", jj, ii);
      }
    }else{
      for (int jj=0; jj<n; jj++)
      {
        std::stringstream buf;
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+ii*n+jj);
                    //printf("%f ", temp);
          buf  << *temp <<" ";
                  //if(jj==101) printf("%f ", *temp);
        }
        outseis << buf.str() << "\n";
            //printf("\n%d %d row, col", jj, ii);
      }
    }
    free(data_host);
  }



static void Print_matrix_to_file(const char nmfile[], float* mat, int n, int m, int r_c){
    //const char nmfile[] = "out.txt";
    std::ofstream outseis(nmfile); // output, normal file
    float *data_host;
    data_host=(float*)malloc(n*m*sizeof(float));
    cudaMemcpy(data_host, mat, n*m*sizeof(float), cudaMemcpyDeviceToHost);  // this won't work, will throw error
    if(r_c==0){
      for (int jj=0; jj<n; jj++)
      {
        std::stringstream buf;
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+jj*m+ii);
                    //printf("%f ", temp);
          
          buf << std::setprecision(8) << *temp <<" ";
          //buf << *temp <<" ";
                  //if(jj==101) printf("%f ", *temp);
        }
        outseis << buf.str() << "\n";
            //printf("\n%d %d row, col", jj, ii);
      }
    }else{
      for (int jj=0; jj<n; jj++)
      {
        std::stringstream buf;
        int ii;
        for (ii=0; ii<m; ii++)
        {
          float* temp=(float *)(data_host+ii*n+jj);
                    //printf("%f ", temp);
          buf << std::setprecision(8) << *temp <<" ";
          //buf << *temp <<" ";
                  //if(jj==101) printf("%f ", *temp);
        }
        outseis << buf.str() << "\n";
            //printf("\n%d %d row, col", jj, ii);
      }
    }
    free(data_host);
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


int main(int argc, char* argv[]) {
   int seqLength=100;
   int numLayers=4;
   int hiddenSize=320;
   int input_dim=2*hiddenSize;
   int output_dim=46;
   float* x_in;
   float* x_out;
   float* x_in_d;
   float* x_out_d;
    float* x_in_d1;
   float* x_out_d1;
    
 float* x_out_soft;

   if (argc == 4) {
      seqLength = atoi(argv[1]);
      numLayers =  atoi(argv[2]);
      hiddenSize =  atoi(argv[3]);
   }
   else if (argc == 1) {
      seqLength = 100;
      numLayers = 4;
      hiddenSize = 320;
   }
   
   //x_in = (float*)malloc(seqLength*input_dim*sizeof(float));
   cudaMallocHost((void**)&x_in, seqLength*input_dim*sizeof(float)); //pinned memory
   cudaMallocHost((void**)&x_out, seqLength*output_dim*sizeof(float)); //pinned memory
   //x_out = (float*)malloc(seqLength*output_dim*sizeof(float));  
   
   cudaMalloc((void**)&x_in_d, seqLength * input_dim * sizeof(float));
   cudaMalloc((void**)&x_out_d, seqLength * output_dim * sizeof(float));
   cudaMalloc((void**)&x_in_d1, seqLength * input_dim * sizeof(float));
   cudaMalloc((void**)&x_out_d1, seqLength * output_dim * sizeof(float));
   
   cudaMalloc((void**)&x_out_soft, seqLength * output_dim * sizeof(float));
   
   //srand (time(NULL));
  	
   for(int i=0; i<seqLength * input_dim; i++){
   		if(((rand() % 10) % 2)==0)
        x_in[i]=(rand() % 10)/10.f;
      else
        x_in[i]=-(rand() % 10)/10.f;
   		//printf("%f ", x_in[i]);
   }
   
   Net* N;
   Net* N1;
   Net* N2;
   
   Layer* L;
   AffineTransform* A;
   N = new Net(4,hiddenSize);
   N1 = new Net(4,hiddenSize);
   N2 = new Net(4,hiddenSize);
   
   N->Resize(seqLength); 
   N1->Resize(seqLength); 
   N2->Resize(seqLength);
   	for(int i=0; i<numLayers; i++){
   		if(i==0)
			L = new Layer(input_dim,2*hiddenSize,hiddenSize);
		else
			L = new Layer(2*hiddenSize,2*hiddenSize,hiddenSize);
		L->Init();
		if(DEBUG){	
   			char s1[2]="H";
   			char s2[5]=".txt";
   			char s3[5]="bias";
   			char s4[3]="Wx";
   			char s5[3]="Wh";
   			char pi[8]="phole_i";
   			char pf[8]="phole_f";
   			char po[8]="phole_o";
   			char result[19];
   			char result1[13];
   			char result2[11];
   			sprintf(result1,"%s%d%s",s3,i,s2);
   			Print_matrix_to_file(result1, L->bias_fw(), 4*hiddenSize, 1, 1);
   			sprintf(result2,"%s%d%s",s4,i,s2);
   			if(i==0)
   				Print_matrix_to_file(result2, L->wei_gifo_x_fw(), 4*hiddenSize, input_dim, 1);
   			else
   				Print_matrix_to_file(result2, L->wei_gifo_x_fw(), 4*hiddenSize, 2*hiddenSize, 1);
   			sprintf(result2,"%s%d%s",s5,i,s2);
   			Print_matrix_to_file(result2, L->wei_gifo_m_fw(), 4*hiddenSize, hiddenSize, 1); 
 			sprintf(result,"%s%d%s",pi,i,s2);
   			Print_matrix_to_file(result, L->phole_i_c_fw(), hiddenSize, 1, 1);
   			sprintf(result,"%s%d%s",pf,i,s2);
   			Print_matrix_to_file(result, L->phole_f_c_fw(), hiddenSize, 1, 1);
   			sprintf(result,"%s%d%s",po,i,s2);
   			Print_matrix_to_file(result, L->phole_o_c_fw(), hiddenSize, 1, 1);
  	
  			char s6[8]="bw.txt";
   			sprintf(result1,"%s%d%s",s3,i,s6);
   			Print_matrix_to_file(result1, L->bias_bw(), 4*hiddenSize, 1, 1);
   			sprintf(result2,"%s%d%s",s4,i,s6);
   			if(i==0)
   				Print_matrix_to_file(result2, L->wei_gifo_x_bw(), 4*hiddenSize, input_dim, 1);
   			else
   				Print_matrix_to_file(result2, L->wei_gifo_x_bw(), 4*hiddenSize, 2*hiddenSize, 1);
   			sprintf(result2,"%s%d%s",s5,i,s6);
   			Print_matrix_to_file(result2, L->wei_gifo_m_bw(), 4*hiddenSize, hiddenSize, 1); 
 			sprintf(result,"%s%d%s",pi,i,s6);
   			Print_matrix_to_file(result, L->phole_i_c_bw(), hiddenSize, 1, 1);
   			sprintf(result,"%s%d%s",pf,i,s6);
   			Print_matrix_to_file(result, L->phole_f_c_bw(), hiddenSize, 1, 1);
   			sprintf(result,"%s%d%s",po,i,s6);
   			Print_matrix_to_file(result, L->phole_o_c_bw(), hiddenSize, 1, 1);		
 		}
		N->AppendLayer(L);
		N1->AppendLayer(L);
		N2->AppendLayer(L);		
   	}
   	A = new AffineTransform(2*hiddenSize, output_dim); 
   	A->Init();
	if(DEBUG){
   		Print_matrix_to_file("WA.txt", A->wei_affine(), output_dim, input_dim, 1);
   		Print_matrix_to_file("bA.txt", A->bias(), output_dim, 1, 1);
	 }
	N->AppendAffineTransformLayer(A);
	N1->AppendAffineTransformLayer(A);
	N2->AppendAffineTransformLayer(A);
 	cudaMemcpy( x_in_d, x_in, seqLength * input_dim *sizeof(float), cudaMemcpyHostToDevice);
 	if(DEBUG) Print_matrix_to_file1("X.txt", x_in_d, input_dim, seqLength, 1);
 	cudaEvent_t start, stop;
	float elapsedTime=0.f;
	cublasHandle_t handle;
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;

  	size_t dimBlock = output_dim > CU1DBLOCK ? CU1DBLOCK : output_dim;
    size_t dimGrid = seqLength;
    

  	cublasCreate(&handle);

	cudaStreamCreate(&stream1);
  	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	pthread_t th1;
	pthread_t th2;
	pthread_t th3;    	
	struct arg_struct args1;
    	args1.net = N;
    	args1.seq = seqLength;
	args1.in = x_in_d;
	args1.out = x_out_d;
	args1.handle = handle;
	args1.str = stream1;
		
	struct arg_struct args2;
    	args2.net = N1;
    	args2.seq = seqLength;
	args2.in = x_in_d1;
	args2.out = x_out_d1;
	args2.handle = handle;
	args2.str = stream2;
	
	struct arg_struct args3;
    	args3.net = N1;
    	args3.seq = seqLength;
	args3.in = x_in_d1;
	args3.out = x_out_d1;
	args3.handle = handle;
	args3.str = stream3;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
  		
	cudaMemcpy( x_in_d, x_in, seqLength * input_dim *sizeof(float), cudaMemcpyHostToDevice);
	
    	if (pthread_create(&th1, NULL, &launch_net, (void *)&args1) != 0) {
        	//printf("Uh-oh!\n");
        	return -1;
   	 }
	cudaMemcpy( x_in_d1, x_in, seqLength * input_dim *sizeof(float), cudaMemcpyHostToDevice);
 		
	if (pthread_create(&th2, NULL, &launch_net, (void *)&args2) != 0) {
        	//printf("Uh-oh!\n");
        	return -1;
   	 }
	cudaMemcpy( x_in_d1, x_in, seqLength * input_dim *sizeof(float), cudaMemcpyHostToDevice);
 	
	if (pthread_create(&th3, NULL, &launch_net, (void *)&args3) != 0) {
        	//printf("Uh-oh!\n");
        	return -1;
   	 }
	pthread_join(th1, NULL);
	pthread_join(th2, NULL);
	pthread_join(th3, NULL);

 	//N->Feedforward(handle, stream1, x_in_d, x_out_d, seqLength);
	
	//N1->Feedforward(handle, stream2, x_in_d1, x_out_d1, seqLength);
 	//softmax_reduce_w(dimGrid, dimBlock, stream, x_out_soft, x_out_d, seqLength, output_dim);
 	cudaMemcpy( x_out, x_out_soft, seqLength * output_dim *sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); 	
	cudaEventRecord(stop);
 	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	printf("%f ", elapsedTime/3);
	if(DEBUG){
	 	Print_matrix_to_file("X1.txt", N->PropagateBuffer(), 2 * hiddenSize, seqLength, 1);
	 	Print_matrix_to_file("X2.txt", N->PropagateBuffer() + 2 * hiddenSize*seqLength, 2 * hiddenSize, seqLength, 1);
	 	Print_matrix_to_file("Xout.txt", x_out_d, output_dim, seqLength, 1);
	 	Print_matrix_to_file("Xsoft.txt", x_out_soft, output_dim, seqLength, 1);
 	}
 	delete N;
 	//cudaErrCheck(cudaMemcpy( devciao1, ciao1, sizeof(float), cudaMemcpyHostToDevice));    
}
