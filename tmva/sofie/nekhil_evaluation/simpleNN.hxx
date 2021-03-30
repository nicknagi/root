//Code generated automatically by TMVA for Inference of Model file [simpleNN.onnx] at [Tue Mar 30 14:46:19 2021] 
#include<vector>
namespace TMVA_SOFIE_simpleNN{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_output[100];
float tensor_2[100];
std::vector<float> infer(float* tensor_1,float* tensor_input){
	for (int id = 0; id < 100 ; id++){
		tensor_2[id] = tensor_input[id] + tensor_1[id];
	}
	for (int id = 0; id < 100 ; id++){
		tensor_output[id] = ((tensor_2[id] > 0 )? tensor_2[id] : 0);
	}
	std::vector<float> ret (tensor_output, tensor_output + sizeof(tensor_output) / sizeof(tensor_output[0]));
	return ret;
}
} //TMVA_SOFIE_simpleNN
