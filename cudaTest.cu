#include "cudaTest.h"
#include <array>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

using testType = float;

#define nDim 1 

#define BLOCKSIZE_T 128

template<int BlockSize, typename T, class Tensor>
__global__ void TensorDataTransfer(Tensor A, Tensor B){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    B.t_data[tid] = A.t_data[tid];
}


void cudaBWTest()
{
    std::array<int, nDim> lengths;
    std::array<int, nDim> strides;
    for(int i = 0; i < nDim; i++){
        lengths[i] = 1024;
        if(i == 0)
            strides[i] = 1;
        else
            strides[i] = lengths[i - 1];
    }
    testTensor<testType, nDim> HostA(lengths, strides);
    testTensor<testType, nDim> HostB(lengths, strides);

    HostA.t_data = static_cast<testType* >(malloc(HostA.get_tensor_space_size() * sizeof(testType)));
    HostB.t_data = static_cast<testType* >(malloc(HostB.get_tensor_space_size() * sizeof(testType)));


    testTensor<testType, nDim> A_device;
    for(int i = 0; i < nDim; i++){
        A_device.lengths[i] = HostA.lengths[i];
        A_device.strides[i] = HostA.strides[i];
    }
    cudaMalloc(&A_device.t_data, A_device.get_tensor_space_size() * sizeof(testType));
    cudaMemcpy(A_device.t_data, HostA.t_data, A_device.get_tensor_space_size() * sizeof(testType), cudaMemcpyHostToDevice);


    testTensor<testType, nDim> B_device;
    for(int i = 0; i < nDim; i++){
        B_device.lengths[i] = HostB.lengths[i];
        B_device.strides[i] = HostB.strides[i];
    }
    cudaMalloc(&B_device.t_data, B_device.get_tensor_space_size() * sizeof(testType));
    //cudaMemcpy(B_device.t_data, B.t_data, B_device.get_tensor_space_size() * sizeof(testType), cudaMemcpyHostToDevice);

    size_t ele_size = HostA.get_tensor_size();

    dim3 dim_block(BLOCKSIZE_T, 1, 1);
    dim3 dim_grid(ele_size / BLOCKSIZE_T, 1, 1);

    auto startTime = high_resolution_clock::now();

    TensorDataTransfer<BLOCKSIZE_T, testType, decltype(B_device)><<<dim_grid, dim_block>>>(A_device, B_device);

    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(endTime - startTime); 
    std::cout << "cuda runtime" <<double(duration.count()) / 1000 << "ms" << std::endl;

    cudaMemcpy(HostB.t_data, B_device.t_data, B_device.get_tensor_space_size() * sizeof(testType), cudaMemcpyDeviceToHost);
    cudaFree(A_device.t_data);
    cudaFree(B_device.t_data);

    free(HostA.t_data);
    free(HostB.t_data);

}