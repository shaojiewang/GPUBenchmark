
#pragma once

#include<array>
#include<numeric>

template <typename T, int nDim>
struct testTensor
{
    /* data */
    std::array<int, nDim> lengths;
    std::array<int, nDim> strides;

    T* t_data;

    testTensor() = default;

    testTensor(std::array<int, nDim> lengths_, std::array<int, nDim> strides_)
        : lengths(lengths_), strides(strides_)
    {

    }

    ~testTensor() = default;

    size_t get_tensor_space_size(){
        size_t t_size = 1;
        for(int i = 0; i < nDim; i++)
        {
            t_size += (lengths[i] - 1) * strides[i]; 
        }
        return t_size;
    }

    size_t get_tensor_size(){
        return std::accumulate(lengths.begin(), lengths.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }

};


void cudaBWTest();