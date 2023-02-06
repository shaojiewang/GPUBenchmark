cmake_minimum_required(VERSION 3.8)
project(CUDA_TEST_BENCH LANGUAGES CXX CUDA)

add_library(cudaTest cudaTest.cu cudaTest.h)
target_compile_features(cudaTest PUBLIC cxx_std_11)

add_executable(test test.cc)
target_link_libraries(test cudaTest)