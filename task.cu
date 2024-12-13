#include "task.h"
#include <cuda_runtime.h>
#include <iostream>

// Constructor to initialize task dimensions and compute padded dimensions
Task::Task(int M, int N, int R)
    : M(M), N(N), R(R),
      M_pad((M + 15) / 16 * 16), N_pad((N + 15) / 16 * 16), R_pad((R + 15) / 16 * 16),
      d_pDense(nullptr), d_Qmat(nullptr), d_Rmat(nullptr),
      d_localB(nullptr), d_localC(nullptr), d_localMat(nullptr) {}

// Destructor to free GPU memory
Task::~Task() {
    cudaFree(d_pDense);
    cudaFree(d_Qmat);
    cudaFree(d_Rmat);
    cudaFree(d_localB);
    cudaFree(d_localC);
    cudaFree(d_localMat);
}

// Method to allocate GPU memory for task matrices using padded dimensions
void Task::allocateDeviceMemory(int nRHS) {
    cudaMalloc(&d_pDense, M_pad * N_pad * sizeof(cuComplex));
    cudaMalloc(&d_Qmat, M_pad * R_pad * sizeof(cuComplex));
    cudaMalloc(&d_Rmat, R_pad * N_pad * sizeof(cuComplex));
    cudaMalloc(&d_localB, N_pad * nRHS * sizeof(cuComplex));
    cudaMalloc(&d_localC, M_pad * nRHS * sizeof(cuComplex));
    cudaMalloc(&d_localMat, M_pad * N_pad * sizeof(cuComplex));
}
