#include "taskBin.h"
#include <iostream>

// Default constructor
TaskBin::TaskBin() : M_pad(0), N_pad(0), R_pad(0) {
    cudaStreamCreate(&stream);
}

// Constructor to initialize bin dimensions and CUDA stream
TaskBin::TaskBin(int M_pad, int N_pad, int R_pad)
    : M_pad(M_pad), N_pad(N_pad), R_pad(R_pad) {
    cudaStreamCreate(&stream);
}

// Destructor to destroy CUDA stream
TaskBin::~TaskBin() {
    cudaStreamDestroy(stream);
}

// Method to add a task to the bin
void TaskBin::addTask(const Task &task) {
    tasks.push_back(task);
}

// Method to pad and compute matrix multiplication using cuBLAS batch mode
void TaskBin::padAndCompute(cublasHandle_t &handle) {
    int batchSize = tasks.size();
    std::vector<const cuComplex *> d_A(batchSize);
    std::vector<const cuComplex *> d_B(batchSize);
    std::vector<cuComplex *> d_C(batchSize);

    for (int i = 0; i < batchSize; ++i) {
        d_A[i] = tasks[i].d_Qmat;
        d_B[i] = tasks[i].d_Rmat;
        d_C[i] = tasks[i].d_localMat;
    }

    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);

    cublasSetStream(handle, stream);
    cublasCgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       M_pad, R_pad, N_pad,
                       &alpha,
                       d_A.data(), M_pad,
                       d_B.data(), N_pad,
                       &beta,
                       d_C.data(), M_pad,
                       batchSize);
}
