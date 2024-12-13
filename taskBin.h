#ifndef TASKBIN_H
#define TASKBIN_H

#include "task.h"
#include <vector>
#include <map>
#include <tuple>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

class TaskBin {
public:
    int M_pad = 0, N_pad = 0, R_pad = 0; // Bin info based on padded dimensions
    std::vector<Task> tasks;
    cudaStream_t stream;

    TaskBin(); // Default constructor
    TaskBin(int M_pad, int N_pad, int R_pad);
    ~TaskBin();

    void addTask(const Task &task);
    void padAndCompute(cublasHandle_t &handle);
};

#endif // TASKBIN_H
