#ifndef TASK_H
#define TASK_H

#include <cuda_runtime.h>
#include <cuComplex.h>

// Task struct to hold task-related data
struct Task {
    int M, N, R;         // Original dimensions
    int M_pad, N_pad, R_pad; // Padded dimensions
    cuComplex *d_pDense;
    cuComplex *d_Qmat;
    cuComplex *d_Rmat;
    cuComplex *d_localB;
    cuComplex *d_localC;
    cuComplex *d_localMat;

    Task(int M, int N, int R);
    ~Task();

    void allocateDeviceMemory(int nRHS);
};

#endif // TASK_H

