#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>


void __global__ matrix_scatter(const cuComplex* A_global, cuComplex** A_array, const int m, const int k, const int batch_sizes);

void __global__ matrix_gather(cuComplex* C_global, cuComplex** C_array, const int m, const int n, const int batch_sizes);
