#include "kernel.cuh"

void __global__ matrix_scatter(const cuComplex* A_global, cuComplex** A_array, const int m, const int k ,const int dim_global) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //A_global and A_array are column major
    if (ix < m && iy < k) 
    {
        {
            A_array[blockIdx.z][ix + iy * m] = A_global[ix + iy * dim_global];
        }
    }
}

void __global__ matrix_gather(cuComplex* C_global, cuComplex** C_array, const int m, const int n, const int dim_global)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < m && iy < n)
    {
        C_global[ix + iy * dim_global] = C_array[blockIdx.z][ix + iy * m];
    }
}