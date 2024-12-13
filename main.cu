#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <map>

#include "kernel.cuh"

#define CUDA_CHECK(call)                                                                                           \
    do                                                                                                             \
    {                                                                                                              \
        cudaError_t err = call;                                                                                    \
        if (err != cudaSuccess)                                                                                    \
        {                                                                                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                    \
        }                                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                              \
    do                                                                                  \
    {                                                                                   \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            fprintf(stderr, "cuBLAS error in %s (%s:%d)\n", #call, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

const int group_count = 3;
const int batch_sizes[group_count] = {8192};
const int m[group_count] = {32, 64, 128};
const int n[group_count] = {32, 64, 128};
const int k[group_count] = {32, 64, 128};

const int matrix_dim_global = 1024;

using data_type = cuComplex;

void initialize_matrix(data_type **matrices, int rows, int cols, int batch_num)
{
    data_type value = make_cuComplex(1.f, 0.f);
    for (int ibatch = 0; ibatch < batch_num; ibatch++)
    {
        for (int i = 0; i < rows * cols; i++)
        {
            matrices[ibatch][i] = value;
        }
    }
}

void print_matrix(int rows, int cols, const data_type *matrix, int ld)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cuComplex val = matrix[i + j * ld];
            printf("(%f, %f) ", cuCrealf(val), cuCimagf(val));
        }
        printf("\n");
    }
}

std::vector<std::tuple<int, int, int>> parseMILFile(const std::string &filePath)
{
    std::vector<std::tuple<int, int, int>> parsedData;
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return parsedData;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream lineStream(line);
        std::string word;
        std::vector<std::string> words;
        while (lineStream >> word)
        {
            words.push_back(word);
        }

        // Check if the line has at least 7 elements (based on the given format)
        if (words.size() >= 7)
        {
            int M = std::stoi(words[4]);
            int N = std::stoi(words[5]);
            int R = std::stoi(words[6]);
            parsedData.emplace_back(M, N, R);
        }
    }

    file.close();
    return parsedData;
}

std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>>
binByMultipleOf16(const std::vector<std::tuple<int, int, int>> &data)
{
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> bins;

    for (const auto &entry : data)
    {
        int M = std::get<0>(entry);
        int N = std::get<1>(entry);
        int R = std::get<2>(entry);

        // Calculate the bin key
        int binM = (M+15) / 16;
        int binN = (N+15) / 16;
        int binR = (R+15) / 16;
        std::tuple<int, int, int> binKey = std::make_tuple(binM, binN, binR);

        // Insert the entry into the appropriate bin
        bins[binKey].emplace_back(M, N, R);
    }

    return bins;
}

// Helper function to print the bins
void printBins(const std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> &bins)
{
    for (const auto &[key, values] : bins)
    {
        std::cout << "Bin (" << std::get<0>(key) << ", " << std::get<1>(key) << ", " << std::get<2>(key) << "):\n";
        for (const auto &value : values)
        {
            std::cout << "    M: " << std::get<0>(value)
                      << ", N: " << std::get<1>(value)
                      << ", R: " << std::get<2>(value) << "\n";
        }
    }
}

// Helper function to print the size of each bin
void printBinsSize(const std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> &bins) {
    std::cout << "Bin Sizes:\n";
    for (const auto &[key, values] : bins) {
        std::cout << "Bin (" << std::get<0>(key)*16 << ", " << std::get<1>(key)*16 << ", " << std::get<2>(key)*16 << "): "
                  << values.size() << " elements\n";
    }
}

// Function to split the bins into two maps: R = 0 and R > 0
std::pair<
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>>,
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>>>
splitBins(const std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> &bins) {
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> binsR0;
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>> binsRPositive;

    for (const auto &[key, values] : bins) {
        for (const auto &value : values) {
            int R = std::get<2>(value);
            if (R == 0) {
                binsR0[key].emplace_back(value);
            } else {
                binsRPositive[key].emplace_back(value);
            }
        }
    }

    return {binsR0, binsRPositive};
}


int main()
{

    const std::string filePath = "MIL_Thread_combo_small.txt";
    auto data = parseMILFile(filePath);
     auto bins = binByMultipleOf16(data);
    //printBins(bins);
   // printBinsSize(bins);

    auto [binsR0, binsRPositive] = splitBins(bins);

    std::cout << "Bins with R = 0:\n";
    printBinsSize(binsR0);

    std::cout << "\nBins with R > 0:\n";
    printBinsSize(binsRPositive);


    cublasHandle_t cublasH[group_count];
    cudaStream_t streams[group_count];

    data_type **A[group_count], **B[group_count], **C[group_count];
    data_type **d_A[group_count], **d_B[group_count], **d_C[group_count];
    data_type **d_A_array[group_count], **d_B_array[group_count], **d_C_array[group_count];

    data_type *d_A_global, *d_B_global, *d_C_global;

    CUDA_CHECK(cudaMalloc((void **)&d_A_global, sizeof(data_type) * matrix_dim_global * matrix_dim_global));
    CUDA_CHECK(cudaMalloc((void **)&d_B_global, sizeof(data_type) * matrix_dim_global * matrix_dim_global));
    CUDA_CHECK(cudaMalloc((void **)&d_C_global, sizeof(data_type) * matrix_dim_global * matrix_dim_global));

    for (int g = 0; g < group_count; g++)
    {
        int batch_size = batch_sizes[g];
        int lm = m[g], ln = n[g], lk = k[g];
        A[g] = new data_type *[batch_size];
        B[g] = new data_type *[batch_size];
        C[g] = new data_type *[batch_size];

        for (int i = 0; i < batch_size; i++)
        {
            A[g][i] = new data_type[lm * lk];
            B[g][i] = new data_type[lk * ln];
            C[g][i] = new data_type[lm * ln];
        }

        initialize_matrix(A[g], lm, lk, batch_size);
        initialize_matrix(B[g], lk, ln, batch_size);
    }

    // print matrix
    /*
    for (int g = 0; g < group_count; g++)
    {
        int batch_num =  batch_sizes[g];
        for (int i = 0; i < batch_num; i++)
        {
            printf("Group %d, Matrix %d:\n", g, i);
            print_matrix(m[g], k[g], A[g][i], m[g]);
            print_matrix(k[g], n[g], B[g][i], k[g]);
        }
    }
*/
    const data_type alpha = make_cuComplex(1.f, 0.f);
    const data_type beta = make_cuComplex(0.f, 0.f);

    for (int g = 0; g < group_count; g++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
        CUBLAS_CHECK(cublasCreate(&cublasH[g]));
        CUBLAS_CHECK(cublasSetStream(cublasH[g], streams[g]));

        int batch_num = batch_sizes[g];

        // d_A, d_B, d_C are device pointers of size batch_num to store the device pointers of batch_num matrices

        d_A[g] = new data_type *[batch_num];
        d_B[g] = new data_type *[batch_num];
        d_C[g] = new data_type *[batch_num];

        CUDA_CHECK(cudaMalloc((void **)&d_A_array[g], sizeof(data_type *) * batch_num));
        CUDA_CHECK(cudaMalloc((void **)&d_B_array[g], sizeof(data_type *) * batch_num));
        CUDA_CHECK(cudaMalloc((void **)&d_C_array[g], sizeof(data_type *) * batch_num));

        for (int i = 0; i < batch_num; i++)
        {
            size_t A_size = sizeof(data_type) * m[g] * k[g];
            size_t B_size = sizeof(data_type) * k[g] * n[g];
            size_t C_size = sizeof(data_type) * m[g] * n[g];

            CUDA_CHECK(cudaMalloc((void **)&d_A[g][i], sizeof(data_type) * A_size));
            CUDA_CHECK(cudaMalloc((void **)&d_B[g][i], sizeof(data_type) * B_size));
            CUDA_CHECK(cudaMalloc((void **)&d_C[g][i], sizeof(data_type) * C_size));

            CUDA_CHECK(cudaMemcpyAsync(d_A[g][i], A[g][i], A_size, cudaMemcpyHostToDevice, streams[g]));
            CUDA_CHECK(cudaMemcpyAsync(d_B[g][i], B[g][i], B_size, cudaMemcpyHostToDevice, streams[g]));

            // d_Ai, d_Bi, d_Ci are device pointers, but in host memory.
            // CUDA_CHECK(cudaMemcpyAsync(&d_A[g][i], &d_Ai, sizeof(data_type *), cudaMemcpyHostToDevice, streams[g]));
            // CUDA_CHECK(cudaMemcpyAsync(&d_B[g][i], &d_Bi, sizeof(data_type *), cudaMemcpyHostToDevice, streams[g]));
            // CUDA_CHECK(cudaMemcpyAsync(&d_C[g][i], &d_Ci, sizeof(data_type *), cudaMemcpyHostToDevice, streams[g]));

            // printf("Group %d, Matrix %d, d_Ci=%p, d_C[g]=%p\n", g, i, d_Ci, d_C[g]);

            // print d_A, A_B, d_C
            // printf("Group %d, Matrix %d, d_A=%p, d_B=%p, d_C=%p\n", g, i, d_A[g][i], d_B[g][i], d_C[g][i]);
        }

        // copy d_A, d_B, d_C to device memory
        CUDA_CHECK(cudaMemcpyAsync(d_A_array[g], d_A[g], sizeof(data_type *) * batch_num, cudaMemcpyHostToDevice, streams[g]));
        CUDA_CHECK(cudaMemcpyAsync(d_B_array[g], d_B[g], sizeof(data_type *) * batch_num, cudaMemcpyHostToDevice, streams[g]));
        CUDA_CHECK(cudaMemcpyAsync(d_C_array[g], d_C[g], sizeof(data_type *) * batch_num, cudaMemcpyHostToDevice, streams[g]));

        // d_A, d_B, d_C are device pointers, but their inside values, such as d_C[g][i] are in device memory, which cannot be access directly in host code
    }

    for (int g = 0; g < group_count; g++)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int g = 0; g < group_count; g++)
    {

        dim3 threads_scatter = dim3(32, 32);
        dim3 blocks_scatter = dim3((m[g] + threads_scatter.x - 1) / threads_scatter.x, (k[g] + threads_scatter.y - 1) / threads_scatter.y, batch_sizes[g]);

        matrix_scatter<<<blocks_scatter, threads_scatter, 0, streams[g]>>>(d_A_global, d_A_array[g], m[g], k[g], matrix_dim_global);

        CUBLAS_CHECK(cublasCgemmBatched(
            cublasH[g],
            CUBLAS_OP_N, CUBLAS_OP_N,
            m[g], n[g], k[g],
            &alpha,
            d_A_array[g], m[g],
            d_B_array[g], k[g],
            &beta,
            d_C_array[g], m[g],
            batch_sizes[g]));

        dim3 threads_gather = dim3(32, 32);
        dim3 blocks_gather = dim3((m[g] + threads_gather.x - 1) / threads_gather.x, (n[g] + threads_gather.y - 1) / threads_gather.y, batch_sizes[g]);

        matrix_gather<<<blocks_gather, threads_gather, 0, streams[g]>>>(d_C_global, d_C_array[g], m[g], n[g], matrix_dim_global);
    }

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %f ms\n", milliseconds);
    // calculate GFLOPS
    size_t total_flops = 0;
    for (int g = 0; g < group_count; g++)
    {
        total_flops += 6 * m[g] * n[g] * k[g] * batch_sizes[g];
    }
    printf("Total FLOPS: %lu\n", total_flops);
    float gflop = total_flops / 1e9;
    printf("GFLOP: %f\n", total_flops / 1e9);

    float gflops = gflop * 1e3 / (milliseconds);
    printf("GFLOP/S: %f\n", gflops);

    /*


    for (int g = 0; g < group_count; g++)
    {
        int batch_num =  batch_sizes[g];
        for (int i = 0; i < batch_num; i++)
        {
            printf("Group %d, Matrix %d:\n", g, i);
            printf("C[%d][%d]=%p\n", g, i, C[g][i]);
            printf("d_C[%d][%d]=%p\n", g, i, d_C[g][i]);
            CUDA_CHECK(cudaMemcpyAsync(C[g][i], d_C[g][i], sizeof(data_type) * m[g] * n[g], cudaMemcpyDeviceToHost, streams[g]));
        }
    }

    for (int g = 0; g < group_count; g++)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));

        for (int i = 0; i < group_sizes[g]; i++)
        {
            printf("Group %d, Matrix %d:\n", g, i);
            print_matrix(m[g], n[g], C[g][i], m[g]);
        }

        for (int i = 0; i < group_sizes[g]; i++)
        {
            delete[] A[g][i];
            delete[] B[g][i];
            delete[] C[g][i];
        }

        delete[] A[g];
        delete[] B[g];
        delete[] C[g];

        cudaFree(d_A[g]);
        cudaFree(d_B[g]);
        cudaFree(d_C[g]);

        cublasDestroy(cublasH[g]);
        cudaStreamDestroy(streams[g]);
    }
    */
    return EXIT_SUCCESS;
}
