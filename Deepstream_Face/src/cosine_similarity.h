#ifndef __COSINE_SIMILARITY_H__
#define __COSINE_SIMILARITY_H__
#include <algorithm>
#include <chrono>

#include <iostream>
#include "cblas.h"
#include "cuda_runtime_api.h"
#include <cublasLt.h>

void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);

class CosineSimilarityCalculator {
    /*
    Calculate C = A x B
    Input:
        A: m x k, row-major matrix
        B: n x k, row-major matrix
    Output:
        C: m x n, row-major matrix
    NOTE: Since cuBLAS use column-major matrix as input, we need to transpose A (transA=CUBLAS_OP_T).
    */
  public:
    CosineSimilarityCalculator();
    ~CosineSimilarityCalculator();
    void init(float *knownEmbeds, int numRow, int numCol);
    void calculate(float *embeds, int embedCount, float *outputs);

  private:
    cudaDataType_t dataType = CUDA_R_32F;
    cublasLtHandle_t ltHandle;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    void *workspace;
    const size_t workspaceSize = 1024 * 1024 * 4;
    cudaStream_t stream;
    float *dA, *dB, *dC;
    const float alpha = 1, beta = 0;
    int m, n, k, lda, ldb, ldc;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
};

#endif