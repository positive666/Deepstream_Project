#ifndef ARCFACE_TRT_ARCFACE_H
#define ARCFACE_TRT_ARCFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cblas.h"
#include <cublasLt.h>
#include <vector>

using namespace nvinfer1;

void checkCublasStatus(cublasStatus_t status);
float cosine_similarity(std::vector<float> &A, std::vector<float> &B);
std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
                                                        std::vector<struct KnownID> &B, const int size,
                                                        bool normalize = false);
void cublas_batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs);
void batch_cosine_similarity(std::vector<std::vector<float>> A, std::vector<struct KnownID> B, int size,
                             float *outputs);
void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs);
void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                     std::vector<struct CroppedFace> &croppedFaces);
class CosineSimilarityCalculator {
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


class ArcFace {
public:
    ArcFace(const std::string &config_file);
    ~ArcFace();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
    void initKnownEmbeds(int num);
    void initCosSim();
    float *featureMatching();
    //std::tuple<std::vector<std::string>, std::vector<float>> getOutputs(float *output_sims);
   void visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims);
    void addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox);
    void resetVariables();
    // void initKnownEmbeds(int num);
    // void initCosSim();
private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> &image);
    void ReshapeandNormalize(float out[], cv::Mat &feature, const int &MAT_SIZE, const int &outSize);

    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
  //  CosineSimilarityCalculator cossim;
};

#endif //ARCFACE_TRT_ARCFACE_H
