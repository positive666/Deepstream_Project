#include "arcface.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"
#include<fstream>
#include<iostream>
#include <faiss/AutoTune.h>
//#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <cmath>
#include <map>
#include "json.hpp"
using json = nlohmann::json;//注明空间



void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed with status " << status << "\n";
        throw std::logic_error("cuBLAS API failed");
    }
}
void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("CUDA API failed");
    }
}


CosineSimilarityCalculator::CosineSimilarityCalculator() {
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaStreamCreate(&stream));
}
void CosineSimilarityCalculator::init(float *knownEmbeds, int numRow, int numCol) {
    /*
    Calculate C = A x B
    Input:
        A: m x k, row-major matrix
        B: n x k, row-major matrix
    Output:
        C: m x n, row-major matrix
    NOTE: Since cuBLAS use column-major matrix as input, we need to transpose A (transA=CUBLAS_OP_T).
    */
    m = static_cast<const int>(numRow);
    k = static_cast<const int>(numCol);
    lda = static_cast<const int>(numCol);
    ldb = static_cast<const int>(numCol);
    ldc = static_cast<const int>(numRow);

    // alloc and copy known embeddings to GPU
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(float)));
    checkCudaStatus(cudaMemcpyAsync(dA, knownEmbeds, m * k * sizeof(float), cudaMemcpyHostToDevice, stream));

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    // here we just need to set the transforms for A and B
    // checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, dataType));
    // checkCublasStatus(
    //     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    // checkCublasStatus(
    //     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));
}

void CosineSimilarityCalculator::calculate(float *embeds, int embedCount, float *outputs) {
    n = embedCount;

    // Allocate arrays on GPU
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(float)));
    checkCudaStatus(cudaMemcpyAsync(dB, embeds, k * n * sizeof(float), cudaMemcpyHostToDevice, stream));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    cublasLtMatrixLayout_t Bdesc = NULL, Cdesc = NULL;
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dataType, m, n, ldc));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Do the actual multiplication
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));

    // Cleanup: descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));

    // Copy the result on host memory
    checkCudaStatus(cudaMemcpyAsync(outputs, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // CUDA stream sync
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Free GPU memory
    checkCudaStatus(cudaFree(dB));
    checkCudaStatus(cudaFree(dC));
}

CosineSimilarityCalculator::~CosineSimilarityCalculator() {
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(dA));
    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
}

void cublas_batch_cosine_similarity(float *A, float *B, int m, int n, int k, float *outputs) {
    const int lda = k, ldb = k, ldc = m;
    const float alpha = 1, beta = 0;

    cublasLtHandle_t ltHandle;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    void *workspace;
    size_t workspaceSize = 1024 * 1024 * 4;
    cudaStream_t stream;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    checkCudaStatus(cudaStreamCreate(&stream));

    // Allocate arrays on GPU
    auto start = std::chrono::high_resolution_clock::now();
    float *dA, *dB, *dC;
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(float)));

    checkCudaStatus(cudaMemcpyAsync(dA, A, m * k * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(dB, B, k * n * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\tAllo & cpy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    // cuBLASLt init
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    // checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32F));
    // checkCublasStatus(
    //     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    // checkCublasStatus(
    //     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Do the actual multiplication
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, 0));

    // Cleanup: descriptors are no longer needed as all GPU work was already enqueued
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "\tMatmul & cleanup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    // Copy the result on host memory
    checkCudaStatus(cudaMemcpyAsync(outputs, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Free GPU memory
    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(dA));
    checkCudaStatus(cudaFree(dB));
    checkCudaStatus(cudaFree(dC));
    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "\tCpy & free: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";
}

void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (blasint)embedCount, (blasint)classCount, size, 1, A, size, B,
                size, 0, outputs, classCount);
}

// std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
//                                                         std::vector<struct KnownID> &B, const int size,
//                                                         bool normalize) {
//     std::vector<std::vector<float>> outputs;
//     if (normalize) {
//         // Calculate cosine similarity
//         for (int A_index = 0; A_index < A.size(); ++A_index) {
//             std::vector<float> output;
//             for (int B_index = 0; B_index < B.size(); ++B_index) {
//                 float *p_A = &A[A_index][0];
//                 float *p_B = &B[B_index].embeddedFace[0];
//                 float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1);
//                 output.push_back(sim);
//             }
//             outputs.push_back(output);
//         }
//     } else {
//         // Pre-calculate norm for all elements
//         std::vector<float> A_norms, B_norms;
//         for (int i = 0; i < A.size(); ++i) {
//             float *p = &A[i][0];
//             float norm = cblas_snrm2((blasint)size, p, 1);
//             A_norms.push_back(norm);
//         }
//         for (int i = 0; i < B.size(); ++i) {
//             float *p = &B[i].embeddedFace[0];
//             float norm = cblas_snrm2((blasint)size, p, 1);
//             B_norms.push_back(norm);
//         }
//         // Calculate cosine similarity
//         for (int A_index = 0; A_index < A.size(); ++A_index) {
//             std::vector<float> output;
//             for (int B_index = 0; B_index < B.size(); ++B_index) {
//                 float *p_A = &A[A_index][0];
//                 float *p_B = &B[B_index].embeddedFace[0];
//                 float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1) / (A_norms[A_index] * B_norms[B_index]);
//                 output.push_back(sim);
//             }
//             outputs.push_back(output);
//         }
//     }
//     return outputs;
// }

float cosine_similarity(std::vector<float> &A, std::vector<float> &B) {
    if (A.size() != B.size()) {
        std::cout << A.size() << " " << B.size() << std::endl;
        throw std::logic_error("Vector A and Vector B are not the same size");
    }

    // Prevent Division by zero
    if (A.size() < 1) {
        throw std::logic_error("Vector A and Vector B are empty");
    }

    float *p_A = &A[0];
    float *p_B = &B[0];
    float mul = cblas_sdot((blasint)(A.size()), p_A, 1, p_B, 1);
    float d_a = cblas_sdot((blasint)(A.size()), p_A, 1, p_A, 1);
    float d_b = cblas_sdot((blasint)(A.size()), p_B, 1, p_B, 1);

    if (d_a == 0.0f || d_b == 0.0f) {
        throw std::logic_error("cosine similarity is not defined whenever one or both "
                               "input vectors are zero-vectors.");
    }

    return mul / (sqrt(d_a) * sqrt(d_b));
}

// void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
//                      std::vector<struct CroppedFace> &croppedFaces) {
//     for (std::vector<struct Bbox>::iterator it = outputBbox.begin(); it != outputBbox.end(); it++) {
//         cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
//         cv::Mat tempCrop = frame(facePos);
//         struct CroppedFace currFace;
//         cv::resize(tempCrop, currFace.faceMat, cv::Size(resize_h, resize_w), 0, 0, cv::INTER_CUBIC);
//         currFace.face = currFace.faceMat.clone();
//         currFace.x1 = it->x1;
//         currFace.y1 = it->y1;
//         currFace.x2 = it->x2;
//         currFace.y2 = it->y2;
//         croppedFaces.push_back(currFace);
//     }
// }

ArcFace::ArcFace(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["arcface"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
}

ArcFace::~ArcFace() = default;

void ArcFace::LoadEngine() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
        assert(engine != nullptr);
    }
}

bool ArcFace::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> sample_images = readFolder(folder_name);
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;
    //std::cout<<"test sieze:"<<sample_images.size()<<std::endl;
    EngineInference(sample_images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void build_database(std::vector<std::string> &name,float  *fea){
	
	//std::cout<<fea.size();
	int d=512;   //设定维度
	int ncentroids=200;  //
	int nq=name.size();
	float *xb=new float[d*name.size()];
	float *xq = new float[d * nq];
	/* faiss::IndexFlatL2 coarse_quantizer (d); */
    // a reasonable number of centroids to index nb vectors
   /*  faiss::IndexIVFPQ index (&coarse_quantizer, d,
                             ncentroids, 4, 8);  */
	/*  faiss::IndexHNSWFlat index_hsnw(fea, 512);
	
	std::cout<<
	faiss::IndexIDMap index(&index_hsnw);
	index.add_with_ids(5, fea, name); */
	
	faiss::IndexFlatL2 index(d);           // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    std::cout<<"build database:"<<nq<<std::endl;
	//FILE *fr;
    //fr=fopen("../save.faissindex","wb");
   //faiss::write_index(&index, fr);
	index.add(name.size(),fea);  //添加索引向量
    //faiss::read_index("../save.txt");
	faiss::write_index(&index, "../save_new.faissindex");
	printf("ntotal = %ld\n", index.ntotal);
	 faiss::Index *index_new = faiss::read_index("/home/project/Deepstream_Project/Deepstream_Face/arcface/save_new.faissindex") ;
	int k=1;
	       // search xq
        long *I = new long[k * nq];     //？
        float *D = new float[k * nq];     //？
        index.search(nq, fea, k, D, I); 
        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }
		delete [] I;
		delete [] D;

	//FILE *fr;
	//fr=fopen(output_file,"wb");
	//faiss::write_index(&index, fr); 
}

void ArcFace::EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                              const std::vector<int64_t> &bufferSize, cudaStream_t stream) {
   
    int index = 0;
    int batch_id = 0;
	
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    cv::Mat face_feature(image_list.size(), outSize, CV_32FC1);
	
	std::vector<float>fea;
    std::vector<std::vector<float>> Data;
	std::vector<std::string> id_name;
    float total_time = 0;
	std::map<std::string, cv::Mat> write_txt;
    std::ofstream wout("face_writer.txt");
    if (!wout.is_open()) {
      std::cout << "File is open fail!" << std::endl;
    }
   // std::cout<<"size:"<<image_list.size()<<std::endl;
	//faiss::IndexFlatL2 coarse_quantizer (512);
	//std::fstream fout("/home/project/Deepstream_Project/Deepstream_Face/arcface/saveface.txt");
    for (const std::string &image_name : image_list)
    {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data)
        {
            vec_Mat[batch_id] = src_img.clone();
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size())
        {
            auto t_start_pre = std::chrono::high_resolution_clock::now();
           // std::cout << "prepareImage" << std::endl;
            std::vector<float>curInput = prepareImage(vec_Mat);
            auto t_end_pre = std::chrono::high_resolution_clock::now();
            float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
            std::cout << "prepare image take: " << total_pre << " ms." << std::endl;
            total_time += total_pre;
            batch_id = 0;
            if (!curInput.data()) {
                std::cout << "prepare images ERROR!" << std::endl;
                continue;
            }
            // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
            std::cout << "host2device" << std::endl;
            cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);
             std::cout<<"读取次数："<<index<<std::endl;
            // do inference
            std::cout << "execute" << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(BATCH_SIZE, buffers);
            auto t_end = std::chrono::high_resolution_clock::now();
            float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            std::cout << "Inference take: " << total_inf << " ms." << std::endl;
            total_time += total_inf;
            std::cout << "execute success" << std::endl;
          //  std::cout << "device2host" << std::endl;
          //  std::cout << "post process" << std::endl;
            auto r_start = std::chrono::high_resolution_clock::now();
            float out[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int rowSize = index % BATCH_SIZE == 0 ? BATCH_SIZE : index % BATCH_SIZE;
            cv::Mat feature(rowSize, outSize, CV_32FC1);
			//std::cout<<"featur:"<<out<<std::endl;
			std::cout<<" before rowSize:"<<rowSize<<std::endl;
			std::cout<<"owwwwSize:"<<outSize<<std::endl;
            ReshapeandNormalize(out, feature, rowSize, outSize);
			std::cout<<" before2 rowSize:"<<feature.rows<<std::endl;
			std::cout<<" before2 rowSize:"<<feature.cols<<std::endl;
            feature.copyTo(face_feature.rowRange(index - rowSize, index));
			std::cout<<face_feature<<std::endl;
		    //std::cout<<" after 行:"<<face_feature.rows<<std::endl;
			//std::cout<<" after 列:"<<face_feature.cols<<std::endl;
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            //std::cout << "Post process take: " << total_res << " ms." << std::endl;
         
            total_time += total_res;
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
			id_name.emplace_back(image_name);
			//fea.emplace_back(feature[0]);
			//std::cout<<fea<<std::endl;
            //unsigned char *array=new unsigned char[face_feature.rows*face_feature.cols];
		   /*  if (feature.isContinuous())
                  array =feature(); */
			/* for(int j=0;j<feature.cols;j++)
				  fea.emplace_back(feature.at<float>(0,j));
			float result_sim= cosine_similarity( fea,  fea);   
            std::cout<<" consine sim:"<<result_sim<<std::endl;   */  
           // memcpy(fea,feature.data,sizeof(float)*4);
            //fea.push_back (feature.at<uchar>(0));
			//fea.emplace_back(array);
			/* for(auto i:fea)
				std::cout<<i<<std::endl; */
			std::cout<<"feature cols:"<<face_feature.cols<<std::endl;
			std::cout<<"feature rows:"<<face_feature.rows<<std::endl;
			write_txt.insert(make_pair(image_name, face_feature));
			// out<<make_pair(image_name, feature)<<"\n";
            //out<<image_name,fea)<<"\n";
		   // std::cout<<"xxxx:"<write_txt[image_name]<<"\n";
		    //out<<
			//std::cout<<"feasize:"<<fea.size()<<std::endl;
			//Data.emplace_back(fea);
			//fea.clear();
			//fout<<image_name<<"\n";
			//fout<<feature<<"\n";
        }
    }
	//std::cout<<"size:"<<index<<std::endl;
	/* for(int i =0;i<512;i++)
		std::cout<<Data[0][i]<<std::endl; */
	/* float *xb= new float[512*image_list.size()];
	for(int i=0;i<image_list.size();i++)
		for(int j=0;j<512;j++){       
         //std::cout<<"s";
		  xb[512 * i + j]=Data[i][j];
	}
	//for(auto i:id_name )
	//std::cout<<"nameL:"<<i<<std::endl;
    // std::cout<<"id name size:"<<id_name.size()<<std::endl;
    build_database(id_name,xb); */
	std::map<std::string ,cv::Mat >::iterator iter = write_txt.begin();
   /*  while (iter != write_txt.end()){
		std::cout <<"xxx:"<< iter->first << ":" << write_txt[iter->first]<<std::endl; 
		iter++; //保存
    }   */
     
	for(;iter!=write_txt.end();iter++){
			wout<<iter->first<<":"<<iter->second<<std::endl; 	
	}

     
    // iter = write_txt.begin();
    
	wout.close(); 
	//从文本中读出
/* 	std::ifstream ins("face_writer.txt");
std::map<std::string ,std::vector<float> value> your_map;
while(!ins.eof()){
std::string key;
std::vector<float> value;
ins>>key>>value;
your_map.insert(make_pair(key,value));
}
for(auto  itr=your_map.begin();itr!=your_map.end();itr++)
     std::cout<<"The"<<itr->first<<"th word is"<<itr->second<<std::endl; */

		  //std::cout<<i()<<std::endl;
        	//fout<<i<<"\n";
	//fout.close();
	
	

   
	
	std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    cv::Mat similarity = face_feature * face_feature.t();
    std::cout << "The similarity matrix of the image folder is:\n" << (similarity + 1) / 2 << "!" << std::endl;
}

std::vector<float> ArcFace::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);

        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        for (int i = 0; i < INPUT_CHANNEL; ++i)
        {
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

void ArcFace::ReshapeandNormalize(float *out, cv::Mat &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        cv::normalize(onefeature, onefeature);
        onefeature.copyTo(feature.row(i));
    }
}
