#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <cmath>
#include "arcface.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"
#include "json.hpp"

using json = nlohmann::json;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0 // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 112;
static const int INPUT_W = 112;
static const int OUTPUT_SIZE = 128;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
	float *gamma = (float*)weightMap[lname + "_gamma"].values;
	int len = weightMap[lname + "_gamma"].count;

	float *scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	float *scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval_1[i] = -1.0;
		scval_2[i] = -gamma[i];
	}
	Weights scale_1{ DataType::kFLOAT, scval_1, len };
	Weights scale_2{ DataType::kFLOAT, scval_2, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = 0.0;
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	auto relu1 = network->addActivation(input, ActivationType::kRELU);
	assert(relu1);
	IScaleLayer* scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
	assert(scale1);
	auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
	assert(relu2);
	IScaleLayer* scale2 = network->addScale(*relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
	assert(scale2);
	IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
	assert(ew1);
	return ew1;
}

ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k = 3, int p = 1, int s = 2, int groups=1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(groups);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_batchnorm", 1e-3);
    assert(bn1);
    auto act1 = addPRelu(network, weightMap, *bn1->getOutput(0), lname + "_relu");
    assert(act1);
    return act1;
}

ILayer* conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int k = 3, int p = 1, int s = 1, int groups=1) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(groups);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_batchnorm", 1e-3);
    assert(bn1);
    return bn1;
}

ILayer* DepthWise(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, groups, DimsHW{1, 1}, weightMap[lname + "_conv_sep_conv2d_weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{0, 0});
    conv1->setNbGroups(1);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_conv_sep_batchnorm", 1e-3);
    assert(bn1);
    auto act1 = addPRelu(network, weightMap, *bn1->getOutput(0), lname + "_conv_sep_relu");
    assert(act1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*act1->getOutput(0), groups, DimsHW{3, 3}, weightMap[lname + "_conv_dw_conv2d_weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(groups);
    auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "_conv_dw_batchnorm", 1e-3);
    assert(bn2);
    auto act2 = addPRelu(network, weightMap, *bn2->getOutput(0), lname + "_conv_dw_relu");
    assert(act2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*act2->getOutput(0), oup, DimsHW{1, 1}, weightMap[lname + "_conv_proj_conv2d_weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{0, 0});
    conv3->setNbGroups(1);
    auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "_conv_proj_batchnorm", 1e-3);
    assert(bn3);
    return bn3;
}


ILayer* DWResidual(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups, int s) {

    auto dw1 = DepthWise(network, weightMap, input, lname, inp, oup, groups, s);
    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *dw1->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    return ew1;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("/home/project/Deepstream_Project/Deepstream_Face/tensorrtx-arcfacemobilenet-master/arcface-mobilefacenet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto conv_1 = conv_bn_relu(network, weightMap, *data, "conv_1", 64, 3, 1, 2);
    auto conv_2_dw = conv_bn_relu(network, weightMap, *conv_1->getOutput(0), "conv_2_dw", 64, 3, 1, 1, 64);
    auto conv_23 = DepthWise(network, weightMap, *conv_2_dw->getOutput(0), "dconv_23", 64, 64, 128, 2);
    auto res_3_block0 = DWResidual(network, weightMap, *conv_23->getOutput(0), "res_3_block0", 64, 64, 128, 1);
    auto res_3_block1 = DWResidual(network, weightMap, *res_3_block0->getOutput(0), "res_3_block1", 64, 64, 128, 1);
    auto res_3_block2 = DWResidual(network, weightMap, *res_3_block1->getOutput(0), "res_3_block2", 64, 64, 128, 1);
    auto res_3_block3 = DWResidual(network, weightMap, *res_3_block2->getOutput(0), "res_3_block3", 64, 64, 128, 1);
    auto conv_34 = DepthWise(network, weightMap, *res_3_block3->getOutput(0), "dconv_34", 64, 128, 256, 2);
    auto res_4_block0 = DWResidual(network, weightMap, *conv_34->getOutput(0), "res_4_block0", 128, 128, 256, 1);
    auto res_4_block1 = DWResidual(network, weightMap, *res_4_block0->getOutput(0), "res_4_block1", 128, 128, 256, 1);
    auto res_4_block2 = DWResidual(network, weightMap, *res_4_block1->getOutput(0), "res_4_block2", 128, 128, 256, 1);
    auto res_4_block3 = DWResidual(network, weightMap, *res_4_block2->getOutput(0), "res_4_block3", 128, 128, 256, 1);
    auto res_4_block4 = DWResidual(network, weightMap, *res_4_block3->getOutput(0), "res_4_block4", 128, 128, 256, 1);
    auto res_4_block5 = DWResidual(network, weightMap, *res_4_block4->getOutput(0), "res_4_block5", 128, 128, 256, 1);
    auto conv_45 = DepthWise(network, weightMap, *res_4_block5->getOutput(0), "dconv_45", 128, 128, 512, 2);
    auto res_5_block0 = DWResidual(network, weightMap, *conv_45->getOutput(0), "res_5_block0", 128, 128, 256, 1);
    auto res_5_block1 = DWResidual(network, weightMap, *res_5_block0->getOutput(0), "res_5_block1", 128, 128, 256, 1);
    auto conv_6_sep = conv_bn_relu(network, weightMap, *res_5_block1->getOutput(0), "conv_6sep", 512, 1, 0, 1);
    auto conv_6dw7_7 = conv_bn(network, weightMap, *conv_6_sep->getOutput(0), "conv_6dw7_7", 512, 7, 0, 1, 512);
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*conv_6dw7_7->getOutput(0), 128, weightMap["fc1_weight"], weightMap["pre_fc1_bias"]);
    assert(fc1);
    auto bn1 = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);
    assert(bn1);
    bn1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*bn1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


std::vector<std::string> GetListFolders(const std::string& path, const std::string& exten = "*")
{
    std::vector<std::string> list;
    list.clear();

    DIR* dp = nullptr;
    struct dirent* dirp = nullptr;
    if ((dp = opendir(path.c_str())) == nullptr) {
        return list;
    }

    while ((dirp = readdir(dp)) != nullptr) {
        if (dirp->d_type == DT_DIR && strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0) {
            if (exten.compare("*") == 0)
                list.emplace_back(static_cast<std::string>(dirp->d_name));
            else
                if (std::string(dirp->d_name).find(exten) != std::string::npos)
                    list.emplace_back(static_cast<std::string>(dirp->d_name));
        }
    }

    closedir(dp);

    return list;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
	bool Create_Embeds=false;
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("arcface-mobilefacenet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("arcface-mobilefacenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } 
	else if (argc == 3 and std::string(argv[1])=="-c" )
	{   
        std::cout<<"选择创建数据集"<<std::endl;
	    Create_Embeds=true;
        std::ifstream file("arcface-mobilefacenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }	
	}
	else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./arcface-mobilefacenet -s  // serialize model to plan file" << std::endl;	
        std::cerr << "./arcface-mobilefacenet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
	
    delete[] trtModelStream;
    int batch_id = 0;
    //static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //static float prob[BATCH_SIZE * OUTPUT_SIZE];
    //cv::Mat img = cv::imread(imagefile);
	if (Create_Embeds==true)
	{   
        std::cout<<"[INFO]开始进行特征库创建"<<std::endl;
		
		nlohmann::json js;	
		json array = json::array();
		nlohmann::json jss;	
		json array2 = json::array();
		float sum=0;
		float p2[128];
		for(auto i:p2)
			i=0;
		//float *tensor_out=new float[image_list.size()*128];
       // std::vector<std::string> image_list = readFolder("/home/project/code/face_recognition-pytorch/data_DB/捷克-教练-希尔哈维");
		std::string file_path=std::string(argv[2]);
		std::cout<<"确认加载的目录："<<file_path<<std::endl;
		std::vector<std::string>filePaths=GetListFolders(file_path);
		
		//cv::glob(file_path, filePaths,false);
		
        for( auto &names:filePaths){
			std::cout<<"名字:"<<names<<std::endl;
			std::vector<std::string> image_list = readFolder(file_path+"/"+names);
			cv::Mat Sum_mat=cv::Mat::zeros(1,128,CV_32FC1);
			 int  count=0;
		for(const std::string &image_name : image_list){
	       
			float *out=new float[128];
			std::cout << "Processing: " << image_name << std::endl;
			cv::Mat img = cv::imread(image_name);
			cv::resize(img, img, cv::Size(112,112));
			if (img.data)
			{
				//vec_Mat[batch_id] = img.clone();
				
				batch_id++;
			}
			else  return -1;
	  
			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
				data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
				data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
				data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
			}
			
			std::cout<<"inference..."<<std::endl;
			doInference(*context, data, prob, BATCH_SIZE);
			std::cout<<"inference over..."<<std::endl;
			std::cout<<sizeof(prob)/sizeof(float)<<std::endl;
		
			cv::Mat feature(1, 128, CV_32FC1,prob);
			
			cv::Mat out_norm;
			
		    float embed[128];
		   
			cv::normalize(feature, out_norm);
			Sum_mat+=feature;
			//std::cout<<"calculate norm tensor"<<out_norm<<std::endl;
			memcpy(embed,out_norm.data,sizeof(float)*128);
			//memcpy(out_norm.data,p2,sizeof(out_norm.data));
			array=embed;
			//std::cout<<"tensor"<<feature<<std::endl;

			std::string name_id=image_name.substr(0,image_name.size()-4);
			std::cout<<"图片名字:"<<name_id<<std::endl;
			//js[name_id]=array;
			//std::cout << js.dump(2) << std::endl;
			
			js[name_id]=array;
			std::ofstream os("face.json");
			os<<js.dump(1)<<std::endl;
			count++;
		    //std::cout<<"getbase_data"<<std::endl;
		//empty_array_explicit(out_norm);
		//std::copy(prob,prob+128,tensor_out);
      }
	    std::cout<<"开始计算特征平均值，数量："<<count<<std::endl;
		Sum_mat/=count;
		
		cv::normalize(Sum_mat, Sum_mat);
		
		//unsigned char *ay=new unsigned char[128];
		//std::vector<uchar>sb;
		//sb=Sum_mat.data;
		//float save_mean[128];
		/* ay=Sum_mat.data;
		if (Sum_mat.isContinuous())
		{
			std::cout<<"success"<<std::endl;
			
		} */
		float coco[128];
        memcpy(coco,Sum_mat.data,sizeof(float)*128);
		/* for(auto i:p2)
			i=i/image_list.size(); */
	    std::cout<<"check names!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<names<<std::endl;
		jss[names]=coco;
		std::ofstream os3("mean_face.json");
		os3<<jss.dump(1)<<std::endl;
		std::cout<<"getbase_data:"<<jss.dump(1)<<std::endl;
	}
		
    
	return 1;
	}
	
	//Inference_file(*context,"/home/project/球赛人员头像/捷克-教练-希尔哈维");
	
	cv::Mat img = cv::imread("/home/project/球赛人员头像/裁判-比约恩·凯珀斯/2.jpg");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(128, 1, CV_32FC1, prob);
    cv::Mat out_norm;
    cv::normalize(out, out_norm);
    //std::vector<std::string> &image_list,
	 std::cout<<"1111111:"<<out_norm<<std::endl;
    img = cv::imread("/home/project/球赛人员头像/裁判-比约恩·凯珀斯/1.jpg");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
     
	 
	 
    // Run inference
    start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out1(1, 128, CV_32FC1, prob);
    cv::Mat out_norm1;
    cv::normalize(out1, out_norm1);
    std::cout<<"22222222222:"<<out_norm1<<std::endl;
    cv::Mat res = out_norm1 * out_norm;
   // wout<<out_norm1<<std::endl;
	//wout<<out_norm.t()<<std::endl;
    std::cout << "similarity score: " << *(float*)res.data << std::endl;
    //wout.close();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    //Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << p_out_norm[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}



