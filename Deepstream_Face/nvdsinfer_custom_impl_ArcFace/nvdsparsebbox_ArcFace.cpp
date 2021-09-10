/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* This custom post processing parser is for centernet face detection model */
#include <cstring>
#include <iostream>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <tuple>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include "cblas.h"
#include <cublasLt.h>
#define IMAGE_HEIGHT 112
#define IMAGE_WIDTH  112
#define INPUT_CHANNEL 3
#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include "nvmsgbroker.h"
#define SO_PATH "/opt/nvidia/deepstream/deepstream/lib/"
#include<thread>
#define KAFKA_KEY "kafka"
#define KAFKA_PROTO_SO "libnvds_kafka_proto.so"
#define KAFKA_PROTO_PATH SO_PATH KAFKA_PROTO_SO
#define KAFKA_CFG_FILE "./cfg_kafka.txt"
#define KAFKA_CONN_STR "localhost;9092" // broker;port
#define  KAFKA_TPOIC  "qucikstart-event"
int test_id=0;
//using faiss::Index;
#include <jsoncpp/json/json.h>
struct recogn{
	char *flag ="people_face";
	char *descript="description_null";
	char *mes="unkown";
    
	
	
};
struct test_info
{
    int test_id;
    char *test_key;
    char *proto_key;
    char *proto_path;
    char *cfg_file;
    char *conn_str;
    int cb_count;
    int consumed_count;

    test_info() {}
    test_info(int id, const char* test_key, const char* proto_key, const char* proto_path, const char* cfg_file, const char* conn_str)
    {
        load_info(id, test_key, proto_key, proto_path, cfg_file, conn_str);
    }

    void load_info(int id, const char* test_key, const char* proto_key, const char* proto_path, const char* cfg_file, const char* conn_str)
    {
        this->test_id = id;
        this->test_key = strdup(test_key);
        this->proto_key = strdup(proto_key);
        this->proto_path = strdup(proto_path);
        this->cfg_file = (cfg_file != nullptr) ? strdup(cfg_file) : nullptr;
        this->conn_str = (conn_str != nullptr) ? strdup(conn_str) : nullptr;
        this->cb_count = 0;
        this->consumed_count = 0;
    }

    ~test_info()
    {
        free(test_key);
        free(proto_key);
        free(proto_path);
        if (cfg_file != nullptr)
            free(cfg_file);
        if (conn_str != nullptr)
            free(conn_str);
    }
};

std::vector<long double> split(const std::string& str, const std::string& delim)
{
    std::vector<long double> res;
    if("" == str) return res;
    char * strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p){
        long double f = std::stold(p);
        res.push_back(f);
        p = strtok(NULL, d);
    }
    return res;
}

std::string to_string_with_high_precision(double value, int precision = 20)
{
    std::stringstream ss;
    ss.precision(precision);

    ss << value;
    return ss.str();
}
// Global info map for tests
// The entry for each test holds information for launching
// the test as well as keeping track of callback counts etc.
std::map<char*, test_info> g_info_map;

void test_connect_cb(NvMsgBrokerClientHandle h_ptr, NvMsgBrokerErrorType status)
{
    if (status == NV_MSGBROKER_API_OK)
        printf("Connect succeeded\n");
    else
        printf("Connect failed\n");
}

void test_send_cb(void *user_ptr,  NvMsgBrokerErrorType flag)
{
    int count = -1;
    int id = -1;
    char *key = (char*)user_ptr;
    std::map<char*, test_info>::iterator iter = g_info_map.find(key);
    if (iter != g_info_map.end()) {
        test_info &ti = iter->second;
        count = ++ti.cb_count;
        id = ti.test_id;
    } else {
        printf("test_send_cb: Failed to find test info for %s\n", key);
    }

    if (flag == NV_MSGBROKER_API_OK)
        printf("Test %d: async send[%d] succeeded for %s\n", id, count, key);
    else
        printf("Test %d: async send[%d] failed for %s\n", id, count, key);
}

void test_subscribe_cb(NvMsgBrokerErrorType flag, void *msg, int msglen, char *topic, void *user_ptr)
{
    int count = -1;
    int id = -1;
    char *key = (char*)user_ptr;
    std::map<char*, test_info>::iterator iter = g_info_map.find(key);
    if (iter != g_info_map.end()) {
        test_info &ti = iter->second;
        count = ++ti.consumed_count;
        id = ti.test_id;
    } else {
        printf("subscribe_cb: Failed to find test info for %s\n", key);
    }

    if(flag == NV_MSGBROKER_API_ERR) {
        printf("Test %d: Error in consuming message[%d] from broker\n", id, count);
    }
    else {
        printf("Test %d: Consuming message[%d], on topic[%s]. Payload= %.*s\n", id, count, topic, msglen, (const char *) msg);
    }
}

int run_test(char* test_key,recogn pop)
{
    std::map<char*, test_info>::iterator iter = g_info_map.find(test_key);
    if (iter == g_info_map.end()) {
        printf("Failed to find test info for %s\n", test_key);
        return -1;
    }
    test_info &ti = iter->second;
	std::cout<<"1:"<<ti.proto_path<<std::endl;
	std::cout<<"2:"<<ti.conn_str<<std::endl;
	std::cout<<"3:"<<ti.cfg_file<<std::endl;
    NvMsgBrokerClientHandle conn_handle;
	Json::Value json_temp;
	json_temp["face"] = Json::Value(pop.mes);
	//json_temp["age"] = Json::Value(18);
	Json::Value root; 
	root["object name "] = json_temp; 
	root["object des"].append(pop.descript);
	root["object des"].append(pop.flag);
	//root["key_array"].append(1234); 
   
    std::string out = root.toStyledString();
	//root.clear()
	const char*SEND_MSG=( out.c_str());
    std::cout << "message "<<SEND_MSG << std::endl;
	 
    conn_handle = nv_msgbroker_connect(ti.conn_str, ti.proto_path, test_connect_cb, ti.cfg_file);
    if (!conn_handle) {
        printf("Test %d: Connect failed for %s [%s:%s].\n", ti.test_id, ti.conn_str, ti.test_key, ti.proto_key);
        return -1;
    }
    std::cout<<"状态xxxxxxxxxxxxxxxxxxxxxxxx"<<conn_handle<<std::endl;
    //Subscribe to topics
    const char *topics[] = {KAFKA_TPOIC};
    int num_topics=1;
    NvMsgBrokerErrorType ret = nv_msgbroker_subscribe(conn_handle, (char **)topics, num_topics, test_subscribe_cb, ti.test_key);
    switch (ret) {
        case NV_MSGBROKER_API_ERR:
            printf("Test %d: Subscription to topic[s] failed for %s(%s)\n", ti.test_id, ti.test_key, ti.proto_key);
            return -1;
        case NV_MSGBROKER_API_NOT_SUPPORTED:
            printf("Test %d: Subscription not supported for %s(%s). Skipping subscription.\n", ti.test_id, ti.test_key, ti.proto_key);
            break;
    }

    NvMsgBrokerClientMsg msg;
    msg.topic = strdup(KAFKA_TPOIC);
    msg.payload = const_cast<char *>(SEND_MSG);
    msg.payload_len = strlen(SEND_MSG);
    for(int i = 0; i < 1; i++) {
      if (nv_msgbroker_send_async(conn_handle, msg, test_send_cb, ti.test_key) != NV_MSGBROKER_API_OK)
	    printf("Test %d: send [%d] failed for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
      else{
	    printf("Test %d: sending [%d] asynchronously for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
	 // usleep(100);  
	  }  //10ms sleep
    }
    free(msg.topic);
    //printf("Test %d: Disconnecting... in 3 secs\n", ti.test_id);
    //sleep(3);
    nv_msgbroker_disconnect(conn_handle);
}

std::vector<std::string>readFolder(const std::string &image_path)
{
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.push_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}



/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomTfSSD( std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
       NvDsInferNetworkInfo const& networkInfo,
       float classifierThreshold,
       std::vector<NvDsInferAttribute> &attrList,
       std::string &descString);

/* This is a smaple bbox parsing function for the centernet face detection onnx model*/
struct ArcFace 
{
	int inputHeight;
	int inputWidth;
	//int INPUT_CHANNEL;
	float visualizeThreshold;
	int postNmsTopN;
	int outputBboxSize;
    //float face_data[512];
	std::vector<float> classifierRegressorStd;
};

void ReshapeandNormalize(float *out, cv::Mat &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {   
       // std::cout<< "reshape"<<std::endl;
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        cv::normalize(onefeature, onefeature);
        onefeature.copyTo(feature.row(i));
    }
}

std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(112 * 112 * 3);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(112, 112));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(3);
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
/* void CenterFace::NmsDetect(std::vector<FaceRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const FaceRes &left, const FaceRes &right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
            if (iou > nms_threshold)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes &det)
    { return det.confidence == 0; }), detections.end());
} */


recogn faiss_search(cv::Mat &query){
	recogn pop;
	int vs_flag=1;
	int maxsize=query.cols;
	float *xq= new float[maxsize];
	for(int j=0;j<maxsize;j++)
		  xq[j]=query.at<float>(0,j);		
    int k=5; 
    faiss::Index *index_new = faiss::read_index("/home/project/Deepstream_Project/Deepstream_Face/save_new.faissindex") ;
/* 	for(int i=0;i<maxsize;i++)
	std::cout<<xq[i]<<std::endl; */
	long *I = new long[k * 1];     //
    float *D = new float[k * 1];     //
    index_new->search(1, xq, k, D, I); 
	/* for(auto i:I)
		std::cout<<i<<"\n"; */
	/* for(int i=0;i<1;i++)
		std::cout<<i<<"\n" */;
	  //printf("D=\n");
        for(int i = 0; i < 1; i++) {
            for(int j = 0; j < k; j++){
              std::cout<<"当前ID和检测库中的最近"<<k<<"个距离为"<< D[i * k + j]<<std::endl;
			 // pop.descript="当前Face_ID和检测库中的最近的距离为"+reinterpret_cast<char*>(D)[i * k + j]+"->";
		      if(D[i * k + j]>0.2){
			    std::cout<<"当前特征比对相似度不满足阈值,未找到该身份ID"<<std::endl;
				pop.descript="->当前特征比对相似度不满足阈值,未找到该身份ID";
		         vs_flag=0;
			  }
			}
        }
	//printf("I (1 first results)=\n");
	if(vs_flag==1){
    for(int i = 0; i < 1; i++) {
            for(int j = 0; j < k; j++){
				 pop.descript="找到当前ID和检测库中最相似ID";
                std::cout<<"当前ID和检测库中最像的ID："<< I[i * k + j]<<std::endl;
			     pop.mes= reinterpret_cast<char*> (I[i * k + j]);
                 //return pop;
			}
           // printf("\n");
        }
	}
		delete [] I;
		delete [] D;
	
	return pop;
	
   // printf("is_trained = %s\n", index.is_trained ? "true" : "false");
  /*  •Index *index_cpu_to_gpu = index_cpu_to_gpu(resource, dev_no, index): 复制索引到GPU
•Index *index_gpu_to_cpu = index_gpu_to_cpu(index):从GPU到CPU
•index_cpu_to_gpu_multiple: uses an IndexShards or IndexProxy to copy the index to several GPUs. */
	
}
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

int  get_embedding(std::string file,std::map<std::string, std::vector<long double>> &map_txt,std::vector<float>out){
	
	//std::vector<float> res;
     std::ifstream ous(file);
    if (!ous) 
		return -1;
    std::string str_line = "";
    std::string str_key = "";
    std::string str_num = "";
    while(getline(ous,str_line)){
        std::string::size_type middle_pos = 0;
        std::string::size_type left_pos = 0;
        std::string::size_type right_pos = 0;
        if (str_line.npos != (middle_pos = str_line.find(":")))
        {
            str_key = str_line.substr(0, middle_pos);

        }
        if ((str_line.npos != (left_pos = str_line.find("["))) && (str_line.npos != (right_pos = str_line.find("]"))))
        {
            str_num = str_line.substr(left_pos + 1, right_pos-left_pos-1);
            //str_num = ClearAllSpace(str_num);
            std::vector<long double> vec_txt = split(str_num, ",");
            map_txt.insert(make_pair(str_key, vec_txt));

        }
    }
    ous.close();
    ous.clear();

    auto it = map_txt.end(); 
	it--;
    for (auto it2 = (it->second).begin(); it2 != (it->second).end(); it2++)
    {
        std::cout.setf(std::ios::left);
        std::cout.width(15);
        out.emplace_back(float(*it2));
        std::cout << *it2 << std::endl;
    } 
	 std::map<std::string, std::vector<long double>> ::iterator it3;
	// for(auto  itr= map_txt.begin();itr!= map_txt.end();itr++){
    //     std::cout<<"xxxxxxxxxxxx:"<<itr->first<<std::endl;
    // }
	//  for(auto it3= map_txt.begin(); it3!= map_txt.end(); it3++){
    //     std::cout<<"Key:"<<it3->first<<std::endl;
	// 	for (auto it2 = it3->second.begin(); it2 != it3->second.end(); it2++)
	// 	std::cout << "small key= " << it2->first << " value= " << it2->second << std::endl; 
    //  }
	return 1;
}
static std::vector < std::vector< std:: string > > labels { {
    "coupe1", "largevehicle1", "sedan1", "suv1", "truck1", "van1"} };
/* customcenternetface */
/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
extern "C" bool NvDsInferParseCustomArcFace( std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
       std::vector<NvDsInferAttribute> &attrList,
       std::string &descString)
{   
	 auto layerFinder = [&outputLayersInfo](const std::string &name)
		-> const NvDsInferLayerInfo * {
		for (auto &layer : outputLayersInfo)
		{

			if (layer.dataType == FLOAT &&
				(layer.layerName && name == layer.layerName))
			{
				return &layer;
			}
		}
		return nullptr;
	}; 
	int outputBBoxLayerIndex = -1;
	/* for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strstr(outputLayersInfo[i].layerName, "fc1") != nullptr) {
            outputBBoxLayerIndex = i;
        }
       
    }
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;
		
	NvDsInferDimsCHW outputBBoxDims;
	//NvDsInferDimsNCHW  ouuttttt;
	getDimsCHWFromDims(outputBBoxDims,
                       outputLayersInfo[outputBBoxLayerIndex].dims); */
	/* std::vector<float>curInput = prepareImage(vec_Mat); //接数据库的图像
	unsigned int numAttributes = outputLayersInfo.size();
	objectList.clear(); */
	//std::cout<<"xx"<<  *outputBboxBuffer  <<std::endl;
	const NvDsInferLayerInfo *face_tensor = layerFinder("fc1");
	
	//    std::cout<<"width"<<&networkInfo.width<<std::endl;
    /*  for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
        if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
          cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
              info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
      } */
	float *outbuffer =(float *)face_tensor[0].buffer; 
    //  std::cout<<face_tensor<<std::endl;
	int tensor_512 = face_tensor->inferDims.d[0];
	
	
	//; //#heatmap.size[2];
	//int fea_w = face_tensor->inferDims.d[2];//;//heatmap.size[3];
	float (*array)[512] = (float(*)[512]) face_tensor->buffer;
	// cv::Mat xout( 512, 1, CV_32FC1, outbuffer);
	  //cv::Mat out_norm;
	   //cv::normalize(xout, out_norm); 
	   
	//std::cout<<"fea:"<<*array<<std::endl;
    /*  for(int m =0;m<face_tensor->inferDims.numElements;m++){
          std::cout<<" "<< (*array)[m];
          } */ 
		  
	int rowSize = 1;
		  
    cv::Mat feature(rowSize, 512,CV_32FC1); 
    ReshapeandNormalize(*array, feature, rowSize, 512);
	 //std::cout<<"result:"<<feature<<std::endl; 
	 //std::cout<<"rows:"<<feature.rows<<std::endl;
	// std::cout<<"cols:"<<feature.cols<<std::endl;
	std::vector<float> sim_fea1;
    std::vector<float> sim_fea2;
	recogn pop=faiss_search(feature);
	std::map<std::string, std::vector<long double>> res;
    get_embedding("/home/project/Deepstream_Project/Deepstream_Face/arcface/build/face_writer.txt", res,sim_fea1);

    for(int j=0;j<feature.cols;j++)
				  sim_fea2.emplace_back(feature.at<float>(0,j));
    // float *xb= new float[512*30];
	// for(int i=0;i<30;i++)
	// 	for(int j=0;j<512;j++){       
    //      //std::cout<<"s";
	// 	  //xb[512 * i + j]=Data[i][j];
    //       sim_fea2.emplace_back();
	// }
    auto it = res.end(); 
	it--;
    for (auto it2 = 0; it2 < sim_fea1.size(); it2++)
    {
        std::cout << "it2 sizexxxxxxxxxxxxxxx:"<<sim_fea1[it2] << std::endl;
         std::cout << "faa2 sizexxxxxxxxxxxxxxxxxxx:"<<sim_fea2.size() << std::endl;
       // sim_fea1.emplace_back(float(*it2));   
       // float gs=cosine_similarity(sim_fea1[0],sim_fea2);
        sim_fea2.clear();
        //sim_fea1.clear();
        //sim_fea1.emplace_back(float(*it2));
        //std::cout<<"验证相似度！！！！！！！！！！："<<gs<<std::endl;
        // if(gs>0.9)
        //   std::cout<<"BINGO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!！！！！！！！！！！："<<std::endl;
    } 
    


	NvDsInferAttribute attr;
	std::string text="人物身份";
	//attr.attributeLabel="人物1";
	
   // test_info kafka_test(test_id++, KAFKA_KEY, KAFKA_KEY, KAFKA_PROTO_PATH, KAFKA_CFG_FILE, KAFKA_CONN_STR);
  
		
	//std::cout<<"descStringtring:"<<descString<<std::endl;	
    // To disable a test, comment out the line below that adds it to g_info_map.
  //  g_info_map[kafka_test.test_key] = kafka_test;
    
    //float probability =1.0 ;     
    float maxProbability = -254;
    bool attrFound = true;
    attr.attributeIndex = 1;
    attr.attributeValue = 0;
    attr.attributeConfidence = 1.0;
      if (attrFound)
        {
            if (labels.size() > attr.attributeIndex &&
                    attr.attributeValue < labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                descString.append(attr.attributeLabel).append(" ");
        }
    // if (attrFound)
    // {
    //     attr.attributeLabel = const_cast<char *>(text.c_str());
	// 	//attr.attributeLabel = text;
	// 	attrList.push_back(attr);
	// 	//if (attr.attributeLabel)
    //     //descString.append(attr.attributeLabel).append(" ");
	// 	//std::cout<<"descStringtring:"<<descString<<std::endl; 
	// 	std::cout<<"descStringtring2"<<text.c_str()<<std::endl;
    // }
  // 	std::cout<<"descStringtring2"<<text.c_str()<<std::endl;
	return true;
}
extern "C"
bool NvDsInferClassiferParseCustomSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);

extern "C"
bool NvDsInferClassiferParseCustomSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString)
{
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();
    std::cout<<"numberxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:"<<numAttributes<<std::endl;
    /* Iterate through all the output coverage layers of the classifier.
    */
   for (unsigned int l = 0; l < numAttributes; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;
         
        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound)
        {
            if (labels.size() > attr.attributeIndex &&
                    attr.attributeValue < labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel =nullptr;
            attrList.push_back(attr);
			//std::cout<<"atr:"<<attr.attributeLabel<<std::endl;
            if (attr.attributeLabel)
                descString.append(attr.attributeLabel).append(" ");
        }
   }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomSoftmax);
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomArcFace);