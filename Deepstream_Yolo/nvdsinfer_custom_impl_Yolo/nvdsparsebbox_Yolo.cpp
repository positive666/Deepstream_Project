/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <dlfcn.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"
#include <map>
#include <string>
#include <vector>
#include "nvmsgbroker.h"
#include<thread>
#include "rdkafkacpp.h"
#include <csignal>
#define SO_PATH "/opt/nvidia/deepstream/deepstream/lib/"

/* #define KAFKA_KEY "kafka"
#define KAFKA_PROTO_SO "libnvds_kafka_proto.so"
#define KAFKA_PROTO_PATH SO_PATH KAFKA_PROTO_SO
#define KAFKA_CFG_FILE "./cfg_kafka.txt"
#define KAFKA_CONN_STR "localhost;9092" // broker;port
#define KAFKA_TPOIC "qucikstart-events"
#include <jsoncpp/json/json.h> */
#include <fstream>

static bool run = true;
static void sigterm (int sig) {
    run = false;
}

/* class ExampleDeliveryReportCb : public RdKafka::DeliveryReportCb {
public:
    void dr_cb (RdKafka::Message &message) {
        std::cout << "Message delivery for (" << message.len() << " bytes): " <<
                     message.errstr() << std::endl;
        if (message.key())
            std::cout << "Key: " << *(message.key()) << ";" << std::endl;
    }
}; */

/* class ExampleEventCb : public RdKafka::EventCb {
public:
    void event_cb (RdKafka::Event &event) {
        switch (event.type())
        {
        case RdKafka::Event::EVENT_ERROR:
            std::cerr << "ERROR (" << RdKafka::err2str(event.err()) << "): " <<
                         event.str() << std::endl;
            if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN)
                run = false;
            break;

        case RdKafka::Event::EVENT_STATS:
            std::cerr << "\"STATS\": " << event.str() << std::endl;
            break;

        case RdKafka::Event::EVENT_LOG:
            fprintf(stderr, "LOG-%i-%s: %s\n",
                    event.severity(), event.fac().c_str(), event.str().c_str());
            break;

        default:
            std::cerr << "EVENT " << event.type() <<
                         " (" << RdKafka::err2str(event.err()) << "): " <<
                         event.str() << std::endl;
            break;
        }
    }
}; */

/* void libkafka(std::vector<NvDsInferParseObjectInfo>& objectList)
{  
    Json::Value json_temp;
	int count=0;
	//Json::Value Te;
	Json::Value local;
	//Json::Value event_time;
	//Json::Value E_time;
	Json::Value BBox;
	
    //json_temp["位置"] = Json::Value();
	//std::vector<string> Root;
	Json::Value root; 
	//std::cout<<"obj size:"<<obj.size()<<std::endl;
	for(auto& pop : objectList){
		
		json_temp["广告"]=Json::Value("1");
		local["left"]=(pop.left);
		local["top"]=(pop.top);
		local["width"]=(pop.width);
		local["height"]=(pop.height);
		BBox["EventTime"]="触发时间";
		BBox["type"]=json_temp;
		BBox["box"]=(local);
		BBox["目标id"]=count++;
		root.append(BBox);
	}
  
    std::string out = root.toStyledString();
	//std::cout<<writer.write(root)<<std::endl;
	const char*SEND_MSG=(out.c_str());
	std::string brokers = "localhost:9092";
    std::string errstr;
    std::string topic_str="yolov5";
    int32_t partition = RdKafka::Topic::PARTITION_UA;

    RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    RdKafka::Conf *tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);
  
    conf->set("bootstrap.servers", brokers, errstr);
    std::cout<<"xxxxxxxxxxxxxxxx:"<<errstr<<std::endl;
    ExampleEventCb ex_event_cb;
    conf->set("event_cb", &ex_event_cb, errstr);

    signal(SIGINT, sigterm);
    signal(SIGTERM, sigterm);

    ExampleDeliveryReportCb ex_dr_cb;
    conf->set("dr_cb", &ex_dr_cb, errstr);

    RdKafka::Producer *producer = RdKafka::Producer::create(conf, errstr);
    if (!producer) {
        std::cerr << "Failed to create producer: " << errstr << std::endl;
        exit(1);
    }
    std::cout << "% Created producer " << producer->name() << std::endl;

    RdKafka::Topic *topic = RdKafka::Topic::create(producer, topic_str,
                                                   tconf, errstr);
    if (!topic) {
        std::cerr << "Failed to create topic: " << errstr << std::endl;
        exit(1);
    }

   // for (std::string line; run && std::getline(std::cin, line);) {
        if (objectList.empty()) {
            producer->poll(0);
            //continue;
        }

        RdKafka::ErrorCode resp =
                producer->produce(topic, partition,
                                  RdKafka::Producer::RK_MSG_COPY ,
                                  const_cast<char *>(SEND_MSG), strlen(SEND_MSG),
                                  NULL, NULL);
								  std::cerr << "% Produced message :" <<SEND_MSG<<std::endl;
        if (resp != RdKafka::ERR_NO_ERROR)
            std::cerr << "% Produce failed: " <<
                         RdKafka::err2str(resp) << std::endl;
        else
            std::cerr << "% Produced message " <<
                         std::endl;

        producer->poll(0);
   // }
    
    run = true;
   // 退出前处理完输出队列中的消息
    while (run && producer->outq_len() > 0) {
        std::cerr << "Waiting for " << producer->outq_len() << std::endl;
        producer->poll(100);
    }

    delete conf;
    delete tconf;
    delete topic;
    delete producer;

    RdKafka::wait_destroyed(1000);

	
	
	
	
	
} */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);
extern "C" bool NvDsInferParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloTLT(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);
	
static const int NUM_CLASSES_YOLO = 15;
#define NMS_THRESH 0.4
#define CONF_THRESH 0.35
#define BATCH_SIZE 1

//ADD
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

std::vector<std::string> m_classes;

/* struct test_info
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
    test_info(const test_info& tmp){
		this->test_id = tmp.test_id;
        this->test_key = strdup(tmp.test_key);
        this->proto_key = strdup(tmp.proto_key);
        this->proto_path = strdup(tmp.proto_path);
        this->cfg_file = (tmp.cfg_file != nullptr) ? strdup(tmp.cfg_file) : nullptr;
        this->conn_str = (tmp.conn_str != nullptr) ? strdup(tmp.conn_str) : nullptr;
        this->cb_count = 0;
        this->consumed_count = 0;
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
    {  //if (test_key != nullptr)
        free(test_key);
	//if (proto_key != nullptr)
        free(proto_key);
	//if (proto_path != nullptr)
        free(proto_path);
        if (cfg_file != nullptr)
            free(cfg_file);
        if (conn_str != nullptr)
            free(conn_str);
    }
}; */
 
 
 
/* std::map<char*, test_info> g_info_map;   

void test_connect_cb(NvMsgBrokerClientHandle h_ptr, NvMsgBrokerErrorType status)
{
    if (status == NV_MSGBROKER_API_OK)
        printf("Connect succeeded\n");
    else{
        printf("Connect failed\n");		
	}
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
} */
  
/* int  run_test(char *test_key,std::vector<NvDsInferParseObjectInfo>obj )
{   	
    //加载类别
    
  
	
   // test_info kafka_test(test_id, KAFKA_KEY, KAFKA_KEY, MOATA, KAFKA_CFG_FILE, KAFKA_CONN_STR);
    // To disable a test, comment out the line below that adds it to g_info_map.
    //g_info_map[kafka_test.test_key] = kafka_test;
//	for (auto it = g_info_map.begin(); it != g_info_map.end(); ++it) {
	 
    
    std::cout<<"run test!!!"<<std::endl;
	 //char *test_key=it->first;
  for (auto iter = g_info_map.begin(); iter != g_info_map.end(); ++iter) 
        //printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
		std::cout<<"AFTERRRRRRRRRRRRRRRR:"<<iter->first<<" "<<iter->second.proto_key<<std::endl;
   // std::cout<<"开始处理kafka消息！"<<g_info_map.find(test_key)<<std::endl;
	//std::cout<<""<<std::endl;d
    std::map<char*, test_info>::iterator iter = g_info_map.find(test_key);
	//std::cout<<"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa："<<g_info_map.size()<<std::endl;
    if (iter == g_info_map.end()) {
        printf("Failed to find test info for %s\n", test_key);
        return -1;
    } 
	int count=0;
	//g_info_map.clear();
	test_info &ti=iter->second;
	//g_info_map.erase(iter);
    NvMsgBrokerClientHandle conn_handle;
	std::cout<<"1:"<<g_info_map.find(test_key)->second.proto_path<<std::endl;
	std::cout<<"2:"<<g_info_map.find(test_key)->second.conn_str<<std::endl;
	std::cout<<"3:"<<g_info_map.find(test_key)->second.cfg_file<<std::endl;
	
	conn_handle = nv_msgbroker_connect(ti.conn_str, ti.proto_path, test_connect_cb, ti.cfg_file); 
	std::cout<<"状态"<<conn_handle<<std::endl;
    if (!conn_handle) {
         printf("Test %d: Connect failed for %s [%s:%s].\n", ti.test_id, ti.conn_str, ti.test_key, ti.proto_key);
		// g_info_map.erase(test_key);
         return -1;
    }
    
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
 
   
	Json::Value json_temp;
	//Json::Value Te;
	Json::Value local;
	//Json::Value event_time;
	//Json::Value E_time;
	Json::Value BBox;
	Json::FastWriter writer;
	
    //json_temp["位置"] = Json::Value();
	//std::vector<string> Root;
	Json::Value root; 
	std::cout<<"obj size:"<<obj.size()<<std::endl;
	for(auto& pop : obj){
		
		json_temp["广告"]=Json::Value("1");
		local["left"]=(pop.left);
		local["top"]=(pop.top);
		local["width"]=(pop.width);
		local["height"]=(pop.height);
		BBox["EventTime"]="触发时间";
		BBox["type"]=json_temp;
		BBox["box"]=(local);
		BBox["目标id"]=count++;
		root.append(BBox);
	}
  
    std::string out = root.toStyledString();
	//std::cout<<writer.write(root)<<std::endl;
	const char*SEND_MSG=(out.c_str());
    std::cout << "message :"<<SEND_MSG << std::endl;
	 
   
    NvMsgBrokerClientMsg msg;
    msg.topic = strdup(KAFKA_TPOIC);
    msg.payload = const_cast<char *>(SEND_MSG);
    msg.payload_len = strlen(SEND_MSG);
    for(int i = 0; i < 1; i++) {
      if (nv_msgbroker_send_async(conn_handle, msg, test_send_cb, ti.test_key) != NV_MSGBROKER_API_OK)
	    printf("Test %d: send [%d] failed for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
      else{
	    printf("Test %d: sending [%d] asynchronously for %s(%s)\n", ti.test_id, i, ti.test_key, ti.proto_key);
    // unsleep(100);  
	 }  //10ms sleep
   }
      free(msg.topic);
	 
     
     //printf("Test %d: Disconnecting... in 3 secs\n", ti.test_id);
   // sleep(3);
	 //m_classes.clear();
 //}
	
	nv_msgbroker_disconnect(conn_handle);
	//}
	//m_classes.clear();
} */


float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int test_id=0;

	
	/* test_info kafka_test(test_id++, KAFKA_KEY, KAFKA_KEY, KAFKA_PROTO_PATH, KAFKA_CFG_FILE, KAFKA_CONN_STR); */
/* This is a sample bounding box parsing function for the sample YoloV5m detector model */
extern "C"  bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<Detection> res;
    
    nms(res, (float*)(outputLayersInfo[0].buffer), CONF_THRESH, NMS_THRESH);
    //std::cout<<"Nms done sucessfully----"<<std::endl;
	//std::cerr<<KAFKA_CFG_FILE<<KAFKA_CFG_FILE<<"....."<<KAFKA_CONN_STR<<std::endl;
    
    //printf("Refer to nvds log file for log output\n");
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
	
       /*  std::cout << static_cast<unsigned int>(r.bbox[0]) << "," << static_cast<unsigned int>(r.bbox[1]) << "," << static_cast<unsigned int>(r.bbox[2]) << "," 
                  << static_cast<unsigned int>(r.bbox[3]) << "," << static_cast<unsigned int>(r.class_id) << "," << static_cast<unsigned int>(r.conf) << std::endl; */
	    
		objectList.push_back(oinfo);
	
        ///auto iter = g_info_map.begin();
        //printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
				//std::thread(run_test, iter->first, pop).join();
				//run_test(iter->first ,oinfo);
		//std::cout<<"检测容量："<<g_info_map<<std::endl;
		//std::cout<<m_classes.size()<<std::endl;
		// for (auto iter = g_info_map.begin(); iter != g_info_map.end(); ++iter) {
         // printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
    //} 
    //} 
    }
	
	
	
	//std::cout<<"进行一次推理！"<<objectList.size()<<std::endl;
	//test_info tt{kafka_test};
	//g_info_map[kafka_test.test_key] = tt;
	// if(!objectList.empty()){
		 
        /* std::string classesFile = "/home/project/Deepstream_Project/Deepstream_Yolo/lables_ads.txt";
	     std::ifstream ifs(classesFile.c_str());
	     std::string line; */
	   /*   while (getline(ifs, line))
         m_classes.push_back(line); */
	
	    
        // To disable a test, comment out the line below that adds it to g_info_map.
        
		std::cout<<"检测到目标 "<<std::endl;
	 //for (auto iter = g_info_map.begin(); iter != g_info_map.end(); ++iter) {
		//std::cout<<kafka_test<<std::endl;
       /*  if(!objectList.empty())	   
			libkafka(objectList); */
        //printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
	    // printf("Starting %s: %s\n",iter->second); 
     /*   for (auto iter = g_info_map.begin(); iter != g_info_map.end(); ++iter) {
        //printf("Starting %s: %s\n", iter->second.test_key, iter->second.proto_key);
		std::cout<<"BEFORE:"<<iter->first<<" "<<iter->second.proto_key<<std::endl;
        std::thread(run_test, iter->first,objectList).join();
	
       // break;
    }  */
//	 }
	// }
	    
    //}  
	//g_info_map.clear();
	//g_info_map.clear();
   // printf("Done. All tests finished successfully\n");
    return true;
}
/* This is a sample bounding box parsing function for the sample YoloV4 detector model */
static bool NvDsInferParseYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<Detection> res;

    nms(res, (float*)(outputLayersInfo[0].buffer), CONF_THRESH, NMS_THRESH);
    //std::cout<<"Nms done sucessfully----"<<std::endl;
    
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
        //std::cout << static_cast<unsigned int>(r.bbox[0]) << "," << static_cast<unsigned int>(r.bbox[1]) << "," << static_cast<unsigned int>(r.bbox[2]) << "," 
        //          << static_cast<unsigned int>(r.bbox[3]) << "," << static_cast<unsigned int>(r.class_id) << "," << static_cast<unsigned int>(r.conf) << std::endl;
	    objectList.push_back(oinfo);        
    }
    
    return true;
}

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)]);
                const float bh
                    = ph * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV3Tensor(
    const float* detections, const std::vector<int> &mask, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static inline std::vector<const NvDsInferLayerInfo*>
SortLayers(const std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo*> outLayers;
    for (auto const &layer : outputLayersInfo) {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](const NvDsInferLayerInfo* a, const NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}

static bool NvDsInferParseYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const std::vector<float> &anchors,
    const std::vector<std::vector<int>> &masks)
{
    const uint kNUM_BBOXES = 3;

    const std::vector<const NvDsInferLayerInfo*> sortedLayers =
        SortLayers (outputLayersInfo);

    if (sortedLayers.size() != masks.size()) {
        std::cerr << "ERROR: yoloV3 output layer.size: " << sortedLayers.size()
                  << " does not match mask.size: " << masks.size() << std::endl;
        return false;
    }

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint idx = 0; idx < masks.size(); ++idx) {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid

        assert(layer.inferDims.numDims == 3);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];
        const uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));

        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloV3Tensor((const float*)(layer.buffer), masks[idx], anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                       NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }


    objectList = objects;

    return true;
}


/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYoloV5(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}
extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
	
    return NvDsInferParseYoloV4 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}
extern "C" bool NvDsInferParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const std::vector<float> kANCHORS = {
        10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
        45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    static const std::vector<std::vector<int>> kMASKS = {
        {6, 7, 8},
        {3, 4, 5},
        {0, 1, 2}};
    return NvDsInferParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

extern "C" bool NvDsInferParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const std::vector<float> kANCHORS = {
        10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    static const std::vector<std::vector<int>> kMASKS = {
        {3, 4, 5},
        //{0, 1, 2}}; // as per output result, select {1,2,3}
        {1, 2, 3}};

    return NvDsInferParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

static bool NvDsInferParseYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // copy anchor data from yolov2.cfg file
    std::vector<float> anchors = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
        5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    const uint kNUM_BBOXES = 5;

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    assert(layer.inferDims.numDims == 3);
    const uint gridSizeH = layer.inferDims.d[1];
    const uint gridSizeW = layer.inferDims.d[2];
    const uint stride = DIVUP(networkInfo.width, gridSizeW);
    assert(stride == DIVUP(networkInfo.height, gridSizeH));
    for (auto& anchor : anchors) {
        anchor *= stride;
    }
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor((const float*)(layer.buffer), anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                   NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseCustomYoloTLT(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const int topK = 200;
    const int* keepCount = static_cast <const int*>(outputLayersInfo.at(0).buffer);
    const float* boxes = static_cast <const float*>(outputLayersInfo.at(1).buffer);
    const float* scores = static_cast <const float*>(outputLayersInfo.at(2).buffer);
    const float* cls = static_cast <const float*>(outputLayersInfo.at(3).buffer);

    for (int i = 0; (i < keepCount[0]) && (objectList.size() <= topK); ++i)
    {
        const float* loc = &boxes[0] + (i * 4);
        const float* conf = &scores[0] + i;
        const float* cls_id = &cls[0] + i;

        if(conf[0] > 1.001)
            continue;

        if((loc[0] < 0) || (loc[1] < 0) || (loc[2] < 0) || (loc[3] < 0))
            continue;

        if((loc[0] > networkInfo.width) || (loc[2] > networkInfo.width) || (loc[1] > networkInfo.height) || (loc[3] > networkInfo.width))
           continue;

        if((loc[2] < loc[0]) || (loc[3] < loc[1]))
            continue;

        if(((loc[3] - loc[1]) > networkInfo.height) || ((loc[2]-loc[0]) > networkInfo.width))
            continue;

        NvDsInferParseObjectInfo curObj{static_cast<unsigned int>(cls_id[0]),
                                        loc[0],loc[1],(loc[2]-loc[0]),
                                        (loc[3]-loc[1]), conf[0]};
        objectList.push_back(curObj);

    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3Tiny);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloTLT);
