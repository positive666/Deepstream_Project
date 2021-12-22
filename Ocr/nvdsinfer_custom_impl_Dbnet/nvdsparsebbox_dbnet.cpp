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
#include "clipper/clipper.hpp"
#include "common.hpp"
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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define KAFKA_KEY "kafka"
#define KAFKA_PROTO_SO "libnvds_kafka_proto.so"
#define KAFKA_PROTO_PATH SO_PATH KAFKA_PROTO_SO
#define KAFKA_CFG_FILE "./cfg_kafka.txt"
#define KAFKA_CONN_STR "localhost;9092" // broker;port
#define KAFKA_TPOIC "qucikstart-events"
#include <jsoncpp/json/json.h>
#include <fstream>
#include <opencv2/imgproc/types_c.h>

#define EXPANDRATIO 1.7
#define BOX_MINI_SIZE 3
#define SCORE_THRESHOLD 0.4
#define BOX_THRESHOLD 0.3
static bool run = true;
static void sigterm (int sig) {
    run = false;
}

class ExampleDeliveryReportCb : public RdKafka::DeliveryReportCb {
public:
    void dr_cb (RdKafka::Message &message) {
        std::cout << "Message delivery for (" << message.len() << " bytes): " <<
                     message.errstr() << std::endl;
        if (message.key())
            std::cout << "Key: " << *(message.key()) << ";" << std::endl;
    }
};

class ExampleEventCb : public RdKafka::EventCb {
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
};

void libkafka(std::vector<NvDsInferParseObjectInfo>& objectList)
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
                                  RdKafka::Producer::RK_MSG_COPY /* Copy payload */,
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
}


	
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


cv::RotatedRect expandBox(cv::Point2f temp[], float ratio)
{
    ClipperLib::Path path = {
        {ClipperLib::cInt(temp[0].x), ClipperLib::cInt(temp[0].y)},
        {ClipperLib::cInt(temp[1].x), ClipperLib::cInt(temp[1].y)},
        {ClipperLib::cInt(temp[2].x), ClipperLib::cInt(temp[2].y)},
        {ClipperLib::cInt(temp[3].x), ClipperLib::cInt(temp[3].y)}};
    double area = ClipperLib::Area(path);
    double distance;
    double length = 0.0;
    for (int i = 0; i < 4; i++) {
        length = length + sqrtf(powf((temp[i].x - temp[(i + 1) % 4].x), 2) +
                                powf((temp[i].y - temp[(i + 1) % 4].y), 2));
    }

    distance = area * ratio / length;

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::JoinType::jtRound,
                   ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths paths;
    offset.Execute(paths, distance);
    
    std::vector<cv::Point> contour;
    for (int i = 0; i < paths[0].size(); i++) {
        contour.emplace_back(paths[0][i].X, paths[0][i].Y);
    }
    offset.Clear();
    return cv::minAreaRect(contour);
}

float paddimg(cv::Mat& In_Out_img, int shortsize = 736) {
    int w = In_Out_img.cols;
    int h = In_Out_img.rows;
    float scale = 1.f;
    if (w < h) {
        scale = (float)shortsize / w;
        h = scale * h;
        w = shortsize;
    }
    else {
        scale = (float)shortsize / h;
        w = scale * w;
        h = shortsize;
    }

    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }

    cv::resize(In_Out_img, In_Out_img, cv::Size(w, h));
    return scale;
}

bool get_mini_boxes(cv::RotatedRect& rotated_rect, cv::Point2f rect[],
                    int min_size)
{
    
    cv::Point2f temp_rect[4];
    rotated_rect.points(temp_rect);
	//std::cout<<"xin:"<<temp_rect[0]<<",yin:"<<temp_rect[1]<<",xmax:"<<temp_rect[2]<<",ymax:"<<temp_rect[3]<<std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (temp_rect[i].x > temp_rect[j].x) {
                cv::Point2f temp;
                temp = temp_rect[i];
                temp_rect[i] = temp_rect[j];
                temp_rect[j] = temp;
            }
        }
    }
    int index0 = 0;
    int index1 = 1;
    int index2 = 2;
    int index3 = 3;
    if (temp_rect[1].y > temp_rect[0].y) {
        index0 = 0;
        index3 = 1;
    } else {
        index0 = 1;
        index3 = 0;
    }
    if (temp_rect[3].y > temp_rect[2].y) {
        index1 = 2;
        index2 = 3;
    } else {
        index1 = 3;
        index2 = 2;
    }   

    rect[0] = temp_rect[index0];  // Left top coordinate
    rect[1] = temp_rect[index1];  // Left bottom coordinate
    rect[2] = temp_rect[index2];  // Right bottom coordinate
    rect[3] = temp_rect[index3];  // Right top coordinate
	//std::cout<<"xim:"<<rect[0]<<",ymin:"<<rect[1]<<",xmax:"<<rect[2]<<",ymax:"<<rect[3]<<std::endl;
    if (rotated_rect.size.width < min_size ||
        rotated_rect.size.height < min_size ) {
		
        return false;
    } else {
		
        return true;
    }
}

float get_box_score(float* map, cv::Point2f rect[], int width, int height,
                    float threshold)
{
 // std::cout<<"computr!!!!!"<<std::endl;
    int xmin = width - 1;
    int ymin = height - 1;
    int xmax = 0;
    int ymax = 0;

    for (int j = 0; j < 4; j++) {
        if (rect[j].x < xmin) {
            xmin = rect[j].x;
        }
        if (rect[j].y < ymin) {
            ymin = (rect[j].y >0)?rect[j].y:0;
        }
        if (rect[j].x > xmax) {
            xmax = rect[j].x;
        }
        if (rect[j].y > ymax) {
            ymax = rect[j].y;
        }
    }
	//std::cout<<"xim:"<<xmin<<",ymin:"<<ymin<<",xmax:"<<xmax<<",ymax:"<<ymax<<std::endl;
    float sum = 0;
    int num = 0;
	//fillPoly(map, rect, 1, , Scalar(0,0,255));
    

	 if(xmin <0 || ymin<0 ||xmax<0||ymax<0)
	{
		//std::cout<<"drop zero"<<std::endl;
		return 0;
		
		
	} 
	
	
	
    for (int i = ymin; i <= ymax; i++) {
		
        for (int j = xmin; j <= xmax; j++) {
            if (map[i * width + j] > threshold) {
                sum = sum + map[i * width + j];
                num++;
            }
        }
    }

    return sum / num;
}
/* This is a sample bounding box parsing function for the sample YoloV5m detector model */
extern "C"  bool NvDsInferParseDB(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (1 != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << 1<< std::endl;
    }

    std::vector<Detection> res;
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
	
	objectList.clear();
	const NvDsInferLayerInfo *out = layerFinder("out1");
	////    std::cout<<"width"<<&networkInfo.width<<std::endl;
//(float*)(outputLayersInfo[0].buffer)
	if (!out)
	{
		std::cerr << "ERROR: some layers missing or unsupported data types "
				  << "in output tensors" << std::endl;
		return false;
	}
	
	 
	int h_net=out->inferDims.d[1];//; //#heatmap.size[2];
	int w_net=out->inferDims.d[2];
	
	//std::cout<<"width"<<h_net<<std::endl;
	float * output_tenosr=(float *) out->buffer;
	cv::Mat map = cv::Mat::zeros(cv::Size(w_net,h_net),CV_8UC1); 
	
	for (int h = 0; h < h_net; ++h) {
            uchar *ptr = map.ptr(h);
            for (int w = 0; w < w_net; ++w) {
				//std::cout<<"id""<<heatmap[i*w+j]<<std::endl;
                ptr[w] = (output_tenosr[h * (w_net)+ w] > 0.3) ? 255 : 0;
            }
        }
		
     //提取最小外接矩形
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    cv::findContours(map, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE) ;
  
  
    std::vector<cv::RotatedRect> box(contours.size());
    cv::Point2f rect[4];
    cv::Point2f order_rect[4];
    
    for (int i = 0; i < contours.size(); i++) {
			//std::cout<<"get current contour"<<std::endl;
			
            cv::RotatedRect rotated_rect = cv::minAreaRect(cv::Mat(contours[i]));
				//std::cout<<"最小外接矩形的中心点："<<rotated_rect.points <<std::endl;
         	//std::cout<<"最小外接矩形的角度："<<rotated_rect.angle <<std::endl;
			
			/*  if (rotated_rect.angle>=-45)) {
                std::cout << " no  box too s......." <<  std::endl;
                continue;
            }
			 */
			
            if (!get_mini_boxes(rotated_rect, rect, BOX_MINI_SIZE)) {
                std::cout << "box too small......." <<  std::endl;
                continue;
            }
            // drop low score boxes
            float score = get_box_score(output_tenosr, rect,w_net, h_net, SCORE_THRESHOLD);
			//std::cout<<"get !!!!!!!!! score boxes"<<std::endl;
            if (score < BOX_THRESHOLD) {
                std::cout << "score too low = " << score << ", threshold = " << BOX_THRESHOLD <<  std::endl;
                continue;
            }
			//std::cout<<"drop low score boxes done"<<std::endl;	
            // Scaling the predict boxes depend on EXPANDRATIO
            cv::RotatedRect expandbox = expandBox(rect, EXPANDRATIO);
            expandbox.points(rect);
			
	
			
            if (!get_mini_boxes(expandbox, rect, BOX_MINI_SIZE + 2)) {  
                continue;
            }
           // std::cout<<"Restore the coordinates to the original image"<<std::endl;
            // Restore the coordinates to the original image
            for (int j = 0; j < 4; j++) {
 
				//过滤因为扩大box导致的部分超界坐标
				if(rect[j].x<0){
					rect[j].x=0;
				}
				if(rect[j].x>(w_net-1)){
					rect[j].x=w_net-1;
				}
				if(rect[j].y<0){
					rect[j].y=0;
				}
				if(rect[j].y>(h_net-1)){
					rect[j].y=h_net-1;
				}
				order_rect[j].x =
					floor(rect[j].x / w_net * w_net);
				// floor(rect[j].x/scale);
				order_rect[j].y =
					floor(rect[j].y / h_net * h_net);
            // floor(rect[j].y/scale);
        
				//std::cout<<"[]0x:"<<order_rect[k].x<<std::endl;
				//std::cout<<"[]1y:"<<order_rect[k].y<<std::endl;
				//std::cout<<"[]2:"<<<<std::endl;
				///std::cout<<"[]3:"<<<<std::endl;
            }
			//std::cout<<"Restore the coordinates end !!!!!!!!!!"<<std::endl;
			NvDsInferParseObjectInfo oinfo;
			oinfo.detectionConfidence = 0.99;
            oinfo.classId = 0;
			oinfo.left    = static_cast<unsigned int>(order_rect[0].x);
			oinfo.top    = static_cast<unsigned int>(order_rect[0].y);
			oinfo.width  = static_cast<unsigned int>(order_rect[1].x-order_rect[0].x);
			oinfo.height = static_cast<unsigned int>(order_rect[3].y-order_rect[0].y);
			//std::cout<<"left"<<oinfo.left<<std::endl;
			//drop scale screen ration box
			//if(oinfo.left<h_net*0.55)

			objectList.push_back(oinfo);
		    
            //cv::rectangle(src_img, cv::Point(order_rect[0].x,order_rect[0].y), cv::Point(order_rect[2].x,order_rect[2].y), cv::Scalar(0, 0, 255), 2, 8);
            //std::cout << "After LT =  " << order_rect[0] << ", After RD = " << order_rect[2] <<  std::endl;            
        } 
		
   
		
        //std::cout << "write image done." << std::endl;
    return true;
}


extern "C" bool NvDsInferParseCustomDB(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseDB(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDB);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3Tiny);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloTLT);
