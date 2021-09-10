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
#include "nvdsinfer_custom_impl.h"
#include <cassert>
//#include <cmath>
#include <tuple>
#include <memory>
#include <opencv2/opencv.hpp>

#define CLIP(a, min, max) (MAX(MIN(a, max), min))

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomTfSSD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
										  NvDsInferNetworkInfo const &networkInfo,
										  NvDsInferParseDetectionParams const &detectionParams,
										  std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a smaple bbox parsing function for the centernet face detection onnx model*/
struct FrcnnParams
{
	int inputHeight;
	int inputWidth;
	int outputClassSize;
	float visualizeThreshold;
	int postNmsTopN;
	int outputBboxSize;
	std::vector<float> classifierRegressorStd;
};

struct FaceInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float landmarks[10];
};

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
/* NMS for centernet */
static void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold)
{
	std::sort(input.begin(), input.end(),
			  [](const FaceInfo &a, const FaceInfo &b) {
				  return a.score > b.score;
			  });

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1; //std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2; //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > nmsthreshold)
				merged[j] = 1;
		}
	}
}
/* For CenterNetFacedetection */
//extern "C"
static std::vector<int> getIds(float *heatmap, int h, int w, float thresh)
{
	std::vector<int> ids;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{

			//			std::cout<<"ids"<<heatmap[i*w+j]<<std::endl;
			if (heatmap[i * w + j] > thresh)
			{
				//				std::array<int, 2> id = { i,j };
				ids.push_back(i);
				ids.push_back(j);
				//	std::cout<<"print ids"<<i<<std::endl;
			}
		}
	}
	return ids;
}

/* customcenternetface */
extern "C" bool NvDsInferParseCustomCenterFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
												  NvDsInferNetworkInfo const &networkInfo,
												  NvDsInferParseDetectionParams const &detectionParams,
												  std::vector<NvDsInferObjectDetectionInfo> &objectList)
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
	objectList.clear();
	const NvDsInferLayerInfo *heatmap = layerFinder("537");
	const NvDsInferLayerInfo *scale = layerFinder("538");
	const NvDsInferLayerInfo *offset = layerFinder("539");
	const NvDsInferLayerInfo *landmarks = layerFinder("540");
	//    std::cout<<"width"<<&networkInfo.width<<std::endl;

	if (!heatmap || !scale || !offset || !landmarks)
	{
		std::cerr << "ERROR: some layers missing or unsupported data types "
				  << "in output tensors" << std::endl;
		return false;
	}
    
	int fea_h = heatmap->inferDims.d[1];//; //#heatmap.size[2];
	int fea_w = heatmap->inferDims.d[2];//;//heatmap.size[3];
	int spacial_size = fea_h* fea_w;
 //	std::cout<<"features"<<fea_h<<"width"<<fea_w<<std::endl;
	//std::cout<<"0:"<<heatmap->inferDims.d[0]<<"1:"<<heatmap->inferDims.d[1]<<std::endl;
	float *heatmap_ = (float *)(heatmap->buffer);

	float *scale0 = (float *)(scale->buffer);
	float *scale1 = scale0 + spacial_size;

	float *offset0 = (float *)(offset->buffer);
	float *offset1 = offset0 + spacial_size;
	float *lm = (float *)landmarks->buffer;

	float scoreThresh = 0.5;
	std::vector<int> ids = getIds(heatmap_, fea_h, fea_w, scoreThresh);
	//?? d_w, d_h
	//std::cerr<<"getids:"<<ids.size()<<std::endl;
	int width = networkInfo.width;
	int height = networkInfo.height; 
	int d_h = (int)(std::ceil(height / 32) * 32);
	int d_w = (int)(std::ceil(width / 32) * 32);
	//	int d_scale_h = height/d_h ;
	//	int d_scale_w = width/d_w ;
	//	float scale_w = (float)width / (float)d_w;
	//	float scale_h = (float)height / (float)d_h;
	std::vector<FaceInfo> faces_tmp;
	std::vector<FaceInfo> faces;
	for (int i = 0; i < ids.size() / 2; i++)
	{
		int id_h = ids[2 * i];
		int id_w = ids[2 * i + 1];
		int index = id_h * fea_w + id_w;

		float s0 = std::exp(scale0[index]) * 4;
		float s1 = std::exp(scale1[index]) * 4;
		float o0 = offset0[index];
		float o1 = offset1[index];
		float x1 = std::max(0., (id_w + o1 + 0.5) * 4 - s1 / 2);
		float y1 = std::max(0., (id_h + o0 + 0.5) * 4 - s0 / 2);
		float x2 = 0, y2 = 0;
		x1 = std::min(x1, (float)d_w);
		y1 = std::min(y1, (float)d_h);
		x2 = std::min(x1 + s1, (float)d_w);
		y2 = std::min(y1 + s0, (float)d_h);

		FaceInfo facebox;
		facebox.x1 = x1;
		facebox.y1 = y1;
		facebox.x2 = x2;
		facebox.y2 = y2;
		facebox.score = heatmap_[index];
		for (int j = 0; j < 5; j++)
		{
			facebox.landmarks[2 * j] = x1 + lm[(2 * j + 1) * spacial_size + index] * s1;
			facebox.landmarks[2 * j + 1] = y1 + lm[(2 * j) * spacial_size + index] * s0;
		}
		//std::cout<<"face:"<<x1<<","<<y1<<"---------"<<x2<<","<<y2<<std::endl;
		faces_tmp.push_back(facebox);
	}
   //std::cout<<"decode end!"<<std::endl;
	const float threshold = 0.4;
	nms(faces_tmp, faces, threshold);
	for (int k = 0; k < faces.size(); k++)
	{
		NvDsInferObjectDetectionInfo object;
		//Clip object box co-ordinates to network resolution
		object.left = CLIP(faces[k].x1, 0, networkInfo.width - 1);
		object.top = CLIP(faces[k].y1, 0, networkInfo.height - 1);
		object.width = CLIP((faces[k].x2 - faces[k].x1), 0, networkInfo.width - 1);
		object.height = CLIP((faces[k].y2 - faces[k].y1), 0, networkInfo.height - 1);
		
		
		float wide_dist = pow((faces[k].landmarks[0] - faces[k].landmarks[2]) * (faces[k].landmarks[0] - faces[k].landmarks[2]) + (faces[k].landmarks[1] - faces[k].landmarks[3]) *(faces[k].landmarks[1] - faces[k].landmarks[3]),0.5);
        float high_dist = pow((faces[k].landmarks[0] - faces[k].landmarks[6]) * (faces[k].landmarks[0] - faces[k].landmarks[6]) + (faces[k].landmarks[1] - faces[k].landmarks[7]) *(faces[k].landmarks[1] - faces[k].landmarks[7]),0.5);
        float dist_rate = high_dist / wide_dist;

        float dist_C = pow((faces[k].landmarks[6] - faces[k].landmarks[4]) * (faces[k].landmarks[6] - faces[k].landmarks[4]) + (faces[k].landmarks[7] - faces[k].landmarks[5]) *(faces[k].landmarks[7] - faces[k].landmarks[5]),0.5);
        float dist_D = pow((faces[k].landmarks[8] - faces[k].landmarks[4]) * (faces[k].landmarks[8] - faces[k].landmarks[4]) + (faces[k].landmarks[9] - faces[k].landmarks[5]) *(faces[k].landmarks[9] - faces[k].landmarks[5]),0.5);
        float width_rate =fabs(dist_C / dist_D - 1);
		
		/* object.left = faces[k].x1;
		object.top = faces[k].y1;
		object.width = faces[k].x2 - faces[k].x1;
		object.height = faces[k].y2 - faces[k].y1;
        std::cout<<"h:"<<object.height<<std::endl;
		
		std::cout<<"w:"<<object.width<<std::endl; */
		std::cout<<"now width_rate:"<<width_rate<<std::endl;
		std::cout<<"now dist_rate:"<<dist_rate<<std::endl;
		if (object.width && object.height && dist_rate < 1.5 && width_rate <0.7)
        {   //std::cout<<"biu!!!!!!"<<std::endl;
          std::cout<<" ================ "<<dist_rate<< " ---- "<<width_rate << " ---- "<< high_dist<< " ---- "<< wide_dist <<std::endl;
          object.detectionConfidence = 0.99;
          object.classId = 0;
          objectList.push_back(object);
        }
	}
	return true;
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCenterFace);