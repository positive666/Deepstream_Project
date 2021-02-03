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

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#define MIN_V_OVERLAPS 0.6
#define MIN_SIZE_SIM 0.7
typedef struct node{
	int index;
	struct node *next;
	struct node *right;

}node;


int frame=0;
static const int NUM_CLASSES_YOLO = 2;

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

static unsigned clamp(const uint val, const uint minVal, const uint maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.left = x - bw / 2;
    b.width = bw;

    b.top = y - bh / 2;
    b.height = bh;

    b.left = clamp(b.left, 0, netW);
    b.width = clamp(b.width, 0, netW);
    b.top = clamp(b.top, 0, netH);
    b.height = clamp(b.height, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (((bbi.left + bbi.width) > netW) || ((bbi.top + bbi.height) > netH)) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) 
					 {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });
    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

static std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(const float nmsThresh,
        std::vector<NvDsInferParseObjectInfo>& binfo,
        const uint numClasses)
{
    std::vector<NvDsInferParseObjectInfo> result;
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const std::vector<float> &anchors,
    const uint gridSize, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const float probThresh, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
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

                if (maxProb > probThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}


static std::vector<NvDsInferParseObjectInfo>
decodeYoloV3Tensor(
    const float* detections, const std::vector<int> &mask, const std::vector<float> &anchors,
    const uint gridSize, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const float probThresh, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
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

                if (maxProb > probThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
                }
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
    std::sort (outLayers.begin(), outLayers.end(),
      [](const NvDsInferLayerInfo *a, const NvDsInferLayerInfo *b){
          return a->dims.d[1] < b->dims.d[1];
      });
    return outLayers;
}
					
static bool sortFun(const NvDsInferParseObjectInfo& p1, const NvDsInferParseObjectInfo& p2)
{
	return p1.left < p2.left;//升序排列  
}

float max1(float a, float b)
{
	return a > b ?a : b;
}

float min1(float a, float b)
{
	return a > b ? b : a;
}

int meet_v_iou(int index1, int index2, std::vector<NvDsInferParseObjectInfo> binfo)//计算两个BOX的重合度
{   //竖直方向重合程度
	float h = binfo[index1].height +1;
	float h1 = binfo[index2].height +1;

	float y = max1(binfo[index1].top, binfo[index2].top);
	float y1 = min1(binfo[index1].top + binfo[index1].height , binfo[index2].top + binfo[index2].height );

	float overlap = max1(0, y1 - y + 1) / min1(h, h1);

	float similarity = min1(h, h1) / max1(h, h1);

	if (overlap >= MIN_V_OVERLAPS && similarity >= MIN_SIZE_SIM)//判断
		return 1;
	return 0;
}

node *get_successions(int index, node* boxes_table, std::vector<NvDsInferParseObjectInfo> binfo, int len, int img_w)
//node *get_successions(int index, node* boxes_table, float text_proposals[][4], float score[], int len, int img_w)
{
	
    node *ret=new node[sizeof(node)];
	ret->index = -1;
	ret->right = NULL;
    // 防止和自己重合
	for (int start = binfo[index].left + 1; start < (int)min1(binfo[index].left + 30, img_w); start++)
	{
		node  head = boxes_table[start], *cur = &head;
		while (cur && cur->index != -1)
		{
			if (meet_v_iou(cur->index, index, binfo))
			{
				//add a new node to the list
				if (ret->index == -1)
				{
					ret->index = cur->index;
				}
				else
				{
					// node *n = (node*)malloc(sizeof(node));
                    node *n=new node[sizeof(node)];
					n->index = cur->index;
					n->right = ret->right;
					ret->right = n;
				}
			}
			cur = cur->right;
		}
		if (ret->index != -1)
			return ret;
	}
	return ret;
}

//检测左边30像素的BOX的INDEX
node *get_precursors(int index, node* boxes_table, std::vector<NvDsInferParseObjectInfo> binfo, int len, int img_w)
{
	
    node *ret=new node[sizeof(node)];
	ret->index = -1;
	ret->right = NULL;
	for (int start = binfo[index].left - 1; start >= (int)max1(binfo[index].left - 50, 0); start--)
	{
		node  head = boxes_table[start], *cur = &head;
		while (cur && cur->index != -1)
		{
			if (meet_v_iou(cur->index, index, binfo))
			{
				//加入 new node to the list
				if (ret->index == -1)
				{
					ret->index = cur->index;
				}
				else
				{
					node *n = new node[sizeof(node)];
					n->index = cur->index;
					n->right = ret->right;
					ret->right = n;
				}
			}
			cur = cur->right;
		}
		if (ret->index != -1)
			return ret;
	}
	return ret;
}

int is_successions(int index, int succession_index, node *boxes_table, std::vector<NvDsInferParseObjectInfo> binfo, int len, int img_w)
{
	node *ret = get_precursors(succession_index, boxes_table, binfo, len, img_w);
	node *precursors = ret;
	//get the max index 
	float max = 0;
	int max_index = -1;
	while (precursors  && precursors->index != -1){
		if (binfo[precursors->index].detectionConfidence>max)
		{
			max = binfo[precursors->index].detectionConfidence;
			max_index = precursors->index;
		}
		precursors = precursors->right;
	}

	//释放内存
	while (ret)
	{
		node *tmp = ret;
		ret = ret->right;
		delete tmp;
	}

	if (binfo[index].detectionConfidence >= binfo[max_index].detectionConfidence)
		return 1;
	return 0;
}

static void gen_graph(std::vector<NvDsInferParseObjectInfo> binfo,
	const uint img_w, const uint img_h, unsigned char **graph)
	//gen_graph(float text_proposals[][4], float score[], int len,int img_h, int img_w, unsigned char **graph)
{
	int len = binfo.size();
	//创建 boxes table 和 初始化：对图像中水平方向的每个像素点 建立列表
	node * boxes_table = new node[img_w*(sizeof(node))];
	for (int i = 0; i < img_w; i++)
	{
		boxes_table[i].index = -1;
		boxes_table[i].right = NULL;
	}
	
	for (int i = 0; i < len; i++)
	{
		int x = binfo[i].left;
		//first time to insert
		if (boxes_table[x].index == -1)
		{
			boxes_table[x].index = i;
		}
		else
		{
			//插入新的节点 
			// node *n = (node*)malloc(sizeof(node));
            node *n=new node[sizeof(node)];
			n->index = i;
			n->right = boxes_table[x].right;
			boxes_table[x].right = n;
		}
	}
	for (int i = 0; i < len; i++)
	{
		node *ret = get_successions(i, boxes_table, binfo, len, img_w);//检测右边配对算法
		node *successions = ret;//index：对应的Box和右侧30个像素的所有BOX 竖直放重合度高的BOX的INDEX
		//得到 max index 
		float max = 0;
		int max_index = -1;
		while (successions  && successions->index != -1){
			if (binfo[successions->index].detectionConfidence>max)
			{
				max = binfo[successions->index].detectionConfidence;
				max_index = successions->index;
			}
			successions = successions->right;
		}
		if (max_index == -1) continue;
      //      留下该index的box对应successions的box相应scores最高的index

		if (is_successions(i, max_index, boxes_table, binfo, len, img_w))
		{
			graph[i][max_index] = 1;
		}
		//FREE 内存
		while (ret){
			node* tmp = ret;
			ret = ret->right;
			delete tmp;
		}

	}
	//释放内存
	for (int i = 0; i < img_w; i++)
	{
		node *n = boxes_table[i].right;
		while (n){
			node *tmp = n;
			n = n->right;
			delete tmp;
		}
	}
	delete []boxes_table;

}


void sub_graphs_connected(unsigned char **graph, int len, node *sub_graphs)
{
	node *cur = sub_graphs;
	for (int i = 0; i < len; i++)
	{
		//找到没有前节点的
		int j = 0;
		while (j<len && graph[j][i] == 0) j++;
		if (j < len)
			continue;
		//判断当前 node 有没有 next node
		j = 0;
		while (j<len && graph[i][j] == 0) j++;
		if (j == len)
			continue;

		//add the first node to sub_graphs
		if (cur->index == -1){
			cur->index = i;
		}
		else
		{
			node *n = (node *)malloc(sizeof(node));
			n->index = i;
			n->next = NULL;
			n->right = NULL;
			cur->next = n;
			cur = n;
		}

		//judge the node whether it includes next node
		int k = i;
		node *rcur = cur;
		while (rcur->right) rcur = rcur->right;
		while (1)
		{
			j = 0;
			while (j<len && graph[k][j] == 0) j++;
			if (j == len)
				break;
			k = j;
			// node *n = (node*)malloc(sizeof(node));
            node *n=new node[sizeof(node)];
			n->index = j;
			n->right = NULL;
			rcur->right = n;
			rcur = n;
		}


	}
}



static std::vector<NvDsInferParseObjectInfo>
get_text_lines(std::vector<NvDsInferParseObjectInfo> binfo,const uint img_w,const uint img_h)
//get_text_lines(float text_proposals[][4], float score[], int len, int img_h, int img_w, float rbox_sco[][9], int *rlen)
{
	int len = binfo.size();
	// unsigned char **graph = (unsigned char**)malloc(len*sizeof(unsigned char*));
    unsigned char **graph=new unsigned char*[len];//分配二维数组内存：
	std::vector<NvDsInferParseObjectInfo> result_objects;
	for(int i = 0; i < len; i++)
	{
		// graph[i] = (unsigned char*)malloc(len*sizeof(unsigned char));
        graph[i]= new unsigned char[len];
		// memset(graph[i], 0, len);
        
	}
    //初始化
    for(int i=0;i<len;++i)
     for(int j=0;j<len;++j)
       graph[i][j]=0;
	gen_graph(binfo, img_w, img_h, graph);

	//创建一个空链表sub_graphs
    node *sub_graphs=new node[sizeof(node)];
	sub_graphs->index = -1;
	sub_graphs->right = NULL;
	sub_graphs->next = NULL;
	sub_graphs_connected(graph, len, sub_graphs);

	//std::cout<<"3333333::::"<<len<<std::endl;
	node *cur = sub_graphs;//text proposal
	while (cur && cur->index != -1)
	{
		node *lcur = cur;
        
		float y1 = binfo[lcur->index].top;
		float y2 = binfo[lcur->index].top + binfo[lcur->index].height ;
		float x1 = binfo[lcur->index].left;
		float x2 = binfo[lcur->index].left + binfo[lcur->index].width ;
		int num = 0;
		float sco = 0;
		while (lcur){
			sco +=  binfo[lcur->index].detectionConfidence;
			
			if (binfo[lcur->index].left < x1)
				x1 = binfo[lcur->index].left;
			
			if (binfo[lcur->index].left + binfo[lcur->index].width >x2)
				x2 = binfo[lcur->index].left + binfo[lcur->index].width ;

			if (binfo[lcur->index].top < y1)
				y1 = binfo[lcur->index].top;
			
			if (binfo[lcur->index].top + binfo[lcur->index].height >y2)
				y2 = binfo[lcur->index].top + binfo[lcur->index].height ;
			lcur = lcur->right;
			num++;
		}
                // # the score of a text line is the average score of the scores
            // # of all text proposals contained in the text line
		sco = sco / num;
		int m = (x2 - x1) / num * 0.5;//offset
		x1 = x1 - m;
		x2 = x2 + m;
		
		NvDsInferParseObjectInfo p;
		p.left = x1;
		p.top = y1;
		//p.width = (x2+x1)/2.0;
		p.width = x2-x1;
		//p.height = (y1+y2)/2.0;
		p.height = y2-y1;
		p.detectionConfidence = sco;
		p.classId = 1;
		if(p.top>400 )
		result_objects.push_back(p);
		//std::cout<<"img_hs"<<img_h<<std::endl;
		cur = cur->next;//继续下个
	}
	std::cout<<"result_objects"<<result_objects.size()<<std::endl;
   
    delete []graph;
	
	while (sub_graphs){
		node *cur = sub_graphs;
		sub_graphs = sub_graphs->next;
		while (cur) {
			node *tmp = cur;
			cur = cur->right;
			free(tmp);
		}
	}
	

	//std::cout<<"55555"<<std::endl;
	return result_objects;
}  


static std::vector<NvDsInferParseObjectInfo>
text_line_drop(std::vector<NvDsInferParseObjectInfo>& boxes,const uint width,const uint height)
{
    std::vector<NvDsInferParseObjectInfo> result;
    //std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
	std::vector<int> boxIndex;
	std::vector<NvDsInferParseObjectInfo> imp_objects;
	std::vector<NvDsInferParseObjectInfo> tmp_imp_objects;
	std::vector<NvDsInferParseObjectInfo> result_objects;
	int i = 0;
	float nmsThresh =0.2; //0.3
    for (auto& box : boxes)
    {
		
        //std::cout<<"zzzzzz:"<<box.left<<"###"<<box.classId<<"  qqqqq  "<<box.detectionConfidence<<std::endl;
		if (box.detectionConfidence > 0.3)  //0.5
		{
			boxIndex.push_back(i); 
		}
		i++;
    }
	for(auto& index:boxIndex)
	{
		imp_objects.push_back(boxes[index]);
	}
	sort(imp_objects.begin(), imp_objects.end(), sortFun);
	
	tmp_imp_objects = nonMaximumSuppression(nmsThresh, imp_objects);
	//std::cout<<"22222:::"<<boxes.size()<<std::endl;
	result_objects = get_text_lines(tmp_imp_objects,width,height);/* 在此将碎片框合并成一行，存储在*/
	//std::cout<<"66666::: "<<"result"<<result_objects.size()<<std::endl;
	return result_objects;
}

static bool NvDsInferParseYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const std::vector<float> &anchors,
    const std::vector<std::vector<int>> &masks)
{
	//std::cout<<"nvdsparsebbox_Yolo:710 *******   start yolov3 parse"<<std::endl;
    const uint kNUM_BBOXES = 3;
    static const float kNMS_THRESH = 0.3f;
    static const float kPROB_THRESH = 0.1f; //0.7 0.1
	clock_t t_start; 
	clock_t t_end3;
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
	//std::cout<<"masks.size:"<<masks.size()<<"networkInfo:"<<networkInfo.width<<" "<<networkInfo.height<<std::endl;
    for (uint idx = 0; idx < masks.size(); ++idx) {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid
        assert (layer.dims.numDims == 3);
        const uint gridSize = layer.dims.d[1];
        const uint stride = networkInfo.width / gridSize;
		//std::cout<<"gridSize: "<<gridSize<<std::endl;
        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloV3Tensor((const float*)(layer.buffer), masks[idx], anchors, gridSize, stride, kNUM_BBOXES,
                       NUM_CLASSES_YOLO, kPROB_THRESH, networkInfo.width, networkInfo.height);
					   //std::cout<<"outObjs: "<<outObjs.size()<<"outobjs[1]:"<<std::endl;
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }
	//在此添加文本线构造函数即可
	//std::cout<<"11111::::"<<objects.size()<<""<<std::endl;
	
	
	
      
     
	  t_start = clock(); 
	objectList = text_line_drop(objects,networkInfo.width,networkInfo.height);
	//objectList=objects;
	t_end3 = clock();
	clock_t t3 = t_end3 - t_start;
    double time_taken3 = ((double)t3)/CLOCKS_PER_SEC; // in seconds 

	//std::cout<<"ago: "<<objects.size()<<std::endl;
	//std::cout<<"TIME: "<<time_taken3<<std::endl;
	/* for(auto& index:objects)
	{
		std::cout<<index.left<<" "<<index.top<<" "<<index.width<<" "<<index.height
		<<" "<<index.classId<<" "<<index.detectionConfidence<<std::endl;
	} */

	std::cout<<"after: "<<objectList.size()<<std::endl;
	/* for(auto& index:objectList)
	{
		std::cout<<index.left<<" "<<index.top<<" "<<index.width<<" "<<index.height
		<<" "<<index.classId<<" "<<index.detectionConfidence<<std::endl;
	} */
    //objectList.clear();
    //objectList = nmsAllClasses(kNMS_THRESH, objects, NUM_CLASSES_YOLO);
    //此项操作可以不要，因为已经使用文本线构造算法，包含了NMS，准确的说只剩下一两个框了
	frame ++;
    return true;
}


/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
	//std::cout<<"nvdsparsebbox_Yolo:788   &&&&&&&&&"<<std::endl;
    static const std::vector<float> kANCHORS = {
       16,11, 16,16, 16,23, 16,33, 16,48, 16,97, 16,139, 16,198, 16,283};
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
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const float nmsThreshold, const float probthreshold)
{
    static const std::vector<float> kANCHORS = {
        18.3273602, 21.6763191, 59.9827194, 66.0009613,
        106.829758, 175.178879, 252.250244, 112.888962,
        312.656647, 293.384949 };
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

    assert (layer.dims.numDims == 3);
    const uint gridSize = layer.dims.d[1];
    const uint stride = networkInfo.width / gridSize;
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor((const float*)(layer.buffer), kANCHORS, gridSize, stride, kNUM_BBOXES,
                   NUM_CLASSES_YOLO, probthreshold, networkInfo.width, networkInfo.height);

    objectList.clear();
    objectList = nmsAllClasses(nmsThreshold, objects, NUM_CLASSES_YOLO);

    return true;
	
}

extern "C" bool NvDsInferParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const float kNMS_THRESH = 0.5f;
    static const float kPROB_THRESH = 0.6f;

    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kNMS_THRESH, kPROB_THRESH);
}

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const float kNMS_THRESH = 0.5f;
    static const float kPROB_THRESH = 0.6f;
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kNMS_THRESH, kPROB_THRESH);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3Tiny);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
