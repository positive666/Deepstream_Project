#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>




extern "C" bool NvDsInferParseCustomText_new(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
       std::vector<NvDsInferAttribute> &attrList,
       std::string &descString);

std::vector<int> CTCGreedyDecoderLayer(float * probs,int N_,int T_,int C_)
{
	std::vector<std::vector<int>> res_all;
	std::vector<int> res_sque;
	for (int n = 0; n < N_; ++n) {
		int prev_class_idx = -1;

		for (int t = 0; t<T_; ++t) {
			// get maximum probability and its index
			int max_class_idx = 0;
			
			float max_prob = probs[0];
			++probs;
			for (int c = 1; c < C_; ++c, ++probs) {
				if (*probs > max_prob) {
					max_class_idx = c;
					max_prob = *probs;
				}
			}
			//std::cout<<"res: "<<max_prob<<std::endl;
			if (max_class_idx != 0
				&& !(max_class_idx == prev_class_idx)) 
				{
					//std::cout<<max_class_idx<<" "<<std::endl;
					res_sque.push_back(max_class_idx);
				}
				prev_class_idx = max_class_idx;
			}	
		//res_all.push_back(res_sque);
		}
	
	return res_sque;
}



std::vector<std::string> *InputData_To_Vector()
{
  std::vector<std::string> *p = new std::vector<std::string>;
  //std::ifstream infile("/home/data/installation/deepstream_sdk_v4.0_x86_64/sources/text_yolo4/char_std_5990.txt");
  std::ifstream infile("/home/data/installation/deepstream_sdk_v4.0_x86_64/sources/text_yolo4/char_std_5530.txt");
  std::string number;
  while(! infile.eof())
  {
    infile >> number;
	//std::cout<<number<<std::endl;
    p->push_back(number);
  }
  p->pop_back(); //此处要将最后一个数字弹出，是因为上述循环将最后一个数字读取了两次
  return p;
}


/* extern "C" bool NvDsInferParseCustomText_new(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    int outputBBoxLayerIndex = -1;
	//std::cout<<"@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$$$!!!！！!!!!!!"<<std::endl;
	for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strstr(outputLayersInfo[i].layerName, "1158") != nullptr) 
        //if (strstr(outputLayersInfo[i].layerName, "58") != nullptr) 
		{
            outputBBoxLayerIndex = i;
        }
       
    }
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;
		
	NvDsInferDimsCHW outputBBoxDims;
	//NvDsInferDimsNCHW  ouuttttt;
	getDimsCHWFromDims(outputBBoxDims,
                       outputLayersInfo[outputBBoxLayerIndex].dims);
	
	
	
	
	std::vector<int> char_rec;
	std::vector<std::vector<int>> res_all;
	char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,25,5990);
	//char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,101,5530);
	
	std::vector<std::string> *file_to_vector = InputData_To_Vector();
	//std::cout<<"size:"<<file_to_vector->size()<<std::endl;
	//std::cout<<"result: "<<std::endl;

	

	std::ofstream file;
	if (file.bad())
	{
		std::cout << "cannot open file" << std::endl;;

	}
	file.open("context.txt", std::ios::app);
	

	
		//std::cout<<"结果: "<<std::endl;
		for (int i =0;i<char_rec.size();i++)
		{
			
			std::cout<<file_to_vector->at(char_rec[i])<<" ";
			file << file_to_vector->at(char_rec[i]) <<" ";
		}
		std::cout<<"\n";
	
	
	file <<"\n";
    return true;
}  */

extern "C" bool NvDsInferParseCustomText_new(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
       std::vector<NvDsInferAttribute> &attrList,
       std::string &descString)
{
    int outputBBoxLayerIndex = -1;
	unsigned int numAttributes = outputLayersInfo.size();
    std::cout<<"@@@@@start secondary @@@@@@@@@@$$$$$$$$$$$$$$$$$$$$$!!!!!!"<<std::endl;
	for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strstr(outputLayersInfo[i].layerName, "58") != nullptr) {
            outputBBoxLayerIndex = i;
        }
       
    }
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;
		
	NvDsInferDimsCHW outputBBoxDims;
	//NvDsInferDimsNCHW  ouuttttt;
	getDimsCHWFromDims(outputBBoxDims,
                       outputLayersInfo[outputBBoxLayerIndex].dims);

	std::vector<int> char_rec;
	std::vector<std::vector<int>> res_all;
	//char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,25,5990);//int
	char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,101,5530);//int  作者训的
	//char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,71,5991);//int
	//char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,71,5530);//int
	
	std::vector<std::string> * file_to_vector=InputData_To_Vector();
	
	std::cout<<"result: "<<char_rec.size()<<std::endl;
	std::ofstream file;
	if (file.bad())
	{
		std::cout << "cannot open file" << std::endl;;

	}
	/* file.open("context.txt", std::ios::app);
	for (int i =0;i<char_rec.size();i++)
		{
			
			std::cout<<file_to_vector->at(char_rec[i])<<" ";
			file << file_to_vector->at(char_rec[i]) <<" ";
		}
		std::cout<<"\n"; */
		
/* 	std::string text="中国人我是";
	NvDsInferAttribute attr;
	attr.attributeIndex = 1;
    attr.attributeValue = 0;
    attr.attributeConfidence = 1.0;
	attr.attributeLabel = text.c_str();
	descString.append(attr.attributeLabel).append(" "); */
	
	float maxProbability = -254;
    bool attrFound = false;
    NvDsInferAttribute attr;
	std::string text;
	//char a[char_rec.size()+1];
    //char *text=a;
	for (unsigned int i =0;i<char_rec.size();i++)
		{   
	        
			attrFound = true;
			attr.attributeIndex = char_rec[i];
			
			attr.attributeValue = 0;
			attr.attributeConfidence =1.0;
		//text.append(file_to_vector->at(char_rec[i]));
		text.append(file_to_vector->at(char_rec[i])).append(" ");
		//text+=file_to_vector->at(char_rec[i]);
		//sprintf(text,"%s",(file_to_vector->at(char_rec[i])).c_str());
		}
	
	if(attrFound){
		attr.attributeLabel = text.c_str();
		//attr.attributeLabel = text;
		attrList.push_back(attr);
		 if (attr.attributeLabel)
              descString.append(attr.attributeLabel).append(" ");
		std::cout<<"descStringtring:"<<descString<<std::endl;
		std::cout<<"descStringtring2"<<text.c_str()<<std::endl;
		}
	//file <<"\n";
    return true;
}



//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomText_new);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomText_new);