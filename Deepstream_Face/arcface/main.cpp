#include "arcfacem.h"
#include<iostream>
#include<vector>
using namespace std;
int main()
{
	arcfacem af;
	af.WTSToEngine("../arcface-mobilefacenet.wts","../arcface-mobilefacenet.engine");
	af.Init("../arcface-mobilefacenet.engine");
	float *p1=new float[128];
	af.Inference_file("../joey0.ppm",p1);
	float *p2=new float[128];
	af.Inference_file("../joey1.ppm",p2);
	float ret = af.Compare(p1,p2);
	std::cout<<"compare result:"<<ret<<std::endl;
	af.UnInit();
	return 0;
}
