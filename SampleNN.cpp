#include "SampleNN.h"

//http://www.cnblogs.com/ronny/p/ann_01.html
//http://www.cnblogs.com/ronny/p/ann_02.html
SamleNN::SamleNN(void)
	: actual_output(0)
{
	getSamplesData();
	initialWgt();
}


SamleNN::~SamleNN(void)
{
	inputData.clear();
	weight.clear();
}

//样本获取：
void SamleNN::getSamplesData(void)
{
	const int iterations = 15000; // 15000个样本
	for (int i = 0; i < iterations; i++)
	{
		int index = i % 4;
		vector<double> dvect(4, 0);
		dvect[index] = 1;
		for (size_t i = 0; i != dvect.size(); i++)
		{
			dvect[i] += (5e-3*rand() / RAND_MAX - 2.5e-3);
		}
		inputData.push_back(dvect);
	}
}

//用[0,0.05]之间的随机值初始化权重。
void SamleNN::initialWgt(void)
{
	// 4个连结和一个偏置w0
	for (int i = 0; i != 5; i++)
	{
		weight.push_back(0.05*rand()/RAND_MAX);
	}
}

//向前计算
void SamleNN::cmtForward(const vector<double>& inVect)
{
	double dsum = weight[4];//先把偏置加上
	for (size_t i = 0; i != inVect.size(); i++)
	{
		dsum += (inVect[i] * weight[i]);
	}
	actual_output = 1 / (1 + exp(-1*dsum));
}

//更新权重
void SamleNN::updataWgt(const vector<double>& inVect, const double true_output)
{
	double learnRate = 0.05; // 权重更新参数
	for (size_t i = 0; i != weight.size() - 1; i++)
	{
		weight[i] += (learnRate*(true_output - actual_output)*actual_output*(1 - actual_output)*inVect[i]);
	}
	// w0单独计算
	weight[4] += (learnRate*(true_output - actual_output)*actual_output*(1 - actual_output)*1);
}
