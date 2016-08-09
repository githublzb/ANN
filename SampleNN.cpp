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

//������ȡ��
void SamleNN::getSamplesData(void)
{
	const int iterations = 15000; // 15000������
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

//��[0,0.05]֮������ֵ��ʼ��Ȩ�ء�
void SamleNN::initialWgt(void)
{
	// 4�������һ��ƫ��w0
	for (int i = 0; i != 5; i++)
	{
		weight.push_back(0.05*rand()/RAND_MAX);
	}
}

//��ǰ����
void SamleNN::cmtForward(const vector<double>& inVect)
{
	double dsum = weight[4];//�Ȱ�ƫ�ü���
	for (size_t i = 0; i != inVect.size(); i++)
	{
		dsum += (inVect[i] * weight[i]);
	}
	actual_output = 1 / (1 + exp(-1*dsum));
}

//����Ȩ��
void SamleNN::updataWgt(const vector<double>& inVect, const double true_output)
{
	double learnRate = 0.05; // Ȩ�ظ��²���
	for (size_t i = 0; i != weight.size() - 1; i++)
	{
		weight[i] += (learnRate*(true_output - actual_output)*actual_output*(1 - actual_output)*inVect[i]);
	}
	// w0��������
	weight[4] += (learnRate*(true_output - actual_output)*actual_output*(1 - actual_output)*1);
}
