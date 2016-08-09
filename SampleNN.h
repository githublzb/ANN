#pragma once
#include<vector>
using namespace std;


class SamleNN
{
public:
	SamleNN(void);
	~SamleNN(void);
	void getSamplesData(void);
	void initialWgt(void);
	void cmtForward(const vector<double>& inVect);
	void updataWgt(const vector<double>& inVect, const double true_output);
private:
	//权值，初始化后的大小为5，其中weight[4]为偏置
	vector<double> weight;
public:
	vector<vector<double>> inputData;	
	double actual_output;

};

