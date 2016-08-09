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
	//Ȩֵ����ʼ����Ĵ�СΪ5������weight[4]Ϊƫ��
	vector<double> weight;
public:
	vector<vector<double>> inputData;	
	double actual_output;

};

