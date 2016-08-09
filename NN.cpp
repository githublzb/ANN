#include "comm.hpp"
#include "SampleNN.h"

int NN()
{
	int i;
	SamleNN nn;
	double data;
	for (i = 0; i < 15000; i++)
	{
		data = ((i % 4) + 1) * 0.25;
		nn.cmtForward( nn.inputData[i]);
		nn.updataWgt(nn.inputData[i], data);
		if (i > 14000)
		{
			cout<<"input: " \
				<<nn.inputData[i][0]<<" " \
				<<nn.inputData[i][1]<<" " \
				<<nn.inputData[i][2]<<" " \
				<<nn.inputData[i][3]<<" " \
				<<"output:" \
				<<nn.actual_output<<endl;
		}
	}
	return 0;
}

int main()
{
	NN();
	return 0;
}