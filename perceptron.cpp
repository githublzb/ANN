#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
using namespace std;
#include <cv.h>
#include <cxcore.h>
#include <cxerror.h>
#include <highgui.h>
#include <ml.h>
#include <cxmisc.h>

//PerceptronLearning Algorithm(PLA)

const double alpha = 1.5;       //学习参数

#define COUNT	50
#define HEIGHT  512  
#define WIDTH   512  
#define RANDOM_X (rand() % WIDTH) //通过取余取得指定范围的随机数  
#define RANDOM_Y (rand() % HEIGHT) //通过取余取得指定范围的随机数  


CvMat *mat_point = NULL;
CvMat *mat_weight = NULL;

int init_data()
{
	int i, j;
	int *data;
	//初始化样本
	mat_point = cvCreateMat(COUNT, 4, CV_32SC1);
	cvSetZero(mat_point);
	srand(time(NULL));  
	for (i = 0; i< mat_point->height; i++)
	{
		data = (int *)(mat_point->data.ptr + i * mat_point->step);
		data[0] = RANDOM_X;
		data[1] = RANDOM_Y;
		data[2] = 1;
		if (data[0] < 200)
			data[3] = 1;
		else
			data[3] = -1;
	}
	//初始化权值
	mat_weight = cvCreateMat(1, 4, CV_64FC1);
	cvSetZero(mat_weight);

	return 0;
}

int release_data()
{
	cvReleaseMat(&mat_point);
	cvReleaseMat(&mat_weight);
	return 0;
}

int show_update(int delay)
{
	int i;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	double x_temp;
	int *point;
	double *weight;
	IplImage *p_image = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
	cvZero(p_image);
	//绘制样本点
	CvScalar scalar = cvScalar(255, 255, 255, 0);
	for (i = 0; i < COUNT; i++)
	{
		point = (int *)(mat_point->data.ptr + mat_point->step * i);
		if (point[3] == 1)
		{
			scalar = cvScalar(0, 0, 255, 0);
		}
		else
			scalar = cvScalar(0, 255, 0, 0);
		cvLine(p_image, cvPoint(point[0] - 2, point[1]),cvPoint(point[0] + 2, point[1]), scalar, 1);
		cvLine(p_image, cvPoint(point[0], point[1] - 2),cvPoint(point[0], point[1] + 2), scalar, 1);
	}
	//绘制分界面
	weight = mat_weight->data.db;
	if (weight[0] < -0.0001 || weight[0] > 0.0001)
	{
		for (y1 = 0; y1 < HEIGHT; y1++)
		{
			x_temp = (-1 * weight[2] - y1 * weight[1])/weight[0];
			if (x_temp >= 0 && x_temp < HEIGHT)
			{
				x1 = x_temp;
				break;
			}
		}

	}
	if (x1 < 0 || x1 >= HEIGHT)
	{
		x1 = 0;
	}

	if (weight[0] < -0.0001 || weight[0] > 0.0001)
	{
		for (y2 = HEIGHT -1 ; y2 > 0; y2--)
		{
			x_temp = (-1 * weight[2] - y2 * weight[1])/weight[0];
			if (x_temp >= 0 && x_temp < HEIGHT)
			{
				x2 = x_temp;
				break;
			}
		}
	}
	if (x2 < 0 || x2 >= HEIGHT)
	{
		x2 = 0;
	}

	scalar = cvScalar(255, 0, 0, 0);
	cvLine(p_image, cvPoint(x1, y1),cvPoint(x2, y2), scalar, 2);
	//显示
	cvNamedWindow("perceptron", 1);
	cvShowImage("perceptron", p_image);
	cvWaitKey(delay);
	cvDestroyWindow("perceptron");
	cvReleaseImage(&p_image);
	return 0;
}

int compute(int *point, double *weight)
{
	double sum =0.0;
	int i;
	for (i = 0; i < 3; ++i)
	{
		sum += point[i] * weight[i];
	}
	if(sum > 0.0)
		return 1;
	else
		return -1;
}


int perceptron()
{
	bool bLearningOK = false;
	int count = 0;
	int *point;
	double *weight;
	//感知器学习算法
	init_data();
	while(!bLearningOK)
	{
		bLearningOK = true;
		for (int i = 0 ; i < COUNT ; ++i)
		{
			point = (int *)(mat_point->data.ptr + mat_point->step * i);
			weight = mat_weight->data.db;
			//计算输出，比较后矫正权值
			int output = compute(point,weight);
			if(output!= point[3])
			{
				for(int w = 0 ; w <3 ; ++w)
				{
					weight[w] += alpha * point[3] * point[w];
				}
				bLearningOK = false;
			}
		}
		count++;
// 		cout<<count<<endl;
//  	show_update(50);
	}
	show_update(0);
	release_data();
	return 0;
}

int main()
{
	perceptron();
	return 0;
}