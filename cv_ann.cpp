#include "comm_def_col.h"
//http://www.cnblogs.com/mfryf/archive/2013/05/29/3105262.html
//http://blog.csdn.net/lpr_pro/article/details/4382917
//http://blog.csdn.net/xiaowei_cqu/article/details/9027617
//车牌识别
//http://blog.jobbole.com/84234/
//http://www.cnblogs.com/mfryf/archive/2013/05/29/3105262.html

int cv_ann()
{
	//Setup the BPNetwork  
	CvANN_MLP bp;   
	// Set up BPNetwork's parameters  
	CvANN_MLP_TrainParams params;  
	params.train_method=CvANN_MLP_TrainParams::BACKPROP;  //(Back Propagation,BP)反向传播算法
	params.bp_dw_scale=0.1;  
	params.bp_moment_scale=0.1;  

	// Set up training data  
	float labels[10][2] = {{0.9,0.1},{0.1,0.9},{0.9,0.1},{0.1,0.9},{0.9,0.1},{0.9,0.1},{0.1,0.9},{0.1,0.9},{0.9,0.1},{0.9,0.1}};  
	//这里对于样本标记为0.1和0.9而非0和1，主要是考虑到sigmoid函数的输出为一般为0和1之间的数，只有在输入趋近于-∞和+∞才逐渐趋近于0和1，而不可能达到。
	Mat labelsMat(10, 2, CV_32FC1, labels);  

	float trainingData[10][2] = { {11,12},{111,112}, {21,22}, {211,212},{51,32}, {71,42}, {441,412},{311,312}, {41,62}, {81,52} };  
	Mat trainingDataMat(10, 2, CV_32FC1, trainingData);  
 	Mat layerSizes=(Mat_<int>(1,5) << 2, 2, 2, 2, 2); //5层：输入层，3层隐藏层和输出层，每层均为两个perceptron
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM ,选用sigmoid作为激励函数
	bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);  //训练

	// Data for visual representation  
	int width = 512, height = 512;  
	Mat image = Mat::zeros(height, width, CV_8UC3);  
	Vec3b green(0,255,0), blue (255,0,0);  
	// Show the decision regions
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)  
		{  
			Mat sampleMat = (Mat_<float>(1,2) << i,j);  
			Mat responseMat;  
			bp.predict(sampleMat,responseMat);  
			float* p=responseMat.ptr<float>(0);  
			//
			if (p[0] > p[1])
			{
				image.at<Vec3b>(j, i)  = green;  
			} 
			else
			{
				image.at<Vec3b>(j, i)  = blue;  
			}
		}  
	}
	// Show the training data  
	int thickness = -1;  
	int lineType = 8;  
	circle( image, Point(111,  112), 5, Scalar(  0,   0,   0), thickness, lineType); 
	circle( image, Point(211,  212), 5, Scalar(  0,   0,   0), thickness, lineType);  
	circle( image, Point(441,  412), 5, Scalar(  0,   0,   0), thickness, lineType);  
	circle( image, Point(311,  312), 5, Scalar(  0,   0,   0), thickness, lineType);  
	circle( image, Point(11,  12), 5, Scalar(255, 255, 255), thickness, lineType);  
	circle( image, Point(21, 22), 5, Scalar(255, 255, 255), thickness, lineType);       
	circle( image, Point(51,  32), 5, Scalar(255, 255, 255), thickness, lineType);  
	circle( image, Point(71, 42), 5, Scalar(255, 255, 255), thickness, lineType);       
	circle( image, Point(41,  62), 5, Scalar(255, 255, 255), thickness, lineType);  
	circle( image, Point(81, 52), 5, Scalar(255, 255, 255), thickness, lineType);       

	imwrite("result.png", image);        // save the image   

	imshow("BP Simple Example", image); // show it to the user  
	waitKey(0); 
	return 0;
}
