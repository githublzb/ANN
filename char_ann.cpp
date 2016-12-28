#include "comm_def_col.h"

#define HORIZONTAL		1
#define VERTICAL		0

const string g_path = "./training_05/";
const string g_exten = "*.bmp";


static Mat ProjectedHistogram(Mat img, int t)
{
	int sz=(t)?img.rows:img.cols;
	Mat mhist=Mat::zeros(1,sz,CV_32F);

	for(int j=0; j<sz; j++){
		Mat data=(t)?img.row(j):img.col(j);
		mhist.at<float>(j)= countNonZero(data);
	}

	//Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if(max>0)
		mhist.convertTo(mhist,-1 , 1.0f/max, 0);

	return mhist;
}

static Mat features(Mat in, int sizeData){
	//Histogram features
	Mat vhist=ProjectedHistogram(in,VERTICAL);
	Mat hhist=ProjectedHistogram(in,HORIZONTAL);

	//Low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData) );


	//Last 10 is the number of moments components
	int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;

	Mat out=Mat::zeros(1,numCols,CV_32F);
	//Asign values to feature
	int j=0;
	for(int i=0; i<vhist.cols; i++)
	{
		out.at<float>(j)=vhist.at<float>(i);
		j++;
	}
	for(int i=0; i<hhist.cols; i++)
	{
		out.at<float>(j)=hhist.at<float>(i);
		j++;
	}
	for(int x=0; x<lowData.cols; x++)
	{
		for(int y=0; y<lowData.rows; y++){
			out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
			j++;
		}
	}
	return out;
}


static int save_xml()
{
	cv::Directory dir;
	FileStorage fs_train("cny_chars_train.xml", FileStorage::WRITE);
	FileStorage fs_test("cny_chars_test.xml", FileStorage::WRITE);
	Mat train;
	Mat label_1;
	Mat test;
	Mat label_2;
	for (int k = 0; k < 10; k++)
	{
		char buf[8] = {0};
		sprintf(buf, "%02d", k);
		string sub_path = buf;
		vector<string> file_name = dir.GetListFiles(g_path + sub_path, g_exten);
		for (int i = 0; i < 40 && i < file_name.size(); i++) 
		{
			cout<<file_name[i]<<endl;
			Mat image = imread(file_name[i], IMREAD_GRAYSCALE);
			Mat feat = features(image, 5);
			train.push_back(feat);
			label_1.push_back(k);
		}
		for (int i = 40; i < 50 && i < file_name.size(); i++) 
		{
			cout<<file_name[i]<<endl;
			Mat image = imread(file_name[i], IMREAD_GRAYSCALE);
			Mat feat = features(image, 5);
			test.push_back(feat);
			label_2.push_back(k);
		}
	}
	train.convertTo(train, CV_32FC1);

	test.convertTo(test, CV_32FC1);	

	fs_train << "train" << train;
	fs_train << "label" << label_1;
	fs_test << "test" << test;
	fs_test << "label" << label_2;

	fs_train.release();
	fs_test.release();
	return 0;
}

class OCR
{
public:
	OCR();
	OCR(string train_file);
	bool trained;
	int classify(Mat f);
	void train(Mat train, Mat label, int nlayers);
	void test();
protected:
private:
	CvANN_MLP  ann;
};

int OCR::classify(Mat f)
{
	int result=-1;
	Mat output(1, 10, CV_32FC1);
	ann.predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
	//We need know where in output is the max val, the x (cols) is the class.
	return maxLoc.x;
}

void OCR::train(Mat TrainData, Mat label, int nlayers)
{
	Mat layers(1,3,CV_32SC1);
	layers.at<int>(0)= TrainData.cols;
	layers.at<int>(1)= nlayers;
	layers.at<int>(2)= 10;
	ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	//Prepare trainClases
	//Create a mat with n trained data by m classes
	Mat trainClasses;
	trainClasses.create( TrainData.rows, 10, CV_32FC1 );
	for( int i = 0; i <  trainClasses.rows; i++ )
	{
		for( int k = 0; k < trainClasses.cols; k++ )
		{
			//If class of data i is same than a k class
			if( k == label.at<int>(i) )
				trainClasses.at<float>(i,k) = 1;
			else
				trainClasses.at<float>(i,k) = 0;
		}
	}
	Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );

	//Learn classifier
	ann.train( TrainData, trainClasses, weights );
	trained=true;
}

OCR::OCR()
{
	trained=false;
}

OCR::OCR(string train_file)
{
	trained=false;
	//Read file storage.
	FileStorage fs;
	fs.open("cny_chars_train.xml", FileStorage::READ);
	Mat TrainingData;
	Mat Classes;
	fs["train"] >> TrainingData;
	fs["label"] >> Classes;
	train(TrainingData, Classes, 30);
}

void OCR::test()
{
	if (trained == false)
		return;
	//Read file storage.
	FileStorage fs;
	fs.open("cny_chars_test.xml", FileStorage::READ);
	Mat TestData;
	Mat Classes;
	fs["test"] >> TestData;
	fs["label"] >> Classes;
	for (int i = 0; i < TestData.rows; i++)
	{
		Mat data = TestData.row(i);
		int result = classify(data);
		bool ret = (Classes.at<int>(i) == result);
		cout<<"result is:"<<result<<"("<<(ret?"TURN":"FALSE")<<")"<<endl;
	}	
}



int char_ann()
{

	save_xml();
	OCR ocr("aaa");
	ocr.test();
	return 0;
}