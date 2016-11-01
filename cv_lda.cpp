#include "comm_def_col.h"
//http://www.blogbus.com/shijuanfeng-logs/220816641.html
//http://blog.csdn.net/jialeheyeshu/article/details/52822010
int cv_lda(void)  
{  
	//sampledata  
	double sampledata[6][2]={{0,1},{0,2},{2,4},{8,0},{8,2},{9,4}};  
	Mat mat=Mat(6,2,CV_64FC1,sampledata);  
	//labels  
	vector<int>labels;  

	for(int i=0;i<mat.rows;i++)  
	{  
		if(i<mat.rows/2)  
		{  
			labels.push_back(0);  
		}  
		else  
		{  
			labels.push_back(1);  
		}  
	}  

	//do LDA  
	//��ʼ�������㣬���캯���д��м���  
	LDA lda=LDA(mat,labels,1);  
	//get the eigenvector  
	//�����������  
	Mat eivector=lda.eigenvectors().clone();  

	cout<<"����������double������:"<<endl;  
	for(int i=0;i<eivector.rows;i++)  
	{  
		for(int j=0;j<eivector.cols;j++)  
		{  
			cout<<eivector.ptr<double>(i)[j]<<" ";  
		}  
		cout<<endl;  
	}  


	//------------------------------������������------------  
	//�������������⣬�����������ݼ�������  
	int classNum=2;  
	vector<Mat> classmean(classNum);  
	vector<int> setNum(classNum);  

	for(int i=0;i<classNum;i++)  
	{  
		classmean[i]=Mat::zeros(1,mat.cols,mat.type());  //��ʼ�����о�ֵΪ0  
		setNum[i]=0;  //ÿһ���е���Ŀ��  
	}  

	Mat instance;  
	for(int i=0;i<mat.rows;i++)  
	{  
		instance=mat.row(i);//��ȡ��i��  
		if(labels[i]==0)  //�����ǩΪ0  
		{     
			add(classmean[0], instance, classmean[0]);  //�������  
			setNum[0]++;  //�������  
		}  
		else if(labels[i]==1)  //���ڵ�1��Ĵ���  
		{  
			add(classmean[1], instance, classmean[1]);  
			setNum[1]++;  
		}  
		else  
		{}  
	}  
	for(int i=0;i<classNum;i++)   //����ÿһ��ľ�ֵ  
	{  
		classmean[i].convertTo(classmean[i],CV_64FC1,1.0/static_cast<double>(setNum[i]));  
	}  
	//----------------------------------END��������-------------------------  


	vector<Mat> cluster(classNum);  //һ��2��  


	cout<<"����������"<<endl;  
	cout<<eivector<<endl;   //��ʱ������������һ��������  


	cout<<endl<<endl;  
	cout<<"��һ�ַ�ʽ(�ֶ�����)��"<<endl;  
	//1.ͶӰ�ĵ�һ�ַ�ʽ��Y=X*W  
	//�еĽ̳�д��Y=W^T*X,����ʱ��X��������������������Ҫ��wת�ã�  
	for(int i=0;i<classNum;i++)  
	{  
		cluster[i]=Mat::zeros(1,1,mat.type()); //��ʼ��0  
		//����������ת��ͬ���ֵ���)  
		cluster[i]=classmean[i]*eivector;  
	}  

	cout<<"The project cluster center is:"<<endl;  //�����ֵ��ͶӰ  
	for(int i=0;i<classNum;i++) //����������ĵ�ͶӰֵ  
	{  
		cout<<cluster[i].at<double>(0)<<endl;  
	}  

	//2.�ڶ��ַ�ʽʹ�����ú�������  
	//��һ������  
	cout<<endl<<"�ڶ��ַ�ʽ:";  
	cout<<endl<<"��һ�����ֵ��ͶӰ:"<<endl;  
	cout<<lda.project(classmean[0]).at<double>(0);  
	cout<<endl<<"�ڶ������ֵ��ͶӰ"<<endl;  
	cout<<lda.project(classmean[1]).at<double>(0);  


	system("pause");  
	return 0;  
}  