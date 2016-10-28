#include "comm_def_col.h"

#define K 5
#define TH 0.02			//阈值
#define SAMPLE_NUM 100	//样本数量
#define HEIGHT	512
#define WIDTH	512
#define RANDOM_X (rand() % WIDTH) //通过取余取得指定范围的随机数
#define RANDOM_Y (rand() % HEIGHT) //通过取余取得指定范围的随机数
#define RANDOM_COLOR (rand() % 255)

typedef struct _feature 
{
	double x;			//特征x
	double y;			//特征y
} FEATURE;

typedef struct _sample 
{
	FEATURE feature;
	int cluster;		//所属的类
} SAMPLE;

typedef struct _cluster
{
	FEATURE center;
	FEATURE pre_center;
	int count;			//样本个数
} CLUSTER;

typedef struct _color
{
	unsigned char val[3];
} COLOR;

static CLUSTER c[K];
static COLOR color[K];
static SAMPLE s[SAMPLE_NUM];
static IplImage *p_image = NULL;

static double dist(FEATURE f1, FEATURE f2)
{
	double x = f1.x - f2.x;
	double y = f1.y - f2.y;
	return static_cast<double>(sqrt(x * x + y * y));
}

static void update_center()
{
	double x[K],y[K];
	memset(x,0,sizeof(x));
	memset(y,0,sizeof(y));
	for(int i = 0; i < SAMPLE_NUM; i++)
	{
		x[s[i].cluster] += s[i].feature.x;
		y[s[i].cluster] += s[i].feature.y;
	}
	for(int i = 0; i < K; i++)
	{
		c[i].pre_center = c[i].center;
		c[i].center.x = x[i] / c[i].count;
		c[i].center.y = y[i] / c[i].count;
		c[i].count = 0;
	}
}

static bool good_result()
{
	for(int i = 0; i < K; i++)
	{
		if(dist(c[i].center,c[i].pre_center) > TH)
			return false;
	}
	return true;
}

static void show_outcome()
{
	unsigned char *data = NULL;
	for(int y = 0; y < HEIGHT; y++)//这里将平面中所有的点都标记，就可以看到平面是怎样被划分的了
	{
		data = (unsigned char *)(p_image->widthStep * y + p_image->imageData);
		for(int x = 0; x < WIDTH; x++)
		{
			double min_dist = 1000;
			int min_k = 0;
			FEATURE f;
			f.x = x;
			f.y = y;
			for(int i = 0; i < K; i++)
			{
				double tmp = dist(c[i].center, f); 
				if(tmp < min_dist)
				{
					min_dist = tmp;
					min_k = i; 
				}
			}
			*(data + (x * p_image->nChannels + 0)) = color[min_k].val[0];
			*(data + (x * p_image->nChannels + 1)) = color[min_k].val[1];
			*(data + (x * p_image->nChannels + 2)) = color[min_k].val[2];
			*(data + (x * p_image->nChannels + 3)) = 200;
// 			IMG_B(img,x,y) = color[min_k].val[0];
// 			IMG_G(img,x,y) = color[min_k].val[1];
// 			IMG_R(img,x,y) = color[min_k].val[2];
// 			IMG_A(img,x,y) = 200;//4通道图像，就是说可以是透明的，纯试验而已，哪知道直接显示没效果，要保存之后才能看出来。
		}
	}
		CvScalar scalar = cvScalar(255,255,255,255);
		for(int i = 0; i < SAMPLE_NUM; i++)//画每个样本点
		{
			int x = static_cast<int>(s[i].feature.x);
			int y = static_cast<int>(s[i].feature.y);
			cvLine(p_image,cvPoint(x - 5,y),cvPoint(x + 5,y),scalar,2);
			cvLine(p_image,cvPoint(x,y - 5),cvPoint(x,y + 5),scalar,2);
		}
		for(int i = 0;i < K; i++)//画聚类中心
		{
			int x = static_cast<int>(c[i].center.x);
			int y = static_cast<int>(c[i].center.y);
			cvCircle(p_image, cvPoint(x,y), 10, scalar,2);
		}
		cvNamedWindow("Kmeans");
		cvShowImage("Kmeans", p_image);
		cvWaitKey(1000);//100毫秒是个差不多的数值，可以完整的看到聚类过程
		cvDestroyWindow("Kmeans");
}

static void init()
{
	srand(time(NULL));
	for (int i = 0; i < SAMPLE_NUM; i++)	//随即生成样本
	{
		s[i].feature.x = RANDOM_X;
		if (s[i].feature.x < 6)
			s[i].feature.x = 6;
		if (s[i].feature.x > 507)
			s[i].feature.x = 507;
		s[i].feature.y = RANDOM_Y;
		if (s[i].feature.y < 6)
			s[i].feature.y = 6;
		if (s[i].feature.y > 507)
			s[i].feature.y = 507;

	}

	for (int i = 0; i < K; i++)				//初始化类的中心和类的颜色
	{
		c[i].center = s[i].feature;
		c[i].pre_center = s[i].feature;
		c[i].pre_center.x += (20 * TH);
		c[i].pre_center.y += (20 * TH);
		c[i].count = 0;

		color[i].val[0] = (unsigned char)RANDOM_COLOR;
		color[i].val[1] = (unsigned char)RANDOM_COLOR;
		color[i].val[2] = (unsigned char)RANDOM_COLOR;
	}
}
int kmeans_test()
{
	int iter_times = 0;//迭代次数
	init();//全局数据初始化
	p_image = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 4);
	cvSetZero(p_image);
	while(!good_result())//检查是否是需要的聚类中心
	{
		for(int i = 0; i < SAMPLE_NUM; i++)
		{
			double min_dist = 10000;
			int min_k = 0;
			for(int j = 0; j < K; j++)
			{
				double tmp = dist(c[j].center, s[i].feature); 
				if(tmp < min_dist)
				{
					min_dist = tmp;
					min_k = j; 
				}
			}
			s[i].cluster = min_k;//确定样本所属的新类
			c[min_k].count++;//更新该类中样本的个数
		}
		update_center();//更新聚类中心
		iter_times++;
		show_outcome();
	}
	cvReleaseImage(&p_image);
	cvWaitKey();
	return 0;
}

