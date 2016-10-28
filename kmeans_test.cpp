#include "comm_def_col.h"

#define K 5
#define TH 0.02			//��ֵ
#define SAMPLE_NUM 100	//��������
#define HEIGHT	512
#define WIDTH	512
#define RANDOM_X (rand() % WIDTH) //ͨ��ȡ��ȡ��ָ����Χ�������
#define RANDOM_Y (rand() % HEIGHT) //ͨ��ȡ��ȡ��ָ����Χ�������
#define RANDOM_COLOR (rand() % 255)

typedef struct _feature 
{
	double x;			//����x
	double y;			//����y
} FEATURE;

typedef struct _sample 
{
	FEATURE feature;
	int cluster;		//��������
} SAMPLE;

typedef struct _cluster
{
	FEATURE center;
	FEATURE pre_center;
	int count;			//��������
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
	for(int y = 0; y < HEIGHT; y++)//���ｫƽ�������еĵ㶼��ǣ��Ϳ��Կ���ƽ�������������ֵ���
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
// 			IMG_A(img,x,y) = 200;//4ͨ��ͼ�񣬾���˵������͸���ģ���������ѣ���֪��ֱ����ʾûЧ����Ҫ����֮����ܿ�������
		}
	}
		CvScalar scalar = cvScalar(255,255,255,255);
		for(int i = 0; i < SAMPLE_NUM; i++)//��ÿ��������
		{
			int x = static_cast<int>(s[i].feature.x);
			int y = static_cast<int>(s[i].feature.y);
			cvLine(p_image,cvPoint(x - 5,y),cvPoint(x + 5,y),scalar,2);
			cvLine(p_image,cvPoint(x,y - 5),cvPoint(x,y + 5),scalar,2);
		}
		for(int i = 0;i < K; i++)//����������
		{
			int x = static_cast<int>(c[i].center.x);
			int y = static_cast<int>(c[i].center.y);
			cvCircle(p_image, cvPoint(x,y), 10, scalar,2);
		}
		cvNamedWindow("Kmeans");
		cvShowImage("Kmeans", p_image);
		cvWaitKey(1000);//100�����Ǹ�������ֵ�����������Ŀ����������
		cvDestroyWindow("Kmeans");
}

static void init()
{
	srand(time(NULL));
	for (int i = 0; i < SAMPLE_NUM; i++)	//�漴��������
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

	for (int i = 0; i < K; i++)				//��ʼ��������ĺ������ɫ
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
	int iter_times = 0;//��������
	init();//ȫ�����ݳ�ʼ��
	p_image = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 4);
	cvSetZero(p_image);
	while(!good_result())//����Ƿ�����Ҫ�ľ�������
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
			s[i].cluster = min_k;//ȷ����������������
			c[min_k].count++;//���¸����������ĸ���
		}
		update_center();//���¾�������
		iter_times++;
		show_outcome();
	}
	cvReleaseImage(&p_image);
	cvWaitKey();
	return 0;
}

