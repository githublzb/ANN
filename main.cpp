#include "comm_def_col.h"
extern int NN();
extern int cv_ann();
extern int perceptron();
extern int kmeans_test();
//线性判别分析
extern int cv_lda();

extern int char_ann();

extern int adaptive_thresh( int, string file_name);

int main()
{
//	NN();
//	cv_ann();
//	cv_lda();
//	perceptron();
//	kmeans_test();
//	char_ann();
	adaptive_thresh(1, "fingerprint.png");
	system("pause");
	return 0;
}