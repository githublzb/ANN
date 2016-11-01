#include "comm_def_col.h"
extern int NN();
extern int cv_ann();
extern int perceptron();
extern int kmeans_test();
//线性判别分析
extern int cv_lda();

int main()
{
//	NN();
//	cv_ann();
	cv_lda();
//	perceptron();
//	kmeans_test();
	return 0;
}