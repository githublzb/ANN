#include "comm_def_col.h"
#pragma once

//连线
class NNconnection
{
public:
	NNconnection(void);
	~NNconnection(void);
public: 
	//连接线的权重在整个权重向量中的索引
	unsigned weightIdx; 
	//连接的前面一层结点的索引
	unsigned neuralIdx; 
};

//节点
class NNneural
{
public:
	NNneural(void);
	~NNneural(void);
public: 
	//输出
	double output;
	//若干连线
	vector<NNconnection> m_connection;
};


class NNlayer
{
public:
//	NNlayer(void);
	~NNlayer(void);
public:                                                                                      
	NNlayer(){ preLayer = NULL; }                                                            
	NNlayer *preLayer;                                                                       
	vector<NNneural> m_neurals;                                                              
	vector<double> m_weights;                                                                
	void addNeurals(unsigned num, unsigned preNumNeurals);                                   
	void backPropagate(vector<double>& dErrWrtDxn, vector<double>& dErrWrtDxnm, double eta);
};

class NeuralNetwork
{
public:
	NeuralNetwork(void);
	~NeuralNetwork(void);
private:                                                                    
	unsigned nLayer;						// 网络层数                                            
	vector<unsigned> nodes;					// 每层的结点数                                 
	vector<double> actualOutput;			// 每次迭代的输出结果                      
	double etaLearningRate;					// 权值学习率                                
	unsigned iterNum;						// 迭代次数                                        
public:                                                                     
	vector<NNlayer*> m_layers;															// 整个网络层                               
	void create(unsigned num_layers,unsigned * ar_nodes);								// 创建网络      
	void initializeNetwork();															// 初始化网络，包括设置权值等                
	void forwardCalculate(vector<double>& invect,vector<double>& outvect);				// 向前计算
	void backPropagate(vector<double>& tVect,vector<double>& oVect);					///反向传播
	void train(vector<vector<double>>& inputVect,vector<vector<double>>& outputVect);	//训练 
	void classifer(vector<double>& inVect,vector<double>& outVect);						// 分类 

};





