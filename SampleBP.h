#include "comm_def_col.h"
#pragma once

//����
class NNconnection
{
public:
	NNconnection(void);
	~NNconnection(void);
public: 
	//�����ߵ�Ȩ��������Ȩ�������е�����
	unsigned weightIdx; 
	//���ӵ�ǰ��һ���������
	unsigned neuralIdx; 
};

//�ڵ�
class NNneural
{
public:
	NNneural(void);
	~NNneural(void);
public: 
	//���
	double output;
	//��������
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
	unsigned nLayer;						// �������                                            
	vector<unsigned> nodes;					// ÿ��Ľ����                                 
	vector<double> actualOutput;			// ÿ�ε�����������                      
	double etaLearningRate;					// Ȩֵѧϰ��                                
	unsigned iterNum;						// ��������                                        
public:                                                                     
	vector<NNlayer*> m_layers;															// ���������                               
	void create(unsigned num_layers,unsigned * ar_nodes);								// ��������      
	void initializeNetwork();															// ��ʼ�����磬��������Ȩֵ��                
	void forwardCalculate(vector<double>& invect,vector<double>& outvect);				// ��ǰ����
	void backPropagate(vector<double>& tVect,vector<double>& oVect);					///���򴫲�
	void train(vector<vector<double>>& inputVect,vector<vector<double>>& outputVect);	//ѵ�� 
	void classifer(vector<double>& inVect,vector<double>& outVect);						// ���� 

};





