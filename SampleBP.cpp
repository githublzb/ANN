#include "SampleBP.h"


NNlayer::NNlayer(void)
{
}


NNlayer::~NNlayer(void)
{
}


void NNlayer::addNeurals(unsigned num, unsigned preNumNeural) 
{ 
	for (vector<NNneural>::size_type i = 0; i != num; i++) 
	{ 
		NNneural sneural; 
		sneural.output = 0; 
		for (vector<NNconnection>::size_type k = 0; k != preNumNeural+1; k++) 
		{ 
			NNconnection sconnection; 
			sconnection.weightIdx = i*(preNumNeural + 1) + k; // 设置权重索引 
			sconnection.neuralIdx = k;    // 设置前层结点索引 
			sneural.m_connection.push_back(sconnection); 
		} 
		m_neurals.push_back(sneural); 
	} 
}   


void NNlayer::backPropagate(vector<double>& dErrWrtDxn,vector<double>& dErrWrtDxnm,double eta)
{
	double output;
	vector<double> dErrWrtDyn(dErrWrtDxn.size());
	for (vector<NNneural>::size_type i = 0; i != m_neurals.size(); i++)
	{
		output = m_neurals[i].output;
		dErrWrtDyn[i] = DSIGMOID(output)*dErrWrtDxn[i];
	}
	unsigned ii(0);
	vector<NNneural>::iterator nit = m_neurals.begin();
	vector<double> dErrWrtDwn(m_weights.size(),0);
	while(nit != m_neurals.end())
	{
		for (vector<NNconnection>::size_type k = 0; k != (*nit).m_connection.size(); k++)
		{
			if (k == (*nit).m_connection.size() - 1)
				output = 1;
			else
				output = preLayer->m_neurals[(*nit).m_connection[k].neuralIdx].output;
			dErrWrtDwn[(*nit).m_connection[k].weightIdx] += output*dErrWrtDyn[ii];
		}

		++nit;
		++ii;
	}
	unsigned j(0);
	nit = m_neurals.begin();
	while (nit != m_neurals.end())
	{
		for (vector<NNconnection>::size_type k = 0; k != (*nit).m_connection.size()-1; k++)
		{
			dErrWrtDxnm[(*nit).m_connection[k].neuralIdx] += dErrWrtDyn[j] * m_weights[(*nit).m_connection[k].weightIdx];
		}
		++j;
		++nit;
	}
	for (vector<double>::size_type i = 0; i != m_weights.size(); i++)
	{    
		m_weights[i] -= eta*dErrWrtDwn[i];
	}
} 

NNneural::NNneural(void)
{
}


NNneural::~NNneural(void)
{
}


NNconnection::NNconnection(void)
{
}


NNconnection::~NNconnection(void)
{
}


NeuralNetwork::NeuralNetwork(void)
{
}


NeuralNetwork::~NeuralNetwork(void)
{
}

void NeuralNetwork::initializeNetwork() 
{ 
	// 初始化网络，主要是创建各层和各层的结点，并给权重向量赋初值 
	for (vector<NNlayer*>::size_type i = 0; i != nLayer; i++) 
	{ 
		NNlayer* ptrLayer = new NNlayer; 
		if (i == 0) 
		{ 
			ptrLayer->addNeurals(nodes[i],0); 
		} 
		else 
		{ 
			ptrLayer->preLayer = m_layers[i - 1]; 
			ptrLayer->addNeurals(nodes[i],nodes[i-1]); 
			unsigned num_weights = nodes[i] * (nodes[i-1]+1); // 有一个是bias 
			for (vector<double>::size_type k = 0; k != num_weights; k++) 
			{ 
				// 初始化权重在0~0.05 
				ptrLayer->m_weights.push_back(0.05*rand()/RAND_MAX); 
			} 
		} 
		m_layers.push_back(ptrLayer); 
	} 
}   


void NeuralNetwork::forwardCalculate(vector<double>& invect, vector<double>& outvect)
{
	actualOutput.clear();
	vector<NNlayer*>::iterator layerIt = m_layers.begin();
	while (layerIt != m_layers.end())
	{
		if (layerIt == m_layers.begin())
		{
			// 第一层
			for (vector<NNneural>::size_type k = 0; k != (*layerIt)->m_neurals.size(); k++)
			{
				(*layerIt)->m_neurals[k].output = invect[k];
			}
		}
		else
		{
			vector<NNneural>::iterator neuralIt = (*layerIt)->m_neurals.begin();
			int neuralIdx = 0;
			while (neuralIt != (*layerIt)->m_neurals.end())
			{
				vector<NNconnection>::size_type num_connection = (*neuralIt).m_connection.size();
				double dsum = (*layerIt)->m_weights[num_connection*(neuralIdx + 1) - 1]; // 先将偏置加上
				for (vector<NNconnection>::size_type i = 0; i != num_connection - 1; i++)
				{
					// sum=sum of xi*wi
					unsigned wgtIndex = (*neuralIt).m_connection[i].weightIdx;
					unsigned neuIndex = (*neuralIt).m_connection[i].neuralIdx;
					dsum += ((*layerIt)->preLayer->m_neurals[neuIndex].output*(*layerIt)->m_weights[wgtIndex]);
				}
				neuralIt->output = SIGMOID(dsum);
				neuralIdx++;
				neuralIt++;
			}
		}
		++layerIt;
	}
	// 将最后一层的结果传递给输出
	NNlayer* lastLayer = m_layers[m_layers.size() - 1];
	vector<NNneural>::iterator neuralIt = lastLayer->m_neurals.begin();
	while (neuralIt != lastLayer->m_neurals.end())
	{
		outvect.push_back(neuralIt->output);
		++neuralIt;
	}
}   



void NeuralNetwork::backPropagate(vector<double>& tVect, vector<double>& oVect)
{
	// lit是最后一层的迭代器
	vector<NNlayer*>::iterator lit = m_layers.end() - 1;
	// dErrWrtDxLast是最后一层所有结点的误差
	vector<double> dErrWrtDxLast((*lit)->m_neurals.size());
	// 所有层的误差
	vector<vector<double>> diffVect(nLayer);
	for (vector<NNneural>::size_type i = 0; i != (*lit)->m_neurals.size();i++)
	{
		dErrWrtDxLast[i] = oVect[i] - tVect[i];
	}
	diffVect[nLayer - 1] = dErrWrtDxLast;
	// 先将其他层的误差都设为0
	for (unsigned i = 0; i < nLayer - 1; i++)
	{
		diffVect[i].resize(m_layers[i]->m_neurals.size(),0.0);
	}

	vector<NNlayer*>::size_type i = m_layers.size()-1;
	for (lit; lit>m_layers.begin(); lit--)
	{
		(*lit)->backPropagate(diffVect[i],diffVect[i-1],etaLearningRate);
		--i;
	}
	diffVect.clear();
}





