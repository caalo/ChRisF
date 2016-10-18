#ifndef __CRFMODEL
#define __CRFMODEL

#include <iostream>
#include <algorithm>  

#include "crfmath.h"
#include <map>
#include <string>
#include "crflinearalgebra.h"
#include "crfchunkkeys.h"
#include <limits>
#include <cstring>
#include <assert.h> 

using namespace std;

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;

typedef vector<vector<vector<int> > > Ukeys; //indexed by nodes, states
enum LossFunction {NotLearning, Bayes, Pairwise, CrossEntropyAdadelta, Probit, PairwiseCrossEntropy, CrossEntropy, CrossEntropyMomentum, CrossEntropyUniform};

struct Potentials {
	vector<Mat> M;
	Mat unary;
	Mat beta;
};

class CRFChunkModel {
	private:

	int nStates, sqStates;
	LossFunction lossFunction;
	float *features, *expectedFeatures, *weights;
	float *weights2; //used only for probit, pairwise
	float *expectedFeatures2; //used only for pairwise 
	float *features2; //used only for pairwise
	float *blankweights; //used only for crossentropy uniform
	double ZNorm;
	vector<Ukeys> allUnaryKeys;

	float *gradient;

	/*accumulated gradient, update. For ADADELTA learning rate implemenation.
	Cross entropy for now. */
	//CrossEntropyMomentum also uses accuUpdate, lastUpdated, t
	float *accuGrad;
	float *accuUpdate;
	int *lastUpdated;
	int t;
	int clipcount;
	float clip;

	//debug
	//vector<float> update1, update2, rate1, rate2;
	//vector<float> overallUpdate, overallRate;
	//vector<float> sampleProbs;
	//vector<float> ratio;
	//vector<float> samplefeedback;

	
	float lrate;
	float lambda; //regularizer

	int DEFAULT_WEIGHT;

	void getPotentials(float *w, Potentials &p, int nNodes, Ukeys& ukeys) {
		//first, construct unary potentials
		vector<int> keys;
		p.unary.zeros(nNodes, nStates);
		for(int i = 0; i < nNodes; i++) {
			for(int j = 0; j < nStates; j++) {	
				for (int k : ukeys[i][j]) {
					assert(w[k] != std::numeric_limits<double>::infinity());
					assert(w[k] != -std::numeric_limits<double>::infinity());
					p.unary(i, j) += w[k];
				}
			}
		}
		
		//second, construct pairwise potentials.
		Mat pairwise(nStates, nStates);
		for(int i = 0; i < nStates; i++) {
			for(int j = 0; j < nStates; j++) { 
				keys.clear();
				ChunkKeys::getPairwise(keys, i, j);
				if(keys.empty())
					pairwise(i, j) = -std::numeric_limits<double>::infinity(); //these are prohibited transitions.
				for(int k : keys) {
					assert(w[k] != std::numeric_limits<double>::infinity());
					assert(w[k] != -std::numeric_limits<double>::infinity());
					pairwise(i, j) += w[k];
				}
			}
		}

		//now, put both together in M matrix of potentials.
		p.M.resize(nNodes);
		for (int i = 1; i < nNodes; i++) {	
			p.M[i].zeros(nStates, nStates);
			for (int j = 0; j < nStates; j++) {
				p.M[i].row(j) = (pairwise.row(j) + p.unary.row(i));
			}
		}

/*
		Vec end(nStates); //copy over end weights...
		for(int i = 0; i < nStates; i++) {
			assert(weights[ChunkKeys::getEnd(i)] != std::numeric_limits<double>::infinity());
			assert(weights[ChunkKeys::getEnd(i)] != -std::numeric_limits<double>::infinity());
			end(i) = weights[ChunkKeys::getEnd(i)];
		}

		//now, put both together in M matrix of potentials.
		p.M.resize(nNodes);
		for (int i = 1; i < nNodes; i++) {	
			p.M[i].zeros(nStates, nStates);
			if(i == nNodes - 1) { //if the last node, we add the end weights to the potentials.
				for (int j = 0; j < nStates; j++) {
					p.M[i].row(j) = (pairwise.row(j) + p.unary.row(i) + end);
				}
			} else { //otherwise, we're fine.
				for (int j = 0; j < nStates; j++) {
					p.M[i].row(j) = (pairwise.row(j) + p.unary.row(i));
				}
			}
		}
*/
	}




	void getFeatures(float *f, StrMat &Xdata, Vec &Ydata, Ukeys& ukeys) {
		int nNodes = Ydata.size();
		vector<int> keys;
		
		for(int i = 1; i < nNodes; i++) {
			keys.clear();
			ChunkKeys::getPairwise(keys, Ydata(i - 1), Ydata(i));
			for(int k: keys)
				f[k] += 1;
		}


		for(int i = 0; i < nNodes; i++) {
			for(int k : ukeys[i][(int)Ydata(i)]) { 
				f[k] += 1;
			} 
		}

		f[ChunkKeys::getStart(Ydata(0))] = 1;
		f[ChunkKeys::getEnd(Ydata(nNodes - 1))] = 1;
	
	}


	void getExpectedFeatures(float *ef, Potentials& p, Ukeys& ukeys) { 
		int nNodes = p.M.size();

		/*
			***A NOTE ABOUT INDEXING***: We are following Laffetry's indexing MINUS ONE, 
			as we don't use start and stop states and it does not make sense to
			index everything starting from 1. We incorporate start and stop
			weights separately. 
			Thus: 
			M, P1 are indexed from 1,...,nNodes - 1. M.slice(0), P1.slice(0) are never used!
			alpha, beta are indexed from 0,...,nNodes - 1
			Y, X, are indexed from 0,...,nNodes - 1

		*/

		//From now on forwards, all work in log space to prevent underflow. In paper, we have M = exp(M), but log(exp(M)) = M. So leave as is.
		vector<Mat> &M = p.M;
		Mat &unaryPotentials = p.unary;

		//alpha
		Mat alpha;
		alpha.ones(nNodes, nStates); //log space
		alpha = alpha * -std::numeric_limits<double>::infinity(); 
		for(int i = 0; i < nStates; i++) 
			//alpha(0, i) = weights[ChunkKeys::getStart(i)] + unaryPotentials(0, i); 
			alpha(0, i) = unaryPotentials(0, i); 

		for (int t = 1; t < nNodes; t++) { //recurse
			alpha.row(t) = CRFMath::logDotVM(alpha.row(t - 1), M[t]);  //alpha.row(t) = alpha.row(t - 1) * M.slice(t); 
		}
		//beta
		Mat beta;
		beta.ones(nNodes, nStates); //Work in log space to prevent underflow.
		beta = beta * -std::numeric_limits<double>::infinity();
		//Vec end(nStates); //copy over end weights...
		//for(int i = 0; i < nStates; i++) 
		//	end(i) = weights[ChunkKeys::getEnd(i)];
		//beta.row(nNodes - 1) = end;
		beta.row(nNodes - 1).zeros(nStates);

		for (int t = nNodes - 2; t >= 0; t--) { //recurse
			beta.row(t) = CRFMath::logDotMV(M[t + 1], beta.row(t + 1)); 
		}
		p.beta = beta; //TODO THIS NEEDS REDOING

		//Znorm
		//double ZNorm = CRFMath::logDotProd(alpha.row(nNodes - 1), end); 
		ZNorm = CRFMath::logSumExp(alpha.row(nNodes - 1));

		//Probability matrix P1: P(y', y|x)
		vector<Mat> P1(nNodes);
		for (int i = 1; i < nNodes; i++) {
			P1[i].zeros(nStates, nStates);
			for (int yp = 0; yp < nStates; yp++) {
				for (int y = 0; y < nStates; y++) {
					P1[i](yp, y) = alpha(i - 1, yp) + M[i](yp, y) + beta(i, y); 
					P1[i](yp, y) -= ZNorm;
				}
			}
		}

		//Probability matrix P2: P(y_i | x)
		Mat P2(nNodes, nStates);
		for (int i = 0; i < nNodes; i++) {
			for (int y = 0; y < nStates; y++) {
				P2(i, y) = alpha(i, y) + beta(i, y); 
				P2(i, y) -= ZNorm; 
			}
		}

		//EXPECTED FEATURES: pariwise
		vector<int> keys;
		for(int y1 = 0; y1 < nStates; y1++) {
			for(int y2 = 0; y2 < nStates; y2++) {
				for(int i = 1; i < nNodes; i++){
					keys.clear();
					ChunkKeys::getPairwise(keys, y1, y2);
					for(int k : keys) {
						ef[k] += exp(P1[i](y1, y2));
						assert(ef[k] != std::numeric_limits<double>::infinity());
						assert(ef[k] != -std::numeric_limits<double>::infinity());
						assert(!std::isnan(ef[k])); 
					}
				}
			}
		}

		//unary
		for (int i = 0; i < nNodes; i++) {
			for (int j = 0; j < nStates; j++) {	
				for (int k : ukeys[i][j]) {
					ef[k] += exp(P2(i, j)); 
					assert(ef[k] != std::numeric_limits<double>::infinity());
					assert(ef[k] != -std::numeric_limits<double>::infinity());
					assert(!std::isnan(ef[k])); 
				}
			}
		}

		//start, stop 
		for(int i = 0; i < nStates; i++) {
			//ef[ChunkKeys::getStart(i)] = exp(weights[ChunkKeys::getStart(i)] + beta(0, i) - ZNorm);
			//assert(ef[ChunkKeys::getStart(i)] != std::numeric_limits<double>::infinity());
			//ef[ChunkKeys::getEnd(i)] = exp(weights[ChunkKeys::getEnd(i)] + alpha(nNodes - 1, i) - ZNorm);
			//assert(ef[ChunkKeys::getEnd(i)] != std::numeric_limits<double>::infinity());
		}
		

	}

	Vec sampleModel(StrMat &Xdata, Potentials& p) {
		int nNodes = Xdata.size();
		vector<Mat> &M = p.M;
		Vec sampled(nNodes);
		//start -> first node
		Vec startParams(nStates);
		for (int i = 0; i < nStates; i++) {
			//startParams(i) = weights[ChunkKeys::getStart(i)] + p.unary(0, i) + beta(0, i); //in log space
			startParams(i) = p.unary(0, i) + p.beta(0, i); 
		}
		startParams = startParams - CRFMath::logSumExp(startParams); //normalize it.
		startParams.takeExp();
		sampled(0) = CRFMath::sampleMultinomial(startParams);
		//the rest.
		for (int i = 1; i < nNodes; i++) {
			Vec params(nStates);
			for (int j = 0; j < nStates; j++) {
				params(j) = M[i](sampled(i - 1), j) + p.beta(i, j); //in log space
			}
			params = params - CRFMath::logSumExp(params);
			params.takeExp();
			sampled(i) = CRFMath::sampleMultinomial(params);
		}
		return sampled;

	}


	Vec viterbi(Potentials p) { //can be called internally
		vector<Mat> &M = p.M;
		int nNodes = M.size();		
		Mat V;
		V.ones(nNodes, nStates);
		V = V * -std::numeric_limits<double>::infinity();

		//base case
		for(int i = 0; i < nStates; i++) 
			//V(0, i) = p.unary(0, i) + weights[ChunkKeys::getStart(i)]; 
			V(0, i) = p.unary(0, i); 

		Mat path(nNodes, nStates);

		//induction cases
		for (int i = 1; i < nNodes; i++) {
			for (int y = 0; y < nStates; y++) {
				for (int yp = 0; yp < nStates; yp++) { //look at previous probability and paths in order to maximize current situation
					double newval = V(i - 1, yp) + M[i](yp, y); //log space
					double& val = V(i, y);
					if (newval > val) {
						path(i, y) = yp;
						val = newval;
					}
				}	 			
			}
		}
		
		//finally, pick our path out of the nStates possibilities. this part is stupidly long
		Vec lastRow = V.row(V.nRows() - 1);
		double maxVal = -std::numeric_limits<double>::infinity();  //0 in normal space
		int maxIndex = 0;
		for (int i = 0; i < V.nCols(); i++) {
			double newMax = maxVal;
			newMax = std::max(lastRow(i), maxVal);
			if (newMax > maxVal)
	 			maxIndex = i;
			maxVal = newMax;
		}

		//backtracking
		Vec bestPath(nNodes);
		bestPath(nNodes - 1) = maxIndex;
		for (int t = nNodes - 1; t > 0; t--)
			 bestPath(t - 1) = path(t, bestPath(t));
			
		return bestPath;
	}

	float getModelProb(float *w, Vec &Ydata, Ukeys &ukeys) {
		//By definition of CRF, compute p(Y | Xdata).
		//We will need pairwise + unary weights
		//NEEDS TO RUN GETEXPECTEDFEATURES BEFORE CALLING THIS METHOD
		int nNodes = Ydata.size();
		vector<int> keys;
		float prob = 0;
		for(int i = 1; i < nNodes; i++) {
			keys.clear();
			ChunkKeys::getPairwise(keys, Ydata(i - 1), Ydata(i));
			for(int k: keys)
				prob += w[k];
		}

		for(int i = 0; i < nNodes; i++) {
			for(int k : ukeys[i][(int)Ydata(i)]) { 
				prob += w[k];
			} 
		}
		//normalize it...
		//cout << "before exp: " << prob << " znorm: " << ZNorm << endl;
		prob = std::exp(prob - ZNorm); //need expectedfeature
		return prob;
	}

	void reset() {
		std::fill_n(features, CRFMath::HASH_SIZE, 0);
		std::fill_n(expectedFeatures, CRFMath::HASH_SIZE, 0);

		if(lossFunction == Pairwise || lossFunction == PairwiseCrossEntropy) {
			std::fill_n(features2, CRFMath::HASH_SIZE, 0);
			std::fill_n(expectedFeatures2, CRFMath::HASH_SIZE, 0);
			std::fill_n(weights2, CRFMath::HASH_SIZE, DEFAULT_WEIGHT);
		} else if(lossFunction == Probit) {
			std::fill_n(weights2, CRFMath::HASH_SIZE, 0);
		}
	}


	//DEBUG METHODS

	double l2norm(float *a) {
		double result = 0;
		for(int i = 0; i < CRFMath::HASH_SIZE; i++) {
			result += pow(a[i], 2);
		}
		return sqrt(result);
	}

	int getNumFeatures() {
		int n = 0;
		for(int i = 0; i < CRFMath::HASH_SIZE; i++)
			if(weights[i] != DEFAULT_WEIGHT)
				n++;
		return n;
	}

	float maxOfVector(vector<float> &v) {
		float max = 0;
		for(int i = 0; i < v.size(); i++)
			max = std::max(max, v[i]);
		return max;
	}

	float minOfVector(vector<float> &v) {
		float min = 0;
		for(int i = 0; i < v.size(); i++)
			min = std::min(min, v[i]);
		return min;
	}
	float varOfVector(vector<float> &v) {
		float avg = 0;
		float var = 0;
		for(int i = 0; i < v.size(); i++)
			avg += v[i];
		avg = avg / v.size();
		for(int i = 0; i < v.size(); i++)
			var += pow(v[i] - avg, 2);
		var = var / v.size();
		return var;
	}
	float avgOfVector(vector<float> &v) {
		float avg = 0;
		for(int i = 0; i < v.size(); i++)
			avg += v[i];
		avg = avg / v.size();
		return avg;
	}
	void saveVector(vector<float> &v, string fname) {
		ofstream myfile;
		myfile.open (fname);
		for(int i = 0; i < v.size(); i++) {
			myfile << v[i] << endl;
		}
		myfile.close();
	}

	public:

	void init(int s, float l, float lam, float c, LossFunction lossF) {
		nStates = s;
		sqStates = sqrt(nStates);
		lrate = l;	
		lambda = lam;
		lossFunction = lossF;		
		DEFAULT_WEIGHT = 1;
		clip = c;
		
		weights = new float[CRFMath::HASH_SIZE];
		std::fill_n(weights, CRFMath::HASH_SIZE, DEFAULT_WEIGHT);

		gradient = new float[CRFMath::HASH_SIZE];
		std::fill_n(gradient, CRFMath::HASH_SIZE, 0);
		
		features = new float[CRFMath::HASH_SIZE];
		if(lossFunction == Pairwise || lossFunction == PairwiseCrossEntropy) {
			features2 = new float[CRFMath::HASH_SIZE];	
			weights2 = new float[CRFMath::HASH_SIZE];	
			expectedFeatures2 = new float[CRFMath::HASH_SIZE];	
		}
		expectedFeatures = new float[CRFMath::HASH_SIZE];
		if(lossFunction == Probit) 
			weights2 = new float[CRFMath::HASH_SIZE];
		if(lossFunction == CrossEntropyAdadelta || lossFunction == CrossEntropyMomentum) {
			accuGrad = new float[CRFMath::HASH_SIZE];
			accuUpdate = new float[CRFMath::HASH_SIZE];
			lastUpdated = new int[CRFMath::HASH_SIZE];
			std::fill_n(accuGrad, CRFMath::HASH_SIZE, 0);
			std::fill_n(accuUpdate, CRFMath::HASH_SIZE, 0);
			std::fill_n(lastUpdated, CRFMath::HASH_SIZE, 0);
		}
		if(lossFunction == CrossEntropyUniform) {
			blankweights = new float[CRFMath::HASH_SIZE];
			std::fill_n(blankweights, CRFMath::HASH_SIZE, DEFAULT_WEIGHT);
		}
		t = 0;
		clipcount = 0;
	}

	//load weights constructor
	void init(int s, float l, float lam, float c, vector<pair<int, float> > toLoad, LossFunction lossF) {
		nStates = s;
		sqStates = sqrt(nStates);
		lrate = l;
		lambda = lam;
		lossFunction = lossF;	
		DEFAULT_WEIGHT = 1;
		clip = c;

		weights = new float[CRFMath::HASH_SIZE];

		std::fill_n(weights, CRFMath::HASH_SIZE, DEFAULT_WEIGHT);

		gradient = new float[CRFMath::HASH_SIZE];
		std::fill_n(gradient, CRFMath::HASH_SIZE, 0);

		for(int i = 0; i < toLoad.size(); i++) {
			weights[toLoad[i].first] = toLoad[i].second;
		}
		
		features = new float[CRFMath::HASH_SIZE];
		if(lossFunction == Pairwise || lossFunction == PairwiseCrossEntropy) {
			features2 = new float[CRFMath::HASH_SIZE];	
			weights2 = new float[CRFMath::HASH_SIZE];	
			expectedFeatures2 = new float[CRFMath::HASH_SIZE];	
		}
		expectedFeatures = new float[CRFMath::HASH_SIZE];
		if(lossFunction == Probit) 
			weights2 = new float[CRFMath::HASH_SIZE];

		if(lossFunction == CrossEntropyAdadelta || lossFunction == CrossEntropyMomentum) {
			accuGrad = new float[CRFMath::HASH_SIZE];
			accuUpdate = new float[CRFMath::HASH_SIZE];
			lastUpdated = new int[CRFMath::HASH_SIZE];
			std::fill_n(accuGrad, CRFMath::HASH_SIZE, 0);
			std::fill_n(accuUpdate, CRFMath::HASH_SIZE, 0);
			std::fill_n(lastUpdated, CRFMath::HASH_SIZE, 0);
		}
		t = 0;

	}



	~CRFChunkModel() {
		delete[] weights;
		delete[] expectedFeatures;
		delete[] features;
		delete[] gradient;
		if(lossFunction == Pairwise) {
			delete[] weights2;
			delete[] expectedFeatures2;
			delete[] features2;
		}else if(lossFunction == Probit) {
			delete[] weights2;
		}
		if(lossFunction == CrossEntropyAdadelta || lossFunction == CrossEntropyMomentum) {
			delete[] accuGrad;
			delete[] accuUpdate;
			delete[] lastUpdated;
		}
		if(lossFunction == CrossEntropyUniform)
			delete[] blankweights;



	}

	vector<pair<int, float> > getModel() {
		vector<pair<int, float> > list;
		for(int i = 0; i < CRFMath::HASH_SIZE; i++) {
			if (weights[i] != DEFAULT_WEIGHT) {
				pair<int, float> p;
				p.first = i;
				p.second = weights[i];
				list.push_back(p);
			}
		}
		return list;
	}

	vector<pair<int, float> > getGradient() {
		vector<pair<int, float> > list;
		for(int i = 0; i < CRFMath::HASH_SIZE; i++) {
			if (gradient[i] != 0) {
				pair<int, float> p;
				p.first = i;
				p.second = gradient[i];
				list.push_back(p);
			}
		}
		return list;
	}

	void resetGradient() {
		std::fill_n(gradient, CRFMath::HASH_SIZE, 0);
	}



	void banditUpdate(int sentenceNum, int epoch, StrMat &Xdata, Vec& Ydata) {
		if(lossFunction == NotLearning)
			return;

		if(t ==  std::numeric_limits<int>::max() - 1)
			t = 0;
		else
			t++;


		reset(); //reset features, expectedFeatures
		int nNodes = Ydata.size();
/*
		//debug 
		if(sentenceNum == 7935) {
			cout << "Clip %: " << (float)clipcount/(float)sentenceNum << endl;
			clipcount = 0;
		}
	
		if(sentenceNum == 7935) {
			cout << "sampled probs avg: " << avgOfVector(sampleProbs) << " sampled probs var: " << varOfVector(sampleProbs) << endl;
			cout << "sampled feedback avg: " << avgOfVector(samplefeedback) << " sampled feedback var: " << varOfVector(samplefeedback) << endl;
			cout << "updateConst avg: " << avgOfVector(ratio) << " updateConst var: " << varOfVector(ratio) << endl;
			cout << "norm of weights: " << l2norm(weights) << endl;
			//saveVector(sampleProbs, "feb16_xentropy/sampleProbs" + to_string(epoch) + ".txt");
			//saveVector(samplefeedback, "feb16_xentropy/samplefeedback" + to_string(epoch) + ".txt");
			//saveVector(ratio, "feb16_xentropy/ratio" + to_string(epoch) + ".txt");
			sampleProbs.clear();
			samplefeedback.clear();
			ratio.clear();
		}*/
		//if(sentenceNum == 999) {
	//		cout << "Var of gradient: " << varOfVector(gradvar) << endl;
	//		gradvar.clear();
	//	}
		/*if(sentenceNum == 7935) {	
			float a1 = 0;
			float a1_1 = 0;
			float a2 = 0;
			float a2_1 = 0;		

			for(int i = 0; i < update1.size(); i++) {
				a1 += update1[i];
				a1_1 += rate1[i];
			}	
			a1 = a1/update1.size();
			a1_1 = a1_1/update1.size();

			for(int i = 0; i < update2.size(); i++) {
				a2 += update2[i];
				a2_1 += rate2[i];
			}	
			a2 = a2/update2.size();
			a2_1 = a2_1/update2.size();

			cout << "BIII_avgupdate: " << a1 << endl;
			cout << "BIII_avgrate: " << a1_1 << endl;

			cout << "IIII_avgupdate: " << a2 << endl;
			cout << "IIII_avgrate: " << a2_1 << endl;

			cout << "Max overall update: " << maxOfVector(overallUpdate) << endl;
			cout << "Min overall update: " << minOfVector(overallUpdate) << endl;

			cout << "Max overall rate: " << maxOfVector(overallRate) << endl;
			cout << "Min overall rate: " << minOfVector(overallRate) << endl;

			update1.clear();	
			update2.clear();
			rate1.clear();
			rate2.clear();
			overallUpdate.clear();
			overallRate.clear();

		}*/

		//calculate unary keys for this sentence, then store it in allUnaryKeys.
		//after the first epoch, this is never called again, saving time to compute unary keys.
		if (allUnaryKeys.size() <= sentenceNum) {
			assert(allUnaryKeys.size() == sentenceNum);
			allUnaryKeys.push_back(Ukeys(nNodes, vector<vector<int> >(nStates)));
			Ukeys& ukeys = allUnaryKeys[sentenceNum];
			for (int i = 0; i < nNodes; i++) {
				for (int j = 0; j < nStates; j++) {
					ChunkKeys::getUnary(ukeys[i][j], i, j, Xdata);
				}
			}
		}

		Ukeys& currentUKey = allUnaryKeys[sentenceNum];

		Potentials p;
		getPotentials(weights, p, nNodes, currentUKey);
		Vec Ysample;
		if(lossFunction != Probit || lossFunction != CrossEntropyUniform) {
			getExpectedFeatures(expectedFeatures, p, currentUKey);
			Ysample = sampleModel(Xdata, p);
			getFeatures(features, Xdata, Ysample, currentUKey);	
		}

		float feedback;
		float updateConst; //used only for x-entropy, probit

		if(lossFunction == Bayes) {
			feedback = 1 - CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;
			assert(feedback >= 0 && feedback <= 1);
			float modelProb = getModelProb(weights, Ysample, currentUKey);
			//sampleProbs.push_back(modelProb);
			//samplefeedback.push_back(1 - feedback);
			if(modelProb < clip) {
				modelProb = clip;
				clipcount++;
				//cout << "clip, resulting updateconst: " << feedback / modelProb << ", feedback: " << feedback << endl;
			}
			//ratio.push_back((1 - feedback) / modelProb);

		} else if(lossFunction == CrossEntropy || lossFunction == CrossEntropyAdadelta || lossFunction == CrossEntropyMomentum) {
			feedback = CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;
			assert(feedback >= 0 && feedback <= 1);
			float modelProb = getModelProb(weights, Ysample, currentUKey);
			//const float clip = .0001;
			if(modelProb == 0) {
				//updateConst = 0; //so we don't divide by zero 
			//	cout << "update of 0" << endl;
				modelProb = clip;
				updateConst = feedback / modelProb;
				clipcount++;
			} else {
				//float normByLength = .5 * nNodes * std::log(nStates); //nStates ^ nNodes..log space it 
				//float normByLength = std::max(.15 * nNodes, 1.0) * std::log(nStates);
				//updateConst = feedback * std::exp(-std::log(modelProb) - normByLength);
				//cout << modelProb << endl;
				if(modelProb < clip) {
					modelProb = clip;
					clipcount++;
					//cout << "clip, resulting updateconst: " << feedback / modelProb << ", feedback: " << feedback << endl;
				}
				updateConst = feedback / modelProb;
			}
			//sampleProbs.push_back(modelProb);
			//samplefeedback.push_back(feedback);
			//ratio.push_back(updateConst);
		}else if(lossFunction == CrossEntropyUniform) {
			//use weights of 1.
			Potentials p2;
			getPotentials(blankweights, p2, nNodes, currentUKey);
			Vec Ysample;
			getExpectedFeatures(expectedFeatures, p2, currentUKey);
			Ysample = sampleModel(Xdata, p2);
			getFeatures(features, Xdata, Ysample, currentUKey);
			feedback = CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;	
			updateConst = feedback;
			
		} else if(lossFunction == Pairwise) {
			vector<int> keys;
			for(int i = 0; i < nStates; i++)
				for(int j = 0; j < nStates; j++)
					ChunkKeys::getPairwise(keys, i, j);
			for(int k : keys) 
				weights2[k] = -weights[k];
			for (int i = 0; i < nNodes; i++)
				for (int j = 0; j < nStates; j++)
					for (int k : currentUKey[i][j])
						weights2[k] = -weights[k];
			Potentials p2;
			getPotentials(weights2, p2, nNodes, currentUKey);
			getExpectedFeatures(expectedFeatures2, p2, currentUKey);
			Vec Ysample2 = sampleModel(Xdata, p2);
			getFeatures(features2, Xdata, Ysample2, currentUKey);		
			
			float l1 = CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;
			float l2 = CRFMath::F1_chunk(Ysample2, Ydata, sqStates, false).F1;
			if (l1 > l2) {
				feedback = 0; //a loss function
				//feedback = 1 - (l1 - l2);
			} else {
				feedback = 1;
			}
		}else if (lossFunction == PairwiseCrossEntropy) {
			float modelProb = getModelProb(weights, Ysample, currentUKey);
			vector<int> keys;
			for(int i = 0; i < nStates; i++)
				for(int j = 0; j < nStates; j++)
					ChunkKeys::getPairwise(keys, i, j);
			for(int k : keys) 
				weights2[k] = -weights[k];
			for (int i = 0; i < nNodes; i++)
				for (int j = 0; j < nStates; j++)
					for (int k : currentUKey[i][j])
						weights2[k] = -weights[k];
			Potentials p2;
			getPotentials(weights2, p2, nNodes, currentUKey);
			getExpectedFeatures(expectedFeatures2, p2, currentUKey);
			Vec Ysample2 = sampleModel(Xdata, p2);
			float modelProb2 = getModelProb(weights2, Ysample2, currentUKey);
			getFeatures(features2, Xdata, Ysample2, currentUKey);	

			float l1 = CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;
			float l2 = CRFMath::F1_chunk(Ysample2, Ydata, sqStates, false).F1;
			if (l1 > l2) {
				//feedback = 0; //a gain function
				feedback = l1 - l2;
			} else {
				feedback = 0;
			}	

			if(modelProb == 0 || modelProb2 == 0) {
				updateConst = 0; //so we don't divide by zero 
				cout << "update of 0: " << modelProb << ", " << modelProb2 << endl;
			} else {
				float normByLength = nNodes * std::log(nStates); //nStates ^ nNodes.. in log space
				updateConst = feedback * std::exp(std::log(lrate) - (std::log(modelProb) + std::log(modelProb2)) - normByLength);
			}
		}else if (lossFunction == Probit) {
			//perturb weights by a standard normal vector N(0, diag(1))
			vector<int> keys;
			for(int i = 0; i < nStates; i++)
				for(int j = 0; j < nStates; j++)
					ChunkKeys::getPairwise(keys, i, j);
			for(int k : keys) 
				weights2[k] = weights[k] + CRFMath::sampleNormal(0.0, 1.0);
			for (int i = 0; i < nNodes; i++)
				for (int j = 0; j < nStates; j++)
					for (int k : currentUKey[i][j])
						weights2[k] = weights[k] + CRFMath::sampleNormal(0.0, 1.0);
			Potentials p2;
			getPotentials(weights2, p2, nNodes, currentUKey);
			Ysample = viterbi(p2);
			feedback = CRFMath::F1_chunk(Ysample, Ydata, sqStates, false).F1;
		}


		//pairwise weights:
		vector<int> keys;
		for(int i = 0; i < nStates; i++) {
			for(int j = 0; j < nStates; j++) {
				ChunkKeys::getPairwise(keys, i, j);
			}
		}

		for(int k : keys) {
			if(lossFunction == Bayes) {
				weights[k] -= (lrate * feedback) * (features[k] - expectedFeatures[k]) + (lrate * lambda * weights[k]); 
				gradient[k] += (lrate * feedback) * (features[k] - expectedFeatures[k]) + (lrate * lambda * weights[k]); 
			}else if(lossFunction == Pairwise) {
				weights[k] -= (lrate * feedback) * ((features[k] - features2[k]) - (expectedFeatures[k] - expectedFeatures2[k])) + (lrate * lambda * weights[k]);
				gradient[k] += (lrate * feedback) * ((features[k] - features2[k]) - (expectedFeatures[k] - expectedFeatures2[k])) + (lrate * lambda * weights[k]);
			}else if(lossFunction == PairwiseCrossEntropy)
				weights[k] -= updateConst * (-features[k] + features2[k] + expectedFeatures[k] - expectedFeatures2[k]) + (lrate * lambda * weights[k]);
			else if(lossFunction == Probit) 
				weights[k] += lrate * feedback * (weights2[k] - weights[k]);
			else if(lossFunction == CrossEntropyAdadelta) {
				//ADADELTA update
				const float rho_cst = .95;
				const float epsilon = .00001;
				int timeSinceUpdate = 0;
				if(t < lastUpdated[k]) //had gone over max value of int, loop back
					timeSinceUpdate = (std::numeric_limits<int>::max() - lastUpdated[k]) + t;
				else
					timeSinceUpdate = t - lastUpdated[k];
				const float rho = pow(rho_cst, timeSinceUpdate);
				float g = updateConst * (-features[k] + expectedFeatures[k]); //gradient
				accuGrad[k] = rho * accuGrad[k] + (1 - rho) * pow(g, 2);
				float rate = sqrt(accuUpdate[k] + epsilon) / sqrt(accuGrad[k] + epsilon);
				float update = -1 * g * rate;
				accuUpdate[k] =  rho * accuUpdate[k] + (1 - rho) * pow(update, 2);
				weights[k] += update;
				lastUpdated[k] = t;
			}else if(lossFunction == CrossEntropy) {
				weights[k] -= lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
				gradient[k] += lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
			}
			else if (lossFunction == CrossEntropyMomentum) {
				const float meu = std::min(.99, 1 - pow(2, -1 - log2(floor(epoch / (.15 * (1 / lrate))) + 1)));
				int timeSinceUpdate = 0;
				if(t < lastUpdated[k]) //had gone over max value of int, loop back
					timeSinceUpdate = (std::numeric_limits<int>::max() - lastUpdated[k]) + t;
				else
					timeSinceUpdate = t - lastUpdated[k];
				float update = pow(meu, timeSinceUpdate) * accuUpdate[k] - (lrate * updateConst * (-features[k] + expectedFeatures[k])) - (lrate * lambda * weights[k]);
				accuUpdate[k] = update;
				weights[k] += update; 
				lastUpdated[k] = t;
			}else if(lossFunction == CrossEntropyUniform) {
				weights[k] -= lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
			}
			assert (weights[k] != std::numeric_limits<double>::infinity());
		}

		//unary weights
		for (int i = 0; i < nNodes; i++) {
			for (int j = 0; j < nStates; j++) {
				for (int k : currentUKey[i][j]) {
					if(lossFunction == Bayes) {
						weights[k] -= (lrate * feedback) * (features[k] - expectedFeatures[k]) + (lrate * lambda * weights[k]);   
						gradient[k] += (lrate * feedback) * (features[k] - expectedFeatures[k]) + (lrate * lambda * weights[k]);  
					}else if(lossFunction == Pairwise) {
						weights[k] -= (lrate * feedback) * ((features[k] - features2[k]) - (expectedFeatures[k] - expectedFeatures2[k])) + (lrate * lambda * weights[k]);
						gradient[k] += (lrate * feedback) * ((features[k] - features2[k]) - (expectedFeatures[k] - expectedFeatures2[k])) + (lrate * lambda * weights[k]);
					}else if(lossFunction == PairwiseCrossEntropy)
						weights[k] -= updateConst * (-features[k] + features2[k] + expectedFeatures[k] - expectedFeatures2[k]) + (lrate * lambda * weights[k]);
					else if(lossFunction == Probit) 
						weights[k] += lrate * feedback * (weights2[k] - weights[k]);	
					else if(lossFunction == CrossEntropyAdadelta) {
						//ADADELTA update
						const float rho_cst = .95;
						const float epsilon = .00001;
						int timeSinceUpdate = 0;
						if(t < lastUpdated[k]) //had gone over max value of int, loop back
							timeSinceUpdate = (std::numeric_limits<int>::max() - lastUpdated[k]) + t;
						else
							timeSinceUpdate = t - lastUpdated[k];
						const float rho = pow(rho_cst, timeSinceUpdate);
						float g = updateConst * (-features[k] + expectedFeatures[k]); //gradient
						accuGrad[k] = rho * accuGrad[k] + (1 - rho) * pow(g, 2);
						float rate = sqrt(accuUpdate[k] + epsilon) / sqrt(accuGrad[k] + epsilon);
						float update = -1 * g * rate;
						accuUpdate[k] =  rho * accuUpdate[k] + (1 - rho) * pow(update, 2);
						weights[k] += update;
						lastUpdated[k] = t;
					}else if(lossFunction == CrossEntropy) {
						weights[k] -= lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
						gradient[k] += lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
					}else if (lossFunction == CrossEntropyMomentum) {
						int timeSinceUpdate = 0;
						const float meu = std::min(.99, 1 - pow(2, -1 - log2(floor(epoch / (.15 * (1 / lrate))) + 1)));
						if(t < lastUpdated[k]) //had gone over max value of int, loop back
							timeSinceUpdate = (std::numeric_limits<int>::max() - lastUpdated[k]) + t;
						else
							timeSinceUpdate = t - lastUpdated[k];
						float update = pow(meu, timeSinceUpdate) * accuUpdate[k] - (lrate * updateConst * (-features[k] + expectedFeatures[k])) - (lrate * lambda * weights[k]);
						accuUpdate[k] = update;
						weights[k] += update; 
						lastUpdated[k] = t;
					}else if(lossFunction == CrossEntropyUniform) {
						weights[k] -= lrate * updateConst * (-features[k] + expectedFeatures[k]) + (lrate * lambda * weights[k]);
					}
					assert(weights[k] != std::numeric_limits<double>::infinity());
				}
			}
		}


		//start & stop
/*
		for (int i = 0; i < nStates; i++) {
			if(lossFunction == BAYES_LOSS)
				weights[ChunkKeys::getStart(i)] -= (lrate * feedback) * (features[ChunkKeys::getStart(i)] - expectedFeatures[ChunkKeys::getStart(i)]);
			else if(lossFunction == CROSS_ENTROPY_LOSS)
				weights[ChunkKeys::getStart(i)] -= updateConst * (-features[ChunkKeys::getStart(i)] + expectedFeatures[ChunkKeys::getStart(i)]);
			else if(lossFunction == PAIRWISE_PREF_LOSS)
				weights[ChunkKeys::getStart(i)] -= (lrate * feedback) * ((features2[ChunkKeys::getStart(i)] - features[ChunkKeys::getStart(i)]) - expectedFeatures[ChunkKeys::getStart(i)]);		
			assert (weights[ChunkKeys::getStart(i)] != std::numeric_limits<double>::infinity());
			//weights[ChunkKeys::getEnd(i)] -= (lrate * feedback) * (features[ChunkKeys::getEnd(i)] - expectedFeatures[ChunkKeys::getEnd(i)]);
		}*/

		//
		//if(epoch % 20 == 0)
		//	haha(Xdata, p, currentUKey);
		
	}
	
	void fullInformationLearn(int sentenceNum, StrMat &Xdata, Vec &Ydata) {
		//stocastic gradient descent
		reset();
		int nNodes = Ydata.size();

		//calculate unary keys for this sentence, then store it in allUnaryKeys.
		//after the first epoch, this is never called again, saving time to compute unary keys.
		if (allUnaryKeys.size() <= sentenceNum) {
			assert(allUnaryKeys.size() == sentenceNum);
			allUnaryKeys.push_back(Ukeys(nNodes, vector<vector<int> >(nStates)));
			Ukeys& ukeys = allUnaryKeys[sentenceNum];
			for (int i = 0; i < nNodes; i++) {
				for (int j = 0; j < nStates; j++) {
					ChunkKeys::getUnary(ukeys[i][j], i, j, Xdata);
				}
			}
		}

		Ukeys& currentUKey = allUnaryKeys[sentenceNum];

		getFeatures(features, Xdata, Ydata, currentUKey);

		Potentials p;
		getPotentials(weights, p, nNodes, currentUKey);

		getExpectedFeatures(expectedFeatures, p, currentUKey);

		//pairwise weights update:
		vector<int> keys;
		for(int i = 0; i < nStates; i++) {
			for(int j = 0; j < nStates; j++) {
				keys.clear();
				ChunkKeys::getPairwise(keys, i, j);
				for(int k : keys)
					weights[k] += lrate * (features[k] - expectedFeatures[k]);
			}
		}

		//unary weights
		for (int i = 0; i < nNodes; i++) {
			for (int j = 0; j < nStates; j++) {
				for (int k : currentUKey[i][j]) {
					weights[k] += lrate * (features[k] - expectedFeatures[k]);
				}
			}
		}

		//start & stop
		for (int i = 0; i < nStates; i++) {
			//weights[ChunkKeys::getStart(i)] += lrate * (features[ChunkKeys::getStart(i)] - expectedFeatures[ChunkKeys::getStart(i)]);
			//weights[ChunkKeys::getEnd(i)] += lrate * (features[ChunkKeys::getEnd(i)] - expectedFeatures[ChunkKeys::getEnd(i)]);
		}
	}

	Vec decode(StrMat &Xdata) { //external use
		int nNodes = Xdata.size();
 		Ukeys ukeys(nNodes, vector<vector<int> >(nStates));
		for (int i = 0; i < nNodes; i++) {
			for (int j = 0; j < nStates; j++) {
				ChunkKeys::getUnary(ukeys[i][j], i, j, Xdata);
			}
		}
		Potentials p;
		getPotentials(weights, p, nNodes, ukeys);
		return viterbi(p);
	}

	
};


/*
	void haha(StrMat &Xdata, Potentials &p, Ukeys &ukeys) {
		const int nsamps = 1000;
		std::map <std::string, pair<int, double> > samplesMap;
		for(int i = 0; i < nsamps; i++) {
			Vec Ysample = sampleModel(Xdata, p);
			samplesMap[Ysample.toString()] = make_pair(samplesMap[Ysample.toString()].first + 1, getModelProb(Ysample, ukeys));
		}

		std::map<string, pair<int, double> >::iterator iter;
		float total = 0;
		for (iter = samplesMap.begin(); iter != samplesMap.end(); ++iter) {
			//cout << "Sequence: " << iter->first << " counts: " << iter->second << endl;
			total += iter->second.first;
		}
		double mean = 0;
		int n = 0;
		for (iter = samplesMap.begin(); iter != samplesMap.end(); ++iter) {
			//cout << "Sequence: " << iter->first << " rel freq: " << (iter->second)/total << endl;
			///cout << "rel freq: " << (iter->second.first)/total << " vs. model prob: " << iter->second.second << endl;
			mean += (iter->second.first)/total;
			n++;
		}
		mean = mean / n;
		double var = 0;
		double max = 0;
		for (iter = samplesMap.begin(); iter != samplesMap.end(); ++iter) {
			var += pow((iter->second.first)/total - mean, 2);
			if((iter->second.first)/total > max)
				max = (iter->second.first)/total;
		}
		var = var / n;
		//cout << "Var(rel freq of samples): " << var << ", Mean(rel freq of samples): " << mean << ", Max(rel freq of samples): " << max << endl;

	}
*/

#endif
