#ifndef __CRFMATH
#define __CRFMATH
#include <assert.h> 
#include <iostream>
#include <algorithm>  
#include <random>
#include <math.h>
#include "crflinearalgebra.h"

using namespace std;

std::default_random_engine generator;

struct F1Score {
	float F1;
	float precision;
	float recall;
	float ncorrect;

	float nrefchunks;
	float npredchunks;

	F1Score operator+(F1Score a) {
		F1Score result;
		result.F1 = F1 + a.F1;
		result.precision = precision + a.precision;
		result.recall = recall + a.recall;
		result.ncorrect = ncorrect + a.ncorrect;

		result.nrefchunks = nrefchunks + a.nrefchunks;
		result.npredchunks = npredchunks + a.npredchunks;
		return result;
	}

	F1Score operator/(float d) {
		F1Score result;
		result.F1 = F1 / d;
		result.precision = precision / d;
		result.recall = recall / d;
		result.ncorrect = ncorrect / d;

		result.nrefchunks = nrefchunks / d;
		result.npredchunks = npredchunks / d;

		return result;
	}
	
	
	F1Score() {
		F1 = 0;
		precision = 0;
		recall = 0;
		ncorrect = 0;
	}
};

class CRFMath {

	public:

	static void start() {
		generator = std::default_random_engine();
		generator.seed(time(NULL));
	}

	static const int HASH_SIZE = 2000000;
	static unsigned hash(const string str, unsigned int seed = 0) {
		const char* s = str.c_str();
		unsigned int hash = seed;
		while (*s) {
			hash = hash * 101 +  *s++;
		}
		return hash % HASH_SIZE;
	}

	static int sampleMultinomial(Vec p) {
		//given vector p of p_1, ..., p_k parameters, sample multinomial distribution.
		
		//float X = 0;
		
  		std::uniform_real_distribution<double> unifDistribution(0.0, 1.0);
		float X = unifDistribution(generator);
		
		
		//float X = (rand() % 1000000 + .000000001) / 1000000; 

		//p.print("given paraemters");
		//cout << "sums up to: " << accu(p) << endl;
		//if (p.sum() != 1)
			//cout << "WARNING: parameters do not sum to 1: " << p.sum() << endl;
		double sum = 0;
		int i = 0;
		for (i = 0; i < p.size(); i++) {
			sum += p(i);
			if (sum - X >= 0)
				break;
		}

		return i;

	}

	static double sampleNormal(double m, double s) {
		std::normal_distribution<double> distribution(m, s);
		return distribution(generator);
	}

	static float hamming(Vec pred, Vec ref, const int sqstates) {
		assert(pred.size() == ref.size());
		//need to convert pairs --> singleton: range is transformed from 0...8 to 0...2
		for(int i = 0; i < pred.size(); i++) {
			pred(i) = (int)pred(i) % sqstates;
			ref(i) = (int)ref(i) % sqstates;
		}
		float loss = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (pred(i) != ref(i))
				loss++;
		}
		return loss / pred.size();
	}

	//chunking labels. 
	static F1Score F1_chunk(Vec pred, Vec ref, const int sqstates, bool print) {
		assert(pred.size() == ref.size());
		//need to convert pairs --> singleton: range is transformed from 0...8 to 0...2
		for(int i = 0; i < pred.size(); i++) {
			pred(i) = (int)pred(i) % sqstates;
			ref(i) = (int)ref(i) % sqstates;
		}

		if(print) {
			pred.print("pred");
			ref.print("ref");
		}

		int i = 0;
		float nCorrect = 0;
		float nChunksPredicted = 0;
		float nChunksReference = 0;
		bool insideReference = false;
		bool insidePredicted = false;
		bool correct = false;
		for (int i = 0; i < pred.size(); i++) {
			if(ref(i) == 0) { //if ref label is "B"
				if(insideReference) {
					nChunksReference++;
				}
				if(correct && insideReference) {
					if (pred(i) == 0 || pred(i) == 2) {
						nCorrect++;
						correct = false;
					}
				}
				if(pred(i) == ref(i)) {
					correct = true;
				}
				insideReference = true;
			} 
			if(pred(i) == 0) { //if pred label is "B"
				if(insidePredicted)
					nChunksPredicted++;
				insidePredicted = true;
			}
			if(insideReference && ref(i) == 2) { // if ref label is 'O' 
				if(correct) {
					if (pred(i) == 0 || pred(i) == 2) {
						nCorrect++;
						correct = false;
					}
				}
				nChunksReference++;
				insideReference = false;
			}
			if(insidePredicted && pred(i) == 2) { //if pred label is 'O'
				insidePredicted = false;
				nChunksPredicted++;
			}
			if(insideReference && pred(i) != ref(i)) {
				correct = false;
			}

			if(i == pred.size() - 1) {
				//if ref or pred are "I" or "B" at the end of the sentence, call it a chunk anyways.
				if((pred(i) == 1 || pred(i) == 0) && insidePredicted)
					nChunksPredicted++;
				if((ref(i) == 1 || ref(i) == 0) && insideReference) {
					nChunksReference++;
					if(correct && pred(i) != 2)
						correct++; 
				}
				
			}
		}

		F1Score score;
		if(nChunksReference == 0 || nChunksPredicted == 0)
			return score; //a score of 0
		score.recall = nCorrect / nChunksReference;
		score.precision = nCorrect / nChunksPredicted;
		score.ncorrect = nCorrect;
		score.nrefchunks = nChunksReference;
		score.npredchunks = nChunksPredicted;
		if(print)
			cout << "n correct: " << nCorrect << " nChunksRef: " << nChunksReference << " nChunksPred: " << nChunksPredicted << endl;
		if (score.precision + score.recall == 0)
			return score; //a score of 0
		score.F1 = 2 * ((score.precision * score.recall) / (score.precision + score.recall)); //F1 score
		return score;

	}
/*
	static F1Score F1_chunk_old(Vec pred, Vec ref, const int sqstates, bool print) {
		assert(pred.size() == ref.size());
		//need to convert pairs --> singleton: range is transformed from 0...8 to 0...2
		for(int i = 0; i < pred.size(); i++) {
			pred(i) = (int)pred(i) % sqstates;
			ref(i) = (int)ref(i) % sqstates;
		}

		if(print) {
			pred.print("pred");
			ref.print("ref");
		}

		int i = 0;
		float nCorrect = 0;
		int prevPredicted = 2;
		int prevReference = 2;
		float nChunksPredicted = 0;
		float nChunksReference = 0;
		bool insideReference = false;
		bool insidePredicted = false;
		bool correct = false;
		while(i < pred.size()) { //if both not I, the previous chunk should be counted...
			if(insideReference && pred(i) != ref(i)) {
				correct = false;
			}
			if(ref(i) == 0) { //if ref label is "B"
				if(insideReference) {
					nChunksReference++;
				}
				insideReference = true;
				if(pred(i) == ref(i)) {
					if(correct && insideReference) {
						nCorrect++;
					}
					correct = true;
				}
			} 
			if(pred(i) == 0) { //if pred label is "B"
				if(insidePredicted)
					nChunksPredicted++;
				insidePredicted = true;
			}
			if(insideReference && ref(i) == 2) { // if ref label is 'O' 
				if(pred(i) == ref(i) && correct) {
					nCorrect++;
					correct = false;
				}
				nChunksReference++;
				insideReference = false;
			}
			if(insidePredicted && pred(i) == 2) { //if pred label is 'O'
				insidePredicted = false;
				nChunksPredicted++;
			}
			prevReference = ref(i);
			prevPredicted = pred(i);
			i++;
		}

		F1Score score;
		if(nChunksReference == 0 || nChunksPredicted == 0)
			return score; //a score of 0
		score.recall = nCorrect / nChunksReference;
		score.precision = nCorrect / nChunksPredicted;
		if(print)
			cout << "n correct: " << nCorrect << " nChunksRef: " << nChunksReference << " nChunksPred: " << nChunksPredicted << endl;
		if (score.precision + score.recall == 0)
			return score; //a score of 0
		score.F1 = 2 * ((score.precision * score.recall) / (score.precision + score.recall)); //F1 score
		return score;

	}
*/

	static double logDotProd(Vec vec1, Vec vec2) {
		Vec temp(vec1.size());
		for (int i = 0; i < vec1.size(); i++) {
			temp(i) = vec1(i) + vec2(i);
		}
		return logSumExp(temp);
	}

	static Vec logDotVM(Vec vec, Mat mtx) {
		Vec result(mtx.nCols());
		for (int j = 0; j < mtx.nCols(); j++) {
			Vec temp(mtx.nRows());
			for (int i = 0; i < mtx.nRows(); i++) {
				temp(i) = vec(i) + mtx(i, j);
			}
			result(j) = logSumExp(temp);				
		}
		return result;
	}
	
	static Vec logDotMV(Mat mtx, Vec vec) {
		Vec result(mtx.nRows());
		for (int i = 0; i < mtx.nRows(); i++) {
			Vec temp(mtx.nCols());
			for (int j = 0; j < mtx.nCols(); j++) {
				temp(j) = mtx(i, j) + vec(j);
			}	
			result(i) = logSumExp(temp);
		}
		return result;
	}
	
	/* Log-sum-exp computes log[Sum(exp(a_i))]. 'a' is a column vector. 
	    Can also be used to compute sum of vector of probablities that 
		are already in log-space. */
	static double logSumExp(Vec a) {
		double max = a.max();
		//calculate log(exp(max) * sum(exp(a_i - max))) instead
		double sum = 0;
		for (int i = 0; i < a.size(); i++) {
			if (a(i) != -std::numeric_limits<double>::infinity()) //Note: std::exp(-inf) = 0. however, if 'a' contains all -inf entries, we would have std::exp(-inf - -inf) = nan, which is bad. So every time we encounter a(i) = -inf, we don't add to the sum, same as adding 0.
				sum += std::exp(a(i) - max);

		}
		return max + std::log(sum);
	}

};


//old stuff

/*
	
	static void tester() { //tests the multinomial sampler
		mat params = zeros(1, 5);
		mat samples = zeros(1, 5);
		params << .1 << .1 << .5 << .05 << .25 << endr;

		for (int i = 0; i < 20000; i++) {
			samples(sampleMultinomial(params))++;
		}

		cout << "sample distribution" << endl;

		for (int i = 0; i < samples.n_elem; i++) {
			cout << (samples(i) / accu(samples)) << endl;
		}
	}
	

	static double distance(mat a, mat b) {
		if (a.n_elem != b.n_elem) {
			cout << "lengths do not match up!" << endl;
			return -1;
		}
		double dist = 0;
		for (int i = 0; i < a.n_elem; i++) {
			dist += pow(a(i) - b(i), 2);
		}
		return sqrt(dist);
	}
*/

#endif
