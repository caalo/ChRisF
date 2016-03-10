#ifndef __CHUNKKEYS
#define __CHUNKKEYS
#include <assert.h> 
#include <algorithm>  
#include <math.h>
#include "crflinearalgebra.h"
#include "crfmath.h"

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;

//feature definition for chunking task. (possible inheritance?)
class ChunkKeys {

public:

	static void getUnary(vector<int> &keys, int i, int Yi, StrMat &Xdata) {
		//unary features defined here. 
		int nNodes = Xdata.size();
		assert(i >= 0);
		assert(i < nNodes);

		//returns a vector<int> of appropriate keys
		string str;

		for(int j = 0; j < 2; j++) {
			//f1: w_i = w, f8 t_i = t
			str = std::to_string(1 + 7 * j) + std::to_string(Yi) + Xdata[i][j];
			keys.push_back(CRFMath::hash(str));
			//f2: w_i-1 = w, f9 t_i-1 = t
			if(i - 1 >= 0) {
				str = std::to_string(2 + 7 * j) + std::to_string(Yi) + Xdata[i - 1][j];
				keys.push_back(CRFMath::hash(str));
			}
			//f3: w_i+1 = w, f10 t_i+1 = t
			if(i + 1 < nNodes) {
				str = std::to_string(3 + 7 * j) + std::to_string(Yi) + Xdata[i + 1][j];
				keys.push_back(CRFMath::hash(str));
			}
			//f4: w_i-2 = w, ....
			if(i - 2 >= 0) {
				str = std::to_string(4 + 7 * j) + std::to_string(Yi) + Xdata[i - 2][j];
				keys.push_back(CRFMath::hash(str));
			}
			//f5: w_i+2 = w, ...
			if(i + 2 < nNodes) {
				str = std::to_string(5 + 7 * j) + std::to_string(Yi) + Xdata[i + 2][j];
				keys.push_back(CRFMath::hash(str));
			}
			//f6: w_i-1 = w', w_i = w, ...
			if(i - 1 >= 0) {
				str = std::to_string(6 + 7 * j) + std::to_string(Yi) + Xdata[i - 1][j] + Xdata[i][j]; 
				keys.push_back(CRFMath::hash(str));
			}
			//f7: w_i = w, w_i+1 = w', ...
			if(i + 1 < nNodes) {
				str = std::to_string(7 + 7 * j) + std::to_string(Yi) + Xdata[i][j] + Xdata[i + 1][j]; 
				keys.push_back(CRFMath::hash(str));
			}
		}
		//f15: t_i-2 = t', t_i-1 = t
		if(i - 2 >= 0) {
			str = "15" + std::to_string(Yi) + Xdata[i - 2][1] + Xdata[i - 1][1]; 
			keys.push_back(CRFMath::hash(str));
		}
		//f16: t_i+1 = t, t_i+2 = t'
		if(i + 2 < nNodes) {
			str = "16" + std::to_string(Yi) + Xdata[i + 1][1] + Xdata[i + 2][1]; 
			keys.push_back(CRFMath::hash(str));
		}
		//f17: t_i-2 = t'', t_i-1 = t', t_i = t
		if(i - 2 >= 0) {
			str = "17" + std::to_string(Yi) + Xdata[i - 2][1] + Xdata[i - 1][1] + Xdata[i][1]; 
			keys.push_back(CRFMath::hash(str));
		}
		//f18: t_i = t, t_i+1 = t', t_i+2 = t''
		if(i + 2 < nNodes) {
			str = "18" + std::to_string(Yi) + Xdata[i][1] + Xdata[i + 1][1] + Xdata[i + 2][1]; 
			keys.push_back(CRFMath::hash(str));
		}

		//last feature: y_i = y. listed in pairwise feature, but really is an unary with no x compoents. 
		//keys.push_back(CRFMath::hash("label" + to_string(Yi)));

	}

	static void getPairwise(vector<int> &keys, const int Yi, int const Yj) {
		if(Yi == 7) //(2, 1)
			return;

		bool toHash = false;
		if(Yi == 0 || Yi == 3 || Yi == 6) {
			if(Yj == 0 || Yj == 1 || Yj == 2)
				toHash = true; 
		}else if(Yi == 1 || Yi == 4) {
			if(Yj == 3 || Yj == 4 || Yj == 5)
				toHash = true; 
		}else if(Yi == 2 || Yi == 5 || Yi == 8) {
			if(Yj == 6 || Yj == 8)
				toHash = true;
		}
		
		if(toHash) {
			keys.push_back(CRFMath::hash("pw" + to_string(Yi) + to_string(Yj)));
		}
		return;

		//keys.push_back(CRFMath::hash("pw" + to_string(Yj)));
		//keys.push_back(CRFMath::hash("pw" + to_string(Yj % 3)));  //second component
	}


	static int getStart(const int Yi) {
		return CRFMath::hash("start" + to_string(Yi));
	}

	static int getEnd(const int Yi) {
		return CRFMath::hash("end" + to_string(Yi));
	}


};
#endif
