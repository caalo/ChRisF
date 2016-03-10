#include <iostream>

#include "crfchunkmodel.h"
#include "crflinearalgebra.h"
#include "crffileloader.h"
#include "crfmath.h"

using namespace std;

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;

int main(int argc, char* argv[])	{

	//ARGUMENT PARSER
	char* testf;
	char* nsentence;
	char* modelfile;
	bool allSentences = false; //flag: do we want to decode all sentences?
	bool detailedPrint = false; //flag: do we want to have a detailed output?
	if (argc < 4*2) {
		cout << "Not enough or invalid arguments, please try again." << endl;
		cout << "Usage is: ChRisF_decoder -testf -nsentence (all) -loadmodel -print (yes)." << endl;;
		exit(0);
	} else {
		for (int i = 1; i < argc; i++) {
			if (i + 1 != argc) {
				if (strcmp(argv[i], "-testf") == 0) {
					testf = argv[i + 1];
					i++;
				}else if (strcmp(argv[i], "-nsentence") == 0) {
					if (strcmp(argv[i + 1], "all") == 0) {
						allSentences = true;
						nsentence = 0;
					} else {
						nsentence = argv[i + 1];
					}
					i++;
				} else if (strcmp(argv[i], "-loadmodel") == 0) {
					modelfile = argv[i + 1];
					i++;
				}else if (strcmp(argv[i], "-print") == 0) {
					if (strcmp(argv[i + 1], "yes") == 0) 
						detailedPrint = true;
					else
						detailedPrint = false;
					i++;
				} else {
					cout << "Not enough or invalid arguments, please try again." << endl;
					cout << "Usage is: ChRisF_decoder -testf -nsentence (all) -loadmodel -print (yes)." << endl;;
					exit(0);
            		}
			}
		}
	}


	//LOAD DATA
	Mat Y;
	vector<StrMat> X;

	FileLoader::loadChunking(testf, Y, X);

	//SOME PARAMETERS
	int N_SENTENCES = 0;
	if(allSentences)
		N_SENTENCES = Y.nRows();
	else 
		N_SENTENCES = (int)atof(nsentence);

	const int nStates = 3*3;
	const int sqStates = 3;

	//PRINT BEFORE WE START
	cout << "ChRisF Chunking Decoder" << endl;
	cout << "======" << endl;
	cout << "READ IN TEST FILE:" << endl;
	cout << "Total number of sentences: " << Y.nRows() << "\n";
	cout << "USER-SET PARAMETERS..\n";
	cout << "Number of sentences to decode: " << N_SENTENCES << "\n";
	cout << endl;

	//LOAD MODEL
	vector<pair<int, float> > modelWeights;
	FileLoader::loadModel(modelfile, modelWeights);

	//decode
	CRFChunkModel model;
	model.init(nStates, 0, 0, 0, modelWeights, NotLearning);
	float loss_hamming = 0;
	F1Score loss_f1;

	const int N_SENTENCES_PRINT = 5;
	if(detailedPrint) {
		cout << "Some sample decoding comparisions: " << N_SENTENCES_PRINT << endl << endl;
		cout << "original word\tpos\tlabel\tlabel predicted " << endl << endl;
	}

	for (int i = 0; i < N_SENTENCES; i++) {
		Vec Yhat = model.decode(X[i]); //in linearized form
		loss_hamming += CRFMath::hamming(Yhat, Y.row(i), 3);
		loss_f1 = loss_f1 + CRFMath::F1_chunk(Yhat, Y.row(i), sqStates, false);
		
		if(detailedPrint && i < N_SENTENCES_PRINT) {
			for (int j = 0; j < X[i].size(); j++) { //index over jth word in ith sentence
				string a = "";
				string b = "";
				if((int)Y(i, j) % sqStates == 0)
					a = "B";
				else if((int)Y(i, j) % sqStates == 1)
					a = "I";
				else if((int)Y(i, j) % sqStates == 2)
					a = "O";

				if((int)Yhat(j) % sqStates == 0)
					b = "B";
				else if((int)Yhat(j) % sqStates == 1)
					b = "I";
				else if((int)Yhat(j) % sqStates == 2)
					b = "O";

				if (X[i][j][0].size() < 8)
					cout << X[i][j][0] << "\t\t" << X[i][j][1] << "\t" << a << "\t" << b << endl;
				else
					cout << X[i][j][0] << "\t" << X[i][j][1] << "\t" << a << "\t" << b << endl;
			}
			cout << endl;
		}
	}

	loss_hamming = loss_hamming / N_SENTENCES;
	F1Score loss_f1_local = loss_f1 / N_SENTENCES;
	cout << "RESULTS:" << endl;
	cout << "F1 (local): " << loss_f1_local.F1 << endl;
	cout << "Precision (local): " << loss_f1_local.precision << endl;
	cout << "Recall (local): " << loss_f1_local.recall << endl;
	cout << "Hamming: " << loss_hamming << endl;
	float r = loss_f1.ncorrect / loss_f1.nrefchunks;
	float p = loss_f1.ncorrect / loss_f1.npredchunks;
	cout << "Accumulated Recall: " << r << endl;
	cout << "Accumulated Precision: " << p << endl;
	cout << "Accumulated F1: " << 2 * ((p * r) / (p + r)) << endl;
  	return 0;
}

