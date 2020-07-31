#include <iostream>
#include <random>

#include "crfchunkmodel.h"
#include "crflinearalgebra.h"
#include "crffileloader.h"

//

using namespace std;

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;



int main(int argc, char* argv[])	{

	CRFMath::start(); //seed random values for sampling.	
	
	//SOME PARAMETERS
	const float L_RATE = .001;
	const float CLIP = .005;
	const int EPOCHS = 10000;
	const int N_SENTENCES = 100; //7936
	const int N_SENTENCES_DECODE = 1000;
	const int EPOCHS_PER_DECODE = 1; //We decode every ___ epochs.
	const float LAMBDA = .0000001; //regularizer


	//LOAD DATA
	Mat Y;
	vector<StrMat> X;
	FileLoader::loadChunking(argv[1], Y, X);

	//Decode on what file?
	Mat Y2;
	vector<StrMat> X2;
	FileLoader::loadChunking(argv[2], Y2, X2);

	
	int nStates = 3*3;

	//PRINT BEFORE WE START
	int nSentences = Y.nRows();
	cout << "READ IN PARAMETERS:" << endl;
	cout << "Number of sentences: " << nSentences << "\n";
	cout << "nStates: " << nStates << "\n";
	
	cout << "USER-SET PARAMETERS..\n";
	cout << "Learning rate: " << L_RATE << "\n";
	cout << "Regularization rate: " << LAMBDA << endl;
	cout << "Clip prob: " << CLIP << endl;
	cout << "Number of Epochs: " << EPOCHS << "\n";
	cout << "Number of sentence to learn: " << N_SENTENCES << "\n";
	cout << "Number of sentences to decode: " << N_SENTENCES_DECODE << endl;
	cout << "We decode every ___ epochs: " << EPOCHS_PER_DECODE << "\n";


	//INIT OUR MODEL. 

	CRFChunkModel model;

	model.init(nStates, L_RATE, LAMBDA, CLIP, Bayes); 

	

	//Start things
	vector<float> losses;
	for (int e = 0; e < EPOCHS; e++) {
		cout << "\nEPOCH " << e << "\n";

		//LEARNING PROCESS. 
		for (int i = 0; i < N_SENTENCES; i++) {
			//model.fullInformationLearn(i, X[i], Y.row(i));
			model.banditUpdate(i, e, X[i], Y.row(i));
		}

		if (e % EPOCHS_PER_DECODE  == 0) {
			//INFERENCE PROCESS.
			float loss = 0;
			float loss_recall = 0;
			float loss_precision = 0;
			float loss2 = 0;
			for (int i = 0; i < N_SENTENCES_DECODE; i++) {
				Vec Yhat = model.decode(X2[i]); //in linearized form
				loss += CRFMath::F1_chunk(Yhat, Y2.row(i), 3, false).F1;
				loss_recall += CRFMath::F1_chunk(Yhat, Y2.row(i), 3, false).recall;
				loss_precision += CRFMath::F1_chunk(Yhat, Y2.row(i), 3, false).precision;
				loss2 += CRFMath::hamming(Yhat, Y2.row(i), 3);

				/*if(i < 5) {
					int sqStates = 3;
					for (int j = 0; j < Y2.row(i).size(); j++) { //index over jth word in ith sentence
						string a = "";
						string b = "";
						if((int)Y2(i, j) % sqStates == 0)
							a = "B";
						else if((int)Y2(i, j) % sqStates == 1)
							a = "I";
						else if((int)Y2(i, j) % sqStates == 2)
							a = "O";

						if((int)Yhat(j) % sqStates == 0)
							b = "B";
						else if((int)Yhat(j) % sqStates == 1)
							b = "I";
						else if((int)Yhat(j) % sqStates == 2)
							b = "O";

						if (X2[i][j][0].size() < 8)
							cout << X2[i][j][0] << "\t\t" << X2[i][j][1] << "\t" << a << "\t" << b << endl;
						else
							cout << X2[i][j][0] << "\t" << X2[i][j][1] << "\t" << a << "\t" << b << endl;
					}
					cout << "Local F1 score: " << CRFMath::F1_chunk(Yhat, Y2.row(i), 3, true).F1 << " Local F1 precision: " << CRFMath::F1_chunk(Yhat, Y2.row(i), 3, false).precision << " Local F1 recall: " << CRFMath::F1_chunk(Yhat, Y2.row(i), 3, false).recall << endl;
					cout << endl;
				}*/
			}

			loss = loss / N_SENTENCES_DECODE;
			loss_precision = loss_precision / N_SENTENCES_DECODE;
			loss_recall = loss_recall / N_SENTENCES_DECODE;
			loss2 = loss2 / N_SENTENCES_DECODE;
			losses.push_back(loss);
			cout << "F1: " << loss << " Precision: " << loss_precision << " Recall: " << loss_recall << " Hamming: " << loss2 << " \n";

		}

	}


	//save loss to a file
	const string fname = "blah";

	ofstream myfile;
	myfile.open (fname + ".txt");
	for(int i = 0; i < losses.size(); i++) {
		myfile << losses[i] << endl;
	}
	myfile.close();

	//SAVE MODEL
	myfile.open(fname + ".crf");
	vector<pair<int, float> > modelWeights = model.getModel();
	for(int i = 0; i < modelWeights.size(); i++) {
		myfile << modelWeights[i].first << " " << modelWeights[i].second << endl;
	}
	
  	return 0;
}

