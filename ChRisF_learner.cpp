#include <iostream>
#include <random>
#include "crfchunkmodel.h"
#include "crflinearalgebra.h"
#include "crffileloader.h"
#include "crfmath.h"

using namespace std;

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;

int main(int argc, char* argv[])	{
 	CRFMath::start();	

	//ARGUMENT PARSER
	const int FULL_INFORMATION_MODE = 0;
	const int BANDIT_MODE = 1;
	char* loadmodelfile;
	char* trainf;
	char* lrate;
	char* lambda;
	char* clip;
	char* nsentence;
	char* epoch_s;
	char* epoch_e;
	char* modelfile;
	bool allSentences = false; //flag: do we want to learn all of the sentences?
	bool toLoadModel = false; //flag: do we want to load the model from file before learning?
	LossFunction lossFunction; //flag: what loss function to use when optimizing? (See CRFChunkModel for list)
	int train_mode = FULL_INFORMATION_MODE;
	if (argc <8*2) {
		cout << "Not enough or invalid arguments, please try again." << endl;
		cout << "Usage is -mode (full, bandit) -loadmodel (none) -trainf -lrate -lambda -clip -nsentence (full) -epoch -loss: (bayes, crossentropy, crossentropymomentum, crossentropyadadelta pairwise, probit, pairwisecrossentropy, pairwisecontinuous) -savemodel" << endl;
		exit(0);
	} else {
		for (int i = 1; i < argc; i++) {
			if (i + 1 != argc) {
				if (strcmp(argv[i], "-mode") == 0) {
					if (strcmp(argv[i + 1], "full") == 0) {
						train_mode = FULL_INFORMATION_MODE;
					}else if (strcmp(argv[i + 1], "bandit") == 0) {
						train_mode = BANDIT_MODE;
					}
					i++;
				}else if (strcmp(argv[i], "-loadmodel") == 0) {
					if (strcmp(argv[i + 1], "none") == 0) {
						toLoadModel = false;
					} else {
						loadmodelfile = argv[i + 1];
						toLoadModel = true;
					}
					i++;
				}else if (strcmp(argv[i], "-trainf") == 0) {
					trainf = argv[i + 1];
					i++;
				}else if (strcmp(argv[i], "-lrate") == 0) {
					lrate = argv[i + 1];
					i++;
				}else if (strcmp(argv[i], "-lambda") == 0) {
					lambda = argv[i + 1];
					i++;
				}else if(strcmp(argv[i], "-clip") == 0) {
					clip = argv[i + 1];
					i++;
				} else if (strcmp(argv[i], "-nsentence") == 0) {
					if (strcmp(argv[i + 1], "all") == 0) {
						allSentences = true;
					}else {
						nsentence = argv[i + 1];
					}
					i++;
				} else if (strcmp(argv[i], "-epoch_s") == 0) {
					epoch_s = argv[i + 1];
					i++;
				} else if (strcmp(argv[i], "-epoch_e") == 0) {
					epoch_e = argv[i + 1];
					i++;
				}else if (strcmp(argv[i], "-loss") == 0) {
					if (strcmp(argv[i + 1], "bayes") == 0) {
						lossFunction = Bayes;
						cout << "Bayes" << endl;
					} else if (strcmp(argv[i + 1], "crossentropy") == 0) {
						lossFunction = CrossEntropy;
						cout << "Xentropy" << endl;
					} else if (strcmp(argv[i + 1], "pairwise") == 0) {
						lossFunction = Pairwise;
						cout << "Pairwise" << endl;
					}else if (strcmp(argv[i + 1], "probit") == 0) {
						lossFunction = Probit;
						cout << "Probit" << endl;
					}else if (strcmp(argv[i + 1], "pairwisecrossentropy") == 0) {
						lossFunction = PairwiseCrossEntropy;
						cout << "Xentropy pairwise" << endl;
					}else if (strcmp(argv[i + 1], "crossentropyadadelta") == 0) {
						lossFunction = CrossEntropyAdadelta;
						cout << "Xentropy adadelta" << endl;
					}else if (strcmp(argv[i + 1], "crossentropymomentum") == 0) {
						lossFunction = CrossEntropyMomentum;
						cout << "Xentropy momentum" << endl;
					}else if(strcmp(argv[i + 1], "pairwisecontinuous") == 0) {
						lossFunction = PairwiseContinuous;
						cout << "PairwiseContinuous" << endl;
					}
					i++;
				}else if (strcmp(argv[i], "-savemodel") == 0) {
					modelfile = argv[i + 1];
					i++;
				} else {
					cout << "Not enough or invalid arguments, please try again." << endl;
					cout << "Usage is -mode (full, bandit) -loadmodel (none) -trainf -lrate -lambda -clip -nsentence (full) -epoch -loss: (bayes, crossentropy, crossentropymomentum, crossentropyadadelta pairwise, probit, pairwisecrossentropy, pairwisecontinuous) -savemodel" << endl;
				 	exit(0);
            		}
			}
		}
	}

	//LOAD DATA
	Mat Y;
	vector<StrMat> X;
	FileLoader::loadChunking(trainf, Y, X);
	const int nStates = 3*3;
	//SOME PARAMETERS
	const float L_RATE = atof(lrate);
	const int EPOCHS_S = (int)atof(epoch_s);
	const int EPOCHS_E = (int)atof(epoch_e);
	const float LAMBDA = atof(lambda);
	const float CLIP = atof(clip);
	int N_SENTENCES;
	if(allSentences) {
		N_SENTENCES = Y.nRows();
	}else {
		N_SENTENCES = (int)atof(nsentence);
	}

		

	//PRINT BEFORE WE START
	cout << endl;
	cout << "ChRisF Chunking Learner" << endl;
	cout << "======" << endl;
	cout << "READ IN PARAMETERS:" << endl;
	cout << "Number of sentences in dataset: " << Y.nRows() << "\n";
	
	cout << "USER-SET PARAMETERS..\n";
	cout << "Learning rate: " << L_RATE << "\n";
	cout << "Regularization rate: " << LAMBDA << endl;
	cout << "Clip probability (xentropy only): " << CLIP << endl;
	cout << "Number of epochs: " << EPOCHS_S << ", " << EPOCHS_E << "\n";
	cout << "Number of sentence to learn: " << N_SENTENCES << "\n";
	if(train_mode == FULL_INFORMATION_MODE)
		cout << "Learning mode: full information." << endl;
	if(train_mode == BANDIT_MODE)
		cout << "Learning mode: bandit." << endl;

	//INIT OUR MODEL.
	CRFChunkModel model; 
	if(toLoadModel) {  //load from prev model file
		vector<pair<int, float> > modelWeights;
		FileLoader::loadModel(loadmodelfile, modelWeights);
		model.init(nStates, L_RATE, LAMBDA, CLIP, modelWeights, lossFunction);
	}else {
		model.init(nStates, L_RATE, LAMBDA, CLIP, lossFunction); 
	}

	vector<float> losses;
	ofstream myfile;

	for (int e = EPOCHS_S; e < EPOCHS_E; e++) {
		cout << "\nEPOCH " << e << "\n";

		//LEARNING PROCESS. 
		for (int i = 0; i < N_SENTENCES; i++) {
			if (train_mode == FULL_INFORMATION_MODE)
				model.fullInformationLearn(i, X[i], Y.row(i));
			else if (train_mode == BANDIT_MODE)
				model.banditUpdate(i, e, X[i], Y.row(i));
		}

		//SAVE MODEL
		myfile.open(string(modelfile) + "/" + to_string(e) + ".weight");
		vector<pair<int, float> > modelWeights = model.getModel();
		for(int i = 0; i < modelWeights.size(); i++) {
			myfile << modelWeights[i].first << " " << modelWeights[i].second << endl;
		}
		myfile.close();

		//SAVE GRADIENT
		/*
		myfile.open(string(modelfile) + "/" + to_string(e) + ".grad");
		vector<pair<int, float> > modelGrad = model.getGradient();
		for(int i = 0; i < modelGrad.size(); i++) {
			myfile << modelGrad[i].first << " " << modelGrad[i].second << endl;
		}
		myfile.close();

		model.resetGradient();
		*/

		


	}


  	return 0;
}

