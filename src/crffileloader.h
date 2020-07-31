#ifndef __FILELOADER
#define __FILELOADER

#include <iostream>
#include <algorithm>  
//#include <random>
#include <math.h>
#include "crflinearalgebra.h"

using namespace std;

typedef vector<string> StrVec;
typedef vector<StrVec> StrMat;


class FileLoader {
	public:
/*
	//OCR TASK ONLY
	//...given a fold..
	static void loadOCR(char* fileName, vector <Vec> &Y, vector<Mat> &X) {
		int prevSentenceID;
		bool firstLine = true;
		std::ifstream file (fileName, std::ifstream::in);

		if(file) {
			string line;
			Mat x;
			Vec y;
			while(getline(file, line)) {	
				istringstream is(line);
		   		double n;
				double ylabel;
				//first item should be the letter label.
				is >> ylabel;
				//second item should be sentence number.
				int currentSentenceID;
				is >> currentSentenceID;
				if(firstLine) {
					prevSentenceID = currentSentenceID;
					firstLine = false;
				}
				if (currentSentenceID != prevSentenceID)  {
					//we have moved on to a new sentence. add current sentence to our vector of Vec and Mat.
					Y.push_back(y);
					X.push_back(x);
					//restart
					x.clear();
					y.clear();
				
				}
				prevSentenceID = currentSentenceID;
				//third item is fold number.
				is >> n;
				//the rest should be pixels
				Vec pixs;
		   		while(is >> n) {
					pixs.add(n);
				}
				//add everything to our word
				x.addRow(pixs);
				y.add(ylabel);

			}
			file.close();
		} else {
			cout << "Error opening file: " << fileName << endl;
		}		
	}
*/
	//CHUNK TASK
	static void loadChunking(char* fileName, Mat &Y, vector<StrMat> &X) {
		int prevSentenceID;
		bool firstLine = true;
		static int sqStates = 3;
		/*
		The Chunk model is a second order markov CRF. Each y_i = c_i c_i-1 with c_i, c_i-1 ranging 0, 1, 2 to 
		represent B, I, and O respectively. To "linearize" this encoding, we encode in our vector 
		y(i) = sqStates * c_i-1 + c_i, giving possible values 0,...,8. 
		*/

		std::ifstream file (fileName, std::ifstream::in);

		if(file) {
			string line;
			StrMat x; //local to one sentence
			Vec y; //local to one sentence
			int prevY = 2;
			while(getline(file, line)) {	
				istringstream is(line);
		   		string temp;
				//first item should the word.
				is >> temp;
				if (temp.compare("") == 0) {
					//blank line, move on to the next sentence! but first, add end state 
					Y.addRow(y);
					X.push_back(x);
					y.clear();
					x.clear(); 
					prevY = 2; //"assume it is outside for prev". matches what we have in the paper.
				} else {
					StrVec xrow;
					xrow.push_back(temp);
					//second item should be POS tag
					is >> temp;
					xrow.push_back(temp);
					x.push_back(xrow);
					//third item should be chunk tag
					is >> temp;
					if(temp.compare("") == 0)
						cout << "Something's funny with the file." << endl;
					//WHEN ADDING TO VECTOR Y, CONVERT PAIRS TO LINEAR INDEX
					if(temp.compare("B-NP") == 0) {
						y.add(prevY * sqStates + 0); //B 
						prevY = 0;
					} else if(temp.compare("I-NP") == 0) { 
						y.add(prevY * sqStates + 1); //I
						prevY = 1;
					} else {
						y.add(prevY * sqStates + 2); //O
						prevY = 2;
					}
				}
				
				
			}
			file.close();
		} else {
			cout << "Error opening file: " << fileName << endl;
		}		
	}

	static void loadModel(char* fileName, vector<pair<int, float> > &modelWeights) {
		std::ifstream file (fileName, std::ifstream::in);
		if(file) {
			string line;
			while(getline(file, line)) {	
				istringstream is(line);
		   		string temp;
				pair<int, float> p;
				//first item should be the array number
				is >> temp;
				p.first = stoi(temp);
				is >> temp;
				p.second = stof(temp);
				modelWeights.push_back(p);
			}
			file.close();
		} else {
			cout << "Error opening model file: " << fileName << endl;
		}		
	}



};
#endif
