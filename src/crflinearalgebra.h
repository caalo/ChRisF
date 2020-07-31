#ifndef __LINALGEBRA
#define __LINALGEBRA

#include <limits>

#include <iostream>

#include <vector>
#include <fstream>
#include <assert.h> 
#include <sstream>
#include <string>

#include <math.h>
#include <stdlib.h> 

//std::numeric_limits<double>::infinity()

using namespace std;

class Mat;

//typedef Vec std::vector<double>;
//typedef Mat std::vector<std::vector<doubel> >;



class Vec {
	public:
		Vec(int size) {
			assert(size > 0);
			v.resize(size);
			std::fill(v.begin(), v.end(), 0);
		}
		
		Vec(vector<double> &v2) {
			v = v2;
		}

		Vec() {
			//empty constructor for populating vector while reading in a file
		}
		void add(double n) {
			v.push_back(n); //for Vec() constructors.
		}

		int size() {
			return v.size();
		}

		void zeros(int i) {
			assert(i >= 0);
			//clear();
			v.resize(i);
			std::fill(v.begin(), v.end(), 0);
		}

		void ones(int i) {
			assert(i >= 0);
			//clear();
			v.resize(i);
			std::fill(v.begin(), v.end(), 1);
		}

		void ones() {
			std::fill(v.begin(), v.end(), 1);
		}

		void clear() {
			v.clear();
		}

		double max() { 
			double lmax = v[0];
			for (int i = 1; i < size(); i++) {
				if(v[i] > lmax) 
					lmax = v[i];
			}
			return lmax;
		}

		void takeExp() {
			for (int i = 0; i < size(); i++) {
				v[i] = exp(v[i]);
			}
		}

		double sum() {
			double s = 0;
			for (int i = 0; i < size(); i++) {
				s += v[i];
			}
			return s;
		}

		Vec operator+(Vec &a) {
			assert(v.size() == a.size());
			Vec result = *this;
			for(int i = 0; i < v.size(); i++) {
				result(i) += a(i);
			}
			return result;
		}

		Vec operator*(double c) {
			Vec result = *this;
			for(int i = 0; i < v.size(); i++) {
				result(i) *= c;
			}
			return result;
		}

		Vec operator-(Vec &a) {
			assert(v.size() == a.size());
			Vec result = *this;
			for(int i = 0; i < v.size(); i++) {
				result(i) -= a(i);
			}
			return result;
		}

		Vec operator-(double a) {
			Vec result = *this;
			for(int i = 0; i < v.size(); i++) {
				result(i) -= a;
			}
			return result;
		}


		double& operator() (int index) {
			assert(index >= 0);
			assert(index < v.size());
			return v[index];
		}
		
		double operator* (Vec &a) { //vector dot product
			assert(v.size() == a.size());	
			double result = 0;
			for (int i = 0; i < v.size(); i++) {
				result += a(i) * v[i];
			}
			return result;
		}

		Vec operator*(double &n) { // numerical scaling
			Vec result = *this;
			for(int i = 0; i < v.size(); i++) {
				result(i) *= n;
			}
			return result;
		}
		
			
		Vec operator* (Mat &m);
		/*
		void operator=(Vec &a) { //copy over
			assert(v.size() == a.size());	
			for (int i = 0; i < v.size(); i++) {
				v[i] = a(i);
			}
		}*/

		void print(string message) {
			cout << message << endl;
			for (int i = 0; i < v.size() - 1; i++) {
				cout << v[i] << ", ";
			}
			cout << v[v.size() - 1] << endl;
		}

		string toString() {
			string str = "";
			for (int i = 0; i < v.size() - 1; i++) {
				str = str + to_string(v[i]) + ", ";
			}
			return str;
		}


	private:

		//static int ROW = 0; //direction of our vector
		//static int COL = 1; 
		vector<double> v;

};

class Mat {
	public:
		Mat(int r, int c) {
			assert (r > 0);
			assert (c > 0);
			m.resize(r);
			for (int i = 0; i < r; i++) {
				Vec vec(c);
				//m.push_back(vec);
				m[i] = vec;
			}
		}

		Mat() {
			//empty constructor for addRow( ) 	
		}

		//load from file
		Mat(string& fname) {
			std::ifstream file (fname.c_str(), std::ifstream::in);
			if(file) {
				string line;
				while(getline(file, line)) {	
					istringstream is(line);
	    				double n;
					Vec row;
	    				while(is >> n) {
						row.add(n);
					}
					m.push_back(row);
				}
				file.close();
			} else {
				cout << "Error opening file: " << fname << endl;
			}			
		}

		void addRow(Vec v) {
			m.push_back(v);
		}

		void zeros(int r, int c) {
			assert(r >= 0);
			assert(c >= 0);
			//clear();
			m.resize(r);
			for (int i = 0; i < r; i++) {
				Vec vec(c);
				m[i] = vec;
			}
		}

		void ones(int r, int c) {
			assert(r >= 0);
			assert(c >= 0);
			//clear();
			m.resize(r);
			for (int i = 0; i < r; i++) {
				Vec vec;
				vec.ones(c);
				m[i] = vec;
			}
		}

		void ones() {
			for (int i = 0; i < nRows(); i++) {
				m[i].ones();
			}
		}
	
		void takeExp() {
			for(int i = 0; i < nRows(); i++)
				m[i].takeExp();

		}

		void clear() {
			m.clear();
		}
		
		int nRows() {
			return m.size();
		}

		int nCols() {
			return m[0].size();
		}

		Vec& row(int i) {
			assert(i >= 0 && i < nRows());
			return m[i];
		
		}

		Mat operator+(Mat &a) {
			assert(nRows() == a.nRows());
			assert(nCols() == a.nCols());
			Mat result = *this;
			for (int i = 0; i < a.nRows(); i++) {
				for (int j = 0; j < a.nCols(); j++) {
					result(i, j) += a(i, j);
				}
			}
			return result;
		}

		Mat operator-(Mat &a) {
			assert(nRows() == a.nRows());
			assert(nCols() == a.nCols());
			Mat result = *this;
			for (int i = 0; i < a.nRows(); i++) {
				for (int j = 0; j < a.nCols(); j++) {
					result(i, j) -= a(i, j);
				}
			}
			return result;
		}
	
		Vec operator*(Vec &v) {
			assert(v.size() == nCols());
			Vec result(v.size());
			for (int i = 0; i < nRows(); i++) {
				for(int j = 0; j < nCols(); j++) {
					result(i) += m[i](j) * v(j);
				}
			}
			return result;
		}

		
		Mat operator*(Mat &m2) {
			assert(nCols() == m2.nRows());
			Mat result(nRows(), m2.nCols());

			for(int i = 0; i < result.nRows(); i++) {
				for(int j = 0; j < result.nCols(); j++) {
					for (int k = 0; k < nCols(); k++) {
						result(i, j) += m[i](k) * m2(k, j);
					}
				}
			}
			return result;
			
		}

		Mat operator*(double n) {
			Mat result = *this;
			for (int i = 0; i < nRows(); i++) {
				for(int j = 0; j < nCols(); j++) {
					result(i, j) = result(i, j) * n;
				}
			}
			return result;
		}


		double& operator() (int row, int col) {
			assert(row >= 0);
			assert(col >= 0);
			if(row >= nRows()) {
				cout << "requested row: " << row << " nRows: " << nRows() << endl;
				//print("error");
			}
				
			assert(row < nRows());
			//assert(col < nCols()); #COLS WRONG
			return m[row](col);
		}

		void print(string message) 	{
			cout << message << endl;
			cout << "nRows " << nRows() << " nCols " << nCols() << endl;
			for (int i = 0; i < nRows(); i++) {
				for (int j = 0; j < nCols() - 1; j++) {
					cout << m[i](j) << ", ";
				}
				cout << m[i](nCols() - 1) << endl;
			}

		}
		
	private:
		vector<Vec> m;
};

Vec Vec::operator*(Mat& m) { //vec-mat product. left here due to forward declaration.
	assert(size() == m.nRows());
	Vec result(this->size());	
	for (int j = 0; j < m.nCols(); j++) {
		for(int i = 0; i < m.nRows(); i++) {
			result(j) += m(i, j) * v[i];
		}
	}
	//v = result;
	//return *this;
	return result;
}



#endif
