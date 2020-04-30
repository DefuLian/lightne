#define idx(m, k) (m * K + k)
#define code(m, k, d) C[d + D * idx(m,k) ]
#include "mex.h"
#include "string.h"
#include <queue>
//typedef mwSize int;
//function [B] = search(x, C, M, n)
// x is a column vector of D x 1
// C is matrix of shape D x (M x K)
// n is search depth
struct Data
{
	int cmhist_idx;
	mwSize m;
	mwSize k;
	double weight;
	Data(int idx, mwSize m, mwSize k, double w) :m(m), k(k), weight(w), cmhist_idx(idx)
	{
	}
	bool operator<(const Data& rhs) const
	{
		return this->weight < rhs.weight;
	}

};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	double* x = mxGetPr(prhs[0]);
	mwSize D = mxGetM(prhs[0]);
	double* C = mxGetPr(prhs[1]);
	mwSize mk = mxGetN(prhs[1]);
	mwSize M = (mwSize)mxGetScalar(prhs[2]);
	mwSize K = mk / M;
	mwSize n = (mwSize)mxGetScalar(prhs[3]);
	
	bool debug = false;

	plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL);
	double* B = mxGetPr(plhs[0]);

	std::priority_queue<Data> q;
	for (mwSize m = 0; m < M; ++m) {
		for (mwSize k = 0; k < K; ++k) {
			double dist = 0;
			for (mwSize d = 0; d < D; ++d) {
				dist += (code(m, k, d) - x[d]) * (code(m, k, d) - x[d]);
			}
			Data data(0, m, k, dist);
			if (q.size() < n) {
				q.push(data);
			}
			else if (data < q.top()) {
				q.pop();
				q.push(data);
			}
		}
	}
	
	int* cmhist = new int[n * M];
	memset(cmhist, -1, sizeof(int) * n * M);
	double* res = new double[n * D];
	for (mwSize i = 0; i < n; ++i) {
		memcpy((res + i * D), x, sizeof(double) * D);
	}
	int* cmhist_backup = new int[n * M];
	double* res_backup = new double[n * D];
	for (mwSize iter = 1; iter < M + 1; ++iter) {
		memcpy(cmhist_backup, cmhist, sizeof(int) * n * M);
		memcpy(res_backup, res, sizeof(double) * n * D);
		mwSize i = 0;
		while (!q.empty()) {
			auto data = q.top();
			memcpy(cmhist + i * M, cmhist_backup + data.cmhist_idx * M, sizeof(int) * M);
			cmhist[i * M + data.m] = data.k;
			for (mwSize d = 0; d < D; ++d) {
				res[i * D + d] = res_backup[data.cmhist_idx * D + d] - code(data.m, data.k, d);
			}
				
			if (debug) {
				mexPrintf("%d\t%d\t", iter, data.cmhist_idx);
				for (mwSize m = 0; m < M; ++m) {
					if (cmhist[i * M + m] >= 0) {
						mexPrintf("%d:%2d\t", m, cmhist[i * M + m]);
					}
				}
				mexPrintf("%f\t", data.weight);
				double res_norm = 0;
				for (mwSize d = 0; d < D; ++d) {
					res_norm += res[i * D + d] * res[i * D + d];
				}
				mexPrintf("%f\n", res_norm);
			}			
			q.pop();
			i++;
		}
		if (iter == M) {
			break;
		}
		for (mwSize i = 0; i < n; ++i) {
			for (mwSize m = 0; m < M; ++m) {
				if (cmhist[i * M + m] < 0) {
					for (mwSize k = 0; k < K; ++k) {
						double dist = 0;
						for (mwSize d = 0; d < D; ++d) {
							dist += (code(m, k, d) - res[i * D + d]) * (code(m, k, d) - res[i * D + d]);
						}
						Data data(i, m, k, dist);
						if (q.size() < n) {
							q.push(data);
						}
						else if (data < q.top()) {
							q.pop();
							q.push(data);
						}
					}
				}
			}
		}
	}
	
	for (mwSize m = 0; m < M; ++m) {
		B[m] = cmhist[(n-1) * M + m] + 1;
	}
	delete[] cmhist;
	delete[] cmhist_backup;
	delete[] res;
}