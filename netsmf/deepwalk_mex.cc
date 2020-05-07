#include "BinaryGraphWalker.h"
#include "WeightGraphWalker.h"
#include "mex.h"

//function mat = deepwalk_fast(At, T, b, num_thread, rounds, use_log1p, weighted)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	mwIndex* ir = mxGetIr(prhs[0]);
	mwIndex* jc = mxGetJc(prhs[0]);
	mwSize n = mxGetN(prhs[0]);
	double* val = mxGetPr(prhs[0]);
	mwSize T = (mwSize)mxGetScalar(prhs[1]);
	mwSize b = (mwSize)mxGetScalar(prhs[2]);
	mwSize num_thread = (mwSize)mxGetScalar(prhs[3]);
	mwSize rounds = (mwSize)mxGetScalar(prhs[4]);
	mxLogical use_log1p = (mxLogical)mxGetScalar(prhs[5]);
	mxLogical weighted = (mxLogical)mxGetScalar(prhs[6]);

	mwSize nnz = jc[n];
	std::vector<mwIndex> indptr(n + 1);
	std::vector<mwIndex> indices(nnz);
	std::vector<float> data(nnz);

	for (mwIndex v = 0; v < n + 1; ++v) {
		indptr[v] = jc[v];
	}
	for (mwIndex v = 0; v < nnz; ++v) {
		indices[v] = ir[v];
		data[v] = (float)val[v];
	}

	GraphWalker* walker = weighted ?
		(GraphWalker*)new WeightGraphWalker(indices, indptr, data, T, b, num_thread, use_log1p) :
		(GraphWalker*)new BinaryGraphWalker(indices, indptr, T, b, num_thread, use_log1p);
	walker->sampling(rounds, num_thread, "localhost", 2);
	walker->transformation();
	auto ijk = walker->redsvd();
	nnz = ijk->size();
	plhs[0] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
	double* I = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
	double* J = mxGetPr(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
	double* K = mxGetPr(plhs[2]);
	for (mwIndex v = 0; v < nnz; ++v) {
		I[v] = (*ijk)[v].first.first + 1;
		J[v] = (*ijk)[v].first.second + 1;
		K[v] = (*ijk)[v].second;
	}
	delete walker;
	delete ijk;
}