mex -largeArrayDims beam_search.cpp

cd AQ/utils;
mex -largeArrayDims euc_nn_mex.cc
mex -largeArrayDims kmeans_iter_mex.cc
cd ../..

cd netsmf;
mex LDOPTIMFLAGS="$LDOPTIMFLAGS -fopenmp -O2" -lgomp -largeArrayDims deepwalk_mex.cc BinaryGraphWalker.cc WeightGraphWalker.cc GraphWalker.cc; 
cd ..