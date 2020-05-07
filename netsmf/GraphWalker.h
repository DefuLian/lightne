#pragma once
#define _CRT_RAND_S
#include <vector>
#include <iostream>
#include <string>
#include <mex.h>
#include <stdlib.h> 
#define print mexPrintf
#define rand_r rand_s
//using VertexId = unsigned long; //uint32_t;
using VertexId = mwIndex; //uint32_t;
using EdgeId = mwIndex; //uint32_t;
using VertexPair = std::pair<VertexId, VertexId>;
using VertexPairCount = std::pair<VertexPair, unsigned int>;
using ValuedVertexPair = std::pair<std::pair<VertexId, VertexId>, float>;


/* indices, indptr, data
 * indices is array of column indices
 * data is array of corresponding nonzero values
 * indptr points to row starts in indices and data
 * length is n_row + 1, last item = number of values = length of both indices and data
 * nonzero values of the i-th row are data[indptr[i]:indptr[i+1]] with column indices indices[indptr[i]:indptr[i+1]]
 * item (i, j) can be accessed as data[indptr[i]+k], where k is position of j in indices[indptr[i]:indptr[i+1]]
 */

class GraphWalker {
public:
    GraphWalker(const std::vector<VertexId>& indices_,
        const std::vector<VertexId>& indptr_,
        int T, 
        int negative,
        int num_threads,
        bool use_log1p);

    const std::vector<VertexId> indices;
    const std::vector<VertexId> indptr;
    std::vector<float> degree;
    int T;
    int negative;
    int dim;
    int num_threads;
    bool use_log1p;

    std::vector<ValuedVertexPair> *sparsifier_upper, *sparsifier_lower;
    std::vector<ValuedVertexPair> *counter_merged;

    virtual void sampling(int round, int num_threads,
            const std::string& machine,
            int check_point) = 0;
    void transformation();
    const std::vector<ValuedVertexPair>* redsvd();

    static std::vector<ValuedVertexPair>* merge_counters(
        const std::vector<ValuedVertexPair>& counter,
        const std::vector<ValuedVertexPair>& counter_other);
};


