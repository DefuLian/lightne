#include "GraphWalker.h"
#include <numeric> // std::partial_sum
#include <omp.h>


GraphWalker::GraphWalker(const std::vector<VertexId>& indices_,
        const std::vector<VertexId>& indptr_,
        int T_, 
        int negative_,
        int num_threads_,
        bool use_log1p_)
    : indices(indices_), indptr(indptr_), T(T_), negative(negative_), num_threads(num_threads_), use_log1p(use_log1p_){
    
    sparsifier_lower = new std::vector<ValuedVertexPair>();
    sparsifier_upper = new std::vector<ValuedVertexPair>();
    counter_merged = NULL;
}

void GraphWalker::transformation() {

    print("transformation ...\n");
    double M = 0;
    for (auto iter = counter_merged->cbegin(); iter != counter_merged->cend(); ++iter) {
        M += iter->second * 2;
    }
    print("total number of samples=%f\n", M);
    double num_edges = (double)indices.size();
    double vol = 0.0;
    for (auto const& val : degree) {
        vol += val;
    }
    print("vol(G)=%f\n", vol);
    double factor = vol * num_edges / M / negative;
    VertexId src, dst;
    double val;
    std::vector<VertexId> nnz_lower_row(degree.size(), 0);

    size_t nnz_lower = 0;
    sparsifier_upper->clear();
    sparsifier_lower->clear();
    if (use_log1p) {
        print("using log1p...\n");
    } else {
        print("using truncated logarithm...\n");
    }


    for (auto iter = counter_merged->cbegin(); iter != counter_merged->cend(); ++iter) {
        src = iter->first.first;
        dst = iter->first.second;
        val = src != dst ? iter->second : iter->second * 2;
        if(use_log1p)
            val = log1p(val * factor / degree[src] / degree[dst]);
        else
            val = log(val * factor / degree[src] / degree[dst]);
        if (val > 0) {
            sparsifier_upper->push_back(std::make_pair(iter->first, (float)val));
            if (src != dst) {
                ++nnz_lower_row[dst];
                ++nnz_lower;
            }
        }
    }
    print("after log, #nnz in upper triangle and diagonal reduces to %d (from %d)\n", sparsifier_upper->size(), counter_merged->size());
    counter_merged->clear();
    delete counter_merged;


    print("constructing lower triangle ...\n");
    // now, sparsifier stores upper triangle + diagonal
    // we will re-use sparsifier_lower to store lower triangle
    std::vector<VertexId> lower_indptr(degree.size() + 1, 0);
    std::partial_sum(nnz_lower_row.begin(), nnz_lower_row.end(), lower_indptr.begin() + 1);

    sparsifier_lower->resize(nnz_lower);
    print("lower triangle has %d nnz.", nnz_lower);
    for (auto riter = sparsifier_upper->crbegin(); riter != sparsifier_upper->crend(); ++riter) {
        src = riter->first.first;
        dst = riter->first.second;
        if (src == dst) {
            continue;
        }
        auto iter = sparsifier_lower->begin() + lower_indptr[dst] + (--nnz_lower_row[dst]);
        iter->first.first = dst;
        iter->first.second = src;
        iter->second = riter->second;
    }
    print("lower triangle constructed.\n");
}

const std::vector<ValuedVertexPair>* GraphWalker::redsvd() {
   
    print("prepare output ...\n");
    std::vector<ValuedVertexPair>* ll = new std::vector<ValuedVertexPair>();
    ll->insert(ll->end(), sparsifier_lower->begin(), sparsifier_lower->end());
    ll->insert(ll->end(), sparsifier_upper->begin(), sparsifier_upper->end());
    
    sparsifier_upper->clear();
    sparsifier_lower->clear();
    delete sparsifier_upper;
    delete sparsifier_lower;
    return ll;
}

std::vector<ValuedVertexPair>* GraphWalker::merge_counters(const std::vector<ValuedVertexPair>& counter,
        const std::vector<ValuedVertexPair>& counter_other) {
    std::vector<ValuedVertexPair>::const_iterator iter1 = counter.cbegin();
    std::vector<ValuedVertexPair>::const_iterator iter2 = counter_other.cbegin();

    std::vector<ValuedVertexPair> *counter_tmp = new std::vector<ValuedVertexPair>;

    while (iter1 != counter.cend() && iter2 != counter_other.cend()) {
        if (iter1->first < iter2->first) {
            counter_tmp->push_back(*(iter1++));
        } else if (iter1->first > iter2->first) {
            counter_tmp->push_back(*(iter2++));
        } else {
            counter_tmp->push_back(
                    std::make_pair(iter1->first, iter1->second + iter2->second));
            ++iter1;
            ++iter2;
        }
    }

    for (;iter1 != counter.cend(); ++iter1) {
        counter_tmp->push_back(*iter1);
    }

    for (;iter2 != counter_other.cend(); ++iter2) {
        counter_tmp->push_back(*iter2);
    }
    return counter_tmp;
}

