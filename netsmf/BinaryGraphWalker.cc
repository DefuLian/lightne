#include "BinaryGraphWalker.h"

#include <cassert>
#include <numeric>
#include <fstream>
#include <cassert>
#include <functional>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <omp.h>


BinaryGraphWalker::BinaryGraphWalker(const std::vector<VertexId>& indices,
        const std::vector<VertexId>& indptr, 
        int T_,
        int negative_,
        int num_threads_,
        bool use_log1p_)
    : GraphWalker(indices, indptr, T_, negative_, num_threads_, use_log1p_) {
    //assert(indptr.size() == degree.size() + 1);
    
    for (VertexId row = 1; row < indptr.size(); ++row) {
        degree.push_back(indptr[row] - indptr[row - 1]);
    }
    print("unweighted network\n");
}


VertexId BinaryGraphWalker::randomWalk(VertexId u, int step,
        unsigned* seed) const {
    for (;step--;) {
        // u's neighbors are indices[indptr[i]:indptr[i+1]]
        int offset = rand_r(seed) % (indptr[u+1] - indptr[u]);
        u = indices[indptr[u] + offset];
    }
    return u;
}

void BinaryGraphWalker::samplePath(const VertexId u, const VertexId v, int r, unsigned* seed,
        std::vector<VertexPair>& sampled_pairs) const {
    int k = rand_r(seed) % r + 1;
    VertexId u_ = randomWalk(u, k - 1, seed);
    VertexId v_ = randomWalk(v, r - k, seed);
    // add record (u_, v_, 1)

    if (u_ > v_) {
        std::swap(u_, v_);
    }

    sampled_pairs.push_back(std::make_pair(u_, v_));
}

void BinaryGraphWalker::sampling(int round, int num_threads,
        const std::string& machine,
        int check_point) {
    omp_set_num_threads(num_threads);

    std::vector<std::vector<ValuedVertexPair>*> counters;
    for (int i = 0; i < num_threads; ++i) {
        counters.push_back(new std::vector<ValuedVertexPair>);
    }

    #pragma omp parallel default(shared)
    {
        int this_thread = omp_get_thread_num();
        std::string thread_name = std::string("machine_") + machine
            + std::string("_thread_") + std::to_string(this_thread); // + std::string("_time_") + std::to_string(time(0));

        print("[thread %d] thread name is %s\n", this_thread, thread_name.c_str());
        unsigned seed = std::hash<std::string>{}(thread_name);

        std::vector<VertexPair> sampled_pairs;
        std::vector<ValuedVertexPair> *&counter = counters[this_thread];
        std::vector<ValuedVertexPair> *counter_tmp = new std::vector<ValuedVertexPair>;

        print("[thread %d] set seed %d\n", this_thread, seed);
        int my_round= ceil((double)round / num_threads);

        for (int i=0; i<my_round; ++i) {
            for (VertexId u=0; u+1 < indptr.size(); ++u) {
                for (size_t j=indptr[u]; j<indptr[u+1]; ++j) {
                    VertexId v = indices[j];
                    for (int r=1; r<=T; ++r) {
                        // printf("%d %d %d\n", u, v, r);
                        samplePath(u, v, r, &seed, sampled_pairs);
                    }
                }
            }
            if ((i + 1) % check_point == 0 || i + 1 == my_round) {
                float max_val = merge(*counter, *counter_tmp, sampled_pairs);
                std::swap(counter, counter_tmp);
                sampled_pairs.clear();
                counter_tmp->clear();
                print("[thread %d] complete %d rounds, size of counter=%d, counter. max_val=%f\n", this_thread, i + 1, counter->size(), max_val);
            }
        }
        print("[thread %d] finish job\n", this_thread);
        delete counter_tmp;
    }

    // now we have a list of counters, we want to merge them in a binary tree way --- from leaf to root
    while (counters.size() > 1) {
        print("%d counters to merge.\n", counters.size());
        size_t n_half = (counters.size() + 1) >> 1;
        omp_set_num_threads(counters.size() >> 1);

        #pragma omp parallel default(shared)
        {
            int this_thread = omp_get_thread_num();
            print("merge counter %d and %d\n", this_thread, n_half + this_thread);
            std::vector<ValuedVertexPair> *counter_tmp = merge_counters(*counters[this_thread], *counters[n_half + this_thread]);

            delete counters[this_thread];
            delete counters[n_half + this_thread];
            counters[this_thread] = counter_tmp;
        }

        counters.resize(n_half);
    }
    counter_merged = counters[0];
}

float BinaryGraphWalker::merge(const std::vector<ValuedVertexPair>& counter,
        std::vector<ValuedVertexPair>& tmp,
        std::vector<VertexPair>& sampled_pairs) {
    float max_val = 0;
    std::sort(sampled_pairs.begin(), sampled_pairs.end());

    std::vector<ValuedVertexPair>::const_iterator iter = counter.cbegin();
    for (size_t i = 0, j = 0; i < sampled_pairs.size(); i = j) {
        for (j = i + 1; j < sampled_pairs.size() && sampled_pairs[j] == sampled_pairs[i]; ++j);
        for (;iter != counter.end() && iter->first < sampled_pairs[i]; ++iter) {
            max_val = std::max(max_val, iter->second);
            tmp.push_back(*iter);
        }
        if (iter != counter.end() && iter->first == sampled_pairs[i]) {
            max_val = std::max(max_val, j - i + iter->second);
            tmp.push_back(
                    std::make_pair(iter->first, j - i + iter->second));
            ++iter;
        } else {
            max_val = std::max(max_val, float(j - i));
            tmp.push_back(std::make_pair(sampled_pairs[i], float(j - i)));
        }
    }
    for (;iter != counter.end(); ++iter) {
        max_val = std::max(max_val, iter->second);
        tmp.push_back(*iter);
    }
    return max_val;
}




