#include "WeightGraphWalker.h"

#include <cassert>
#include <numeric>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <omp.h>


WeightGraphWalker::WeightGraphWalker(const std::vector<VertexId>& indices,
        const std::vector<VertexId>& indptr, 
        const std::vector<float>& data_,
        int T, 
        int negative,
        int num_threads,
        bool use_log1p)
    : GraphWalker(indices, indptr, T, negative, num_threads, use_log1p), data(data_) {

    for (VertexId v = 0; v < indptr.size() - 1; ++v) {
        degree.push_back(std::accumulate(data.begin() + indptr[v], data.begin() + indptr[v + 1], 0.0));
    }
    prefix_sum.resize(data.size());
    for (VertexId v = 0; v < indptr.size() - 1; ++v) {
        std::partial_sum(data.begin() + indptr[v], data.begin() + indptr[v + 1], prefix_sum.begin() + indptr[v]);
    }
    print("weighted network\n");
}


VertexId WeightGraphWalker::randomWalk(VertexId u, int step, double& Z,
        unsigned* seed) const {
    for (;step--;) {
        // u's neighbors are indices[indptr[i]:indptr[i+1]]
        double ratio = (double)rand_r(seed) / RAND_MAX;
        int head = indptr[u], tail = indptr[u+1] - 1, pos = tail;
        double generalized_out_degree = prefix_sum[tail];
        for (;head < tail;) {
            int mid = (head + tail) >> 1;
            if (prefix_sum[mid] >= ratio * generalized_out_degree) {
                tail= mid - 1;
                pos = mid;
            } else {
                head = mid + 1;
            }
        }

        u = indices[pos];
        Z += 1. / data[pos];
    }
    return u;
}

void WeightGraphWalker::samplePath(VertexId u, VertexId v, double w, int r, unsigned* seed,
        std::vector<ValuedVertexPair>& sampled_pair) const {
    int k = rand_r(seed) % r + 1;
    double Z_half = 1. / w;
    VertexId u_ = randomWalk(u, k - 1, Z_half, seed);
    VertexId v_ = randomWalk(v, r - k, Z_half, seed);
    if (u_ > v_) {
        std::swap(u_, v_);
    }

    // add record (u_, v_, r / Z_half)
    sampled_pair.push_back(std::make_pair(std::make_pair(u_, v_), float(r / Z_half)));
}

void WeightGraphWalker::sampling(int round, int num_threads,
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
            + std::string("_thread_") + std::to_string(this_thread);

        print("[thread %d] thread name is %s\n", this_thread, thread_name);
        unsigned seed = std::hash<std::string>{}(thread_name);

        std::vector<ValuedVertexPair> sampled_pairs;
        std::vector<ValuedVertexPair> *&counter = counters[this_thread];
        std::vector<ValuedVertexPair> *counter_tmp = new std::vector<ValuedVertexPair>;

        print("[thread %d] set seed %d\n", this_thread, seed);
        int my_round= ceil((double)round / num_threads);

        for (int i=0; i<my_round; ++i) {
            for (VertexId u=0; u+1 < indptr.size(); ++u) {
                for (size_t j=indptr[u]; j<indptr[u+1]; ++j) {
                    VertexId v = indices[j];
                    for (int r=1; r<T; ++r) {
                        samplePath(u, v, data[j], r, &seed, sampled_pairs);
                    }
                }
            }
            if ((i + 1) % check_point == 0 || i + 1 == my_round) {
                float max_val = merge(*counter, *counter_tmp, sampled_pairs);
                std::swap(counter, counter_tmp);
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

float WeightGraphWalker::merge(const std::vector<ValuedVertexPair>& counter,
        std::vector<ValuedVertexPair>& tmp,
        std::vector<ValuedVertexPair>& sampled_pairs) {
    float max_val = 0;
    float w;
    std::sort(sampled_pairs.begin(), sampled_pairs.end());

    std::vector<ValuedVertexPair>::const_iterator iter = counter.cbegin();
    for (size_t i = 0, j = 0; i < sampled_pairs.size(); i = j) {
        w = sampled_pairs[i].second;
        for (j = i + 1; j < sampled_pairs.size()
                && sampled_pairs[j].first == sampled_pairs[i].first; ++j) {
            w += sampled_pairs[j].second;
        }
        for (;iter != counter.end() && iter->first < sampled_pairs[i].first; ++iter) {
            max_val = std::max(max_val, iter->second);
            tmp.push_back(*iter);
        }
        if (iter != counter.end() && iter->first == sampled_pairs[i].first) {
            max_val = std::max(max_val, w + iter->second);
            tmp.push_back(
                    std::make_pair(iter->first, w + iter->second));
            ++iter;
        } else {
            max_val = std::max(max_val, w);
            tmp.push_back(std::make_pair(sampled_pairs[i].first, w));
        }
    }
    for (;iter != counter.end(); ++iter) {
        max_val = std::max(max_val, iter->second);
        tmp.push_back(*iter);
    }
    return max_val;
}





