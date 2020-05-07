#pragma once

#include "GraphWalker.h"


class WeightGraphWalker : public GraphWalker {
public:
    WeightGraphWalker(const std::vector<VertexId>& indices,
            const std::vector<VertexId>& indptr,
            const std::vector<float>& data,
            int T,
            int negative,
            int num_threads,
            bool use_log1p);

    void samplePath(VertexId u, VertexId v, double w, int r, unsigned* seed,
            std::vector<ValuedVertexPair>& sampled_pairs) const;
    VertexId randomWalk(VertexId u, int step, double& Z, unsigned* seed) const;
    void sampling(int round, int num_threads,
            const std::string& machine,
            int check_point);

    static float merge(const std::vector<ValuedVertexPair>& counter,
            std::vector<ValuedVertexPair>& tmp,
            std::vector<ValuedVertexPair>& sampled_pairs);

    const std::vector<float> data;
    std::vector<float> prefix_sum;
};
