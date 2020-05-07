function S = deepwalk_fast(A, varargin)
[T, b, num_thread, rounds, use_log1p, weighted] = process_options(varargin, 'T',1, 'b', 1, 'num_thread', 4, 'rounds', 1000, 'use_log1p', false, 'weighted', false);
A = A - spdiags(spdiags(A, 0), 0, size(A,1), size(A,2));
[I,J,K] = deepwalk_mex(A.',T, b, num_thread, rounds, use_log1p, weighted);
S = sparse(I, J, K, size(A,1), size(A, 2));
end