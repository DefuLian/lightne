function [U, S, V] = gsvds(A, B, k, max_iter)
%%% |A^{-1}B - USV|_F^2
if nargin <4
    max_iter = 100;
end
m = size(A, 2);
U = randn(m, k) * 0.01;
for iter=1:max_iter
    V = proj_stiefel_manifold(((U' / A) * B)');
    curr_loss = loss_gmf(A, B, U, V);
    if curr_loss < 1e-2
            break
    end
    fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
    U = A \ (B * V);
end
[U, S, V1] = svd(U, 0);
V = V * V1;
end



