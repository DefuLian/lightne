function [B, Q] = network_hashing(net, varargin)
[ratio, gamma, max_iter, dim, alg, others] = process_options(varargin, 'ratio',1, ...
    'gamma',0, 'max_iter', 50, 'dim', 128, 'alg', 'binary');
net = deepwalk(net, others{:});
[U, S, V] = svds(net, dim);
if strcmp(alg, 'real')
    B = U * diag(sqrt(diag(S)));
    Q = V * diag(sqrt(diag(S)));
else
    B = proj_hamming_balance(U * S);
    M = net;
    Mt = net';
    prev_loss = inf;
    n = length(net);
    mtb = Mt * B;

    for iter=1:max_iter
        Q=proj_stiefel_manifold(mtb);
        if gamma>0
            B_h = sqrt(n) * proj_stiefel_manifold(B);
        else
            B_h = zeros(size(B));
        end
        curr_loss = loss_mf(net, B, Q) + gamma/2*sum(sum((B-B_h).^2));
        fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
        if abs(prev_loss - curr_loss) < 1e-6 * prev_loss
            break
        end
        prev_loss = curr_loss;
        if ratio<1
            s = select(sum(Q.* mtb),ratio);
            B1 = proj_hamming_balance(M*Q(:,s)+gamma*B_h(:,s));
            mtb(:,s) = mtb(:,s) + Mt * (B1-B(:,s));
            B(:,s) = B1;
        else
            B=proj_hamming_balance(M*Q+gamma*B_h);
            mtb=Mt*B;
        end
    end
end
end

function val = loss_mf(net, P, Q)
    val = sum(sum(net.^2)) - 2 * sum(sum((P.' * net) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    val = val / 2;
end
function s = select(b, ratio)
k=length(b);
d = floor(k*ratio);
[~,ind]=sort(b);
ind = ind(1:d);
s = false(k,1);
s(ind) = true;
end

function B = proj_hamming_balance(X)
    n = size(X, 1);
    c = median(X);
    B = (X - repmat(c, n, 1) >  0) * 2 - 1;
end
function W = proj_stiefel_manifold(A)
%%% min_W |A - W|_F^2, s.t. W^T W = I
[U, ~, V] = svd(A, 0);
W = U * V.';
end




