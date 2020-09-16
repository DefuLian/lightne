function [B, Q, varargout] = network_hashing_g(net, varargin)
[ratio, gamma, max_iter, dim, alg, num_codebooks, beta, others] = process_options(varargin, 'ratio',1, ...
    'gamma',0, 'max_iter', 50, 'dim', 128, 'alg', 'binary', 'M', -1, 'beta', 0.05);
beta=gamma;
gamma=1;
M_g = 1/beta * speye(size(net)) - net;
M_l = net;
if strcmp(alg, 'real')
    [U, S, V] = gsvds(M_g, M_l, dim);
    B = U * diag(sqrt(diag(S)));
    Q = V * diag(sqrt(diag(S)));
elseif strcmp(alg, 'binary')
    [U, S, V] = gsvds(M_g, M_l, dim);
    B = proj_hamming_balance(U * S);
    prev_loss = inf;
    n = length(net);
    mtb = (B' / M_g * M_l)';

    for iter=1:max_iter
        Q=proj_stiefel_manifold(mtb);
        if gamma>0
            B_h = sqrt(n) * proj_stiefel_manifold(B);
        else
            B_h = zeros(size(B));
        end
        curr_loss = loss_gmf(M_g, M_l, B, Q) + gamma/2*sum(sum((B-B_h).^2));
        fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
        if abs(prev_loss - curr_loss) < max(1e-6 * prev_loss, 1e-6)
            break
        end
        prev_loss = curr_loss;
        if ratio<1
            s = select(sum(Q.* mtb),ratio);
            B1 = proj_hamming_balance(M_g \ (M_l *Q(:,s))+gamma*B_h(:,s));
            mtb(:,s) = mtb(:,s) + ((B1-B(:,s))' / M_g * M_l )';
            B(:,s) = B1;
        else
            B=proj_hamming_balance(M_g \ (M_l * Q) + gamma*B_h);
            mtb = (B' / M_g * M_l)';
        end
    end
else
    [B, Q, code, codebooks] = network_quantizer_g(M_g, M_l, 'dim', dim, 'M', num_codebooks, 'max_iter', max_iter, 'alg', alg);
    if nargout>2
        varargout{1} = code;
    end
    if nargout>3
        varargout{2} = codebooks;
    end
end
end


function s = select(b, ratio)
k=length(b);
d = floor(k*ratio);
[~,ind]=sort(b);
ind = ind(1:d);
s = false(k,1);
s(ind) = true;
end






