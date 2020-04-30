function [B, C] = network_quantizer(X)
warning('off','MATLAB:rankDeficientMatrix')
%[max_iter, dim, others] = process_options(varargin, 'max_iter', 50, 'dim', 128);
%K = 256;
%net = deepwalk(net, others{:});
%[U, S, V] = svds(net, dim);
%for iter=1:max_iter
%    C = update_codebook(X, B, K);
%    B = update_assignment(
%end
rng(10)
[B, C]= AQ(X, 25, 32);
end

function [B, C] = AQ(X, max_iter, n)
[D, N] = size(X);
K = 256;
M = D / 8;
C = randn(D, M * K);
B = randi(K, M, N);
%B = PQ(X, M, K);
for iter=1:max_iter
    C = update_codebook(X, B, K);
    %error = get_error(X, B, C);
    %fprintf('%d\t1\t%f\n',iter, error);
    B = update_assignment(X, C, M, B, n);
    %B = update_assignment_(X, C, M, n);
    error = get_error(X, B, C);
    fprintf('%d\t2\t%f\n',iter, error);
end
end

function B = PQ(X, M, K)
[D, N] = size(X);
D_ = D/M;
B = zeros(M, N);
for m=1:M
    X_ = X((m-1)*D_+(1:D_), :);
    B(m,:) = kmeans(X_', K);
end
end

function X = decode(B, C)
[M, N] = size(B);
[D, mk] = size(C);
K = mk / M;
X = zeros(D, N);
for m=1:M
    X = X + C(:,(m-1) * K + B(m, :));
end
end

function error = get_error(X, B, C)
e = X - decode(B,C);
error = mean(sqrt(sum(e.^2)));
end

function C = update_codebook(X, B, K)
N = size(X, 2);
M = size(B, 1);
offset = (0:(M-1)) * K;
B1 = offset' + B;
[~, u, v] = find(B1);
A = sparse(v, u, 1, M * K, N);
C = X / A;
end

function B = update_assignment_(X, C, M, n)
%tic;B = search(x, C, M, n);toc
%%% X: DxN, C: Dx(MxK)
N = size(X, 2);
B = zeros(M, N);
for i=1:N
    B(:,i) = beam_search(X(:,i), C, M, n);
    %B(:,i) = local_search(X(:,i), C, M, B(:,i));
end
end

function B = update_assignment(X, C, M, B, n)
if M>4
    K = size(C, 2) / M;
    max_iter = 1;
    X = X - decode(B, C);
    for iter=1:max_iter
        idx_ = sort(datasample(0:M-1, 4, 'Replace', false))';
        idx = idx_ * K + repmat(1:K, 4, 1); idx = reshape(idx',  4*K, 1);
        C_ = C(:, idx);
        B_ = B(idx_+1, :);
        X = X + decode(B_, C_);
        B_ = update_assignment_(X, C_, 4, n);
        X = X - decode(B_, C_);
        B(idx_+1, :) = B_; 
    end
else
    B = update_assignment_(X, C, M, n);
end
end

function b = local_search(x, C, M, b)
max_iter = 20;
mk = size(C, 2);
K = mk/M;
x_ = x;
for m=1:M
    x_ = x_ - C(:,(m-1) * K + b(m));
end
prev_loss = norm(x_)^2;
%fprintf("%f\n", prev_loss);
for iter=1:max_iter
    for m=1:M
        x__ = x_ + C(:,(m-1) * K + b(m));
        c = C(:, (1:K) + (m-1) * K);
        dist = sum(bsxfun(@minus, x__, c).^2); 
        [~, idx] = min(dist);
        if b(m) ~= idx
            b(m) = idx;
            x_ = x__ - C(:,(m-1) * K + b(m));
        end
    end
    curr_loss = norm(x_)^2;
    %fprintf("%f\n", curr_loss);
    if abs(prev_loss - curr_loss)<1e-3
        break
    end
    prev_loss = curr_loss;
end
end