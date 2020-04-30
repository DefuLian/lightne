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
[B, C]= AQ(X, 10);
end

function [B, C] = AQ(X, max_iter)
[D, N] = size(X);
K = 256;
M = D / 8;
C = randn(D, M * K);
B = randi(K, M, N);
%B = PQ(X, M, K);
for iter=1:max_iter
    B = update_assignment(X, C, M, B);
    C = update_codebook(X, B, K);
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

function X = decoder(B, C)
[M, N] = size(B);
[D, mk] = size(C);
K = mk / M;
X = zeros(D, N);
for m=1:M
    X = X + C(:,(m-1) * K + B(m, :));
end
end

function error = get_error(X, B, C)
e = X - decoder(B, C);
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

function B = update_assignment(X, C, M, B)
%tic;B = search(x, C, M, n);toc
%%% X: DxN, C: Dx(MxK)
%N = size(X, 2);
%B = zeros(M, N);
%for i=1:N
%    B(:,i) = beam_search(X(:,i), C, M, n);
    %B(:,i) = local_search(X(:,i), C, M, B(:,i));
%end
B = beam_search_(X, C, B, M, 64);
B = local_search(X, C, M, B);
end

function B = beam_search_(X, C, B, M, n)
sub = 2;
N = size(X, 2);
K = size(C, 2) / M;
X = X - decoder(B, C);
idx_ = sort(datasample(1:M, sub, 'Replace', false))';
idx = (idx_-1) * K + repmat(1:K, sub, 1); 
idx = reshape(idx',  sub*K, 1);
C_ = C(:, idx);
B_ = B(idx_, :);
num_can = floor(min(N*0.1, 1000));
%sample_random = datasample(1:N, num_can*2 , 'Replace', false);
e = sum(X.^2);
[~, samples] = maxk(e, num_can);
%samples = [sample_random, sample_maxe];
%samples = sample_random;
for v=1:length(samples)
    i = samples(v);
    B_(:,i) = beam_search(X(:,i), C_, sub, n);
end
e1 = X - decoder(B_, C_); e1 = sum(e1.^2);
e2 = X - decoder(B(idx_, :), C_); e2 = sum(e2.^2);
update = e1 < e2;
B(idx_, update) = B_(:, update);
end

function B = local_search(X, C, M, B)
X = X - decoder(B, C);
max_iter = 5;
mk = size(C, 2);
K = mk/M;
prev_loss = get_error(X, B, C);
for iter=1:max_iter
    for m=1:M
        b = B(m,:);
        c = C(:, (1:K) + (m-1) * K);
        X = X + decoder(b, c);
        [~,b] = pdist2(c', X', 'euclidean','Smallest',1);
        X = X - decoder(b, c);
        B(m,:) = b;
        
    end
    curr_loss = get_error(X, B, C);
    %fprintf("%d\t%d\t%f\n", iter,m, curr_loss);
    if abs(prev_loss - curr_loss)<1e-3
        break
    end
    prev_loss = curr_loss;
end
end