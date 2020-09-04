function [V, Q, B, C] = network_quantizer_g(M_g, M_l, varargin)
warning('off','MATLAB:rankDeficientMatrix'); rng(10);
[max_iter, dim, num_codebooks, alg, others] = process_options(varargin, 'max_iter', 50, 'dim', 128, ...
    'M', -1, 'alg', 'joint');
if num_codebooks < 0
    num_codebooks = dim / 8;
end
%net = deepwalk(net, others{:});
[U, S, Q] = gsvds(M_g, M_l, dim);
max_inner_iter = 10;
if strcmp(alg, 'opq')
    Xtrain = U * S;
    R_init = eye(dim); num_iter = 50;
    center_init = train_pq(Xtrain*R_init, num_codebooks);
    [C, B, e, Q_] = train_opq_np(Xtrain, num_codebooks, center_init, R_init, max_iter);
    V = zeros(size(Xtrain, 1), dim);
    d = dim / num_codebooks;
    for m=1:num_codebooks
        V(:, (1:d) + (m-1)*d) = C{m}(B(:, m),:);
    end
    B = B';
    C = cell2mat(C)';
    Q = Q * Q_;
elseif strcmp(alg, 'aq')
    Xtrain = U * S;
    [B, C, ~, Q_] = AQ_pipeline(Xtrain', 256*ones(num_codebooks,1), max_iter, 16);
    %B = B{end}; C=C{end}; 
    C = cell2mat(C');
    V = decoder(B, C)';
    Q = Q * double(Q_);
else
prev_loss = inf;
[B, C] = AQ((M_g \ (M_l * Q))', num_codebooks, max_inner_iter);
for iter=1:max_iter
    V = decoder(B, C)';
    curr_loss = loss_gmf(M_g, M_l, V, Q);
    fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
    if abs(prev_loss - curr_loss) < max(1e-6 * prev_loss, 1e-6) 
        break
    end
    prev_loss = curr_loss;
    mtb = (V' / M_g * M_l)';
    Q = proj_stiefel_manifold(mtb);
    [B, C] = AQ((M_g \ (M_l * Q))', num_codebooks, max_inner_iter, B);
end
V = decoder(B, C)';
end
curr_loss = loss_gmf(M_g, M_l, V, Q);
fprintf('The finall loss value: %.3f\n', curr_loss);
end


function [B, C] = AQ(X, M, max_iter, B)
K = 256;
if nargin < 4
    B = PQ(X, M, K);
end
prev_loss = inf;
for iter=1:max_iter
    C = update_codebook(X, B, K);
    B = update_assignment(X, C, M, B);
    curr_loss = get_error(X, B, C);
    fprintf('The %d-th iteration, \tAQloss:%f\n',iter, curr_loss);
    if abs(prev_loss - curr_loss) < 1e-6 * prev_loss 
        break
    end
    prev_loss = curr_loss;
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
B = beam_search_(X, C, B, M, 16);
B = local_search(X, C, M, B);
end

function B = beam_search_(X, C, B, M, n)
sub = min(4, M);
N = size(X, 2);
K = size(C, 2) / M;
X = X - decoder(B, C);
idx_ = sort(datasample(1:M, sub, 'Replace', false))';
idx = (idx_-1) * K + repmat(1:K, sub, 1); 
idx = reshape(idx', sub*K, 1);
C_ = C(:, idx);
B_ = B(idx_, :);
X = X + decoder(B_, C_);
for i=1:N
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
        %[~,b] = pdist2(c', X', 'euclidean','Smallest',1);
        b = assign(X, c)';
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
function b=assign(X, c)
    d = bsxfun(@plus, -2*X'*c, sum(c.*c,1));
    %d = abs(bsxfun(@plus, d', sum(X.*X,1)));
    [~,b] = min(d, [], 2);
end
end
