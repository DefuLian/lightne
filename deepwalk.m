function mat = deepwalk(network, varargin)
[T, b, rank] = process_options(varargin, 'T',1, 'b', 1, 'rank', 1024);
n = length(network);
vol = sum(sum(network));
d_rt = sqrt(sum(network)); d_rt(d_rt<eps) = 1;
D_rt_inv = spdiags(1./d_rt', 0, n, n);
X = D_rt_inv * network * D_rt_inv;
if T<4
    X_power = X;
    mat = X;
    for t = 2:T
        X_power = X_power * X;
        mat = mat + X_power;
    end
    mat = mat * (vol / T / b);
    mat = D_rt_inv * mat * D_rt_inv;
    [I, J, K] = find(mat);
    ind = K>1;
    mat = sparse(I(ind), J(ind), log(K(ind)), n, n);
else
    [V, E] = eigs(X, rank, 'largestreal');
    V = D_rt_inv * V;
    E_f = spdiags(arrayfun(@(x) filter(x, T), spdiags(E)), 0, rank, rank);
    Qt = E_f * V' * (vol / b);
    step_size = 1000;
    num_step = floor((n + step_size-1)/step_size);
    rows = cell(num_step, 1);
    cols = cell(num_step, 1);
    data = cell(num_step, 1);
    for i = 1:num_step
        start_n = (i-1)*step_size + 1;
        end_n = min(i * step_size, n);
        result = V(start_n:end_n,:) * Qt;
        result(result<1) = 0;
        [row, col, val] = find(result);
        rows{i} = row + start_n - 1;
        cols{i} = col;
        data{i} = log(val);
    end
     mat = sparse(cell2mat(rows), cell2mat(cols), cell2mat(data), n, n);
end
end
function v = filter(x, T)
if x>=1
    v = 1;
else
    v = x * (1 - x^T) / (1 - x) / T;
    v = max(0, v);
end
end