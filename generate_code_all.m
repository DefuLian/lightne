function generate_code_all(input, output, varargin)
[network_name, train_ratio, T, b, rank, dim, num_codebooks] = process_options(varargin, 'nn', 'network',...
    'train_ratio', 1, 'T',1, 'b', 1, 'rank', 1024, 'dim', 128, 'M', -1);
load(input, '-mat', network_name)
net = eval(network_name);
if train_ratio<1
    net = split_network(net, 'un', net);
end
net = deepwalk(net, 'T',T, 'b', b, 'rank', rank);
[U, S, V] = svds(net, dim);
[filepath,name,ext] = fileparts(output);

alg = 'real';
B = network_hashing_fast(net, U, S, V, 'alg', alg);
write_out(B, fullfile(filepath, strcat(sprintf('%s_%s',name, alg), ext)))


alg = 'binary';
B = network_hashing_fast(net, U, S, V, 'alg', alg, 'max_iter', 50);
write_out(B, fullfile(filepath, strcat(sprintf('%s_%s',name, alg), ext)))


alg = 'opq';
B = network_hashing_fast(net, U, S, V, 'alg', alg, 'max_iter', 50, 'M', num_codebooks);
write_out(B, fullfile(filepath, strcat(sprintf('%s_%s',name, alg), ext)))

alg = 'joint';
B = network_hashing_fast(net, U, S, V, 'alg', alg, 'max_iter', 50, 'M', num_codebooks);
write_out(B, fullfile(filepath, strcat(sprintf('%s_%s',name, alg), ext)))


alg = 'aq';
B = network_hashing_fast(net, U, S, V, 'alg', alg, 'max_iter', 10, 'M', num_codebooks);
write_out(B, fullfile(filepath, strcat(sprintf('%s_%s',name, alg), ext)))


end

function write_out(B, output)
[n,dim] = size(B,2);
fileid = fopen(output, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(output, [(0:n-1)', B], 'delimiter', ' ', '-append')
end