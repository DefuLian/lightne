function run_hash(input, output, varargin)
[network_name, alg, others] = process_options(varargin, 'nn', 'network', 'alg', 'hash');
load(input, '-mat', network_name)
net = eval(network_name);
B = network_hashing(net, others{:});
n = length(net);
dim = size(B,2);
fileid = fopen(output, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(output, [(0:n-1)', B], 'delimiter', ' ', '-append')
end