function generate_code(input, output, varargin)
[network_name, train_ratio, others] = process_options(varargin, 'nn', 'network', 'train_ratio', 1);
load(input, '-mat', network_name)
net = eval(network_name);
if train_ratio<1
    net = split_network(net, 'un', net);
end
B = network_hashing(net, others{:});
n = length(net);
dim = size(B,2);
fileid = fopen(output, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(output, [(0:n-1)', B], 'delimiter', ' ', '-append')
end