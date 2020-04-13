function evaluate_lp(input, varargin)
[network_name, train_ratio, output, others] = process_options(varargin, 'nn', 'train-test',...
    'train_ratio', 1, 'output', false);
if strcmp(network_name, 'train-test')
    load(input, '-mat', 'train', 'test')
    train = eval('train');
    test = eval('test');
else
    load(input, '-mat', network_name)
    net = eval(network_name);
    [train, test] = split_network(net, 'un', 0.9);
end
if train_ratio<1
    train = split_network(train, 'un', train_ratio);
end
B = network_hashing(train, others{:});
result = evaluate_item(train, test, B, B, -1, 200);
if output
    save(output, 'result');
end
fprintf('ndcg@50=%.4f,AUC=%.4f,MPR=%.4f\n', result.item_ndcg(1,50), result.item_auc, full(result.item_mpr))
end