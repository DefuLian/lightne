M = 4; K = 128; n=32; N = 1000;D=32;
x = rand(D, N);
C = rand(D, M*K);
B = AQ_encoding(x, C1', n);
C1 = network_quantizer(x, B, K);
C_ = mat2cell(C, D, [K K K K]);
C2_ = AQ_update_codebooks(x, B, C_');
C2 = cell2mat(C2_');
norm(x - C(:,B(1)) - C(:,B(2)+1*K) - C(:,B(3)+2*K) - C(:,B(4)+3*K))^2
norm(x - C1(:,B(1)) - C1(:,B(2)+1*K) - C1(:,B(3)+2*K) - C1(:,B(4)+3*K))^2
norm(x - C2(:,B(1)) - C2(:,B(2)+1*K) - C2(:,B(3)+2*K) - C2(:,B(4)+3*K))^2


tic;[B1,e] = AQ_encoding(x, C1', n, false);toc

tic;B = search(x, C, M, n);toc
tic;B = network_quantizer(x, C, M, n);toc

[Bb,Cc]=network_quantizer(x);