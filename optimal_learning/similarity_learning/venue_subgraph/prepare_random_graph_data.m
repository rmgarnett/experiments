%data = sprand(num_nodes, num_nodes, sparsity / 2);
%data = max(data, data');
%data = data - diag(diag(data));
%data = full(double(data > 0));

D = sum(data);
D_normalize = diag(1 ./ sqrt(D));

%L = eye(num_nodes) - D_normalize * data * D_normalize;
L = diag(D) - data;

K = inv(eye(num_nodes) + sigma^2 * L);

responses = randn(1, num_nodes) * chol(K) > 0;
responses = responses(:) + 1;