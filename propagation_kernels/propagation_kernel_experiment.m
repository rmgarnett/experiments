num_folds      = 10;
num_trials     = 10;
num_iterations = 10;

num_nodes   = size(data, 1);
num_classes = max(responses);
num_graphs  = numel(graph_labels);

K_triu = zeros(num_graphs, num_graphs);

r = randperm(num_nodes);

train_ind = r(1:floor(train_fraction * num_nodes));
test_ind  = identity_selector(responses, train_ind);
num_train = numel(train_ind);

probabilities = zeros(num_nodes, num_classes);
for i = 1:numel(train_ind)
  ind = train_ind(i);
  probabilities(ind, responses(ind)) = 1;
end
train_rows = probabilities(train_ind, :);

probabilities(test_ind, :) = 1 / num_classes;

for i = 1:num_iterations
  if (pull_back)
    probabilities(train_ind, :) = train_rows;
  end

  K_triu = K_triu + ...
           lsh_propagation_kernel(graph_ind, probabilities, w, num_vectors, sqrt_flag);

  K = K_triu + triu(K_triu, 1)';

  kernel_function = @(u, v) K(u, v);

  partitions = cvpartition(num_graphs, 'kfold', num_folds);
  accuracies = zeros(num_trials, 1);
  for j = 1:num_trials
    fold_train_ind = find(partitions.training(j));
    fold_test_ind  = find(partitions.test(j));

    svm = svmtrain(fold_train_ind, graph_labels(fold_train_ind), ...
                   'autoscale', false, ...
                   'kernel_function', kernel_function, ...
                   'method', 'qp');
    predictions = svmclassify(svm, fold_test_ind);
    accuracies(j) = mean(predictions == graph_labels(fold_test_ind));
  end

  fprintf('iteration %i, accuracy: %0.3f\n', i, mean(accuracies));

  probabilities = data * probabilities;
end