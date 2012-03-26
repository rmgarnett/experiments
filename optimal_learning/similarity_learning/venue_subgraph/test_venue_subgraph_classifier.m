prepare_random_graph_data;

setup_venue_subgraph_lp;

for train_fraction = [1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1]

  fprintf('with %.1f%% training data:\n', train_fraction * 100);

  permutation = randperm(num_nodes);

  train_ind = permutation(1:ceil(num_nodes * train_fraction))';
  test_ind = identity_selector(responses, train_ind);

  tic;
  [~, accuracy, confusion_matrix] = evaluate_classifier(data, ...
          responses, train_ind, test_ind, probability_function);
  elapsed = toc;

  fprintf('  took: %0.1fs\n', elapsed);
  fprintf('  accuracy: %0.3f\n', accuracy);
  fprintf('  confustion matrix:\n');
  fprintf('    %4i %4i\n', confusion_matrix);
end