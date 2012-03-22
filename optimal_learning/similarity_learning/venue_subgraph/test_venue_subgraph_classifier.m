data_directory = '~/work/data/nips_papers/processed/venue_subgraph/';

load([data_directory 'connected_venue_graph_pca_vectors']);
load([data_directory 'venue_subgraph.mat'], 'connected', 'connected_positive_node_ids');

num_nodes = size(data, 1);
num_edges = nnz(triu(connected));

responses = 2 * ones(num_nodes, 1);
responses(connected_positive_node_ids) = 1;

num_nodes = size(data, 1);

k = 500;
pseudocount = 0.5;
setup_venue_subgraph_knn;

for train_size = [1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1]
  fprintf('with %.1f%% training data:\n', train_size * 100);

  permutation = randperm(num_nodes);

  train_ind = permutation(1:ceil(num_nodes * train_size))';
  test_ind = identity_selector(responses, train_ind);

  [~, accuracy, confusion_matrix] = evaluate_classifier(data, ...
          responses, train_ind, test_ind, probability_function)
end