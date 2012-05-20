required_options = {'num_additional', 'seed', 'num_experiments', ...
                    'num_evaluations', 'max_lookahead', 'report'};
check_required_options;

if (options_defined)

  data_directory = '~/work/tmp/';
  load([data_directory 'citeseer_50_undirected']);
  load([data_directory 'nips']);

  edges = citeseer_50_undirected;
  clear('citeseer_50_undirected');

  num_edges = size(edges, 1);
  
  [~, ~, new_node_labels] = unique([edges(:, 1); edges(:, 2); nips]);

  edges(:, 1) = new_node_labels(1:num_edges);
  edges(:, 2) = new_node_labels((num_edges + 1):(2 * num_edges));
  nips        = new_node_labels((2 * num_edges + 1):end);
  
  num_nodes = max(new_node_labels);
  data = (1:num_nodes)';

  responses = 2 * ones(num_nodes, 1);
  responses(nips) = 1;

  if (~exist('probability_function', 'var'))
    setup_nips_knn_commute;
  end
 
  [results, elapsed] = perform_search_experiment(data, responses, ...
          num_additional, seed, probability_function, probability_bound, ...
          num_experiments, num_evaluations, max_lookahead, report);
end