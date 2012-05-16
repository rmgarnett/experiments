required_options = {'num_additional', 'seed', 'num_experiments', ...
                    'num_evaluations', 'max_lookahead', 'report'};
check_required_options;

if (options_defined)

  data_directory = '~/work/data/citeseer/processed/venue_subgraph/';
  load([data_directory 'connected_venue_graph_pca_vectors'], 'data');
  load([data_directory 'venue_subgraph'], 'connected_positive_node_ids');

  num_observations = size(data, 1);

  responses = 2 * ones(num_observations, 1);
  responses(connected_positive_node_ids) = 1;

  if (~exist('probability_function', 'var'))
    setup_nips_knn;
  end
 
  [results, elapsed] = perform_search_experiment(data, responses, ...
          num_additional, seed, probability_function, probability_bound, ...
          num_experiments, num_evaluations, max_lookahead, report);
end