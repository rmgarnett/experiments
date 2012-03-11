check_search_experiment_options;

if (options_defined)

  data_directory = '~/work/data/nips_papers/processed/top_venues/';
  load([data_directory 'nips_graph_pca_vectors'], 'data');
  load([data_directory 'top_venues_graph'], 'nips_index');

  num_observations = size(data, 1);

  responses = 2 * ones(num_observations, 1);
  responses(nips_index) = 1;

  if (~exist('probability_function', 'var'))
    setup_nips_knn;
  end

  [results, elapsed] = perform_search_experiment(data, responses, ...
          num_additional, seed, probability_function, probability_bound, ...
          num_experiments, num_evaluations, max_lookahead, report);
end