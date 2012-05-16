prepare_random_graph_data;

num_graphs       = 10;
fraction_to_flip = 0.01;

make_flipped_graphs;

num_train_per_graph  = 1;
num_samples          = 100;
h                    = 2;
lookahead            = 1;
num_evaluations      = 10;
verbose              = true;

tolerance            = 1e-1;
max_iterations       = 10;

setup_venue_subgraph_lp;

response_sampling_function = @(data, reponses, train_ind, num_samples) ...
    independent_response_sampler(data, responses, train_ind, ...
        probability_function, num_samples);

if (online)
  test_graph_ind   = unique(graph_ind(graph_ind ~= train_graph_ind));

  sigma_multiplier = 2;
  k                = 3;

  lcbs = @(means, variances) means - sigma_multiplier * sqrt(variances);
    
  train_ind = logical_ind(graph_ind == train_graph_ind, 1);
  for i = 1:numel(test_graph_ind)
    ind = graph_ind == test_graph_ind(i);
    num_nodes = nnz(ind);
    permutation = randperm(num_nodes)';
    train_ind = [train_ind;
                 logical_ind(ind, permutation(1:min(num_nodes, num_train_per_graph)))];
  end
  
  f = @(data, responses) ...
      wl_subtree_kernel(data, responses, graph_ind, train_graph_ind, ...
                        test_graph_ind, h, true);

  utility_function = @(data, responses, train_ind) sum_top_lcb_utility(data, ...
          responses, train_ind, response_sampling_function, f, ...
          num_samples, sigma_multiplier, k);
  
  selection_functions{1} = @(data, responpeses, train_ind) ...
      graph_subset_selection_function(train_ind, graph_ind, train_graph_ind);

  K = f(data, responses);

  [~, ind] = sort(K, 'descend');
  best_similarity = sum(K(ind(1:k)));
  fprintf('best similarity: %f\n', best_similarity);
  
  [means, variances] = calculate_moments(data, responses, train_ind, ...
          response_sampling_function, f, num_samples);
  
  [~, ind] = sort(lcbs(means, variances), 'descend');
  start_similarity = sum(K(ind(1:k)));
  
  fprintf('with no training: %f\n', start_similarity);
  
else

  train_ind = [];
  for i = 1:num_graphs
    ind = (graph_ind == i);
    num_nodes = nnz(ind);
    permutation = randperm(num_nodes)';
    train_ind = [train_ind;
                 logical_ind(ind, permutation(1:min(num_nodes, num_train_per_graph)))];
  end
  
  response_sampling_function = @(data, reponses, train_ind, num_samples) ...
      independent_response_sampler(data, responses, train_ind, ...
          probability_function, num_samples);
  
  f = @(data, responses) ...
      extract_upper_triangle_vector(wl_subtree_kernel(data, responses, ...
          graph_ind, 1:num_graphs, 1:num_graphs, h, true));
  
  selection_functions{1} = @(data, responses, train_ind) ...
      rotating_graph_subset_selection_function(responses, train_ind, graph_ind);
  
  utility_function = @(data, responses, train_ind) ...
      negative_maximum_std_utility(data, responses, train_ind, ...
          response_sampling_function, f, num_samples);
end

random_train_ind  = train_ind;
optimal_train_ind = train_ind;

for i = 1:num_evaluations
  random_chosen_ind = random_sampling(data, responses, random_train_ind, ...
          utility_function, selection_functions, 1, verbose);

  random_train_ind = [random_train_ind; random_chosen_ind];
  
  [means, variances] = calculate_moments(data, responses, random_train_ind, ...
          response_sampling_function, f, num_samples);
  
  [~, ind] = sort(lcbs(means, variances), 'descend');
  random_similarity = sum(K(ind(1:k)));

  fprintf('random training: %f\n', random_similarity);

  optimal_chosen_ind = optimal_learning(data, responses, optimal_train_ind, ...
          utility_function, probability_function, selection_functions, ...
          lookahead, 1, verbose);

  optimal_train_ind = [optimal_train_ind; optimal_chosen_ind];
  
  [means, variances] = calculate_moments(data, responses, optimal_train_ind, ...
          response_sampling_function, f, num_samples);
  
  [~, ind] = sort(lcbs(means, variances), 'descend');
  optimal_similarity = sum(K(ind(1:k)));

  fprintf('optimal training: %f\n', optimal_similarity);

end
