clc;

required_options = ...
    {'data', ...
     'responses', ...
     'graph_ind', ...
     'h', ...
     'num_train_per_graph', ...
     'num_samples', ...
     'probability_function', ...
     'lookahead', ...
     'num_evaluations', ...
     'verbose', ...
    };

check_required_options;

if (options_defined)

  train_ind = [];
  for i = 1:num_graphs
    ind = (graph_ind == i);

    this_num_nodes = nnz(ind);
    permutation    = randperm(this_num_nodes)';
    this_train_ind = permutation(1:min(this_num_nodes, num_train_per_graph));
    
    train_ind = [train_ind;
                 logical_ind(ind, this_train_ind)];
  end

  response_sampling_function = @(data, reponses, train_ind, num_samples) ...
      independent_response_sampler(data, responses, train_ind, ...
          probability_function, num_samples);

  f = @(data, responses) ...
      extract_upper_triangle_vector(wl_subtree_kernel(data, responses, ...
          graph_ind, 1:num_graphs, 1:num_graphs, h, true));

  utility_function = @(data, responses, train_ind) ...
      negative_maximum_std_utility(data, responses, train_ind, ...
          response_sampling_function, f, num_samples);

  selection_functions{1} = @(data, responses, train_ind) ...
      rotating_graph_selection_function(train_ind, graph_ind);

  true_K = f(data, responses);
  performance = @(means, variances) calculate_log_likelihood(true_K, ...
          means, variances);
  
  [means, variances] = calculate_moments(data, responses, train_ind, ...
          response_sampling_function, f, num_samples);

  starting_performance = performance(means, variances);
  fprintf('performance with no training data: %4e\n', starting_performance);

  random_train_ind      = train_ind;
  uncertainty_train_ind = train_ind;
  wdd_train_ind         = train_ind;
  optimal_train_ind     = train_ind;

  random_performances      = [];
  uncertainty_performances = [];
  wdd_performances         = [];
  optimal_performances     = [];

  for i = 1:num_evaluations
    fprintf('performing random sampling: ');
    random_chosen_ind = random_sampling(data, responses, random_train_ind, ...
            utility_function, selection_functions, 1, verbose);
    random_train_ind = [random_train_ind; random_chosen_ind];
  
    [means, variances] = calculate_moments(data, responses, ...
            random_train_ind, response_sampling_function, f, num_samples);
  
    random_performance = performance(means, variances);
    random_performances(end + 1) = random_performance;
    
    fprintf('performing uncertainty sampling: ');
    uncertainty_chosen_ind = uncertainty_sampling(data, responses, ...
            uncertainty_train_ind, probability_function, ...
            selection_functions{1}, 1, verbose);
    uncertainty_train_ind = [uncertainty_train_ind; uncertainty_chosen_ind];
  
    [means, variances] = calculate_moments(data, responses, ...
            uncertainty_train_ind, response_sampling_function, f, num_samples);
  
    uncertainty_performance = performance(means, variances);
    uncertainty_performances(end + 1) = uncertainty_performance;
    
    fprintf('performing wdd sampling: ');
    wdd_chosen_ind = wdd_sampling(data, responses, ...
            wdd_train_ind, probability_function, ...
            selection_functions{1}, 1, verbose);
    wdd_train_ind = [wdd_train_ind; wdd_chosen_ind];
  
    [means, variances] = calculate_moments(data, responses, ...
            wdd_train_ind, response_sampling_function, f, num_samples);
  
    wdd_performance = performance(means, variances);
    wdd_performances(end + 1) = wdd_performance;

    fprintf('performing optimal sampling: ');
    optimal_chosen_ind = optimal_learning(data, responses, ...
            optimal_train_ind, utility_function, probability_function, ...
            selection_functions, lookahead, 1, verbose);
    optimal_train_ind = [optimal_train_ind; optimal_chosen_ind];
  
    [means, variances] = calculate_moments(data, responses, ...
            optimal_train_ind, response_sampling_function, f, num_samples);
  
    optimal_performance = performance(means, variances);
    optimal_performances(end + 1) = optimal_performance;
    fprintf('after %i steps: random: %4e, uncertainty: %4e, wdd: %4e, optimal: %4e\n', ...
            i, ...
            random_performance, ...
            uncertainty_performance, ...
            wdd_performance, ...
            optimal_performance);
  end
end
