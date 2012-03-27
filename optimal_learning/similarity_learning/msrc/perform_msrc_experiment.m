prepare_msrc_data;
setup_msrc_lp;

num_train_per_graph  = 50;
num_response_samples = 100;
num_test_samples     = 1;
h                    = 1;
lookahead            = 1;
num_evaluations      = 10;
verbose              = true;

train_image_ind      = 1;
test_image_ind       = unique(graph_ind(graph_ind ~= train_image_ind));

train_ind = logical_ind(graph_ind == train_image_ind, 1);
for i = 1:numel(test_image_ind)
  num_nodes = nnz(graph_ind == test_image_ind(i));
  permutation = randperm(num_nodes)';
  train_ind = [train_ind; permutation(1:min(num_nodes, num_train_per_graph))];
end

response_sampling_function = @(data, reponses, train_ind, num_samples) ...
    independent_response_sampler(data, responses, train_ind, ...
        probability_function, num_samples);

f = @(data, responses) ...
    wl_subtree_kernel(data, responses, graph_ind, train_image_ind, ...
                      test_image_ind, h, true);

utility_function = @(data, responses, train_ind) ...
    negative_mean_std_utility(data, responses, train_ind, ...
        response_sampling_function, f, ...
        num_response_samples);

selection_functions{1} = @(data, resposes, train_ind) ...
    graph_subset_selection_function(train_ind, graph_ind, train_image_ind);

[chosen_ind, utilities] = optimal_learning(data, responses, train_ind, ...
        utility_function, probability_function, selection_functions, ...
        lookahead, num_evaluations, verbose);