%prepare_venue_subgraph_data;

num_nodes = 100;
sparsity = 0.15;

sigma = 0.2;

num_copies       = 5;
fraction_to_flip = 0.05;

make_flipped_graphs;
setup_venue_subgraph_lp;

num_response_samples = 50;
num_test_samples     = 50;
h                    = 0;
lookahead            = 1;
num_evaluations      = 100;
verbose              = true;

response_sampling_function = @(data, reponses, train_ind) ...
    independent_response_sampler(data, responses, train_ind, ...
        probability_function, num_response_samples);

f = @(data, responses) ...
    extract_upper_triangle_vector(wl_subtree_kernel(data, responses, ...
        graph_ind, h));

utility_function = @(data, responses, train_ind) ...
    negative_mean_std_utility(data, responses, train_ind, ...
                              response_sampling_function, f, ...
                              num_response_samples);

selection_functions{1} = @(data, resposes, train_ind) ...
    random_selector(responses, train_ind, num_test_samples);

[chosen_ind, utilities] = optimal_learning(data, responses, train_ind, ...
        utility_function, probability_function, selection_functions, ...
        lookahead, num_evaluations, verbose);