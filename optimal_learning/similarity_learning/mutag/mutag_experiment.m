data_directory = '~/work/data/biology/mutag/processed/';
load([data_directory 'mutag']);

data = diag(sum(data)) \ data;

num_graphs = max(graph_ind);

train_ind = zeros(num_graphs, 1);
for i = 1:num_graphs
  num_nodes = nnz(graph_ind == i);
  node = randi(num_nodes);
  train_ind(i) = logical_ind(graph_ind == i, node);
end

h                    = 1;
num_response_samples = 100;
num_test_samples     = 1;
num_evaluations      = 100;
lookahead            = 1;
verbose              = true;

setup_mutag_lp;

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
