data_directory = '~/work/data/biology/enzymes/processed/';
load([data_directory 'enzymes']);

num_graphs_to_keep = 10;

D = sum(data);
connected_nodes = (D > 0);

to_keep = (graph_ind <= num_graphs_to_keep) & (connected_nodes');

data = data(to_keep, to_keep);
responses = responses(to_keep);
graph_ind = graph_ind(to_keep);

data = diag(D(to_keep)) \ data;

num_graphs = max(graph_ind);

train_ind = []; %zeros(num_graphs, 1);
for i = 1:num_graphs
  num_nodes = nnz(graph_ind == i);
  r = randperm(num_nodes);
  nodes = r(1:1)';
  train_ind = [train_ind; logical_ind(graph_ind == i, nodes)];
end

h                    = 1;
num_response_samples = 1000;
num_test_samples     = 10;
num_evaluations      = 1000;
lookahead            = 0;
verbose              = true;

setup_mutag_lp;

response_sampling_function = @(data, reponses, train_ind) ...
    independent_response_sampler(data, responses, train_ind, ...
        probability_function, num_response_samples);

f = @(data, responses) ...
    extract_upper_triangle_vector(wl_subtree_kernel(data, responses, ...
        graph_ind, h));

utility_function = @(data, responses, train_ind) ...
    negative_maximum_std_utility(data, responses, train_ind, ...
        response_sampling_function, f, num_response_samples);

selection_functions{1} = @(data, resposes, train_ind) ...
    random_selector(responses, train_ind, num_test_samples);

[chosen_ind, utilities] = optimal_learning(data, responses, train_ind, ...
        utility_function, probability_function, selection_functions, ...
        lookahead, num_evaluations, verbose);
