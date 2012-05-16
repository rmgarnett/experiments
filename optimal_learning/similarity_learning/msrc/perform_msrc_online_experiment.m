prepare_msrc_data;

tolerance      = 1e-1;
max_iterations = 10;

setup_lp;

fill_in_bottom = @(K) (K + triu(K, 1)');
normalize = @(K) (bsxfun(@times, 1 ./ sqrt(diag(K)), ...
                  bsxfun(@times, 1 ./ sqrt(diag(K)'), K)));

train_graph_ind  = 1;
h                = 2;
f                = @(data, responses) ...
                       fill_in_bottom(normalize( ...
                           wl_subtree_kernel_random(data, responses, graph_ind, h)));
num_samples      = 100;
lookahead        = 1;
verbose          = true;
sigma_multiplier = 1;
k                = 5;
num_evaluations  = 10;
train_fraction   = 0.25;

test_graphs = setdiff(graph_ind, train_graph_ind);

train_ind = [];
for i = 1:numel(test_graphs)
  this_graph_inds = find(graph_ind == test_graphs(i));
  num_nodes_this_graph = numel(this_graph_inds);
  
  permutation = randperm(num_nodes_this_graph);

  train_ind = [train_ind; ...
               this_graph_inds(permutation(1:floor(train_fraction * num_nodes_this_graph)))];
          
end

perform_online_similarity_experiment;