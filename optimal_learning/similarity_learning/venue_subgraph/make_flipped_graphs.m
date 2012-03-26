num_total_nodes = num_nodes * num_copies;

big_data      = sparse(num_total_nodes, num_total_nodes);
big_responses = zeros(num_total_nodes, 1);
graph_ind = zeros(num_total_nodes, 1);

flipped_graph     = data;
flipped_responses = responses;

num_edges_to_flip = floor(fraction_to_flip * num_edges);
num_nodes_to_flip = floor(fraction_to_flip * num_nodes);

for i = 1:num_copies
  this_graph_ind = (1 + (i - 1) * num_nodes):(i * num_nodes);
  big_data(this_graph_ind, this_graph_ind) = flipped_graph;
  graph_ind(this_graph_ind) = i;

  %non_leaf_ind = find(sum(flipped_graph) > 1);

  %ind = find(triu(flipped_graph(non_leaf_ind, non_leaf_ind)));
  %permutation = randperm(numel(ind));

  %flipped_graph(ind(non_leaf_ind(permutation(1:num_edges_to_flip)))) = 0;
  %flipped_graph = min(flipped_graph, flipped_graph');

  ind = find(triu(flipped_graph) == 0);
  permutation = randperm(numel(ind));

  flipped_graph(ind(permutation(1:num_edges_to_flip))) = 1;
  flipped_graph = max(flipped_graph, flipped_graph');

  permutation = randperm(num_nodes);
  flipped_responses(permutation(1:num_nodes_to_flip)) = 1 - ...
      flipped_responses(permutation(1:num_nodes_to_flip));

  big_responses(this_graph_ind) = flipped_responses;
end

data = big_data;
clear big_data;

responses = big_responses;
clear big_responses;

data = perform_row_normalization(data);