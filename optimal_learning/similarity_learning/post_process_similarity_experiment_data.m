responses = double(responses);
graph_ind = double(graph_ind);

num_nodes   = size(data, 1);
num_classes = max(responses);
num_graphs  = max(graph_ind);