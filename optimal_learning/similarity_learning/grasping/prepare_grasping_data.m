data_directory = '~/work/data/grasping/processed/';
load([data_directory 'grasping.mat']);

A           = data;
node_labels = responses;
graph_ind   = double(graph_ind);

num_nodes   = size(A, 1);
num_classes = max(node_labels);
num_graphs  = max(graph_ind);
