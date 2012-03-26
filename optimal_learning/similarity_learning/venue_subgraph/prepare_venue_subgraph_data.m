data_directory = '~/tmp/';
%data_directory = '~/work/data/nips_papers/processed/venue_subgraph/';

load([data_directory 'venue_subgraph.mat'], ...
     'connected', ...
     'connected_positive_node_ids');

data = perform_row_normalization(connected);

num_nodes = size(data, 1);
num_edges = nnz(triu(data));

responses = 2 * ones(num_nodes, 1);
responses(connected_positive_node_ids) = 1;