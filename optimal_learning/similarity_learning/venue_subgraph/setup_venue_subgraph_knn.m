%load([data_directory 'connected_venue_graph_pca_vectors']);

weight_function = @(distances) (distances > 0);

pseudocount = 0.05;
%k = 500;

%weights = knn_weights(data, k, weight_function);

probability_function = @(data, responses, train_ind, test_ind) ...
    knn_probability(responses, train_ind, test_ind, weights, pseudocount);