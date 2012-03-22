weight_function = @(distances) (double(distances > 0));
weights = knn_weights(data, k, weight_function);

probability_function = @(data, responses, train_ind, test_ind) ...
    knn_probability(responses, train_ind, test_ind, weights, pseudocount);
