neighbors = neighbors';
similarities = similarities';

weights = sparse(kron((1:num_observations)', ones(num_neighbors, 1)), ...
                 neighbors(:), similarities(:), ...
                 num_observations, num_observations);

max_weights = max(weights);
max_weights = full(max_weights);

probability_function = @(data, responses, train_ind, test_ind) ...
    knn_probability(responses, train_ind, test_ind, weights, pseudocount);

probability_bound = @(data, responses, train_ind, test_ind, num_positives) ...
    knn_probability_bound(responses, train_ind, test_ind, weights, ...
                          max_weights, pseudocount, num_positives);