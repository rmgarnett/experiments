probability_function = @(data, responses, train_ind, test_ind) ...
    label_propagation_probability(data, responses, train_ind, test_ind, ...
        1e-1, 10);
