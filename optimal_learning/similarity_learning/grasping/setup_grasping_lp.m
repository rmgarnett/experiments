required_options = {'tolerance', 'max_iterations'};
check_required_options;

if (options_defined)
  data = perform_row_normalization(data);

  probability_function = @(data, responses, train_ind, test_ind) ...
      label_propagation_probability(data, responses, train_ind, test_ind, ...
          tolerance, max_iterations);
end