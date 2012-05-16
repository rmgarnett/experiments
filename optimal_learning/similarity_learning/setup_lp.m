required_options = {'A', 'tolerance', 'max_iterations'};
check_required_options;

if (options_defined)
  data = bsxfun(@times, 1 ./ sum(A, 2), A);

  probability_function = @(data, responses, train_ind, test_ind) ...
      label_propagation_probability(data, responses, train_ind, test_ind, ...
          tolerance, max_iterations);
end