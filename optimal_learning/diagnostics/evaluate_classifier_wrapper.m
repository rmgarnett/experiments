required_options = {'data', 'responses', 'probability_function', ...
                    'train_fractions'};
check_required_options;

if (options_defined)
  num_points = size(data, 1);

  for train_fraction = train_fractions
    fprintf('with %.1f%% training data:\n', train_fraction * 100);

    permutation = randperm(num_points);
    train_ind = permutation(1:ceil(num_points * train_fraction))';
    test_ind = identity_selector(responses, train_ind);

    tic;
    [log_likelihood, accuracy, confusion_matrix] = ...
        evaluate_classifier(data, responses, train_ind, test_ind, ...
                            probability_function);
    elapsed = toc;

    fprintf('  took: %0.1fs\n', elapsed);
    fprintf('  log likelihood: %0.3f\n', log_likelihood);
    fprintf('  accuracy: %0.3f\n', accuracy);
    fprintf('  confustion matrix:\n');

    digits_required = ceil(1 + max(log10(confusion_matrix(:))));
    format_string = ['%' num2str(digits_required) 'i '];
    confusion_matrix_format = repmat(format_string, 1, num_classes);
    fprintf(['    ' confusion_matrix_format '\n'], confusion_matrix);
  end
end