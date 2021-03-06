function [log_likelihood, accuracy, confusion_matrix] = ...
      evaluate_classifier(data, responses, train_ind, test_ind, ...
                          probability_function)

  num_test = numel(test_ind);
  num_classes = max(responses);

  probabilities = probability_function(data, responses, train_ind, ...
          test_ind);
  [~, predictions] = max(probabilities, [], 2);

  log_likelihood = ...
      sum(log(probabilities(...
          sub2ind(size(probabilities), (1:num_test)', responses(test_ind)))));

  confusion_matrix = sparse(predictions, responses(test_ind), ...
                            ones(num_test, 1), num_classes, num_classes);
  confusion_matrix = full(confusion_matrix);

  accuracy = sum(diag(confusion_matrix)) / num_test;

end