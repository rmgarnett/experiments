function [results, elapsed] = perform_uncertainty_experiment(data, ...
        responses, num_additional, seed, probability_function, ...
        num_experiments, num_evaluations, report)
  
  stream = RandStream('mt19937ar', 'seed', seed);
  RandStream.setDefaultStream(stream);

  utility_function = @(data, responses, train_ind) ...
      count_utility(responses, train_ind);

  expected_utility_function = @(data, responses, train_ind, test_ind) ...
      expected_maximum_variance_utility(data, responses, train_ind, ...
          test_ind, probability_function);

  selection_functions = cell(1);
  selection_functions{1} = @(data, responses, train_ind) ...
      identity_selection_function(responses, train_ind);

  results = zeros(num_experiments, num_evaluations);
  elapsed = zeros(num_experiments, 1);

  for experiment = 1:num_experiments

    % always add one positive point
    r = randperm(nnz(responses == 1));
    train_ind = logical_ind(responses == 1, r(1:(num_additional + 1)));

    r = randperm(nnz(responses == 0));
    train_ind = [train_ind; logical_ind(responses == 0, r(1:(num_additional + 1)))];
    
    start = tic;
    [~, utilities] = optimal_learning(data, responses, train_ind, ...
            selection_functions, probability_function, ...
            expected_utility_function, utility_function, ...
            num_evaluations, 1, true);
    utilities = utilities - utility_function(data, responses, train_ind);
    results(experiment, :) = utilities;
    elapsed(experiment) = toc(start);
    
    fprintf('experiment %i: uncertainty utility: %i, mean: %.2f, took: %.2fs, mean: %.2fs', ...
            experiment, ...
            results(experiment, end), ...
            mean(results(1:experiment, end)), ...
            elapsed(experiment), ...
            mean(elapsed(1:experiment)));
    fprintf('\n');

    for steps = 1:numel(report)
      fprintf('uncertainty utility after %i steps: %i, mean: %.2f', ...
              report(steps), ...
              results(experiment, report(steps)), ...
              mean(results(1:experiment, report(steps))));
      fprintf('\n');
    end
  end
end
