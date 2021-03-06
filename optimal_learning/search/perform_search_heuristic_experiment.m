function [results, elapsed] = perform_search_heuristic_experiment(data, ...
          responses, num_additional, seed, probability_function, ...
          num_experiments, num_evaluations, max_lookahead, report)
  
  stream = RandStream('mt19937ar', 'seed', seed);
  RandStream.setGlobalStream(stream);

  utility_function = @(data, responses, train_ind) ...
      top_probabilities_utility(data, responses, train_ind, ...
                                probability_function, num_evaluations);

  selection_functions = cell(max_lookahead, 1);
  for i = 1:max_lookahead
    selection_functions{i} = @(data, responses, train_ind) ...
        identity_selector(responses, train_ind);
  end

  results = zeros(num_experiments, num_evaluations, max_lookahead);
  elapsed = zeros(num_experiments, max_lookahead);

  for experiment = 1:num_experiments

    % always add one positive point
    r = randperm(nnz(responses == 1));
    train_ind = logical_ind(responses == 1, r(1:(num_additional + 1)));

    r = randperm(nnz(responses ~=1));
    train_ind = [train_ind; logical_ind(responses ~= 1, r(1:(num_additional + 1)))];

    for lookahead = 1:max_lookahead
      start = tic;
      [~, utilities] = optimal_learning(data, responses, train_ind, ...
              utility_function, probability_function, selection_functions, ...
              lookahead, num_evaluations, true);
      utilities = utilities - num_additional;
      results(experiment, :, lookahead) = utilities;
      elapsed(experiment, lookahead) = toc(start);

      for i = 1:lookahead
        fprintf('experiment %i: %i-step utility: %i, mean: %.2f, took: %.2fs, mean: %.2fs', ...
                experiment, ...
                i, ...
                results(experiment, end, i), ...
                mean(results(1:experiment, end, i)), ...
                elapsed(experiment, i), ...
                mean(elapsed(1:experiment, i)));
        fprintf('\n');
      end
    end

    for steps = 1:numel(report)
      for lookahead = 1:max_lookahead
        fprintf('%i-step-utility after %i steps: %i, mean: %.2f', ...
                lookahead, ...
                report(steps), ...
                results(experiment, report(steps), lookahead), ...
                mean(results(1:experiment, report(steps), lookahead)));
        fprintf('\n');
      end
    end
  end
end
