function [utilities, elapsed] = perform_search_experiment(data, ...
          responses, seed, probability_function, probability_bound, ...
          num_experiments, num_evaluations, lookahead, report)

  stream = RandStream('mt19937ar', 'seed', seed);
  RandStream.setGlobalStream(stream);

  % always add one positive point
  r = randperm(nnz(responses == 1));
  train_ind = logical_ind(responses == 1, r(1));

  start = tic;

  [~, utilities] = active_search(data, responses, train_ind, ...
          probability_function, probability_bound, lookahead, ...
          num_evaluations, false);
  utilities = utilities -  1;
  
  elapsed = toc(start);

  for steps = 1:numel(report)
    fprintf('%i-step utility after %i steps: %i\n', ...
            lookahead, ...
            report(steps), ...
            utilities(report(steps)));
  end
end


