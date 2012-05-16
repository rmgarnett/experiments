function wdd_scores = calculate_wdd_scores(data, responses, train_ind, ...
          test_ind, probability_function)

  unlabeled_ind = identity_selector(responses, train_ind);
  
  probabilities = probability_function(data, responses, train_ind, test_ind);
  mean_probability = mean(probabilities);

  kl_divergence = @(p, q) ...
      sum(p .* log(bsxfun(@times, p, 1 ./ q)), 2);

  node_scores = kl_divergence(probabilities, mean_probability);
  network_scores = zeros(numel(test_ind), 1);
  
  for i = 1:numel(test_ind)
    neighbor_ind = find(data(i, test_ind) > 0);
    for j = 1:numel(neighbor_ind)
      network_scores(i) = network_scores(i) + ...
          exp(-sum(kl_divergence(probabilities(i, :), ...
                                 probabilities(neighbor_ind(j), :))));
    end
  end
  
  wdd_scores = -Inf(size(data, 1), 1);
  wdd_scores(test_ind) = node_scores .* network_scores;
  wdd_scores(isnan(wdd_scores)) = 0;
  wdd_scores = wdd_scores(unlabeled_ind);

end