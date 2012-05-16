function log_likelihood = calculate_log_likelihood(x, means, variances)

  log_likelihood = 0;
  for i = 1:numel(means)
    log_likelihood = log_likelihood - normlike([mean(i) ...
                        sqrt(variances(i))], x(i));
  end
  
end