function train_ind = generate_balanced_graph_train_ind(graph_ind, ...
          num_train_per_graph)

  num_graphs = max(graph_ind);

  train_ind = [];
  for i = 1:num_graphs
    ind = (graph_ind == i);

    num_nodes = nnz(ind);
    permutation = randperm(num_nodes);

    train_ind = [train_ind; ...
                 logical_ind(ind, 1:min(num_train_per_graph, num_nodes))];
  end

end