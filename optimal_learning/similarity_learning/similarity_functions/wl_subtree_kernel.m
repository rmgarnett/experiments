% calculates the weisfeiler--lehman subtree graph kernel as
% described in
%
% Shervashidze, N., Schweitzer, P., Jan van Leeuwen, E., Mehlhorn, K.,
% and Borgwardt, K.M. (2010).  Weisfeiler--Lehman graph kernels.
% The Journal of Machine Learning Research, vol. 11, pp. 1--48.
%
% function kernel_matrix = wl_subtree_kernel(data, responses, graph_ind, h)
%
% inputs:
%        data: an (n x n) matrix where n is the total number of
%              nodes appearing in all graphs.  data should be a
%              block-diagonal matrix where the ith block
%              corresponds to the adjacency matrix of the ith
%              graph.  the graph_ind argument, below, provides an
%              index into data for conveniently extracting the
%              adjacency matrix for a specified graph.
%   responses: an (n x 1) vector containing the node labels of the
%              nodes appearing in all graphs.  node labels are
%              expected to be integers from 1..[num_classes]. the
%              entries in responses align with the rows of data, and
%              the graph_ind argument, below, can also be used to
%              index into responses to extract the node labels for a
%              specified graph.
%   graph_ind: an (n x 1) vector indicating which graph each row
%              (node) appearing in data/responses corresponds to.
%              the graphs are expected to be numbered with the
%              integers 1..[num_graphs].
%           h: an integer corresponding to the desired h parameter
%              as defined in the paper above.  if K_i represents
%              the kernel on the feature vectors resulting from the
%              ith application of the wl-subtree transformation,
%              then we return
%
%                 K_1(G, G) + K_2(G, G) + ... + K_h(G, G).
%
% outputs:
%   kernel_matrix: the resulting kernel matrix between the graphs.
%
% copyright (c) roman garnett, 2012.

function kernel_matrix = wl_subtree_kernel(data, responses, graph_ind, ...
          train_graphs, test_graphs, h, normalize)

  normalize = exist('normalize', 'var') && normalize;
  if (normalize)
    original_responses = responses;
  end

  used_graphs = union(train_graphs, test_graphs);

  used_ind = false(size(data, 1), 1);
  for i = 1:numel(used_graphs)
    used_graph_ind = (graph_ind == used_graphs(i));

    used_ind = used_ind | used_graph_ind;
    graph_ind(used_graph_ind) = i;

    train_graphs(train_graphs == used_graphs(i)) = i;
    test_graphs(test_graphs   == used_graphs(i)) = i;
  end

  data      = data(used_ind, used_ind);
  responses = responses(used_ind);
  graph_ind(~used_ind) = 0;

  num_train_graphs = numel(train_graphs);
  num_test_graphs  = numel(test_graphs);
  num_graphs       = numel(used_graphs);

  num_nodes = size(data, 1);

  kernel_matrix = zeros(num_train_graphs, num_test_graphs);

  iteration = 0;
  while (true)
    label_set = unique(responses);
    num_labels = numel(label_set);

    % the contribution to the graph feature vectors at every step
    % is simply the counts of each node label on the graph
    feature_vectors = sparse(zeros(num_graphs, num_labels));
    for i = 1:num_graphs
      ind = (graph_ind == i);
      labels = responses(ind);

      feature_vectors(i, :) = sparse(histc(labels, label_set)');
    end

    % the kernel is the outer product of the feature vectors
    kernel_matrix = kernel_matrix + ...
        feature_vectors(train_graphs, :) * feature_vectors(test_graphs, :)';

    % exit after h applications of the WL transformation
    if (iteration == h)
      break;
    end

    % perform the WL transformation
    signatures = zeros(num_nodes, num_labels + 1, 'uint8');

    % the first entry of each node's signature is its current label
    signatures(:, 1) = uint8(responses);

    for i = 1:num_graphs
      % extract the labels and adjacency matrix for this graph only
      ind = (graph_ind == i);
      labels = responses(ind);
      A = full(data(ind, ind));
      A = (A > 0);

      % the signatures are the counts of the labels surrounding each node
      signatures(ind, 2:end) = uint8(histc(bsxfun(@times, A, labels), label_set)');
    end

    % perform signature compression
    [~, ~, responses] = unique(signatures, 'rows');

    iteration = iteration + 1;
  end

  if (normalize)
    train_variances = zeros(num_train_graphs, 1);
    for i = 1:num_train_graphs
      ind = (graph_ind == train_graphs(i));
      train_variances(i) = wl_subtree_kernel(data(ind, ind), ...
              original_responses(ind), ones(nnz(ind), 1), 1, 1, h, false);
    end

    test_variances = zeros(1, num_test_graphs);
    for i = 1:num_test_graphs
      ind = (graph_ind == test_graphs(i));
      test_variances(i) = wl_subtree_kernel(data(ind, ind), ...
              original_responses(ind), ones(nnz(ind), 1), 1, 1, h, false);
    end

    kernel_matrix = bsxfun(@times, 1 ./ sqrt(train_variances), ...
                           bsxfun(@times, 1 ./ sqrt(test_variances), kernel_matrix));
  end

end