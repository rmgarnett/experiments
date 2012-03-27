required_options = {'num_msrc_classes'};
check_required_options;

if (options_defined)

  data_directory = '~/work/data/images/msrc/processed/';
  load([data_directory 'msrc_' int2str(num_msrc_classes) '_class.mat']);

  if (num_msrc_classes == 9)
    bad_classes = [6 7 9 11];
  end

  bad_ind = ismember(responses, bad_classes);
  bad_graphs = unique(graph_ind(bad_ind));

  to_keep_ind = ~ismember(graph_ind, bad_graphs);

  data      = data(to_keep_ind, to_keep_ind);
  responses = responses(to_keep_ind);
  graph_ind = graph_ind(to_keep_ind);

  unique_graph_inds = unique(graph_ind);
  unique_responses  = unique(responses);

  for i = 1:numel(unique_graph_inds)
    graph_ind(graph_ind == unique_graph_inds(i)) = i;
  end

  for i = 1:numel(unique_responses)
    responses(responses == unique_responses(i)) = i;
  end

  num_nodes   = size(data, 1);
  num_classes = max(responses);
  num_graphs  = max(graph_ind);
end