required_options = {'num_msrc_classes'};
check_required_options;

if (options_defined)

  data_directory = '~/work/data/images/msrc/processed/';
  load([data_directory 'msrc_' int2str(num_msrc_classes) '_class_big.mat']);

end

post_process_similarity_experiment_data;