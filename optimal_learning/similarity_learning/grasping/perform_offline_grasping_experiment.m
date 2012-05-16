prepare_grasping_data;

h                   = 2;
num_train_per_graph = 10;
num_samples         = 100;
lookahead           = 1;
verbose             = true;
num_evaluations     = 9 * num_graphs;

tolerance      = 1e-1;
max_iterations = 10;

setup_lp;

perform_offline_similarity_experiment;
