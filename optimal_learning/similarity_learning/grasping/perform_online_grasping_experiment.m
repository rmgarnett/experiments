prepare_grasping_data;

h              = 2;
train_fraction = 20.05;
num_samples    = 100;
lookahead      = 1;
verbose        = true;
k_fraction     = 5 / num_graphs;

tolerance      = 1e-1;
max_iterations = 10;

setup_lp;

perform_online_similarity_experiment;
