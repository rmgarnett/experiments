prepare_grasping_data;

h                = 2;
num_samples      = 10;
lookahead        = 1;
verbose          = false;
k                = 3;
num_evaluations  = 10;
train_fraction   = 1;
num_iterations   = 5;
use_map          = false;
use_max          = true;
train_graphs     = [2 6 7 15 23 27 31];

setup_lp;
perform_online_similarity_experiment;
