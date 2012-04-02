#include "mex.h"
//#include "matrix.h"

#include <math.h>

#include <time.h>
#include <iostream>

#include <tr1/functional>
#include <tr1/unordered_map>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>

using std::cout;
using std::endl;

using std::tr1::hash;
using std::tr1::unordered_map;

using boost::uniform_real;
using boost::variate_generator;

typedef boost::mt19937 base_generator_type;

/* uniform (0, 1) random number generator */
base_generator_type generator(0);
uniform_real<double> uniform_distribution;
variate_generator<base_generator_type&, uniform_real<double> > uniform_rand(generator, uniform_distribution);

/* convenience macro for indexing 1d arrays as if they were 2d */
#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

/* convenience macros for input/output indices in case we want to change */
#define A_ARG               prhs[0]
#define LABELS_ARG          prhs[1]
#define GRAPH_IND_ARG       prhs[2]
#define TRAIN_GRAPH_IND_ARG prhs[3]
#define TEST_GRAPH_IND_ARG  prhs[4]
#define H_ARG               prhs[5]
#define PROBABILITIES_ARG   prhs[6]
#define NUM_SAMPLES_ARG     prhs[7]

#define MEANS_ARG           plhs[0]
#define VARIANCES_ARG       plhs[1]

/* [means, variances] =
    	 wl_subtree_kernel_moments(A, labels, graph_ind, ...
			     train_graph_ind, test_graph_ind, h, num_samples) */

void generate_samples(int num_nodes, int num_samples, int num_labels, double *probabilities, int *samples);

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{

	/* inputs */
	mwIndex *A_ir, *A_jc;
	double *graph_ind, *labels_in, *train_graph_ind, *test_graph_ind, *h_in,
		*probabilities_in, *probabilities, *num_samples_in;
	int *labels;
	int h, num_samples;

	/* outputs */
	double *means, *variances;

	/* internal variables */
	int i, j, k, row, column, count, iteration, num_nodes, num_labels, num_graphs,
		num_train_graphs, num_test_graphs, num_elements_this_column, *samples;

	double *count_means, *count_variances, *sum_probabilities_squared, *random_offsets, *signatures, r, total;
	unordered_map<double, int, hash<double> > signature_hash;

	clock_t start = clock();

	/* get inputs */
	A_ir              = mxGetIr(A_ARG);
	A_jc              = mxGetJc(A_ARG);
	labels_in         = mxGetPr(LABELS_ARG);
	graph_ind         = mxGetPr(GRAPH_IND_ARG);
	train_graph_ind   = mxGetPr(TRAIN_GRAPH_IND_ARG);
	test_graph_ind    = mxGetPr(TEST_GRAPH_IND_ARG);
	h_in              = mxGetPr(H_ARG);
	probabilities_in  = mxGetPr(PROBABILITIES_ARG);
	num_samples_in    = mxGetPr(NUM_SAMPLES_ARG);

	/* dereference to avoid annoying casting and indexing */
	h = (int)(h_in[0] + 0.5);
	num_samples = (int)(num_samples_in[0] + 0.5);

	/* array sizes */
	num_nodes        = mxGetN(A_ARG);
	num_train_graphs = mxGetN(TRAIN_GRAPH_IND_ARG) * mxGetM(TRAIN_GRAPH_IND_ARG);
	num_test_graphs  = mxGetN(TEST_GRAPH_IND_ARG) * mxGetM(TEST_GRAPH_IND_ARG);

	/* copy label matrix because we will overwrite it */
	labels = (int *)(mxMalloc(num_nodes * sizeof(int)));
	for (i = 0; i < num_nodes; i++)
		labels[i] = (int)(labels_in[i] + 0.5);

	num_labels = 0;
	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if (labels[i] > num_labels)
			num_labels = labels[i];
		if ((int)(graph_ind[i]) > num_graphs)
			num_graphs = (int)(graph_ind[i]);
	}

	/* initialize outputs */
	MEANS_ARG = mxCreateDoubleMatrix(num_train_graphs, num_test_graphs, mxREAL);
	means = mxGetPr(MEANS_ARG);

	VARIANCES_ARG = mxCreateDoubleMatrix(num_train_graphs, num_test_graphs, mxREAL);
	variances = mxGetPr(VARIANCES_ARG);

	/* copy probability matrix because we will overwrite it */
	probabilities = (double *)(mxMalloc(num_nodes * num_labels * sizeof(double)));
	for (i = 0; i < num_nodes; i++)
		for (j = 0; j < num_labels; j++)
			probabilities[INDEX(i, j, num_nodes)] = probabilities_in[INDEX(i, j, num_nodes)];

	count_means               = NULL;
	count_variances           = NULL;
	sum_probabilities_squared = NULL;
	random_offsets            = NULL;
	samples                   = NULL;
	signatures                = NULL;

	iteration = 0;
	while (true) {

		count_means               = (double *)(mxRealloc(count_means,               num_graphs * num_labels * sizeof(double)));
		count_variances           = (double *)(mxRealloc(count_variances,           num_graphs * num_labels * sizeof(double)));
		sum_probabilities_squared = (double *)(mxRealloc(sum_probabilities_squared, num_graphs * num_labels * sizeof(double)));

		for (i = 0; i < num_graphs; i++)
			for (j = 0; j < num_labels; j++) {
				count_means[INDEX(i, j, num_graphs)]               = 0;
				sum_probabilities_squared[INDEX(i, j, num_graphs)] = 0;
		}

		for (i = 0; i < num_nodes; i++)
			for (j = 0; j < num_labels; j++) {
				count_means[INDEX(graph_ind[i] - 1, j, num_graphs)]                += probabilities[INDEX(i, j, num_nodes)];
				sum_probabilities_squared[INDEX(graph_ind[i] - 1, j, num_graphs)]  += probabilities[INDEX(i, j, num_nodes)] *
				 	                                                                    probabilities[INDEX(i, j, num_nodes)];
			}

		for (i = 0; i < num_graphs; i++)
			for (j = 0; j < num_labels; j++) {
				count_variances[INDEX(i, j, num_graphs)] =
					count_means[INDEX(i, j, num_graphs)] * (1 - count_means[INDEX(i, j, num_graphs)] / num_nodes) -
					(sum_probabilities_squared[INDEX(i, j, num_graphs)] -
					 count_means[INDEX(i, j, num_graphs)] * count_means[INDEX(i, j, num_graphs)] / num_nodes);
			}

		for (i = 0; i < num_train_graphs; i++)
 			for (j = 0; j < num_test_graphs; j++)
				for (k = 0; k < num_labels; k++) {

					means[INDEX(i, j, num_train_graphs)] +=
						count_means[INDEX(train_graph_ind[i] - 1, k, num_graphs)] *
						count_means[INDEX( test_graph_ind[j] - 1, k, num_graphs)];

					variances[INDEX(i, j, num_train_graphs)] +=
						count_means    [INDEX(train_graph_ind[i] - 1, k, num_graphs)] *
						count_means    [INDEX(train_graph_ind[i] - 1, k, num_graphs)] *
						count_variances[INDEX( test_graph_ind[j] - 1, k, num_graphs)]
						+
						count_means    [INDEX( test_graph_ind[j] - 1, k, num_graphs)] *
						count_means    [INDEX( test_graph_ind[j] - 1, k, num_graphs)] *
						count_variances[INDEX(train_graph_ind[i] - 1, k, num_graphs)]
						+
						count_variances[INDEX(train_graph_ind[i] - 1, k, num_graphs)] *
						count_variances[INDEX( test_graph_ind[j] - 1, k, num_graphs)];
				}

		if (iteration == h)
			break;

		random_offsets = (double *)(mxRealloc(random_offsets, num_labels * sizeof(double)));
		for (i = 0; i < num_labels; i++)
			random_offsets[i] = uniform_rand();

		signatures = (double *)(mxRealloc(signatures, num_nodes * num_samples * sizeof(double)));
		samples    =    (int *)(mxRealloc(samples,    num_nodes * num_samples * sizeof(int)));

		generate_samples(num_nodes, num_samples, num_labels, probabilities, samples);

		for (i = 0; i < num_nodes; i++) {
			for (j = 0; j < num_samples; j++) {
				signatures[INDEX(i, j, num_nodes)] = samples[INDEX(i, j, num_nodes)];

				count = 0;
				for (column = 0; column < num_nodes; column++) {

					num_elements_this_column = A_jc[column + 1] - A_jc[column];
					for (k = 0; k < num_elements_this_column; k++, count++) {
						row = A_ir[count];
						signatures[INDEX(row, j, num_nodes)] += random_offsets[samples[INDEX(i, j, num_nodes)]];
					}
				}
			}
		}

		num_labels = 0;
		for (i = 0; i < num_nodes; i++)
			for (j = 0; j < num_samples; j++)
				if (signature_hash.count(signatures[INDEX(i, j, num_nodes)]) == 0)
					signature_hash[signatures[INDEX(i, j, num_nodes)]] = num_labels++;

		probabilities = (double *)(mxRealloc(probabilities, num_nodes * num_labels * sizeof(double)));
		for (i = 0; i < num_nodes; i++)
			for (j = 0; j < num_labels; j++)
				probabilities[INDEX(i, j, num_nodes)] = 0;

		for (i = 0; i < num_nodes; i++)
			for (j = 0; j < num_samples; j++)
				probabilities[INDEX(i, signature_hash[signatures[INDEX(i, j, num_nodes)]], num_nodes)] += (1 / num_samples);

		signature_hash.empty();

		iteration++;
	}

	mxFree(labels);
	mxFree(probabilities);
	mxFree(count_means);
	mxFree(count_variances);
	mxFree(random_offsets);
	mxFree(samples);
	mxFree(signatures);

}

void generate_samples(int num_nodes, int num_samples, int num_labels, double *probabilities, int *samples)
{
	int i, j, k;
	double r, total;

	for (i = 0; i < num_nodes; i++) {
		for (j = 0; j < num_samples; j++) {
			r = uniform_rand();
			total = 0;
			samples[INDEX(i, j, num_nodes)] = 0;
			for (k = 0; k < num_labels; k++) {
				total += probabilities[INDEX(i, k, num_nodes)];
				if (total > r)
					break;
				samples[INDEX(i, j, num_nodes)]++;
			}
		}
	}
}
