#include "mex.h"
//#include "matrix.h"

#include <iostream>
#include <math.h>

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

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

/* [kernel_matrix, train_variances, test_variances] =
    	 wl_subtree_kernel_mex(graph_ind, responses, train_graphs, test_graphs, normalize) */

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	mwIndex *data_ir, *data_jc;
	double *graph_ind, *responses, *train_graphs, *test_graphs, *h, *kernel_matrix;
	mxLogical *normalize;

	int i, j, k, row, column, count, iteration, num_nodes, num_labels, num_graphs,
		num_train_graphs, num_test_graphs, num_elements_this_column, *new_responses,
		*feature_vectors;

	double *train_variances, *test_variances, *random_offsets, *signatures;

	unordered_map<double, int, hash<double> > signature_hash;

	base_generator_type generator(0);

	uniform_real<double> uniform_distribution(0, 1);
	boost::variate_generator<base_generator_type&, boost::uniform_real<double> >
		rand(generator, uniform_distribution);

	data_ir = mxGetIr(prhs[0]);
	data_jc = mxGetJc(prhs[0]);
	num_nodes = mxGetM(prhs[0]);

	responses    = mxGetPr(prhs[1]);

	graph_ind    = mxGetPr(prhs[2]);
	train_graphs = mxGetPr(prhs[3]);
	test_graphs  = mxGetPr(prhs[4]);
	h            = mxGetPr(prhs[5]);
	normalize    = mxGetLogicals(prhs[6]);

	num_train_graphs = mxGetN(prhs[3]) * mxGetM(prhs[3]);
	num_test_graphs  = mxGetN(prhs[4]) * mxGetM(prhs[4]);

	plhs[0] = mxCreateDoubleMatrix(num_train_graphs, num_test_graphs, mxREAL);
	kernel_matrix = mxGetPr(plhs[0]);

	num_labels = 0;
	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if (responses[i] > num_labels)
			num_labels = (int)(responses[i]);
		if (graph_ind[i] > num_graphs)
			num_graphs = (int)(graph_ind[i]);
	}

	new_responses = (int *)(mxMalloc(num_nodes * sizeof(int)));
	for (i = 0; i < num_nodes; i++)
		new_responses[i] = (int)(responses[i] + 0.5);

	feature_vectors = NULL;
	random_offsets = NULL;
	signatures = (double *)(mxMalloc(num_nodes * sizeof(double)));

	if (normalize[0] == true) {
		train_variances = (double *)(mxMalloc(num_graphs * sizeof(double)));
		test_variances =  (double *)(mxMalloc(num_graphs * sizeof(double)));
	}

	iteration = 0;
	while (true) {

		feature_vectors = (int *)(mxRealloc(feature_vectors, num_graphs * num_labels * sizeof(int)));
		for (i = 0; i < num_graphs; i++)
			for (j = 0; j < num_labels; j++)
				feature_vectors[INDEX(i, j, num_graphs)] = 0;

		for (i = 0; i < num_nodes; i++)
			feature_vectors[INDEX(graph_ind[i] - 1, new_responses[i] - 1, num_graphs)]++;

		if (normalize[0] == true) {
			for (i = 0; i < num_train_graphs; i++)
				for (j = 0; j < num_labels; j++)
					train_variances[i] +=
						feature_vectors[INDEX(train_graphs[i] - 1, j, num_graphs)] *
						feature_vectors[INDEX(train_graphs[i] - 1, j, num_graphs)];

			for (i = 0; i < num_test_graphs; i++)
				for (j = 0; j < num_labels; j++)
					test_variances[i] +=
						feature_vectors[INDEX(test_graphs[i] - 1, j, num_graphs)] *
						feature_vectors[INDEX(test_graphs[i] - 1, j, num_graphs)];
		}

		for (i = 0; i < num_train_graphs; i++)
			for (j = 0; j < num_test_graphs; j++)
				for (k = 0; k < num_labels; k++) {
					if ((feature_vectors[INDEX(train_graphs[i] - 1, k, num_graphs)] > 0) &&
							(feature_vectors[INDEX(test_graphs[j]  - 1, k, num_graphs)] > 0)) {
						kernel_matrix[INDEX(i, j, num_train_graphs)] +=
							feature_vectors[INDEX(train_graphs[i] - 1, k, num_graphs)] *
							feature_vectors[INDEX(test_graphs[j]  - 1, k, num_graphs)];
					}
				}

		if (iteration == (int)(h[0]))
			break;

		random_offsets = (double *)(mxRealloc(random_offsets, num_labels * sizeof(double)));
		for (i = 0; i < num_labels; i++)
			random_offsets[i] = rand();

		for (i = 0; i < num_nodes; i++)
			signatures[i] = (double)(new_responses[i]);

		count = 0;
		for (column = 0; column < num_nodes; column++) {

			num_elements_this_column = data_jc[column + 1] - data_jc[column];
			for (i = 0; i < num_elements_this_column; i++, count++) {
				row = data_ir[count];
				signatures[row] += random_offsets[new_responses[column] - 1];
			}
		}

		num_labels = 0;
		for (i = 0; i < num_nodes; i++) {

			if (signature_hash.count(signatures[i]) == 0)
				signature_hash[signatures[i]] = ++num_labels;

			new_responses[i] = signature_hash[signatures[i]];
		}
		signature_hash.empty();

		iteration++;
	}

	if (normalize[0] == true)
		for (i = 0; i < num_train_graphs; i++)
			for (j = 0; j < num_test_graphs; j++)
				kernel_matrix[INDEX(i, j, num_train_graphs)] *=
					(1 / sqrt(train_variances[i] * test_variances[j]));

	mxFree(new_responses);
	mxFree(feature_vectors);
	mxFree(random_offsets);
	mxFree(signatures);
	mxFree(train_variances);
	mxFree(test_variances);

}
