#include "mex.h"
//#include "matrix.h"

#include <math.h>

#include <iostream>
using std::cout;
using std::endl;

#include <tr1/functional>
#include <tr1/unordered_map>
using std::tr1::hash;
using std::tr1::unordered_map;

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
using boost::uniform_real;
using boost::variate_generator;

typedef boost::mt19937 base_generator_type;

/* convenience macros for input/output indices in case we want to change */
#define A_ARG               prhs[0]
#define LABELS_ARG          prhs[1]
#define GRAPH_IND_ARG       prhs[2]
#define TRAIN_GRAPH_IND_ARG prhs[3]
#define TEST_GRAPH_IND_ARG  prhs[4]
#define H_ARG               prhs[5]
#define NORMALIZE_ARG       prhs[6]

#define KERNEL_MATRIX_ARG   plhs[0]

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	mwIndex *A_ir, *A_jc;
	double *graph_ind, *labels_in, *train_graphs, *test_graphs, *h_in, *kernel_matrix;
	mxLogical *normalize_in;
	int h, *labels;
	bool normalize;

	int i, j, k, row, column, count, iteration, num_nodes, num_labels, num_graphs,
		num_train_graphs, num_test_graphs, num_elements_this_column, *new_responses,
		*feature_vectors;

	double *train_variances, *test_variances, *random_offsets, *signatures;

	unordered_map<double, int, hash<double> > signature_hash;

	base_generator_type generator(0);

	uniform_real<double> uniform_distribution(0, 1);
	boost::variate_generator<base_generator_type&, boost::uniform_real<double> >
		rand(generator, uniform_distribution);

	A_ir         = mxGetIr(A_ARG);
	A_jc         = mxGetJc(A_ARG);

	labels_in    = mxGetPr(LABELS_ARG);

	graph_ind    = mxGetPr(GRAPH_IND_ARG);
	train_graphs = mxGetPr(TRAIN_GRAPH_IND_ARG);
	test_graphs  = mxGetPr(TEST_GRAPH_IND_ARG);
	h_in         = mxGetPr(H_ARG);
	normalize_in = mxGetLogicals(NORMALIZE_ARG);

	/* dereference to avoid annoying casting and indexing */
	h         = (int)(h_in[0] + 0.5);
	normalize = normalize_in[0];

	/* array sizes */
	num_nodes        = mxGetN(A_ARG);
	num_train_graphs = mxGetN(TRAIN_GRAPH_IND_ARG) * mxGetM(TRAIN_GRAPH_IND_ARG);
	num_test_graphs  = mxGetN(TEST_GRAPH_IND_ARG) * mxGetM(TEST_GRAPH_IND_ARG);

	KERNEL_MATRIX_ARG = mxCreateDoubleMatrix(num_train_graphs, num_test_graphs, mxREAL);
	kernel_matrix = mxGetPr(KERNEL_MATRIX_ARG);

	/* copy label matrix because we will overwrite it */
	labels = (int *)(mxMalloc(num_nodes * sizeof(int)));
	for (i = 0; i < num_nodes; i++)
		labels[i] = (int)(labels_in[i] + 0.5);

	num_labels = 0;
	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if (labels[i] > num_labels)
			num_labels = (int)(labels[i]);
		if ((int)(graph_ind[i]) > num_graphs)
			num_graphs = (int)(graph_ind[i]) + 0.5;
	}

	feature_vectors = NULL;
	random_offsets = NULL;
	signatures = (double *)(mxMalloc(num_nodes * sizeof(double)));

	if (normalize) {
		train_variances = (double *)(mxMalloc(num_graphs * sizeof(double)));
		test_variances =  (double *)(mxMalloc(num_graphs * sizeof(double)));
	}

	iteration = 0;
	while (true) {

		feature_vectors = (int *)(mxRealloc(feature_vectors, num_graphs * num_labels * sizeof(int)));
		for (j = 0; j < num_labels; j++)
			for (i = 0; i < num_graphs; i++)
				feature_vectors[INDEX(i, j, num_graphs)] = 0;

		for (i = 0; i < num_nodes; i++)
			feature_vectors[INDEX(graph_ind[i] - 1, labels[i] - 1, num_graphs)]++;

		if (normalize) {
			for (j = 0; j < num_labels; j++)
				for (i = 0; i < num_train_graphs; i++)
					train_variances[i] +=
						feature_vectors[INDEX(train_graphs[i] - 1, j, num_graphs)] *
						feature_vectors[INDEX(train_graphs[i] - 1, j, num_graphs)];

			for (j = 0; j < num_labels; j++)
				for (i = 0; i < num_test_graphs; i++)
					test_variances[i] +=
						feature_vectors[INDEX(test_graphs[i] - 1, j, num_graphs)] *
						feature_vectors[INDEX(test_graphs[i] - 1, j, num_graphs)];
		}

		for (j = 0; j < num_test_graphs; j++)
			for (i = 0; i < num_train_graphs; i++)
				for (k = 0; k < num_labels; k++) {
					if ((feature_vectors[INDEX(train_graphs[i] - 1, k, num_graphs)] > 0) &&
							(feature_vectors[INDEX(test_graphs[j]  - 1, k, num_graphs)] > 0)) {
						kernel_matrix[INDEX(i, j, num_train_graphs)] +=
							feature_vectors[INDEX(train_graphs[i] - 1, k, num_graphs)] *
							feature_vectors[INDEX(test_graphs[j]  - 1, k, num_graphs)];
					}
				}

		if (iteration == h)
			break;

		random_offsets = (double *)(mxRealloc(random_offsets, num_labels * sizeof(double)));
		for (i = 0; i < num_labels; i++)
			random_offsets[i] = rand();

		for (i = 0; i < num_nodes; i++)
			signatures[i] = (double)(labels[i]);

 		count = 0;
		for (column = 0; column < num_nodes; column++) {

			num_elements_this_column = A_jc[column + 1] - A_jc[column];
			for (i = 0; i < num_elements_this_column; i++, count++) {
				row = A_ir[count];
				signatures[row] += random_offsets[labels[column] - 1];
			}
		}

		num_labels = 0;
		for (i = 0; i < num_nodes; i++) {

			if (signature_hash.count(signatures[i]) == 0)
				signature_hash[signatures[i]] = ++num_labels;

			labels[i] = signature_hash[signatures[i]];
		}
		signature_hash.empty();

		iteration++;
	}

	cout << "count: " << count;
	cout << "{";
	for (i = 0; i < count; i++)
		cout << A_ir[i] << ", ";
	cout << "};" << endl;

	cout << "{";
	for (i = 0; i < num_nodes + 1; i++)
		cout << A_jc[i] << ", ";
	cout << "};" << endl;

	if (normalize)
		for (j = 0; j < num_test_graphs; j++)
			for (i = 0; i < num_train_graphs; i++)
				kernel_matrix[INDEX(i, j, num_train_graphs)] *=
					(1 / sqrt(train_variances[i] * test_variances[j]));

	mxFree(labels);
	mxFree(feature_vectors);
	mxFree(random_offsets);
	mxFree(signatures);
	if (normalize) {
		mxFree(train_variances);
		mxFree(test_variances);
	}
}
