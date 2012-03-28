#include "mex.h"
#include <stdlib.h>
#include "matrix.h"

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

/* feature_vectors = create_feature_vectors(responses, graph_ind) */

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	double *graph_ind, *responses, *train_graphs, *test_graphs, *kernel_matrix,
		*train_variances, *test_variances;

	int i, j, k, num_graphs, num_labels, num_nodes, num_train_graphs, num_test_graphs;
	unsigned short *feature_vectors;

	responses    = mxGetPr(prhs[0]);
	num_nodes    = mxGetM(prhs[0]);
	graph_ind    = mxGetPr(prhs[1]);
	train_graphs = mxGetPr(prhs[2]);
	test_graphs  = mxGetPr(prhs[3]);

	num_train_graphs = mxGetN(prhs[2]) * mxGetM(prhs[2]);
	num_test_graphs  = mxGetN(prhs[3]) * mxGetM(prhs[3]);

	num_labels = 0;
	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if (responses[i] > num_labels)
			num_labels = (int) responses[i];
		if (graph_ind[i] > num_graphs)
			num_graphs = (int) graph_ind[i];
	}

	feature_vectors = (unsigned short *)(malloc(num_graphs * num_labels * sizeof(unsigned short)));
	for (i = 0; i < num_graphs; i++)
		for (j = 0; j < num_labels; j++)
			feature_vectors[INDEX(i, j, num_graphs)] = 0;

	plhs[0] = mxCreateDoubleMatrix(num_train_graphs, num_test_graphs, mxREAL);
	kernel_matrix = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(num_train_graphs, 1, mxREAL);
	train_variances = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(1, num_test_graphs, mxREAL);
	test_variances = mxGetPr(plhs[2]);

	for (i = 0; i < num_nodes; i++)
		feature_vectors[INDEX(graph_ind[i] - 1, responses[i] - 1, num_graphs)] += 1;

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

	free(feature_vectors);
}
