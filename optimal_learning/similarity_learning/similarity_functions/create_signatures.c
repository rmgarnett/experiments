#include "mex.h"
#include "matrix.h"
#include <stdio.h>

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	int i, j, row, column, count, num_labels;
	size_t num_nodes, num_non_zeros;
	mwIndex *data_ir, *data_jc;
	double *responses;
	unsigned short *signatures;

	data_ir = mxGetIr(prhs[0]);
	data_jc = mxGetJc(prhs[0]);
	
	num_non_zeros = mxGetNzmax(prhs[0]);
	
	responses = mxGetPr(prhs[1]);
	
	num_nodes = mxGetM(prhs[0]);

	num_labels = 0;
	for (i = 0; i < num_nodes; i++)
		if (responses[i] > num_labels)
			num_labels = (int) responses[i];

	plhs[0] = mxCreateNumericMatrix(num_nodes, num_labels + 1,
																	mxUINT16_CLASS, mxREAL);
	signatures = mxGetPr(plhs[0]);

	for (i = 0; i < num_nodes; i++)
		signatures[i] = (unsigned short)(responses[i]);

	count = 0;
	for (i = 0; i < num_nodes; i++) {
		column = i;

		int num_elements_this_column = data_jc[i + 1] - data_jc[i];
		for (j = 0; j < num_elements_this_column; j++, count++) {
			row = data_ir[count];
			signatures[row + ((int)(responses[column])) * num_nodes] += 1;
		}
	}

}
