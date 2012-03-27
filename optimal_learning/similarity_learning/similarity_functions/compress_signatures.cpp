#include "mex.h"
#include "matrix.h"

#include <iostream>
#include <string>
#include <sparsehash/dense_hash_map>
#include <tr1/functional>

using google::dense_hash_map;
using std::cout;
using std::endl;
using std::tr1::hash;
using std::string;

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	int i, j, num_signatures;
	size_t num_rows, num_columns;
	unsigned short *signatures, entry;
	double *new_responses;
	
	dense_hash_map<string, double, hash<string> > signature_hash;
	signature_hash.set_empty_key(string());

	num_rows    = mxGetM(prhs[0]);
	num_columns = mxGetN(prhs[0]);

	signatures = (unsigned short *)(mxGetData(prhs[0]));

	plhs[0] = mxCreateDoubleMatrix(num_rows, 1, mxREAL);
	new_responses = mxGetPr(plhs[0]);

	num_signatures = 0;
	for (i = 0; i < num_rows; i++) {
		string this_signature = "";
		
		for (j = 0; j < num_columns; j++) {
			entry = signatures[i + num_rows * j];
			if (entry > 0) {
			  this_signature += j;
				this_signature += " ";
				this_signature += entry;
				this_signature += " ";
			}
		}
	
		if (signature_hash[this_signature] == 0) {
			num_signatures++;
			signature_hash[this_signature] = num_signatures;
		}

		new_responses[i] = signature_hash[this_signature];
	}
}
