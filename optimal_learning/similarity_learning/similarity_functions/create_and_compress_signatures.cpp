#include "mex.h"
//#include "matrix.h"

#include <tr1/functional>
#include <tr1/unordered_map>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>

typedef boost::mt19937 base_generator_type;

using boost::uniform_real;
using boost::variate_generator;

using std::tr1::hash;
using std::tr1::unordered_map;

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	mwIndex *data_ir, *data_jc;
	double *responses, *new_responses;
	size_t num_nodes, num_non_zeros;

	int i, j, row, column, count, num_labels, num_signatures, num_elements_this_column;
	double *random_offsets, response;
	float *signatures;

	unordered_map<float, unsigned short, hash<float> > signature_hash;

	base_generator_type generator(0);

	uniform_real<float> uniform_distribution(0,1);
	boost::variate_generator<base_generator_type&, boost::uniform_real<float> >
		rand(generator, uniform_distribution);

	data_ir = mxGetIr(prhs[0]);
	data_jc = mxGetJc(prhs[0]);
	num_nodes = mxGetM(prhs[0]);

	responses = mxGetPr(prhs[1]);

	plhs[0] = mxCreateDoubleMatrix(num_nodes, 1, mxREAL);
	new_responses = mxGetPr(plhs[0]);

	num_labels = 0;
	for (i = 0; i < num_nodes; i++)
		if (responses[i] > num_labels)
			num_labels = (int) responses[i];

	random_offsets = new double[num_labels];
	signatures = new float[num_nodes];

	for (i = 0; i < num_labels; i++)
		random_offsets[i] = rand();

	for (i = 0; i < num_nodes; i++)
		signatures[i] = (float)(responses[i]);

	count = 0;
	for (column = 0; column < num_nodes; column++) {

		num_elements_this_column = data_jc[column + 1] - data_jc[column];
		for (i = 0; i < num_elements_this_column; i++, count++) {
			row = data_ir[count];
			signatures[row] += random_offsets[(int)(responses[column] - 1)];
		}
	}

	num_signatures = 0;
	for (i = 0; i < num_nodes; i++) {

		response = signature_hash[signatures[i]];
		if (response == 0) {
			num_signatures++;
			response = num_signatures;
			signature_hash[signatures[i]] = response;
		}

		new_responses[i] = response;
	}

}
