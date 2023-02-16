#include "../includes/ann.h"
#include "../includes/af.h"
#include <stdlib.h>
#include <stdio.h>

int main()
{
	ActivationFunction activation_hidden, d_activation_hidden, activation_output, d_activation_output;

	activation_hidden.unary_func = SIGMOID;
	activation_hidden.num_arg = 1;
	activation_output.unary_func = SIGMOID;
	activation_output.num_arg = 1;
	d_activation_hidden.unary_func = D_SIGMOID;
	d_activation_hidden.num_arg = 1;
	d_activation_output.unary_func = D_SIGMOID;
	d_activation_output.num_arg = 1;

	ANN* ann = ANNNew(2, 2, 1, 2, activation_hidden, activation_output, d_activation_hidden, d_activation_output);
	double inputs[] = { .05, .10 };
	double outputs[] = { .01, .99 };
	double weights[] = { .15, .2, .25, .3, .4, .45, .5, .55 };
	double biases[] = { .35, .60 };
	ANNUpdateWeights(ann, weights,biases);

	ANNForwardPropagate(ann, inputs);

	double total_error;
	ANNTotalError(ann, outputs, &total_error);

	ANNBackwardPropagate(ann, inputs, outputs, 0.5);
	printf("%.9f ", total_error);

	scanf("%d");
}