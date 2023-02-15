#include "../includes/ann.h"
#include "../includes/af.h"

int main()
{
    ActivationFunction activation_hidden, activation_output;
    activation_hidden.binary_func = ReLU;
    activation_hidden.num_arg = 2;
    activation_output.unary_func = SIGMOID;
    activation_output.num_arg = 1;

    ANN* ann = ANNNew(2, 2, 1, 2, activation_hidden, activation_output);
	double inputs[] = { .05, .10 };
	double outputs[] = { .01, .99 };
	ANNUpdateWeights(ann, (double[]){ .15, .2, .25, .3, .4, .45, .5, .55 }, (double[]){ .35, .60 });

	ANNForwardPropagate(ann, inputs);
    
	double* total_error = malloc(sizeof(double));
	ANNBackwardPropagate(ann, inputs, outputs, 0.5, "D_SIGMOID");
	printf("%.9f ", *total_error);

	scanf_s("%d");
}