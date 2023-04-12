#include "../includes/ann.h"
#include "../includes/af.h"
#include <stdlib.h>
#include <stdio.h>

int main()
{
	//ActivationFunction *activation_hidden, *d_activation_hidden, *activation_output, *d_activation_output;

	//activation_hidden = SetActivationFunction(1, SIGMOID, NULL, NULL);
	//d_activation_hidden = SetActivationFunction(1, D_SIGMOID, NULL, NULL);
	//activation_output = SetActivationFunction(1, SIGMOID, NULL, NULL);
	//d_activation_output = SetActivationFunction(1, D_SIGMOID, NULL, NULL);

	//ANN* ann = ANNNew(2, 2, 1, 2, activation_hidden, activation_output, d_activation_hidden, d_activation_output);

	//double inputs[] = { .05, .10 };
	//double outputs[] = { .01, .99 };
	//double weights[] = { .15, .2, .25, .3, .4, .45, .5, .55 };
	//double biases[] = { .35, .60 };

	//ANNUpdateWeights(ann, weights,biases);

	//for (unsigned int i = 0; i < 1000000; i++) 
	//{
	//	ANNForwardPropagate(ann, inputs);

	//	double total_error;
	//	ANNTotalError(ann, outputs, &total_error);

	//	ANNBackwardPropagate(ann, inputs, outputs, 0.5);
	//	printf("%.9f\n", total_error);
	//}

	/* Boston Housing Dataset */

	ActivationFunction *activation_hidden, *d_activation_hidden, *activation_output, *d_activation_output;

	activation_hidden = SetActivationFunction(1, ReLU, NULL, NULL);
	d_activation_hidden = SetActivationFunction(1, D_ReLU, NULL, NULL);
	activation_output = SetActivationFunction(1, ReLU, NULL, NULL);
	d_activation_output = SetActivationFunction(1, D_ReLU, NULL, NULL);

	ANN* ann = ANNNew(14, 128, 1, 1, activation_hidden, activation_output, d_activation_hidden, d_activation_output);
	ANNRandomWeights(ann, 0., 1.);
	unsigned int nrows, ncols;

	double** dataset = ANNReadCSV("Z:/c-ann-matrix/c-ann-matrix/tests/BostonHousing.txt", &nrows, &ncols);
	double** train_feature, **test_feature, *train_label, *test_label;

	ANNTrainTestSplit(dataset, nrows, ncols, 0.7, train_feature, test_feature);
	train_label = ANNLabelExtract(train_feature, 13, nrows, ncols);
	test_label = ANNLabelExtract(test_feature, 13, nrows, ncols);

	for (unsigned int i = 0; i < nrows; i++)
	{
		ANNTrain(ann, train_feature[i], &train_label[i], 0.01, 250);
	}
	
	int res;
	scanf("%d", &res);
}