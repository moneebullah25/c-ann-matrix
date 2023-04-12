#include "../includes/ann.h"
#include "../includes/af.h"
#include "../includes/c_matrix.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double hp = 0.5;

struct ActivationFunction {
	int num_arg;
	double(*unary_func)(double);
	double(*binary_func)(double, double);
	void(*softmax_func)(const int, const double*, double*);
};

ActivationFunction* SetActivationFunction(int num_arg, double(*unary_func)(double),
	double(*binary_func)(double, double), void(*softmax_func)(const int, const double*, double*))
{
	if (num_arg < 0) return NULL;
	void *af = malloc(sizeof(ActivationFunction));
	((ActivationFunction*)af)->num_arg = num_arg;
	((ActivationFunction*)af)->unary_func = unary_func;
	((ActivationFunction*)af)->binary_func = binary_func;
	((ActivationFunction*)af)->softmax_func = softmax_func;
	return ((ActivationFunction*)af);
}

struct ANN {
	unsigned int input_neurons_size, hidden_neurons_size, hidden_layer_size, output_neurons_size;
	unsigned int total_weights, total_biases, total_neurons;
	double *weights, *outputs, *biases, *deltas;
	ActivationFunction *activation_hidden, *d_activation_hidden;
	ActivationFunction *activation_output, *d_activation_output;
};

ANN* ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size,
	ActivationFunction *activation_hidden, ActivationFunction *activation_output,
	ActivationFunction *d_activation_hidden, ActivationFunction *d_activation_output)
{
	ANN *ann = (ANN *)malloc(sizeof(ANN));

	ann->input_neurons_size = input_neurons_size;
	ann->hidden_neurons_size = hidden_neurons_size;
	ann->hidden_layer_size = hidden_layer_size;
	ann->output_neurons_size = output_neurons_size;

	ann->total_neurons = input_neurons_size + hidden_neurons_size * hidden_layer_size + output_neurons_size;
	ann->total_weights = (input_neurons_size + 1) * hidden_neurons_size + (hidden_neurons_size + 1) * hidden_layer_size * hidden_neurons_size + (hidden_neurons_size + 1) * output_neurons_size;
	ann->total_biases = hidden_neurons_size * hidden_layer_size + output_neurons_size;

	ann->weights = (double *)malloc(ann->total_weights * sizeof(double));
	ann->biases = (double *)malloc(ann->total_biases * sizeof(double));
	ann->outputs = (double *)malloc(ann->total_neurons * sizeof(double));
	ann->deltas = (double *)malloc(ann->total_neurons * sizeof(double));

	ann->activation_hidden = activation_hidden;
	ann->activation_output = activation_output;
	ann->d_activation_hidden = d_activation_hidden;
	ann->d_activation_output = d_activation_output;

	return ann;
}

double** ANNReadCSV(char* filename, unsigned int* nrows, unsigned int* ncols) {
	FILE* fp;
	char line[1024];
	char* token;
	int row = 0;
	int col = 0;
	double** data;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		perror("Error opening file");
		return NULL;
	}

	// Count number of rows and columns
	while (fgets(line, 1024, fp)) {
		col = 0;
		token = strtok(line, ",");
		while (token != NULL) {
			token = strtok(NULL, ",");
			col++;
		}
		row++;
	}

	// Allocate memory for data array
	data = (double**)malloc(row * sizeof(double*));
	for (int i = 0; i < row; i++) {
		data[i] = (double*)malloc(col * sizeof(double));
	}

	// Reset file pointer and read data
	fseek(fp, 0, SEEK_SET);
	row = 0;
	while (fgets(line, 1024, fp)) {
		col = 0;
		token = strtok(line, ",");
		while (token != NULL) {
			data[row][col] = atof(token);
			token = strtok(NULL, ",");
			col++;
		}
		row++;
	}

	*nrows = row;
	*ncols = col;

	fclose(fp);
	return data;
}

void ANNRandomWeights(ANN *ann, double lower, double upper)
{
	ASSERT(ann && (upper > lower));
	srand((unsigned int)time(NULL)); // Initialization, should only be called once.
	for (unsigned int i = 0; i < ann->total_weights; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->weights[i] = randD;
	}
	for (unsigned int i = 0; i < ann->total_biases; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->biases[i] = randD;
	}
}

void ANNUpdateWeights(ANN *ann, double *weights, double *biases)
{
	ASSERT(ann && weights && biases);
	for (unsigned int i = 0; i < ann->total_weights; i++)
		ann->weights[i] = weights[i];
	for (unsigned int i = 0; i < ann->total_biases; i++)
		ann->biases[i] = biases[i];
}

void ANNForwardPropagate(ANN *ann, double *inputs)
{
	Matrix *input_matrix, *weights_matrix, *bias_matrix, *output_matrix;
	input_matrix = MatrixFrom(ann->input_neurons_size, 1, ann->input_neurons_size, inputs);

	// Propagating input to hidden layer
	weights_matrix = MatrixFrom(ann->hidden_neurons_size, ann->input_neurons_size, ann->total_weights, ann->weights);
	bias_matrix = MatrixFrom(ann->hidden_neurons_size, 1, ann->hidden_neurons_size, ann->biases);
	output_matrix = MatrixAdd(MatrixMultiply(weights_matrix, input_matrix), bias_matrix);

	if (ann->activation_hidden->num_arg == 1)
	{
		for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
		{
			ann->outputs[i] = ann->activation_hidden->unary_func(output_matrix->data[i][0]);
		}
	}
	else
	{
		for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
		{
			ann->outputs[i] = ann->activation_hidden->binary_func(output_matrix->data[i][0], hp);
		}
	}

	// Propagating hidden to output layer
	MatrixFree(input_matrix);
	MatrixFree(weights_matrix);
	MatrixFree(bias_matrix);
	input_matrix = MatrixFrom(ann->hidden_neurons_size, 1, ann->hidden_neurons_size, ann->outputs);

	weights_matrix = MatrixFrom(ann->output_neurons_size, ann->hidden_neurons_size, ann->total_weights + ann->total_biases, ann->weights + ann->total_biases);
	bias_matrix = MatrixFrom(ann->output_neurons_size, 1, ann->output_neurons_size, ann->biases + ann->total_biases);
	output_matrix = MatrixAdd(MatrixMultiply(weights_matrix, input_matrix), bias_matrix);

	if (ann->activation_output->unary_func)
	{
		for (unsigned int i = 0; i < ann->output_neurons_size; i++)
		{
			double s = output_matrix->data[i][0];
			ann->outputs[i + ann->hidden_neurons_size] = (*ann->activation_output->unary_func)(s);
		}
	}
	else if (ann->activation_output->binary_func)
	{
		for (unsigned int i = 0; i < ann->output_neurons_size; i++)
		{
			double s = output_matrix->data[i][0];
			ann->outputs[i + ann->hidden_neurons_size] = (*ann->activation_output->binary_func)(s, hp);
		}
	}

	MatrixFree(input_matrix);
	MatrixFree(weights_matrix);
	MatrixFree(bias_matrix);
	MatrixFree(output_matrix);
}

void ANNTotalError(ANN *ann, double const *outputs, double *result)
{
	ASSERT(ann && outputs && result);

	*result = 0.0;
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
	{
		*result += 0.5 * (ann->outputs[i + ann->hidden_neurons_size] - outputs[i]) *
			(ann->outputs[i + ann->hidden_neurons_size] - outputs[i]);
	}
}

void ANNBackwardPropagate(ANN *ann, double const *inputs, double const *outputs, double learning_rate)
{
	// Calculate the error between the actual output and expected output
	double *error = calloc(sizeof(double), ann->output_neurons_size);
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
		error[i] = outputs[i] - ann->outputs[i + ann->hidden_neurons_size];


	// Calculate delta for the output layer
	double *delta_output = calloc(sizeof(double), ann->output_neurons_size);
	if (ann->d_activation_output->unary_func)
	{
		for (unsigned int i = 0; i < ann->output_neurons_size; i++)
			delta_output[i] = error[i] * ann->d_activation_output->unary_func(ann->outputs[i + ann->hidden_neurons_size]);
	}
	else if (ann->d_activation_output->binary_func)
	{
		for (unsigned int i = 0; i < ann->output_neurons_size; i++)
			delta_output[i] = error[i] * ann->d_activation_output->binary_func(ann->outputs[i + ann->hidden_neurons_size], hp);
	}

	// Calculate delta for the hidden layer
	double *delta_hidden = calloc(sizeof(double), ann->hidden_neurons_size);
	for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
	{
		double sum = 0;
		for (unsigned int j = 0; j < ann->output_neurons_size; j++)
			sum += delta_output[j] * ann->weights[j * ann->hidden_neurons_size + i + ann->total_biases];
		
		if (ann->d_activation_hidden->unary_func)
		{
			delta_hidden[i] = sum * ann->d_activation_hidden->unary_func(ann->outputs[i]);
		}
		else if (ann->d_activation_hidden->binary_func)
		{
			delta_hidden[i] = sum * ann->d_activation_hidden->binary_func(ann->outputs[i], hp);
		}
	}

	// Update the weights and biases of the hidden and output layers
	for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
	{
		for (unsigned int j = 0; j < ann->input_neurons_size; j++)
		{
			double delta_weight = learning_rate * delta_hidden[i] * inputs[j];
			ann->weights[j * ann->hidden_neurons_size + i] += delta_weight;
		}
		double delta_bias = learning_rate * delta_hidden[i];
		ann->biases[i] += delta_bias;
	}

	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
	{
		for (unsigned int j = 0; j < ann->hidden_neurons_size; j++)
		{
			double delta_weight = learning_rate * delta_output[i] * ann->outputs[j];
			ann->weights[j * ann->output_neurons_size + i + ann->total_biases] += delta_weight;
		}
		double delta_bias = learning_rate * delta_output[i];
		ann->biases[i + ann->total_biases] += delta_bias;
	}

	free(error);
	free(delta_output);
	free(delta_hidden);
}


void ANNTrain(ANN* ann, double const *inputs, double const* outputs, double learning_rate, unsigned int epochs_count)
{
	// Train the neural network for the given number of epochs
	for (unsigned int epoch = 1; epoch <= epochs_count; epoch++) {
		
		double total_error = 0.0;

		ANNForwardPropagate(ann, inputs);
		ANNTotalError(ann, outputs, &total_error);
		ANNBackwardPropagate(ann, inputs, outputs, learning_rate);
		
		printf("Epoch %u/%u\n", epoch, epochs_count);
		printf("1/1 [===============================] - loss: %lf\n", total_error);
	}
}

// Function to split the dataset into train and test sets
void ANNTrainTestSplit(double** dataset, unsigned int num_rows, unsigned int num_cols, double ratio, 
	double** train_return, double** test_return)
{
	// Calculate the number of rows in the training set
	unsigned int num_train_rows = (unsigned int)(num_rows * ratio);

	// Allocate memory for the training and test sets
	*train_return = (double*)malloc(num_train_rows * num_cols * sizeof(double));
	*test_return = (double*)malloc((num_rows - num_train_rows) * num_cols * sizeof(double));

	// Copy the training samples to the training set
	for (unsigned int i = 0; i < num_train_rows; i++) {
		for (unsigned int j = 0; j < num_cols; j++) {
			(*train_return)[i * num_cols + j] = dataset[i][j];
		}
	}

	// Copy the testing samples to the testing set
	for (unsigned int i = num_train_rows; i < num_rows; i++) {
		for (unsigned int j = 0; j < num_cols; j++) {
			(*test_return)[(i - num_train_rows) * num_cols + j] = dataset[i][j];
		}
	}
}

// Function to extract the label column from the dataset
double* ANNLabelExtract(double** dataset, unsigned int label_col_index, unsigned int num_rows, unsigned int num_cols) 
{
	// Allocate memory for the label column
	double* label_column = (double*)malloc(num_rows * sizeof(double));

	// Extract the label column and remove it from the dataset
	for (unsigned int i = 0; i < num_rows; i++) {
		label_column[i] = dataset[i][label_col_index];

		for (unsigned int j = label_col_index; j < num_cols - 1; j++) {
			dataset[i][j] = dataset[i][j + 1];
		}
	}

	return label_column;
}

void ANNDeleteFeature(double** dataset, unsigned int feature_index, unsigned int num_rows, unsigned int num_cols) 
{
	// Free memory for the deleted feature column
	for (unsigned int i = 0; i < num_rows; i++) {
		for (unsigned int j = feature_index; j < num_cols - 1; j++) {
			dataset[i][j] = dataset[i][j + 1];
		}
		dataset[i] = realloc(dataset[i], (num_cols - 1) * sizeof(double));
	}
}

void ANNDispose(ANN *ann)
{
	if (ann == NULL)
	{
		return;
	}

	free(ann->weights);
	free(ann->biases);
	free(ann->outputs);
	free(ann);
}
