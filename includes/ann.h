#ifndef _ANN_HEADER_
#define _ANN_HEADER_

typedef struct ActivationFunction ActivationFunction;

ActivationFunction * SetActivationFunction(int num_arg, double(*unary_func)(double),
	double(*binary_func)(double, double), void(*softmax_func)(const int, const double*, double*));

typedef struct ANN ANN;

// Initialize Artificial Neural Network
ANN* ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size,
	ActivationFunction *activation_hidden, ActivationFunction *activation_output,
	ActivationFunction *d_activation_hidden, ActivationFunction *d_activation_output);

// The function takes in a filename as a string and two integer pointers to return 
// the number of rows and columns in the CSV data. It returns a two-dimensional array 
// of doubles representing the data in the CSV file. You can then use this data to train your ANN.
double** ANNReadCSV(char* filename, unsigned int* nrows, unsigned int* ncols);

// Generate Random Weights from lower to upper inclusive
void ANNRandomWeights(ANN* ann, double lower, double upper);

// Update weights based on given array and then free the passed array
void ANNUpdateWeights(ANN* ann, double* weights, double* biases);

/*Copy contents of Total Error E value to total_error after Forward Propogating once
E = SUMMATION(1/2 * (target - output)^2) :: Delta = (target - output)
For PReLU and ELU Alpha must be provided [0, 1]
For rest of Activation Functions Alpha can be of any value since not used*/
void ANNForwardPropagate(ANN* ann, double const *inputs);

void ANNTotalError(ANN* ann, double const* outputs, double* result);

// Return weights list after Back Propagating once and also update the existing weights
void ANNBackwardPropagate(ANN* ann, double const *inputs, double const* outputs, double learning_rate);

void ANNTrain(ANN* ann, double const *inputs, double const* outputs, double learning_rate, unsigned int epochs_count);

// Function to split the dataset into train and test sets
void ANNTrainTestSplit(double** dataset, unsigned int num_rows, unsigned int num_cols, double ratio,
	double** train_return, double** test_return);

// Function to extract the label column from the dataset
double* ANNLabelExtract(double** dataset, unsigned int label_col_index, unsigned int num_rows, unsigned int num_cols);

void ANNDeleteFeature(double** dataset, unsigned int feature_index, unsigned int num_rows, unsigned int num_cols);

// Disposes of all allocated memory for ANN
void ANNDispose(ANN* ann);

#endif