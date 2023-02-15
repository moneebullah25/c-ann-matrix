#ifndef _ANN_HEADER_
#define _ANN_HEADER_

typedef union {
	int num_arg;
    double (*unary_func)(double);
    double (*binary_func)(double, double);
    void (*softmax_func)(const int, const double*, double*);
} ActivationFunction;

typedef struct {
    unsigned int input_neurons_size, hidden_neurons_size, hidden_layer_size, output_neurons_size;
    unsigned int total_weights, total_biases, total_neurons;
    double *weights, *outputs, *biases, *deltas;
    ActivationFunction activation_hidden;
    ActivationFunction activation_output;
} ANN;


// Initialize Artificial Neural Network
ANN* ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size,
	ActivationFunction activation_hidden, ActivationFunction activation_output);

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
void ANNBackwardPropagate(ANN* ann, double const* outputs, double learning_rate);

// Disposes of all allocated memory for ANN
void ANNDispose(ANN* ann);

#endif