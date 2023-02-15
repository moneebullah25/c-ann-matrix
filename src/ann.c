#include "../includes/ann.h"
#include "../includes/af.h"
#include "../includes/c_matrix.h"

#include <stdio.h>
#include <stdlib.h>

ANN *ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size,
            unsigned int hidden_layer_size, unsigned int output_neurons_size,
            ActivationFunction activation_hidden, ActivationFunction activation_output)
{

  ANN *ann = (ANN *)malloc(sizeof(ANN));
  ann->input_neurons_size = input_neurons_size;
  ann->hidden_neurons_size = hidden_neurons_size;
  ann->hidden_layer_size = hidden_layer_size;
  ann->output_neurons_size = output_neurons_size;
  ann->total_neurons = input_neurons_size + hidden_neurons_size * hidden_layer_size + output_neurons_size;
  ann->total_weights = (input_neurons_size + 1) * hidden_neurons_size + (hidden_neurons_size + 1) * hidden_layer_size * hidden_neurons_size + (hidden_neurons_size + 1) * output_neurons_size;
  ann->total_biases = hidden_neurons_size * hidden_layer_size + output_neurons_size;
  ann->weights = (double *)malloc(ann->total_weights, sizeof(double));
  ann->biases = (double *)malloc(ann->total_biases, sizeof(double));
  ann->outputs = (double *)malloc(ann->total_neurons, sizeof(double));
  ann->deltas = (double *)malloc(ann->total_neurons, sizeof(double));
  if (activation_hidden.num_arg == 3)
  {
    fprintf(stderr, "Can't use SOFTMAX on Hidden Layers\n");
    return NULL;
  }
  ann->activation_hidden = activation_hidden;
  ann->activation_output = activation_output;

  return ann;
}

void ANNRandomWeights(ANN *ann, double lower, double upper)
{
  ASSERT(ann && (upper > lower));
  srand(time(NULL)); // Initialization, should only be called once.
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

void ANNForwardPropagate(ANN *ann, double const *inputs)
{
  Matrix *input_matrix, *weights_matrix, *bias_matrix, *output_matrix;
  input_matrix = MatrixFrom(ann->input_neurons_size, 1, ann->input_neurons_size, inputs);

  // Propagating input to hidden layer
  weights_matrix = MatrixFrom(ann->hidden_neurons_size, ann->input_neurons_size, ann->total_weights, ann->weights);
  bias_matrix = MatrixFrom(ann->hidden_neurons_size, 1, ann->hidden_neurons_size, ann->biases);
  output_matrix = MatrixAdd(MatrixMultiply(weights_matrix, input_matrix), bias_matrix);

  if (ann->activation_hidden.num_arg == 1)
  {
    for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
    {
      ann->outputs[i] = ann->activation_hidden.unary_func(output_matrix->data[i][0]);
    }
  }
  else
  {
    for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
    {
      ann->outputs[i] = ann->activation_hidden.binary_func(output_matrix->data[i][0], ann->hidden_neurons_size);
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

  if (ann->activation_output.unary_func)
  {
    for (unsigned int i = 0; i < ann->output_neurons_size; i++)
    {
      double s = output_matrix->data[i][0];
      ann->outputs[i + ann->hidden_neurons_size] = (*ann->activation_output.unary_func)(s);
    }
  }
  else
  {
    for (unsigned int i = 0; i < ann->output_neurons_size; i++)
    {
      double s = output_matrix->data[i][0];
      ann->outputs[i + ann->hidden_neurons_size] = (*ann->activation_output.binary_func)(s, ann->activation_output.num_arg);
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

void ANNBackwardPropagate(ANN *ann, double const *outputs, double learning_rate)
{
  // Calculate the error between the actual output and expected output
  double *error = malloc(sizeof(double) * ann->output_neurons_size);
  for (unsigned int i = 0; i < ann->output_neurons_size; i++)
    error[i] = outputs[i] - ann->outputs[i + ann->hidden_neurons_size];

  // Calculate delta for the output layer
  double *delta_output = malloc(sizeof(double) * ann->output_neurons_size);
  for (unsigned int i = 0; i < ann->output_neurons_size; i++)
    delta_output[i] = error[i] * DERIVATIVE_SIGMOID(ann->outputs[i + ann->hidden_neurons_size]);

  // Calculate delta for the hidden layer
  double *delta_hidden = malloc(sizeof(double) * ann->hidden_neurons_size);
  for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
  {
    double sum = 0;
    for (unsigned int j = 0; j < ann->output_neurons_size; j++)
      sum += delta_output[j] * ann->weights[j * ann->hidden_neurons_size + i + ann->total_biases];
    delta_hidden[i] = sum * DERIVATIVE_ReLU(ann->outputs[i]);
  }

  // Update the weights and biases of the hidden and output layers
  for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
  {
    for (unsigned int j = 0; j < ann->input_neurons_size; j++)
    {
      double delta_weight = learning_rate * delta_hidden[i] * ann->input_neurons[j];
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
