#include "../includes/af.h"
#include <math.h>

double SIGMOID(double x) {
	return 1 / (1 + pow(EULER_CONST, -x));
}

double D_SIGMOID(double x) {
	double s = SIGMOID(x);
	return s * (1 - s);
}

double TANH(double x) {
	return tanh(x);
}

double D_TANH(double x) {
	double t = TANH(x);
	return 1 - t * t;
}

double ReLU(x) {
	return x > 0 ? x : 0;
}

double DReLU(x) {
	return x > 0 ? 1 : 0;
}

double LReLU(double x, double alpha) {
	return x > 0 ? x : alpha * x;
}

double D_LReLU(double x, double alpha) {
	return x > 0 ? 1 : alpha;
}

double PReLU(double x, double alpha) {
	return x > 0 ? x : alpha * x * x;
}

double D_ReLU(double x, double alpha) {
	return x > 0 ? 1 : 2 * alpha * x;
}

double ELU(double x, double alpha) {
	return x >= 0 ? x : alpha * (exp(x) - 1);
}

double D_ELU(double x, double alpha) {
	return x >= 0 ? 1 : ELU(x, alpha) + alpha;
}

double SWISH(double x) {
	return x / (1 + fabs(x));
}

double D_SWISH(double x) {
	double s = SWISH(x);
	return 1 / ((1 + fabs(x)) * (1 + fabs(x))) * (1 - s * s);
}

double GELU(double x) {
	return x * 0.5 * (1 + erf(x / sqrt(2)));
}

double D_GELU(double x) {
	double t = sqrt(2 / PIE_CONST) * exp(-x * x / 2);
	return 0.5 * (1 + erf(x / sqrt(2))) + x * t;
}

void SOFTMAX(const int n, const double *x, double *result) {
	double max_elem = x[0];
	for (int i = 1; i < n; i++) {
		if (x[i] > max_elem) {
			max_elem = x[i];
		}
	}
	double sum = 0;
	for (int i = 0; i < n; i++) {
		result[i] = exp(x[i] - max_elem);
		sum += result[i];
	}
	for (int i = 0; i < n; i++) {
		result[i] /= sum;
	}
}