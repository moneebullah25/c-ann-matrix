#ifndef _AF_HEADER_
#define _AF_HEADER_

#ifndef NULL 
#define NULL 0
#endif

#ifdef DEBUG
#define ASSERT(x) ((void)(!(x) && assert_handler(#x, __FILE__, __LINE__) && (HALT(), 1)))
#else
#define ASSERT(x) ((void)sizeof(x))
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MULT(x, y) (x * y)

#define EULER_CONST 2.71828182845904523536
#define PIE_CONST 3.14159265358979323846
#define DOUBLE_MAX 1.7976931348623158e+308

#define RAND_MAX 0x7fff

double SIGMOID(double x);
double D_SIGMOID(double x);
double TANH(double x);
double D_TANH(double x);
double ReLU(double x);
double DReLU(double x);
double LReLU(double x, double alpha);
double D_LReLU(double x, double alpha);
double PReLU(double x, double alpha);
double D_ReLU(double x, double alpha);
double ELU(double x, double alpha);
double D_ELU(double x, double alpha);
double SWISH(double x);
double D_SWISH(double x);
double GELU(double x);
double D_GELU(double x);

#endif /* _AF_HEADER */