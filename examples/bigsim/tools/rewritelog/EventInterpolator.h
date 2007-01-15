
#include <gsl/gsl_multifit.h>

/**
 @class A class for wrapping the least-square fitting portions of gsl.

  Construct one of these with a filename and it will read in the file
  containing a table of parameters and associated timings(as produced by somthing like a cycle accurate simulator).
  
  Pass in a desired set of parameters to predictTime() to get a predicted time by evaluating the interpolating function at the given parameter vector.

*/
class EventInterpolator{
private: 
  gsl_vector *c;  // coefficients which are produced by least square fit
  gsl_matrix *cov;
  gsl_multifit_linear_workspace * work;
  int n;  // number of sample input evaluations
  int np; // number of input parameters
  int cs; // number of coefficients
  double chisqr;

public:

  double predictTime(double *params);
  double get_chisqr(){if(work!=NULL) return chisqr; else return -1.0;}

  EventInterpolator(char *table_filename);
  ~EventInterpolator();

};


