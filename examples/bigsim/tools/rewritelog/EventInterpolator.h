
#include <gsl/gsl_multifit.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <stdexcept>

#include <map>
#include <string>
#include <utility> // for std::pair
#include <vector>


using namespace std;

/**
 @class A class for wrapping the least-square fitting portions of gsl.

  Construct one of these with a filename and it will read in the file
  containing a table of parameters and associated timings(as produced by somthing like a cycle accurate simulator).

  Pass in a desired set of parameters to predictTime() to get a predicted time by evaluating the interpolating function at the given parameter vector.

*/
class EventInterpolator{
private:

    // For each interpolatable function we record things in these maps:

    map<string, unsigned long> sample_count;
    map<string, gsl_multifit_linear_workspace *> work;
    map<string, gsl_vector *> c; // coefficients which are produced by least square fit
    map<string, gsl_matrix *> cov;
    map<string, double> chisqr;

    map<string, gsl_matrix *> X;  // Each row of matrix is a set of parameters  [1, a, a^2, b, b^2, a*b] for each input parameter set
    map<string, unsigned> Xcount;  // Number of entries in X so far
    map<string, gsl_vector *>y;  // vector of cycle accurate times for each input parameter set

    map<pair<unsigned,unsigned>,pair<string,vector<double> > > eventparams;

    bool canInterpolateName(const string& name);


public:
    double haveNewTiming(const unsigned pe, const unsigned eventid);
    double predictTime(const unsigned pe, const unsigned eventid);
    double predictTime(const pair<string,vector<double> > &p);
    double predictTime(const string &name, const vector<double> &params);


    double get_chisqr(string funcname){if(work[funcname]!=NULL) return chisqr[funcname]; else return -1.0;}

    int EventInterpolator::numCoefficients(string funcname);
    int EventInterpolator::numParameters(string funcname);


    EventInterpolator(char *table_filename);
    ~EventInterpolator();

};


