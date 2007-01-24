
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

typedef pair<string,int> funcIdentifier;
typedef pair<int,vector<double> > fullParams;

class EventInterpolator{
private:
  ofstream log1;

    // For each interpolatable function we record things in these maps:

    map<funcIdentifier, unsigned long> sample_count;
    map<funcIdentifier, gsl_multifit_linear_workspace *> work;
    map<funcIdentifier, gsl_vector *> c; // coefficients which are produced by least square fit
    map<funcIdentifier, gsl_matrix *> cov;
    map<funcIdentifier, double> chisqr;

    map<funcIdentifier, gsl_matrix *> X;  // Each row of matrix is a set of parameters  [1, a, a^2, b, b^2, a*b] for each input parameter set
    map<funcIdentifier, unsigned> Xcount;  // Number of entries in X so far
    map<funcIdentifier, gsl_vector *>y;  // vector of cycle accurate times for each input parameter set

    map<pair<unsigned,unsigned>,pair<funcIdentifier,vector<double> > > eventparams;
    map< pair<funcIdentifier,vector<double> >, double > accurateTimings;


    bool canInterpolateFunc(const funcIdentifier& name);

    map<funcIdentifier,double> min_interpolated_time;
    map<funcIdentifier,double> max_interpolated_time;

    unsigned exact_matches;
    unsigned exact_positive_matches;
    unsigned approx_matches;
    unsigned approx_positive_matches;

public:

    double haveNewTiming(const unsigned pe, const unsigned eventid);

    double predictTime(const unsigned pe, const unsigned eventid);
    double predictTime(const pair<funcIdentifier,vector<double> > &p);
    double predictTime(const funcIdentifier &name, const vector<double> &params);

    bool haveExactTime(const unsigned pe, const unsigned eventid);
    bool haveExactTime(const pair<funcIdentifier,vector<double> > &p);
    bool haveExactTime(const funcIdentifier& name, const vector<double> &p);

    double lookupExactTime(const unsigned pe, const unsigned eventid);
    double lookupExactTime(const pair<funcIdentifier,vector<double> > &p);
    double lookupExactTime(const funcIdentifier& name, const vector<double> &p);

	/** Get the new timing exactly if possible otherwise predict it */
	double getNewTiming(const unsigned pe, const unsigned eventid);

    double get_chisqr(funcIdentifier funcname){if(work[funcname]!=NULL) return chisqr[funcname]; else return -1.0;}

    int numCoefficients(const string &funcname);
    fullParams parseParameters(const string &funcname, istringstream &param_stream, double &time, const bool log);
    fullParams parseParameters(const string &funcname, istringstream &param_stream, const bool log);

    void printMinInterpolatedTimes();

    void printCoefficients();

    void printMatches();


    EventInterpolator(char *table_filename);
    ~EventInterpolator();

};


