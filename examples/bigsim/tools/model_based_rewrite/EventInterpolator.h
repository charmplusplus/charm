
#include <gsl/gsl_multifit.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <stdlib.h> // for rand()

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
typedef multimap< pair<funcIdentifier,vector<double> >, double > timings_type;
typedef vector<long> counterValues;

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
    map<string, int> number_of_coefficients; // records the number of coefficients
    timings_type accurateTimings;

    
		int numX, numY, numZ;
		map<funcIdentifier, vector<double> > model;
		
		std::map<string,double> cycle_accurate_time_sum;


    bool canInterpolateFunc(const funcIdentifier& name);

    map<funcIdentifier,double> min_interpolated_time;
    map<funcIdentifier,double> max_interpolated_time;
		
		map<funcIdentifier,double> sum_newTiming;
    unsigned exact_matches;
    unsigned exact_positive_matches;
    unsigned approx_matches;
    unsigned approx_positive_matches;

    map<counterValues,double> papiTimings; // holds a timing for each set of counter values. This will be used to model the execution time for each sequential execution block just by using the performance counters

public:

    double haveNewTiming(const unsigned pe, const unsigned eventid);

    double predictTime(const unsigned pe, const unsigned eventid);
    double predictTime(const pair<funcIdentifier,vector<double> > &p);
    double predictTime(const funcIdentifier &f, const vector<double> &p);

    bool haveExactTime(const unsigned pe, const unsigned eventid);
    bool haveExactTime(const pair<funcIdentifier,vector<double> > &p);
    bool haveExactTime(const funcIdentifier& f, const vector<double> &p);


	/** Compute average time for the provided event signature(name&parameters) */
    double lookupExactTime(const unsigned pe, const unsigned eventid);
	/** Compute average time for the provided event signature(name&parameters) */
    double lookupExactTime(const pair<funcIdentifier,vector<double> > &p);
	/** Compute average time for the provided event signature(name&parameters) */
    double lookupExactTime(const funcIdentifier& name, const vector<double> &p);

	/** Compute variance of the timings for the provided event signature(name&parameters) */
    double lookupExactVariance(const unsigned pe, const unsigned eventid);
	/** Compute variance of the timings for the provided event signature(name&parameters) */
    double lookupExactVariance(const pair<funcIdentifier,vector<double> > &p);
	/** Compute variance of the timings for the the provided event signature(name&parameters)*/
	double lookupExactVariance(const funcIdentifier& func, const vector<double> &p);

	/** Compute average timing for the provided event signature(name&parameters) */
    double lookupExactMean(const unsigned pe, const unsigned eventid);
	/** Compute average timing for the provided event signature(name&parameters) */
    double lookupExactMean(const pair<funcIdentifier,vector<double> > &p);
	/** Compute average timing for the the provided event signature(name&parameters)*/
    double lookupExactMean(const funcIdentifier& func, const vector<double> &p);

    void analyzeExactVariances();

    /** Get the new timing exactly if possible otherwise predict it */
    double getNewTiming(const unsigned pe, const unsigned eventid);

    double get_chisqr(funcIdentifier funcname){if(work[funcname]!=NULL) return chisqr[funcname]; else return -1.0;}

    int numCoefficients(const string &funcname);
    void recordNumCoefficients(const string &f, int num_params);
    fullParams parseParameters(const string &funcname, istringstream &param_stream, int stepNo);
    counterValues parseCounters(istringstream &line);

    void printMinInterpolatedTimes();

    void printCoefficients();

    void printMatches();

	/**
		Setup an EventInterpolator object from the data in the specified file
		@params sample_rate specifies what fraction of the cycle accurate times
		to keep. The rest will be dropped prior to constructing the best fit
		interpolating model
	*/
    EventInterpolator(const char *table_filename, double sample_rate=1.0);
    void LoadTimingsFile(const char *table_filename);
    void LoadModel();
    void AnalyzeTimings(double sample_rate);
    void AnalyzeTimings_PAPI();
    void LoadParameterFiles();


    ~EventInterpolator();

};


