/*
(Copied out of) Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 11/25/2002
*/
#ifndef __UIUC_OSL_STATISTICS_H
#define __UIUC_OSL_STATISTICS_H

#include "math.h" //for sqrt
#include "stdio.h" //for fprintf
#include "pup.h" //for pup

/**
 * CkSampleT represents a statistical "sample" of some data values.
 * It maintains information it can use to compute means 
 * and variances, max and min. 
 *
 *  The REAL template parameter is the datatype of the values.
 *  The RET template parameter is the datatype of the various 
 *     accumulators used, and the datatype for the derived parameters.
 *
 * The CkSample typedef is just CkSampleT<double,double>
 * Other sensible specializations might include CkSampleT<double, long double>
 * for higher precision accumulation, or CkSampleT<int, double> for computing
 * continuous statistics of a discrete distribution.
 */
template<class real,class ret>
class CkSampleT {
	real lo,hi; //smallest and largest values
	ret sum; //sum of values
	ret sq; //sum of squares of values
	int n; //number of values
public:
	CkSampleT(void) {
		lo=(real)1.0e20; hi=(real)-1.0e20;
		sum=sq=(ret)0;
		n=0;
	}
	/**
	 * Add this value to the sample set.
	 * This function updates the max, min, and mean and variance
	 * for this sample.
	 */
	inline void add(real r) {
		if (r<lo) lo=r;
		if (r>hi) hi=r;
		sum+=(ret)r;
		sq+=(ret)(r*r);
		n++;
	}
	/// Shorthand for add function.
	inline void operator+=(real r) { add(r); }

	/** Add all these samples to this set */
	inline void add(const CkSampleT<real,ret> &errs) {
		if (errs.lo<lo) lo=errs.lo;
		if (errs.hi>hi) hi=errs.hi;
		sum+=errs.sum;
		sq+=errs.sq;
		n+=errs.n;
	}
	
	/**
	 * Return the mean value of this sample--the "average" value, in (value) units.
	 * Computed as the sum of all values divided by the number of values.
	 */
	ret getMean(void) const {
		return sum/n;
	}
	/**
	 * Return the variance of this sample, in (value^2) units.
	 * Computed as the sum of the squares of the diffences between
	 * each value and the mean, divided by the number of values minus 1.
	 */
	ret getVariance(void) const {
		return (sq-sum*sum/n)/(n-1);
	}
	/**
	 * Return the standard deviation of this sample, in (value) units.
	 * Computed as the square root of the variance.
	 */
	ret getStddev(void) const {
		return (ret)sqrt(getVariance());
	}
	/**
	 * Return the smallest value encountered.
	 */
	real getMin(void) const {return lo;}
	/**
	 * Return the largest value encountered.
	 */
	real getMax(void) const {return hi;}
	/**
	 * Return the number of values in this sample.
	 */
	int getCount(void) const {return n;}

	/// Return the RMS value of this variable (square root of sum of squares over number)
	ret getRMS(void) const {return sqrt(sq/n);}
	
	/**
	 * Print a textual description of this sample to this FILE.
	 * For example, a 1,000,000-value sample of a uniform distribution on [0,1] might give:
	 *   ave= 0.500367  stddev= 0.288663  min= 1.27012e-06  max= 0.999999  n= 1000000
	 */
	void print(FILE *dest) {
		fprintf(dest,"ave= %g  stddev= %g  min= %g  max= %g  n= %d\n",
			(double)getMean(), (double)getStddev(), (double)getMin(), (double)getMax(), (int)getCount());
	}

	/**
         * Print a terse textual description of this sample to this FILE.
         */
        void printMinAveMax(FILE *dest) {
		fprintf(dest,"min= %g  ave= %g  max= %g \n",
			(double)getMin(), (double)getMean(), (double)getMax());
	}

	/**
	 * Print a textual description of this sample to stdout.
	 */
	void print(void) {print(stdout);}
	
	/// Pup routine, for migration and parameter marshalling.
	void pup(PUP::er &p) {
		p|lo; p|hi;
		p|sum; p|sq;
		p|n;
	}
	friend inline void operator|(PUP::er &p,CkSampleT<real,ret> &s) {s.pup(p);}
};
typedef CkSampleT<double,double> CkSample;

#endif
