#include "charm++.h"

/* This class is passed as a parameter, so it has to be 
  declared *before* the .decl header! */

/** 
 * Abstract 1d real function class.
 * You inherit from this class to use the numerical integral
 * class below.
 */
class function1d : public PUP::able {
public:
	virtual double evaluate(double x) =0;
	
	PUPable_abstract(function1d);
};

#include "pgm.decl.h"

class main : public Chare
{
private:
  int count; //Number of partial results that have not arrived yet
  double integral; //Partial value of function integral
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void results(double partialIntegral);
};

class integrate : public Chare 
{
private:

public:
  integrate(CkMigrateMessage *m) {}
  integrate(double xStart,double dx, int nSlices, function1d &f);
};
