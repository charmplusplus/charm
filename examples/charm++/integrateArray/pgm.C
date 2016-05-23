/***************************************************************************
  This "integration" program calculates the integral value of function
  f(x)=1.0/x for the range of [1.0, 2.0].

  The idea is to divide the range into "numSlices" of slices, and create 
  "numChares" of parallel objects, or Chares, each taking care of 
  numSlices/numChares of slices. 
  After each chare finishes its portion of computation, a global reduction 
  is performed to sum up the local values and get the final result.
***************************************************************************/

#include "pgm.h"

CProxy_main mainProxy;

// program starts from here
main::main(CkArgMsg *m)
{ 
  if(m->argc < 3) CkAbort("./pgm slices numChares.");
  int numSlices = atoi(m->argv[1]); 
  int numChares = atoi(m->argv[2]);
  mainProxy = thishandle; 	// readonly initialization

  // calculate the number of slices for each chare
  int slicesPerChare=numSlices/numChares;
  numSlices=slicesPerChare*numChares;   //Make slices an even multiple of numChares

  double xLeft=1.0;       		//Left end of integration domain
  double xRight=2.0;      		//Right end of integration domain
  double dx=(xRight-xLeft)/numSlices;	//range between slides

  // create array of chares to do the actual computation
  CProxy_integrate integrateArray = CProxy_integrate::ckNew();
  for (int i=0; i<numChares; i++) {
    int nSlices=slicesPerChare;     // must be a divisor
    double x=xLeft+i*slicesPerChare*dx; // Chare i takes slices starting at [i*slicesPerChare]
    // create chare i that takes slices starting from x for nSlices
    integrateArray[i].insert(x, dx, nSlices); 
  }
  integrateArray.doneInserting();	// done with creation

  // set up reduction handler for sum up the integrals across all chares.
  CkCallback *cb = new CkCallback(CkIndex_main::results(NULL), mainProxy);
  integrateArray.ckSetReductionClient(cb);
}

// reduction handler, it prints out the result and exits.
void main::results(CkReductionMsg *msg) 
{ 
  double integral = *(double *)msg->getData();
  CkPrintf("With all results, integral is: %.15f \n", integral);
  CkExit();
}

integrate::integrate(double xStart,double dx, int nSlices)
{ 
  // calculate the partialIntegral for this chare for
  // range of [xStart, xStart+(nSlices-1)*dx]
  double partialIntegral = 0.0;
  for (int i=0; i<nSlices; i++) {
    double x = xStart+(i+0.5)*dx;
    double f_x=(1.0/x);
    partialIntegral += f_x*dx;
  }

  // reduction to sum the partialIntegral across all chares
  contribute(sizeof(double), &partialIntegral, CkReduction::sum_double);
}

#include "pgm.def.h"
