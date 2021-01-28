#include "pgm.h"

CProxy_main mainProxy;

main::main(CkArgMsg * m)
{ 
  if(m->argc < 3) CkAbort("./pgm slices numChares.");
  int slices = atoi(m->argv[1]); 
  int numChares = atoi(m->argv[2]);
  mainProxy = thishandle; // readonly initialization

  int slicesPerChare=slices/numChares;
  slices=slicesPerChare*numChares; //Make slices an even multiple of numChares

  double xLeft=0.0; //Left end of integration domain
  double xRight=1.0; //Right end of integration domain
  double dx=(xRight-xLeft)/slices;
  for (int i=0; i<numChares; i++) {
    int nSlices=slicesPerChare; // must be a divisor
    double x=xLeft+i*slicesPerChare*dx; //Chare i takes slices starting at [i*slicesPerChare]
    CProxy_integrate::ckNew(x,dx,nSlices); 
  }
  count = numChares;
  integral = 0.0;
}

void main::results(double partialIntegral) 
{ 
  integral += partialIntegral;
  if (0 == --count) {
    CkPrintf("With all results, integral is: %.15f \n", integral);
    CkExit();
  }
}

integrate::integrate(double xStart,double dx, int nSlices)
{ 
  double partialIntegral = 0.0;
  for (int i=0; i<nSlices; i++) {
    double x = xStart+(i+0.5)*dx;
    double f_x=(1.0/(1+x));
    partialIntegral += f_x*dx;
  }

  mainProxy.results(partialIntegral);

  delete this; /*this chare has no more work to do.*/
}

#include "pgm.def.h"
