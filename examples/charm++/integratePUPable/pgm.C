#include "pgm.h"

CProxy_main mainProxy;

/* 
  This is the 1D function we want to integrate.
  Because this class is sent over the network as
  a pointer to its abstract superclass, we need the 
  "PUPable" declarations.
*/
class myFunction : public function1d {
	double height;
public:
	myFunction(double height_=1.0) :height(height_) {}

	virtual double evaluate(double x) {
		return height/(1+x);
	}
	
	//PUP::able support:
	PUPable_decl(myFunction);
	myFunction(CkMigrateMessage *m=0) {}	
	virtual void pup(PUP::er &p) {
		p|height;
	}
};

main::main(CkArgMsg * m)
{ 
  if(m->argc < 3) CkAbort("./pgm slices numChares.");
  int slices = atoi(m->argv[1]); 
  int numChares = atoi(m->argv[2]);
  mainProxy = thishandle; // readonly initialization

  int slicesPerChare=slices/numChares;
  slices=slicesPerChare*numChares; //Make slices an even multiple of numChares

  myFunction *f=new myFunction(1.0);
  double xLeft=0.0; //Left end of integration domain
  double xRight=1.0; //Right end of integration domain
  double dx=(xRight-xLeft)/slices;
  for (int i=0; i<numChares; i++) {
    int nSlices=slicesPerChare; // must be a divisor
    double x=xLeft+i*slicesPerChare*dx; //Chare i takes slices starting at [i*slicesPerChare]
    CProxy_integrate::ckNew(x,dx,nSlices,*f); 
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

integrate::integrate(double xStart,double dx, int nSlices,function1d &f)
{ 
  double partialIntegral = 0.0;
  for (int i=0; i<nSlices; i++) {
    double x = xStart+(i+0.5)*dx;
    partialIntegral += f.evaluate(x)*dx;
  }

  mainProxy.results(partialIntegral);

  delete this; /*this chare has no more work to do.*/
}

#include "pgm.def.h"
