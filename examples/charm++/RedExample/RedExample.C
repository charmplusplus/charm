#include <stdio.h>
#include "RedExample.decl.h"
#include <math.h>


/*readonly*/ CProxy_main mainProxy;
/*readonly*/ int units;
/*readonly*/ double dOne;
/*readonly*/ double dTwo;
#include "RedExample.h"

/*mainchare*/

main::main(CkArgMsg* m)
  {
    //Process command-line arguments
    //Start the computation

    mainProxy = thishandle;
    if(m->argc<2)
      {
	CkExit();
      }
    units=atoi(m->argv[1]);
    dOne=atof(m->argv[2]);
    dTwo=atof(m->argv[3]);
    
    arr = CProxy_RedExample::ckNew(units);

    CkPrintf("RedExample for %d pes on %d units for %f and %f\n",
	     CkNumPes(),units,dOne, dTwo);

    CkCallback *cb = new CkCallback(CkIndex_main::reportIn(NULL),  mainProxy);
    arr.ckSetReductionClient(cb);
    arr.dowork();
  }



#include "RedExample.def.h"
