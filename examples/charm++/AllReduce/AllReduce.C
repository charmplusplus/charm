#include <stdio.h>
#include "AllReduce.decl.h"
#include <math.h>


/*readonly*/ CProxy_main mainProxy;
/*readonly*/ int units;
/*readonly*/ int allredSize;
#include "AllReduce.h"

/*mainchare*/

main::main(CkArgMsg* m)
  {
    //Process command-line arguments
    //Start the computation

    mainProxy = thishandle;
    if(m->argc<2)
      {
	      CkPrintf("Needs number of array elements and allreduce data size\n");
	CkExit();
      }
    units=atoi(m->argv[1]);
    allredSize=atoi(m->argv[2]);
    
    arr = CProxy_AllReduce::ckNew(thisProxy, units);

    CkPrintf("AllReduce for %d pes on %d units for %d size\n",
	     CkNumPes(),units,allredSize);

    arr.init();
    startTime = CkWallTimer();
    arr.dowork();
  }

#include "AllReduce.def.h"
