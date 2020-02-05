#include <stdio.h>
#include "AllReduce.decl.h"
#include <math.h>


#include "AllReduce.h"

/*mainchare*/

main::main(CkArgMsg* m)
  {
    //Process command-line arguments
    //Start the computation

    if(m->argc<2)
      {
	      CkPrintf("Needs number of array elements and allreduce data size\n");
	CkExit();
      }
    units=atoi(m->argv[1]);
    int allredSize=atoi(m->argv[2]);

    delete m;

    arr = CProxy_AllReduce::ckNew(thisProxy, allredSize, units, units);

    CkPrintf("AllReduce for %d pes on %d units for %d size\n",
	     CkNumPes(),units,allredSize);

    startTime = CkWallTimer();
    arr.dowork();
  }

#include "AllReduce.def.h"
