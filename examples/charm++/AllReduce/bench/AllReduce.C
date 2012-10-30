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
	      CkPrintf("Needs number of array elements\n");
	CkExit();
      }
    units=atoi(m->argv[1]);
    // 4 MB size
    allredSize= 4194304; //atoi(m->argv[2]);
    baseSize = 262144;
    currentSize = baseSize;
    sizeInd = 0;
    numItr = 10;
    sizesNo = 5;
    timeForEach = new double[sizesNo];
    iterNo = 0;
    for(int i=0; i<sizesNo; i++)
	    timeForEach[i] = 0.0;
    arr = CProxy_AllReduce::ckNew(thisProxy, units);

    CkPrintf("AllReduce for %d pes on %d units for %d size\n",
	     CkNumPes(),units,allredSize);

    arr.init();
    startTime = CkWallTimer();
    arr.dowork(baseSize);
  }

#include "AllReduce.def.h"
