/** \file rtc.C
 *  Author: Abhinav S Bhatele
 *  E-mail: bhatele@illinois.edu
 *  Date Created: October 12th, 2007
 */

#include <stdio.h>
#include "topo.decl.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;

/*mainchare*/
class Main : public CBase_Main
{
  public:
    Main(CkArgMsg* m) {
#if XT3_TOPOLOGY
      XT3TorusManager xt3tm;
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
      XTTorusManager xttm;
#endif

      mainProxy = thishandle;
      CkPrintf("Testing TopoManager .... \n");
      TopoManager tmgr;
      CkPrintf("Torus Size [%d] [%d] [%d] [%d]\n", tmgr.getDimNX(), tmgr.getDimNY(), tmgr.getDimNZ(), tmgr.getDimNT());

      int x, y, z, t;

      for(int i=0; i<CkNumPes(); i++) {
	tmgr.rankToCoordinates(i, x, y, z, t);
	CkPrintf("---- Processor %d ---> x %d y %d z %d t %d\n", i, x, y, z, t);
#if XT3_TOPOLOGY
	int tmp_t, tmp_x, tmp_y, tmp_z;
	xt3tm.realRankToCoordinates(i, tmp_x, tmp_y, tmp_z, tmp_t);
	CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
	int tmp_t, tmp_x, tmp_y, tmp_z;
	xttm.realRankToCoordinates(i, tmp_x, tmp_y, tmp_z, tmp_t);
	CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#endif
      } // end of for loop

      int size = tmgr.getDimNX() * tmgr.getDimNY() * tmgr.getDimNZ();
      CkPrintf("Torus Contiguity Metric %d : %d [%f] \n", size, CkNumPes()/tmgr.getDimNT(), (float)(CkNumPes())/(tmgr.getDimNT()*size) );
      CkExit();
    };
    
};

#include "topo.def.h"
