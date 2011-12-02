/** \file rtc.C
 *  Author: Abhinav S Bhatele
 *  E-mail: bhatele@illinois.edu
 *  Date Created: October 12th, 2007
 */

#include <stdio.h>
#include "topo.decl.h"
#include "TopoManager.h"

#if CMK_BLUEGENEL
#include <bglpersonality.h>
#elif CMK_BLUEGENEP
#include <dcmf.h>
/** /bgsys/drivers/ppcfloor/comm/include/dcmf.h
 *  /bgsys/drivers/ppcfloor/arch/include/common/bgp_personality.h
 */
#endif

/*readonly*/ CProxy_Main mainProxy;

/*mainchare*/
class Main : public CBase_Main
{
  public:
    Main(CkArgMsg* m) {
#if CMK_BLUEGENEL
      BGLPersonality bgl_p;
      int i = rts_get_personality(&bgl_p, sizeof(BGLPersonality));
#elif CMK_BLUEGENEP
      DCMF_Hardware_t bgp_hwt;
      DCMF_Hardware(&bgp_hwt);
#elif XT3_TOPOLOGY
      XT3TorusManager xt3tm;
#elif XT4_TOPOLOGY || XT5_TOPOLOGY
      XTTorusManager xttm;
#endif

      mainProxy = thishandle;
      CkPrintf("Testing TopoManager .... \n");
      TopoManager tmgr;
      CkPrintf("Torus Size [%d] [%d] [%d] [%d]\n", tmgr.getDimNX(), tmgr.getDimNY(), tmgr.getDimNZ(), tmgr.getDimNT());

#if CMK_BLUEGENEP
      CkPrintf("Torus Size [%d] [%d] [%d] [%d]\n", bgp_hwt.xSize, bgp_hwt.ySize, bgp_hwt.zSize, bgp_hwt.tSize);
#endif
      int x, y, z, t;

      for(int i=0; i<CkNumPes(); i++) {
	tmgr.rankToCoordinates(i, x, y, z, t);
	CkPrintf("---- Processor %d ---> x %d y %d z %d t %d\n", i, x, y, z, t);
#if CMK_BLUEGENEL
	unsigned int tmp_t, tmp_x, tmp_y, tmp_z;
	rts_coordinatesForRank(i, &tmp_x, &tmp_y, &tmp_z, &tmp_t);
	CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#elif CMK_BLUEGENEP
	unsigned int tmp_t, tmp_x, tmp_y, tmp_z;
    #if (DCMF_VERSION_MAJOR >= 3)
	DCMF_NetworkCoord_t nc;
	DCMF_Messager_rank2network(i, DCMF_DEFAULT_NETWORK, &nc);
	tmp_x = nc.torus.x;
	tmp_y = nc.torus.y;
	tmp_z = nc.torus.z;
	tmp_t = nc.torus.t;
    #else
	DCMF_Messager_rank2torus(c, &tmp_x, &tmp_y, &tmp_z, &tmp_t);
    #endif
	CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#elif XT3_TOPOLOGY
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
