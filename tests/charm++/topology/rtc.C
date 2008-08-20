#include <stdio.h>
#include "topo.decl.h"
#include "TopoManager.h"
#ifdef CMK_VERSION_BLUEGENE
#include <bglpersonality.h>
#endif

/*readonly*/ CProxy_Main mainProxy;

/*mainchare*/
class Main : public Chare
{
  public:
    Main(CkArgMsg* m)
    {
#if CMK_VERSION_BLUEGENE
      BGLPersonality bgl_p;
      int i = rts_get_personality(&bgl_p, sizeof(BGLPersonality));
#elif XT3_TOPOLOGY
      XT3TorusManager xt3tm;
#elif XT4_TOPOLOGY
      XT4TorusManager xt4tm;
#endif

      mainProxy = thishandle;
      CkPrintf("Testing TopoManager .... \n");
      TopoManager tmgr;
      CkPrintf("Torus Size [%d] [%d] [%d]\n", tmgr.getDimNX(), tmgr.getDimNY(), tmgr.getDimNZ());
      int x, y, z, t;
      if(CkMyPe()==0) {
        for(int i=0; i<CkNumPes(); i++) {
          tmgr.rankToCoordinates(i, x, y, z, t); 
          CkPrintf("---- Processor %d ---> x %d y %d z %d t %d\n", i, x, y, z, t);
#if CMK_VERSION_BLUEGENE
	  unsigned int tmp_t, tmp_x, tmp_y, tmp_z;
	  rts_coordinatesForRank(i, &tmp_x, &tmp_y, &tmp_z, &tmp_t);
	  CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#elif XT3_TOPOLOGY
	  int tmp_t, tmp_x, tmp_y, tmp_z;
          xt3tm.realRankToCoordinates(i, tmp_x, tmp_y, tmp_z, tmp_t);
	  CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#elif XT4_TOPOLOGY
	  int tmp_t, tmp_x, tmp_y, tmp_z;
          xt4tm.realRankToCoordinates(i, tmp_x, tmp_y, tmp_z, tmp_t);
	  CkPrintf("Real Processor %d ---> x %d y %d z %d t %d\n", i, tmp_x, tmp_y, tmp_z, tmp_t);
#endif
        }
      }
      int size = tmgr.getDimNX() * tmgr.getDimNY() * tmgr.getDimNZ();
      CkPrintf("Torus Contiguity Metric %d : %d [%f] \n", size, CkNumPes()/tmgr.getDimNT(), (float)(CkNumPes())/(tmgr.getDimNT()*size) );
      CkExit();
    };
    
};

#include "topo.def.h"
