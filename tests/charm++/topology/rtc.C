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
#elif CMK_XT3
      CrayTorusManager crtm;
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
#elif CMK_XT3
	  int tmp_t, tmp_x, tmp_y, tmp_z;
          crtm.realRankToCoordinates(i, tmp_x, tmp_y, tmp_z, tmp_t);
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
