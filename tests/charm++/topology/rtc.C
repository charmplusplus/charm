#include <stdio.h>
#include "topo.decl.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;

/*mainchare*/
class Main : public Chare
{
  public:
    Main(CkArgMsg* m)
    {
      mainProxy = thishandle;
      CkPrintf("Testing TopoManager .... Abhinav\n");
      TopoManager tmgr;
      CkPrintf("Torus Size [%d] [%d] [%d]\n", tmgr.getDimNX(), tmgr.getDimNY(), tmgr.getDimNZ());
      int x, y, z, t;
      if(CkMyPe()==0) {
        for(int i=0; i<CkNumPes(); i++) {
          tmgr.rankToCoordinates(i, x, y, z, t); 
          CkPrintf("Processor %d ---> x %d y %d z %d t %d\n", i, x, y, z, t);
        }
      }
      int size = tmgr.getDimNX() * tmgr.getDimNY() * tmgr.getDimNZ();
      CkPrintf("Torus Contiguity Metric %d : %d [%f] \n", size, CkNumPes()/2, (float)(CkNumPes())/(2*size) );
      CkExit();
    };
    
};

#include "topo.def.h"
