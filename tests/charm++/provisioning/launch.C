#include "launch.decl.h"

/* mainchare */
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    delete m;

    CkPrintf("Successfully launched %d PEs, %d processes, %d hosts\n",
             CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes());

    CkExit();
  }
};

#include "launch.def.h"
