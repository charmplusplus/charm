/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "DummyLB.h"

extern int quietModeRequested;

CreateLBFunc_Def(DummyLB, "Dummy load balancer, like a normal one but with empty strategy")

#include "DummyLB.def.h"

DummyLB::DummyLB(const CkLBOptions &opt): CBase_DummyLB(opt)
{
  lbname = (char*)"DummyLB";
  if (CkMyPe() == 0 && !quietModeRequested)
    CkPrintf("CharmLB> DummyLB created.\n");
}

bool DummyLB::QueryBalanceNow(int _step)
{
  return true;
}



/*@}*/
