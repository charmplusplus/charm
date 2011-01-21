/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "DummyLB.h"

CreateLBFunc_Def(DummyLB, "Dummy load balancer, like a normal one but with empty strategy")

#include "DummyLB.def.h"

DummyLB::DummyLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = (char*)"DummyLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] DummyLB created\n",CkMyPe());
}

CmiBool DummyLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}



/*@}*/
