/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "DummyLB.h"


CreateLBFunc_Def(DummyLB);

static void lbinit(void) {
  LBRegisterBalancer("DummyLB", CreateDummyLB, "Dummy load balancer, like a normal one but with empty strategy");
}

#include "DummyLB.def.h"

DummyLB::DummyLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "DummyLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] DummyLB created\n",CkMyPe());
}

CmiBool DummyLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

LBMigrateMsg* DummyLB::Strategy(CentralLB::LDStats* stats, int count)
{

  int sizes=0;
  LBMigrateMsg* msg = new(&sizes,1) LBMigrateMsg;
  msg->n_moves = 0;

  return msg;
};

#endif


/*@}*/
