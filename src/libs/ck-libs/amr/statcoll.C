#include "statcoll.h"

void StatCollector :: registerMe()
{
  msgExpected++;
}

void StatCollector :: unRegisterMe()
{
  msgExpected--;
}

void StatCollector :: migrating()
{
  migCount++;
  unRegisterMe();
  
}

void StatCollector :: incrementAutorefine(void)
{
  aRefineCount++;
  unRegisterMe();
}

void StatCollector :: incrementRefine(void)
{
  refineCount++;
  unRegisterMe();
}

void StatCollector :: sendStat(_DummyMsg *m)
{
  delete m;
  int myPE = CkMyPe();
  _StatCollMsg* msg = new _StatCollMsg(aRefineCount,refineCount, msgExpected, migCount,myPE);
  CProxy_AmrCoordinator coordProxy(coordHandle);
  coordProxy.reportStats(msg);
}
