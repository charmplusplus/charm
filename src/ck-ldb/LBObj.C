#include <converse.h>


#if CMK_LBDB_ON

#include "LBObj.h"
#include "LBOM.h"
#include "LBDB.h"

/*************************************************************
 * LBObj Object-data Code
 *************************************************************/

void LBObj::Clear(void)
{
  data.handle = myhandle;
  data.id = myid;
  data.omHandle = parentOM;
  data.omID = parentDB->LbOM(parentOM)->id();
  data.cpuTime = 0.;
  data.wallTime = 0.;
}

void LBObj::IncrementTime(double walltime, double cputime)
{
  data.wallTime += walltime;
  data.cpuTime += cputime;
}

void LBObj::StartTimer(void)
{
  startWTime = CmiWallTimer();
  startCTime = CmiCpuTimer();
}

void LBObj::StopTimer(double* walltime, double* cputime)
{
  const double endWTime = CmiWallTimer();
  const double endCTime = CmiCpuTimer();
  *walltime = endWTime - startWTime;
  *cputime = endCTime - startCTime;
}

#endif
