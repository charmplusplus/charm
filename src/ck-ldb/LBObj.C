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

#include <converse.h>

#if CMK_LBDB_ON

#include "LBObj.h"
#include "LBOM.h"
#include "LBDBManager.h"

/*************************************************************
 * LBObj Object-data Code
 *************************************************************/

void LBObj::Clear(void)
{
  data.handle = myhandle;
//  data.id = myid;
  data.omHandle = parentOM;
//  data.omID = parentDB->LbOM(parentOM)->id();
  data.cpuTime = 0.;
  data.wallTime = 0.;
}

void LBObj::IncrementTime(double walltime, double cputime)
{
  parentDB->MeasuredObjTime(walltime,cputime);
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
  *walltime = CmiWallTimer() - startWTime;
  *cputime = CmiCpuTimer() - startCTime;
}

#endif

/*@}*/
