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

#include "LBObj.h"
#include "LBOM.h"
#include "LBDBManager.h"

/*************************************************************
 * LBObj Object-data Code
 *************************************************************/

void LBObj::Clear(void)
{
//  data.handle = myhandle;
//  data.id = myid;
//  data.omHandle = parentOM;
//  data.omID = parentDB->LbOM(parentOM)->id();
  data.cpuTime = 0.;
  data.wallTime = 0.;
#if ! COMPRESS_LDB
  data.minWall = 1e6;
  data.maxWall = 0.;
#endif
}

void LBObj::IncrementTime(double walltime, double cputime)
{
  parentDB->MeasuredObjTime(walltime,cputime);
  data.wallTime += walltime;
  data.cpuTime += cputime;
#if ! COMPRESS_LDB
  if (walltime < data.minWall) data.minWall = walltime;
  if (walltime > data.maxWall) data.maxWall = walltime;
#endif
}

#endif

/*@}*/
