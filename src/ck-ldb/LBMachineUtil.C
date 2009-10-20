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

#include <stdlib.h>
#include "LBDatabase.h"
#include "LBMachineUtil.h"

inline void LBMachineUtil::IdleStart(double curWallTime)
{
  start_idle = curWallTime;
}

inline void LBMachineUtil::IdleEnd(double curWallTime)
{
// skip counting idle time in BigSim
  if (state == on) {
    const double stop_idle = curWallTime;
    total_idletime += (stop_idle - start_idle);
  }
}

void LBMachineUtil::staticIdleStart(LBMachineUtil* obj,double curWallTime)
{
  obj->IdleStart(curWallTime);
}
void LBMachineUtil::staticIdleEnd(LBMachineUtil* obj,double curWallTime)
{
  obj->IdleEnd(curWallTime);
}

LBMachineUtil::LBMachineUtil()
{
  state = off;
  total_walltime = total_cputime = 0.0;
  total_idletime = 0;
  start_totalwall = start_totalcpu = -1.;
  total_idletime = 0;
}

void LBMachineUtil::StatsOn()
{
  const double cur_wall = CkWallTimer();
#if CMK_LBDB_CPUTIMER
  const double cur_cpu = CkCpuTimer();
#else
  const double cur_cpu = cur_wall;
#endif

  if (state == off) {
#if ! CMK_BLUEGENE_CHARM
    cancel_idleStart=CcdCallOnConditionKeep(
	 CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)staticIdleStart,(void *)this);
    cancel_idleEnd=CcdCallOnConditionKeep(
         CcdPROCESSOR_END_IDLE,(CcdVoidFn)staticIdleEnd,(void *)this);
#endif
    state = on;
  }

  if (start_totalwall != -1.) {
    total_walltime += (cur_wall - start_totalwall);
    total_cputime += (cur_cpu - start_totalcpu);
  }
  start_totalwall = cur_wall;
  start_totalcpu = cur_cpu;
}

void LBMachineUtil::StatsOff()
{
  if (state == on) {
#if ! CMK_BLUEGENE_CHARM
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,cancel_idleStart);
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE,cancel_idleEnd);
#endif
    state = off;
  }

  if (start_totalwall != -1.) {
    const double cur_wall = CkWallTimer();
#if CMK_LBDB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
#else
    const double cur_cpu = cur_wall;
#endif
    total_walltime += (cur_wall - start_totalwall);
    total_cputime += (cur_cpu - start_totalcpu);
  }
  start_totalwall = start_totalcpu = -1.;
}

void LBMachineUtil::Clear()
{
  total_walltime = total_cputime = 0;

  if (state == off) {
    start_totalwall = start_totalcpu = -1.;
  } else {
    const double cur_wall = CkWallTimer();
#if CMK_LBDB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
#else
    const double cur_cpu = cur_wall;
#endif

    start_totalwall = cur_wall;
    start_totalcpu = cur_cpu;
  }
  total_idletime = 0;
}

void LBMachineUtil::TotalTime(double* walltime, double* cputime)
{
  if (state == on) {
    const double cur_wall = CkWallTimer();
#if CMK_LBDB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
#else
    const double cur_cpu = cur_wall;
#endif
    total_walltime += (cur_wall - start_totalwall);
    total_cputime += (cur_cpu - start_totalcpu);
    start_totalwall = cur_wall;
    start_totalcpu = cur_cpu;
  }
  *walltime = total_walltime;
  *cputime = total_cputime;
}

/*@}*/
