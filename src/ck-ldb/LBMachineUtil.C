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

#include "LBMachineUtil.h"
#include <stdlib.h>

extern "C" void staticIdleStart(LBMachineUtil* obj)
{
  obj->IdleStart();
}

extern "C" void staticIdleEnd(LBMachineUtil* obj)
{
  obj->IdleEnd();
}


void LBMachineUtil::IdleStart()
{
  start_idle = CmiWallTimer();
}

void LBMachineUtil::IdleEnd()
{
  if (state == on) {
    const double stop_idle = CmiWallTimer();
    total_idletime += (stop_idle - start_idle);
  }
}

LBMachineUtil::LBMachineUtil()
{
  state = off;
  total_walltime = total_cputime = 0.0;
  total_idletime = 0;
  start_totalwall = start_totalcpu = -1.;
  total_idletime = 0;
};

void LBMachineUtil::StatsOn()
{
  const double cur_wall = CmiWallTimer();
  const double cur_cpu = CmiCpuTimer();

  if (state == off) {
    cancel_idleStart=CcdCallOnConditionKeep(
	 CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)staticIdleStart,(void *)this);
    cancel_idleEnd=CcdCallOnConditionKeep(
         CcdPROCESSOR_END_IDLE,(CcdVoidFn)staticIdleEnd,(void *)this);
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
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,cancel_idleStart);
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE,cancel_idleEnd);
    state = off;
  }

  if (start_totalwall != -1.) {
    const double cur_wall = CmiWallTimer();
    const double cur_cpu = CmiCpuTimer();
    total_walltime += (cur_wall - start_totalwall);
    total_cputime += (cur_cpu - start_totalcpu);
  }
  start_totalwall = start_totalcpu = -1.;
};

void LBMachineUtil::Clear()
{
  total_walltime = total_cputime = 0;

  if (state == off) {
    start_totalwall = start_totalcpu = -1.;
  } else {
    const double cur_wall = CmiWallTimer();
    const double cur_cpu = CmiCpuTimer();

    start_totalwall = cur_wall;
    start_totalcpu = cur_cpu;
  }
  total_idletime = 0;
}

void LBMachineUtil::TotalTime(double* walltime, double* cputime)
{
  if (state == on) {
    const double cur_wall = CmiWallTimer();
    const double cur_cpu = CmiCpuTimer();
    total_walltime += (cur_wall - start_totalwall);
    total_cputime += (cur_cpu - start_totalcpu);
    start_totalwall = cur_wall;
    start_totalcpu = cur_cpu;
  }
  *walltime = total_walltime;
  *cputime = total_cputime;
}

/*@}*/
