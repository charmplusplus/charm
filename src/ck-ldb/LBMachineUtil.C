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
  total_walltime = 0.0;
  total_idletime = 0.0;
  start_totalwall = -1.;
  start_idle = -1.;
#if CMK_LB_CPUTIMER
  total_cputime = 0.0;
  start_totalcpu = -1.;
#endif
}

void LBMachineUtil::StatsOn()
{
  const double cur_wall = CkWallTimer();
#if CMK_LB_CPUTIMER
  const double cur_cpu = CkCpuTimer();
#endif

  if (state == off) {
#if ! CMK_BIGSIM_CHARM
    cancel_idleStart=CcdCallOnConditionKeep(
	 CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)staticIdleStart,(void *)this);
    cancel_idleEnd=CcdCallOnConditionKeep(
         CcdPROCESSOR_END_IDLE,(CcdVoidFn)staticIdleEnd,(void *)this);
#endif
    state = on;
  }

  if (start_totalwall != -1.) {
    total_walltime += (cur_wall - start_totalwall);
#if CMK_LB_CPUTIMER
    total_cputime += (cur_cpu - start_totalcpu);
#endif
  }
  start_totalwall = cur_wall;
#if CMK_LB_CPUTIMER
  start_totalcpu = cur_cpu;
#endif
}

void LBMachineUtil::StatsOff()
{
  if (state == on) {
#if ! CMK_BIGSIM_CHARM
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,cancel_idleStart);
    CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE,cancel_idleEnd);
#endif
    state = off;
  }

  if (start_totalwall != -1.) {
    const double cur_wall = CkWallTimer();
    total_walltime += (cur_wall - start_totalwall);
#if CMK_LB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
    total_cputime += (cur_cpu - start_totalcpu);
#endif
  }
  start_totalwall = -1.;
#if CMK_LB_CPUTIMER
  start_totalcpu = -1.;
#endif
}

void LBMachineUtil::Clear()
{
  total_walltime = 0.0;
#if CMK_LB_CPUTIMER
  total_cputime = 0.0;
#endif

  if (state == off) {
    start_totalwall = -1.;
#if CMK_LB_CPUTIMER
    start_totalcpu = -1.;
#endif
  } else {
    const double cur_wall = CkWallTimer();
#if CMK_LB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
#endif

    start_totalwall = cur_wall;
#if CMK_LB_CPUTIMER
    start_totalcpu = cur_cpu;
#endif
  }
  total_idletime = 0.0;
  start_idle = -1.;
}

void LBMachineUtil::TotalTime(LBRealType* walltime, LBRealType* cputime)
{
  if (state == on) {
    const double cur_wall = CkWallTimer();
#if CMK_LB_CPUTIMER
    const double cur_cpu = CkCpuTimer();
#endif
    total_walltime += (cur_wall - start_totalwall);
    start_totalwall = cur_wall;
#if CMK_LB_CPUTIMER
    total_cputime += (cur_cpu - start_totalcpu);
    start_totalcpu = cur_cpu;
#endif
  }
  *walltime = total_walltime;
#if CMK_LB_CPUTIMER
  *cputime = total_cputime;
#else
  *cputime = *walltime;
#endif
}

/*@}*/
