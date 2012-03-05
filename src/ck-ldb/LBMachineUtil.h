/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _LDMACHINEUTIL_H_
#define _LDMACHINEUTIL_H_

class LBMachineUtil {
public:
  LBMachineUtil();
  void StatsOn();
  void StatsOff();
  void Clear();
  void TotalTime(LBRealType* walltime, LBRealType* cputime);
  void IdleTime(LBRealType* walltime) { *walltime = total_idletime; };

private:
  enum { off, on } state;
  LBRealType total_walltime;
  LBRealType total_idletime;
  double start_totalwall;
  double start_idle;
#if CMK_LB_CPUTIMER
  LBRealType total_cputime;
  double start_totalcpu;
#endif

  int cancel_idleStart, cancel_idleEnd;

  void IdleStart(double curWallTime);
  void IdleEnd(double curWallTime);
  static void staticIdleStart(LBMachineUtil *util,double curWallTime);
  static void staticIdleEnd(LBMachineUtil *util,double curWallTime);
};

#endif  // _LDMACHINEUTIL_H_

/*@}*/
