/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _LDMACHINEUTIL_H_
#define _LDMACHINEUTIL_H_

#include "converse.h"

class LBMachineUtil {
public:
  LBMachineUtil();
  void StatsOn();
  void StatsOff();
  void Clear();
  void TotalTime(double* walltime, double* cputime);
  void IdleTime(double* walltime) { *walltime = total_idletime; };
  void IdleStart();
  void IdleEnd();
  
private:
  enum { off, on } state;
  double total_walltime;
  double total_cputime;
  double total_idletime;
  double start_totalwall;
  double start_totalcpu;
  double start_idle;

  int cancel_idleStart, cancel_idleEnd;
};

#endif  // _LDMACHINEUTIL_H_
