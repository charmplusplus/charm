#include <converse.h>
#include "commbench.h"

#define NITER 1000000

void timer_init(void)
{
  volatile double starttime, endtime;
  int i;
  EmptyMsg msg;

  starttime = CmiWallTimer();
  for(i=0;i<NITER;i++) CmiCpuTimer();
  endtime = CmiWallTimer();
  CmiPrintf("[timer] (CmiCpuTimer) %le seconds per call\n",
             (endtime-starttime)/NITER);
  starttime = CmiWallTimer();
  for(i=0;i<NITER;i++) CmiWallTimer();
  endtime = CmiWallTimer();
  CmiPrintf("[timer] (CmiWallTimer) %le seconds per call\n",
             (endtime-starttime)/NITER);
  starttime = CmiCpuTimer();
  while((endtime=CmiCpuTimer())==starttime);
  CmiPrintf("[timer] (CmiCpuTimer) %le seconds resolution\n",
             endtime-starttime);
  starttime = CmiWallTimer();
  while((endtime=CmiWallTimer())==starttime);
  CmiPrintf("[timer] (CmiWallTimer) %le seconds resolution\n",
             endtime-starttime);
  CmiSetHandler(&msg, CpvAccess(ack_handler));
  CmiSyncSend(0, sizeof(EmptyMsg), &msg);
}

void timer_moduleinit(void)
{
}

