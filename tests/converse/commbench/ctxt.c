#include <converse.h>
#include "commbench.h"

#define NITER 1000000

void ctxt_init(void)
{
  double starttime, endtime;
  int i;
  EmptyMsg msg;

  starttime = CmiWallTimer();
  for(i=0;i<NITER;i++) CthYield();
  endtime = CmiWallTimer();
  CmiPrintf("[ctxt] Thread Context Switching Overhead = %le seconds\n",
             (endtime-starttime)/NITER);
  CmiSetHandler(&msg, CpvAccess(ack_handler));
  CmiSyncSend(0, sizeof(EmptyMsg), &msg);
}

void ctxt_moduleinit(void)
{
}
