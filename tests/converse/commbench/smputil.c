#include <converse.h>
#include "commbench.h"

#define NVARITER 10000000
#define NLOCKITER 1000000
#define NBARRITER 10000

CpvStaticDeclare(double, privateVar);
CpvStaticDeclare(int, barrIdx);
CsvStaticDeclare(double, sharedVar);

static void barrierHandler(EmptyMsg *msg)
{
  int i;
  double starttime, endtime;
  double loopOverhead;
  if(CmiMyNode() == 0) {
    starttime = CmiWallTimer();
    for(i=0; i<NBARRITER; i++);
    loopOverhead = CmiWallTimer() - starttime;
    for(i=0; i<10; i++)
      CmiNodeBarrier();
    starttime = CmiWallTimer();
    for(i=0; i<NBARRITER; i++)
      CmiNodeBarrier();
    endtime = CmiWallTimer();
    if(CmiMyPe() == 0) {
      CmiPrintf("[smputil] Barrier Overhead: %le seconds\n",
                (endtime - starttime - loopOverhead)/NBARRITER);
      CmiSetHandler(msg, CpvAccess(ack_handler));
      CmiSyncSend(0, sizeof(EmptyMsg), msg);
    }
  }
}

void smputil_init(void)
{
  EmptyMsg msg;
  double starttime, endtime;
  double stackVar=0.0, loopOverhead;
  int i;
  CmiNodeLock lock;

  starttime = CmiWallTimer();
  for(i=0;i<NVARITER;i++);
  loopOverhead = CmiWallTimer() - starttime;
  starttime = CmiWallTimer();
  for(i=0;i<NVARITER;i++)
    stackVar += 1.0;
  endtime = CmiWallTimer();
  CmiPrintf("[smputil] StackVar Access Overhead: %le seconds\n",
             (endtime - starttime - loopOverhead)/NVARITER);
  starttime = CmiWallTimer();
  for(i=0;i<NVARITER;i++)
    CpvAccess(privateVar) += 1.0;
  endtime = CmiWallTimer();
  CmiPrintf("[smputil] ProcPrivateVar Access Overhead: %le seconds\n",
             (endtime - starttime - loopOverhead)/NVARITER);
  starttime = CmiWallTimer();
  for(i=0;i<NVARITER;i++)
    CsvAccess(sharedVar) += 1.0;
  endtime = CmiWallTimer();
  CmiPrintf("[smputil] SharedVar Access Overhead: %le seconds\n",
             (endtime - starttime - loopOverhead)/NVARITER);
  starttime = CmiWallTimer();
  for(i=0;i<NBARRITER;i++);
  loopOverhead = CmiWallTimer() - starttime;
  lock = CmiCreateLock();
  starttime = CmiWallTimer();
  for(i=0;i<NLOCKITER;i++){
    CmiLock(lock);
    CmiUnlock(lock);
  }
  endtime = CmiWallTimer();
  CmiPrintf("[smputil] LockUnlock Overhead: %le seconds\n",
             (endtime - starttime - loopOverhead)/NLOCKITER);
  CmiSetHandler(&msg, CpvAccess(barrIdx));
  CmiSyncBroadcastAll(sizeof(EmptyMsg), &msg);
}

void smputil_moduleinit(void)
{
  CpvInitialize(double, privateVar);
  CpvInitialize(int, barrIdx);
  CsvInitialize(double, sharedVar);

  CpvAccess(barrIdx) = CmiRegisterHandler((CmiHandler)barrierHandler);
}

