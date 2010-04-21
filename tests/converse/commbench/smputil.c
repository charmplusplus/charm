#include <converse.h>
#include "commbench.h"

#define NVARITER 10000000
#define NLOCKITER 1000000
#define NBARRITER 10000

#define NMALLOCITER 100000
#define MALLOCSIZE 257

CpvStaticDeclare(double, privateVar);
CpvStaticDeclare(int, barrIdx);
CpvStaticDeclare(int, memIdx);
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

double memoryAllocTest(){
  double starttime, endtime;  
  double extraOverhead;
  void **ptrs = NULL;
  int i;	
	/* Estimate the malloc overhead */
  ptrs = (void **)malloc(NMALLOCITER*sizeof(void *));
  /* Warm the cache first before estimating the overheads */
  for(i=0; i<NMALLOCITER; i++) ptrs[i] = 0;
  
  starttime = CmiWallTimer();
  for(i=0; i<NMALLOCITER; i++) ptrs[i] = (void *)0xabcd;
  endtime = CmiWallTimer();
  extraOverhead = endtime - starttime;
  
  starttime = CmiWallTimer();
  for(i=0; i<NMALLOCITER; i++) ptrs[i] = CmiAlloc(MALLOCSIZE);
  for(i=0; i<NMALLOCITER; i++) CmiFree(ptrs[i]);
  endtime = CmiWallTimer();
  free(ptrs);
  
  return (endtime-starttime-extraOverhead*2)/NMALLOCITER;  	
}

static void memAllocHandler(EmptyMsg *msg){
	double overhead;
	/* Make sure the memory contention on a node happens roughly at the same time */
	CmiNodeBarrier();
	overhead = memoryAllocTest();	
	CmiNodeBarrier();
	
	if(CmiMyPe()==0){
	  CmiPrintf("[smputil] Estimated CmiAlloc/CmiFree Overhead (w contention): %le seconds\n",overhead);
	  CmiSetHandler(msg, CpvAccess(barrIdx));
	  CmiSyncBroadcastAll(sizeof(EmptyMsg), msg);
	}
	else {
	  CmiFree(msg);
	}
}

void smputil_init(void)
{
  EmptyMsg msg;
  double starttime, endtime;
  double stackVar=0.0, loopOverhead;
  double extraOverhead;
  void **ptrs = NULL;
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

  endtime = memoryAllocTest();
  CmiPrintf("[smputil] Estimated CmiAlloc/CmiFree Overhead (w/o contention): %le seconds\n",endtime);
             
  CmiSetHandler(&msg, CpvAccess(memIdx));
  CmiSyncBroadcastAll(sizeof(EmptyMsg), &msg);
}

void smputil_moduleinit(void)
{
  CpvInitialize(double, privateVar);
  CpvInitialize(int, barrIdx);
  CpvInitialize(int, memIdx);
  CsvInitialize(double, sharedVar);

  CpvAccess(barrIdx) = CmiRegisterHandler((CmiHandler)barrierHandler);
  CpvAccess(memIdx) = CmiRegisterHandler((CmiHandler)memAllocHandler);
}

