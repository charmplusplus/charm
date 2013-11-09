#include "converse.h"

//#define PERFORM_DEBUG 1
#if PERFORM_DEBUG
#define DEBUG(a) a
#else
#define DEBUG(a) 
#endif

int _cleanUp = 0;

static volatile int interopCommThdExit = 0;
CmiNodeLock  interopCommThdExitLock = 0;

#if CMK_USE_LRTS
extern void CommunicationServerThread(int sleepTime);
#else 
void CommunicationServerThread(int sleepTime) { }
#endif

extern int CharmLibInterOperate;
CpvExtern(int,interopExitFlag);

void StartInteropScheduler() {
  DEBUG(printf("[%d]Starting scheduler [%d]/[%d]\n",CmiMyPe(),CmiMyRank(),CmiMyNodeSize());)
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (interopCommThdExit != CmiMyNodeSize()) {
      CommunicationServerThread(5);
    }
    DEBUG(printf("[%d] Commthread Exit Scheduler\n",CmiMyPe());)
    interopCommThdExit = 0;
  } else { 
    CsdScheduler(-1);
  }
}

void StopInteropScheduler() {
  DEBUG(printf("[%d] Exit Scheduler\n",CmiMyPe());)
  CpvAccess(interopExitFlag) = 1;
  CmiLock(interopCommThdExitLock);
  interopCommThdExit++;
  CmiUnlock(interopCommThdExitLock);
}

