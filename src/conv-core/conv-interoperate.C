#include "converse.h"

//#define PERFORM_DEBUG 1
#if PERFORM_DEBUG
#define DEBUG(a) do { a } while(false)
#else
#define DEBUG(a)
#endif

#if CMK_HAS_CXX11_ATOMIC
#include <atomic>
#elif CMK_HAS_CXX0X_CSTDATOMIC
#include <cstdatomic>
#else
#error "Configure should have errored on missing C++11 atomic library support"
#endif

static std::atomic<int> interopCommThdExit{0};

CpvCExtern(int,interopExitFlag);

int _cleanUp = 0;

#if CMK_USE_LRTS
extern void CommunicationServerThread(int sleepTime);
#else
void CommunicationServerThread(int sleepTime) { }
#endif

extern int CharmLibInterOperate;

void StartInteropScheduler() {
  DEBUG(printf("[%d]Starting scheduler [%d]/[%d]\n",CmiMyPe(),CmiMyRank(),CmiMyNodeSize()););
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (interopCommThdExit.load(std::memory_order_relaxed) != CmiMyNodeSize())
    {
      CommunicationServerThread(5);
    }
    DEBUG(printf("[%d] Commthread Exit Scheduler\n",CmiMyPe()););
    interopCommThdExit = 0;
  } else {
    CsdScheduler(-1);
  }
}

void StopInteropScheduler() {
  DEBUG(printf("[%d] Exit Scheduler\n",CmiMyPe()););
  CpvAccess(interopExitFlag) = 1;
  interopCommThdExit++;
}
