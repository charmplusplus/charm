extern "C" void CkExit(void);

#include "mpi-interoperate.h"
//#define PERFORM_DEBUG 1
#if PERFORM_DEBUG
#define DEBUG(a) a
#else
#define DEBUG(a) 
#endif

#if PERFORM_DEBUG
#define DEBUG(a) a
#else
#define DEBUG(a) 
#endif

static int   _libExitStarted = 0;
int    _libExitHandlerIdx;
int _cleanUp = 0;

static volatile int libcommThdExit = 0;
CmiNodeLock  libcommThdExitLock = 0;

#if CMK_CONVERSE_MPI
extern  "C" { extern MPI_Comm charmComm ;}
#endif

extern int _ringexit;		    // for charm exit
extern int _ringtoken;
extern void _initCharm(int unused_argc, char **argv);
extern "C" void CommunicationServerThread(int sleepTime);
extern int CharmLibInterOperate;

extern "C" 
void CharmScheduler() {
  DEBUG(printf("[%d]Starting scheduler [%d]/[%d]\n",CmiMyPe(),CmiMyRank(),CmiMyNodeSize());)
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (libcommThdExit != CmiMyNodeSize()) {
      CommunicationServerThread(5);
    }
    libcommThdExit = 0;
  } else { 
    CsdScheduler(-1);
  }
}

extern "C" 
void CharmScheduler() {
  DEBUG(printf("[%d]Starting scheduler [%d]/[%d]\n",CmiMyPe(),CmiMyRank(),CmiMyNodeSize());)
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (libcommThdExit != CmiMyNodeSize()) {
      CommunicationServerThread(5);
    }
    DEBUG(printf("[%d] Commthread Exit Scheduler\n",CmiMyPe());)
    libcommThdExit = 0;
  } else { 
    CsdScheduler(-1);
  }
}

extern "C"
void StartCharmScheduler() {
  CmiNodeAllBarrier();
  CharmScheduler();
}

extern "C"
void StopCharmScheduler() {
  DEBUG(printf("[%d] Exit Scheduler\n",CmiMyPe());)
  CpvAccess(charmLibExitFlag) = 1;
  CmiLock(libcommThdExitLock);
  libcommThdExit++;
  CmiUnlock(libcommThdExitLock);
}

// triger LibExit on PE 0,
extern "C"
void LibCkExit(void)
{
  // always send to PE 0
  envelope *env = _allocEnv(StartExitMsg);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _libExitHandlerIdx);
  CmiSyncSendAndFree(0, env->getTotalsize(), (char *)env);
}

void _libExitHandler(envelope *env)
{
  DEBUG(printf("[%d] Exit started for %d PE %d nodes\n",CmiMyPe(),CmiNumPes(),CmiNumNodes());)
  switch(env->getMsgtype()) {
    case StartExitMsg:
      DEBUG(printf("[%d] Exit started for %d PE %d nodes\n",CmiMyPe(),CmiNumPes(),CmiNumNodes());)
      CkAssert(CkMyPe()==0);
    case ExitMsg:
      CkAssert(CkMyPe()==0);
      if(_libExitStarted) {
        DEBUG(printf("[%d] Duplicate Exit started for %d PE %d nodes\n",CmiMyPe(),CmiNumPes(),CmiNumNodes());)
        CmiFree(env);
        return;
      }
      _libExitStarted = 1;
      env->setMsgtype(ReqStatMsg);
      env->setSrcPe(CkMyPe());
      // if exit in ring, instead of broadcasting, send in ring
      if (_ringexit){
        const int stride = CkNumPes()/_ringtoken;
        int pe = 0;
        while (pe<CkNumPes()) {
          CmiSyncSend(pe, env->getTotalsize(), (char *)env);
          pe += stride;
        }
        CmiFree(env);
      }else{
        DEBUG(printf("[%d] Broadcast Exit for %d PE %d nodes\n",CmiMyPe(),CmiNumPes(),CmiNumNodes());)
        CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char *)env);
      }	
      break;
    case ReqStatMsg:
      DEBUG(printf("[%d] Receive Exit for %d PE %d nodes\n",CmiMyPe(),CmiNumPes(),CmiNumNodes());)
      if (_ringexit) {
        int stride = CkNumPes()/_ringtoken;
        int pe = CkMyPe()+1;
        if (pe < CkNumPes() && pe % stride != 0)
          CmiSyncSendAndFree(pe, env->getTotalsize(), (char *)env);
        else
          CmiFree(env);
      }
      else
        CmiFree(env);
      //everyone exits here - there may be issues with leftover messages in the queue
      DEBUG(printf("[%d] Am done here\n",CmiMyPe());)
      _libExitStarted = 0;
      StopCharmScheduler();
      break;
    default:
      CmiAbort("Internal Error(_libExitHandler): Unknown-msg-type. Contact Developers.\n");
  }
}

#if CMK_HAS_INTEROP
#if CMK_CONVERSE_MPI
extern "C"
void CharmLibInit(MPI_Comm userComm, int argc, char **argv){
  //note CmiNumNodes and CmiMyNode should just be macros
  charmComm = userComm;
  MPI_Comm_size(charmComm, &_Cmi_numnodes);
  MPI_Comm_rank(charmComm, &_Cmi_mynode);

  CharmLibInterOperate = 1;
  ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
  CharmScheduler();
}
#else 
extern "C"
void CharmLibInit(int userComm, int argc, char **argv){
  CharmLibInterOperate = 1;
  ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
  CharmScheduler();
}
#endif
#else
extern "C"
void CharmLibInit(int userComm, int argc, char **argv){
  CmiAbort("mpi-interoperate not supported in this machine layer");
}
#endif

#undef CkExit
#define CkExit CkExit
extern "C"
void CharmLibExit() {
  _cleanUp = 1;
  CmiNodeAllBarrier();
  if(CkMyPe() == 0) {
    CkExit();
  }
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (1) { CommunicationServerThread(5); }
  } else { 
    CsdScheduler(-1);
  }
}

