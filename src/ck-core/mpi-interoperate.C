#include "mpi-interoperate.h"
//#define PERFORM_DEBUG 1
#if PERFORM_DEBUG
#define DEBUG(a) a
#else
#define DEBUG(a) 
#endif

int _libExitHandlerIdx;
static bool _libExitStarted = false;

extern std::atomic<int> ckExitComplete;
extern std::atomic<int> _cleanUp;

#if CMK_CONVERSE_MPI
extern MPI_Comm charmComm;
#else
typedef int MPI_Comm;
#endif

extern void LrtsDrainResources(); /* used when exit */

extern bool _ringexit;		    // for charm exit
extern int _ringtoken;
extern void _initCharm(int unused_argc, char **argv);
extern void _sendReadonlies();
void CommunicationServerThread(int sleepTime);
extern int CharmLibInterOperate;
extern int userDrivenMode;

void StartInteropScheduler();
void StopInteropScheduler();

void StartCharmScheduler() {
  CmiNodeAllBarrier();
  StartInteropScheduler();
}

void StopCharmScheduler() {
  StopInteropScheduler();
}

// triger LibExit on PE 0,
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
      _libExitStarted = true;
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
      DEBUG(printf("[%d/%d] Am done here\n",CmiMyRank(),CmiMyPe());)
#if !CMK_SMP
      LrtsDrainResources();
#endif
      _libExitStarted = false;
      StopCharmScheduler();
      break;
    default:
      CmiAbort("Internal Error(_libExitHandler): Unknown-msg-type. Contact Developers.\n");
  }
}

// CharmBeginInit calls sets interop flags then calls ConverseInit. This
// initializes the runtime, but does not start the scheduler or send readonlies.
// It returns control the main after main chares have been created.
void CharmBeginInit(int argc, char** argv) {
#if !defined CMK_USE_LRTS || !CMK_USE_LRTS
  CmiAbort("Interop is not supported in non-LRTS machine layers.");
#endif

  userDrivenMode = 1;
  CharmLibInterOperate = true;
  ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
}

// CharmFinishInit broadcasts out readonly data then begins the interop
// scheduler. The split initialization allows interop apps to use readonlies
// without mainchares. They call CharmBeginInit(...), then set readonly values
// and perform other init such as group creation. They they call CharmFinishInit
// to bcast readonlies and group creation messages. Control returns to caller
// after all readonlies are received and all groups are created.
void CharmFinishInit() {
  if (CkMyPe() == 0) {
    _sendReadonlies();
  }
  StartInteropScheduler();
}

// CharmInit is the simplified initialization function for apps which have
// mainchares or don't use readonlies and don't require groups to be created
// before regular execution. It calls both CharmStartInit and CharmFinishInit.
void CharmInit(int argc, char** argv) {
  CharmBeginInit(argc, argv);
  CharmFinishInit();
}

// CharmLibInit is specifically for MPI interop, where MPI applications want
// to call Charm as a library. It does full initialization and starts the
// scheduler. If Charm is built on MPI, multiple Charm instances can be created
// using different communicators.
void CharmLibInit(MPI_Comm userComm, int argc, char **argv) {

#if CMK_USE_LRTS && !CMK_HAS_INTEROP
  if(!userDrivenMode) {
    CmiAbort("mpi-interoperate not supported in this machine layer; did you mean to use CharmInit?");
  }
#endif

#if CMK_CONVERSE_MPI
  if(!userDrivenMode) {
    MPI_Comm_dup(userComm, &charmComm);
  }
#endif

  CharmLibInterOperate = true;
  ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
  StartInteropScheduler();
}

// CharmLibExit is called for both forms of interop when the application is
// done with Charm. In userDrivenMode, this does a full exit and kills the
// application, just like CkExit(). In MPI interop, it just kills the Charm
// instance, but allows the outside application and other Charm instances to
// continue.
#undef CkExit
#define CkExit CKEXIT_0 // CKEXIT_0 and other CkExit macros defined in charm.h
void CharmLibExit() {
  _cleanUp = 1;
  CmiNodeAllBarrier();
  if(CkMyPe() == 0) {
    CkExit();
  }
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (ckExitComplete.load() == 0) { CommunicationServerThread(5); }
  } else { 
    CsdScheduler(-1);
    CmiNodeAllBarrier();
  }
}
