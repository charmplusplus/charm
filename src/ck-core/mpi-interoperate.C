#include "mpi-interoperate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#endif
//#define PERFORM_DEBUG 1
#if PERFORM_DEBUG
#define DEBUG(a) a
#else
#define DEBUG(a) 
#endif

int _libExitHandlerIdx;
int _libStopSchedulerHandlerIdx;
static bool _libExitStarted = false;
// Set when CharmLibExit has torn the runtime down and returned (MPI interop).
static bool _libExitReturned = false;

// The runtime is torn down, so CmiAbort cannot be used.
static void _libOncePerProcess(const char *what)
{
  fprintf(stderr, "%s after CharmLibExit has returned: CharmLibInit and "
          "CharmLibExit can be called once per process.\n", what);
  fflush(stderr);
  abort();
}

extern int _ckExitCode();
extern bool _ckExitSequenceDone();

extern std::atomic<int> ckExitComplete;
extern std::atomic<int> _cleanUp;

extern void LrtsDrainResources(); /* used when exit */

extern bool _ringexit;		    // for charm exit
extern int _ringtoken;
extern void _initCharm(int unused_argc, char **argv);
extern void _registerConverseInitHooks();
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

// Ends the scheduler pass of the PE that receives it (see CharmLibExit).
void _libStopSchedulerHandler(void *msg)
{
  CmiFree(msg);
  CsdExitScheduler();
}

// CharmBeginInit calls sets interop flags then calls ConverseInit. This
// initializes the runtime, but does not start the scheduler or send readonlies.
// It returns control the main after main chares have been created.
void CharmBeginInit(int argc, char** argv) {

  userDrivenMode = 1;
  CharmLibInterOperate = true;
  _registerConverseInitHooks();
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

// MPI interop entry point behind the inline CharmLibInit in
// mpi-interoperate.h, which has already checked that the communicator is
// MPI_COMM_WORLD (up to congruence). Charm++ runs on every process of the job
// and CharmLibExit returns, so the program can keep using MPI afterwards.
// No MPI call is made here; libck does not depend on MPI.
void CharmLibInitWorld(int argc, char **argv) {
  if (_libExitReturned) _libOncePerProcess("CharmLibInit called");
  CmiInteropReturnOnExit = 1;
  CharmLibInterOperate = true;
  _registerConverseInitHooks();
  ConverseInit(argc, argv, (CmiStartFn)_initCharm, 1, 0);
  StartInteropScheduler();
}

// Picks a TCP rendezvous endpoint for LCT's tcp bootstrap: this host's
// address (the IPv4 address its hostname resolves to, or the hostname itself)
// and a port the kernel reports free. The probe socket is closed before
// returning, so another process can take the port before LCT binds it.
// Returns the port, or -1 on failure (always on Windows, where MPI interop is
// not supported).
int CharmInteropPickRendezvous(char *host, int hostlen) {
#if defined(_WIN32)
  return -1;
#else
  char name[256];
  if (gethostname(name, sizeof(name)) != 0) return -1;
  name[sizeof(name) - 1] = '\0';
  snprintf(host, hostlen, "%s", name);
  struct addrinfo hints, *res = nullptr;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(name, nullptr, &hints, &res) == 0 && res) {
    char ip[INET_ADDRSTRLEN];
    struct sockaddr_in *sin = (struct sockaddr_in *)res->ai_addr;
    if (inet_ntop(AF_INET, &sin->sin_addr, ip, sizeof(ip)) &&
        strncmp(ip, "127.", 4) != 0)
      snprintf(host, hostlen, "%s", ip);
    freeaddrinfo(res);
  }
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;
  socklen_t len = sizeof(addr);
  int port = -1;
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0 &&
      getsockname(fd, (struct sockaddr *)&addr, &len) == 0)
    port = ntohs(addr.sin_port);
  close(fd);
  return port;
#endif
}

// Selects LCT's tcp bootstrap for the LCI start that follows. It needs no PMI
// client and no MPI in LCI: the caller supplies rank, size and the rendezvous
// endpoint (rank 0 listens there). An LCT_PMI_BACKEND set by the user is left
// alone.
void CharmInteropSetTcpBootstrap(int rank, int size, int local_rank,
                                 int local_size, const char *host, int port) {
#if defined(_WIN32)
  CmiAbort("MPI interop is not supported on Windows");
#else
  if (getenv("LCT_PMI_BACKEND") == nullptr) {
    char buf[32];
    setenv("LCT_PMI_BACKEND", "tcp", 1);
    snprintf(buf, sizeof(buf), "%d", rank);
    setenv("RANK", buf, 1);
    snprintf(buf, sizeof(buf), "%d", size);
    setenv("WORLD_SIZE", buf, 1);
    snprintf(buf, sizeof(buf), "%d", local_rank);
    setenv("LOCAL_RANK", buf, 1);
    snprintf(buf, sizeof(buf), "%d", local_size);
    setenv("LOCAL_WORLD_SIZE", buf, 1);
    setenv("LCT_MASTER_ADDR", host, 1);
    snprintf(buf, sizeof(buf), "%d", port);
    setenv("LCT_MASTER_PORT", buf, 1);
  }
#endif
}

// MPI interop, library mode, with LCT's tcp bootstrap.
void CharmLibInitWorldTcp(int rank, int size, int local_rank, int local_size,
                          const char *host, int port, int argc, char **argv) {
  CharmInteropSetTcpBootstrap(rank, size, local_rank, local_size, host, port);
  CharmLibInitWorld(argc, argv);
}

// MPI interop, main mode (CharmMpiMain): what the MPI side expects of the
// Charm++ start that follows. Checked by _interopCheckStart.
static bool _mpiMainMode = false;
static int _mpiMainRank = -1, _mpiMainSize = 0;
static bool _mpiMainSerialized = false;
static char _mpiMainUserPmi[32] = "";

// Starts Charm++ as an ordinary run, in a process whose MPI rank is rank of
// size and whose MPI thread level is at least MPI_THREAD_SERIALIZED if
// serialized is nonzero. Does not return.
int CharmMpiMainStart(int rank, int size, int serialized, const char *userPmi,
                      int argc, char **argv) {
  _mpiMainMode = true;
  _mpiMainRank = rank;
  _mpiMainSize = size;
  _mpiMainSerialized = serialized != 0;
  snprintf(_mpiMainUserPmi, sizeof(_mpiMainUserPmi), "%s", userPmi ? userPmi : "");
  return charm_main(argc, argv);
}

// Called by _initCharm on rank 0 of every process. In main mode, aborts
// (through CharmMpiMain's abort hook, which ends the MPI job) unless Charm++
// process i is MPI rank i and the MPI thread level covers this process's PEs.
void _interopCheckStart() {
  if (!_mpiMainMode) return;
  if (CmiMyNode() != _mpiMainRank || CmiNumNodes() != _mpiMainSize)
    CmiAbort("CharmMpiMain: Charm++ process %d of %d is MPI_COMM_WORLD rank %d "
             "of %d; the launcher gave Charm++ and MPI different process "
             "numbering, which MPI interop requires to match.%s",
             CmiMyNode(), CmiNumNodes(), _mpiMainRank, _mpiMainSize,
             _mpiMainUserPmi[0] ? " LCT_PMI_BACKEND is set in the environment, "
             "so LCI did not use the tcp bootstrap that CharmMpiMain sets up; "
             "unset it for MPI interop." : "");
  if (CmiMyNodeSize() > 1 && !_mpiMainSerialized)
    CmiAbort("CharmMpiMain: this process has %d PEs, but MPI provides a thread "
             "level below MPI_THREAD_SERIALIZED. Use an MPI library that "
             "provides it (if the program initializes MPI itself, with "
             "MPI_Init_thread), or run one PE per process.", CmiMyNodeSize());
}

// CharmLibExit is called for both forms of interop when the application is
// done with Charm. After CharmLibInit (MPI interop) it tears the runtime down
// and returns the exit code the application gave to CkExit (0 if it never
// called the real CkExit), leaving MPI to the caller; CharmLibInit and
// CharmLibExit can then not be called again in this process. After CharmInit
// (user-driven) the process exits inside it.
//
// If a mainchare program already ran the exit sequence inside CharmLibInit
// (its CkExit(code) reached ConverseExit on every PE), the sequence is not
// run again: each PE of this process only needs to leave its final scheduler
// pass, so rank 0 sends each of them a stop message.
#undef CkExit
#define CkExit CKEXIT_0 // CKEXIT_0 and other CkExit macros defined in charm.h
int CharmLibExit() {
  if (_libExitReturned) _libOncePerProcess("CharmLibExit called again");
  _cleanUp = 1;
  CmiNodeAllBarrier();
  if (_ckExitSequenceDone()) {
    for (int r = 0; r < CmiMyNodeSize(); r++) {
      char *msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
      CmiSetHandler(msg, _libStopSchedulerHandlerIdx);
      CmiSyncSendAndFree(CmiNodeFirst(CmiMyNode()) + r, CmiMsgHeaderSizeBytes, msg);
    }
  } else if(CkMyPe() == 0) {
    CkExit();
  }
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (ckExitComplete.load() == 0) { CommunicationServerThread(5); }
  } else { 
    CsdScheduler(-1);
    CmiNodeAllBarrier();
    if (CmiInteropReturnOnExit) {
      ConverseInteropFinalize();
      // The program may call MPI_Finalize next: CharmLibInit's abort hook
      // must not call MPI_Abort after that.
      CmiSetAbortHook(nullptr);
      _libExitReturned = true;
    }
  }
  return _ckExitCode();
}
