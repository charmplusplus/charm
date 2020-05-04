#include <stdlib.h>
#include <converse.h>
#include <unistd.h>

CpvDeclare(size_t, msg_size);
CpvDeclare(int, n_iters);
CpvDeclare(int, iter);
CpvStaticDeclare(double, start_time);
CpvStaticDeclare(double, end_time);

CpvDeclare(int, node0_handler);
CpvDeclare(int, node1_handler);
CpvDeclare(int, exit_handler);

void startRing() {
  CpvAccess(iter) = 0;
  char* msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes + CpvAccess(msg_size));
  CmiSetHandler(msg, CpvAccess(node0_handler));
  *((size_t*)(msg + CmiMsgHeaderSizeBytes)) = CpvAccess(msg_size);
  CmiSyncSendAndFree(0, CpvAccess(msg_size), msg);
}

void ringFinished(char *msg) {
  CmiFree(msg);

  CmiPrintf("%lf us one-way\n",
      (1e6 * (CpvAccess(end_time) - CpvAccess(start_time))) / (2. * CpvAccess(n_iters)));

  // Broadcast message for termination
  void* term_msg = CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(term_msg, CpvAccess(exit_handler));
  CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, term_msg);
}

CmiHandler node0HandlerFunc(char* msg) {
  CpvAccess(iter)++;

  // Begin timer for the first iteration
  if (CpvAccess(iter) == 1) CpvAccess(start_time) = CmiWallTimer();

  // Stop timer for the last iteration
  if (CpvAccess(iter) == CpvAccess(n_iters)) {
    CpvAccess(end_time) = CmiWallTimer();
    ringFinished(msg);
  } else {
    CmiSetHandler(msg, CpvAccess(node1_handler));
    *((size_t*)(msg + CmiMsgHeaderSizeBytes)) = CpvAccess(msg_size);
    CmiSyncSendAndFree(1, CpvAccess(msg_size), msg);
  }

  return 0;
}

CmiHandler node1HandlerFunc(char* msg) {
  CpvAccess(msg_size) = *((size_t*)(msg + CmiMsgHeaderSizeBytes));
  CmiSetHandler(msg, CpvAccess(node0_handler));
  CmiSyncSendAndFree(0, CpvAccess(msg_size), msg);

  return 0;
}

CmiHandler exitHandlerFunc(char* msg) {
  CmiFree(msg);
  CsdExitScheduler();

  return 0;
}

CmiStartFn mymain(int argc, char** argv) {
  // Initialize variables
  CpvInitialize(size_t, msg_size);
  CpvInitialize(int, n_iters);
  CpvInitialize(int, iter);
  CpvInitialize(double, start_time);
  CpvInitialize(double, end_time);

  // Register Handlers
  CpvInitialize(int, node0_handler);
  CpvAccess(node0_handler) = CmiRegisterHandler((CmiHandler)node0HandlerFunc);
  CpvInitialize(int, node1_handler);
  CpvAccess(node1_handler) = CmiRegisterHandler((CmiHandler)node1HandlerFunc);
  CpvInitialize(int, exit_handler);
  CpvAccess(exit_handler) = CmiRegisterHandler((CmiHandler)exitHandlerFunc);

  // Initialize CPU affinity
  CmiInitCPUAffinity(argv);

  // Initialize CPU topology
  CmiInitCPUTopology(argv);

  // Wait for all PEs of the node to complete topology init
  CmiNodeAllBarrier();

  // Default parameters
  CpvAccess(msg_size) = 128;
  CpvAccess(n_iters) = 100;

  // Process runtime parameters
  argc = CmiGetArgc(argv);
  int c;
  while ((c = getopt(argc, argv, "s:i:")) != -1) {
    switch (c) {
      case 's':
        CpvAccess(msg_size) = atoi(optarg);
        break;
      case 'i':
        CpvAccess(n_iters) = atoi(optarg);
        break;
      default:
        CmiAbort("Unknown command line argument detected");
    }
  }
  /*
  if (argc == 3) {
    CpvAccess(msg_size) = atoi(argv[1]);
    CpvAccess(n_iters) = atoi(argv[2]);
  } else if (argc == 1) {
    CpvAccess(msg_size) = 128;
    CpvAccess(n_iters) = 100;
  } else {
    CmiAbort("Usage: %s <msg_size> <n_iters>\n", argv[0]);
  }
  */

  if (CmiMyPe() == 0) {
    CmiPrintf("[GPU pingpong]\nMsg size: %d, iterations: %d\n", CpvAccess(msg_size), CpvAccess(n_iters));
  }

  // Start!
  //startRing();

  return 0;
}

int main(int argc, char** argv) {
  ConverseInit(argc, argv, (CmiStartFn)mymain, 0, 0);

  return 0;
}
