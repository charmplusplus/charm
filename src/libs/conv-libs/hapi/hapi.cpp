#include "hapi.h"
#include "converse.h"
#include <stdio.h>

CpvDeclare(int, poll_init_idx);

static void __poll(void* a) {
  printf("[PE %d] Received %d\n", CmiMyPe(), *(int*)a);
  fflush(stdout);
}

static void __poll_init() {
  printf("Calling __poll_init() on PE %d\n", CmiMyPe());
  fflush(stdout);

  /*
  int* a = (int*)CmiAlloc(sizeof(int));
  *a = 3;

  // Needs to be invoked on all PEs
  CcdCallOnConditionKeep(CcdPERIODIC, (CcdVoidFn)__poll, a);
  */
}

// Initialization routine, should be called from user's Main (PE 0)
void hapi_init() {
  printf("[PE %d] HAPI init\n", CmiMyPe());
  fflush(stdout);

  // Register polling handler function
  CpvInitialize(int, poll_init_idx);
  CpvAccess(poll_init_idx) = CmiRegisterHandler((CmiHandler) __poll_init);

  // Broadcast to all PEs to set up polling
  char* msg = (char*)CmiAlloc(CmiMsgHeaderSizeBytes);
  CmiSetHandler(msg, CpvAccess(poll_init_idx));
  CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes, msg);
}

void hapi_test() {}
