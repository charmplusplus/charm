#include "hapi.h"
#include "converse.h"
#include <stdio.h>

void hapi_poll(void* a) {
  printf("[PE %d] Received %d\n", CmiMyPe(), *(int*)a);
  fflush(stdout);
}

void hapi_init() {
  printf("[PE %d] HAPI init\n", CmiMyPe());
  fflush(stdout);

  int* a = (int*)CmiAlloc(sizeof(int));
  *a = 3;

  // Needs to be invoked on all PEs
  CcdCallOnConditionKeep(CcdPERIODIC, (CcdVoidFn)hapi_poll, a);
}

void hapi_test() {}
