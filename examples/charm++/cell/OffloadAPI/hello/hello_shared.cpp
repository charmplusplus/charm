#include <stdio.h>
#include "spert.h"
#include "hello_shared.h"

inline void sayHi(char* msg) {
  printf("\"%s\" from SPE %d...\n",
         msg, (int)getSPEID());
}

#ifdef __cplusplus
extern "C"
#endif
void funcLookup(int funcIndex,
    void* readWritePtr, int readWriteLen,
    void* readOnlyPtr, int readOnlyLen,
    void* writeOnlyPtr, int writeOnlyLen,
    DMAListEntry* dmaList) {

  switch (funcIndex) {

    case SPE_FUNC_INDEX_INIT:  break;
    case SPE_FUNC_INDEX_CLOSE: break;

    case FUNC_SAYHI:
      sayHi((char*)readOnlyPtr);
      break;

    default:
      printf("ERROR :: Invalid funcIndex (%d)\n",
             funcIndex);
      break;
  }
}
