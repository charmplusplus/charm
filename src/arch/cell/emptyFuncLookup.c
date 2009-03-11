#include <stdlib.h>
#include <stdio.h>

#include "spert.h"

#ifdef __cplusplus
extern "C"
#endif
void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen,
                DMAListEntry* dmaList
	       ) {

  switch (funcIndex) {

    case SPE_FUNC_INDEX_INIT:
    case SPE_FUNC_INDEX_CLOSE:
      break;

    default:
      printf("!!! WARNING !!! :: Call into empty spert!... ignoring...\n");
      break;
  }
}
