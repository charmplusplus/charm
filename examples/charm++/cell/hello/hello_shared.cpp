#include <stdio.h>

#include "hello_shared.h"


void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen
               ) {

  switch (funcIndex) {

    case FUNC_SAYHI: sayHi((char*)readWritePtr, (char*)readOnlyPtr); break;

    default:
      printf("!!! WARNING !!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", funcIndex);
      break;
  }
}


void sayHi(char* readWritePtr, char* readOnlyPtr) {
  printf("I was told to say \"Hi\"... so \"Hi\"... ok... later...\n");
  if (readWritePtr != NULL)
    printf("   readWritePtr -> \"%s\"\n", readWritePtr);
  if (readOnlyPtr != NULL)
    printf("    readOnlyPtr -> \"%s\"\n", readOnlyPtr);
}
