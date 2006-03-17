#include <stdio.h>

#include "hello_shared.h"


void funcLookup(int funcIndex, void* data, int dataLen, void* msg, int msgLen) {

  switch (funcIndex) {

    case FUNC_SAYHI: sayHi((char*)data, (char*)msg); break;

    default:
      printf("!!! WARNING !!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", funcIndex);
      break;
  }
}


void sayHi(char* data, char* msg) {
  printf("I was told to say \"Hi\"... so \"Hi\"... ok... later...\n");
  if (data != NULL)
    printf("   data -> \"%s\"\n", data);
  if (msg != NULL)
    printf("    msg -> \"%s\"\n", msg);
}
