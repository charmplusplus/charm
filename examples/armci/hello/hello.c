#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <armci.h>

#define MAX_PROCESSORS 2

int main(int argc, char * argv[]) {
  void *baseAddress[MAX_PROCESSORS];
  char *myBuffer;
  int thisImage;

  // initialize
  ARMCI_Init();
  ARMCI_Myid(&thisImage);

  // allocate data (collective operation)
  ARMCI_Malloc(baseAddress, strlen("hello")+1);

  if (thisImage == 0) {
    sprintf((char *)baseAddress[0], "%s", "hello");
  } else if (thisImage == 1) {
    sprintf((char *)baseAddress[1], "%s", "world");
  }

  // allocate space for local buffer
  myBuffer = (char *)ARMCI_Malloc_local(strlen("hello")+1);

  ARMCI_Barrier();

  if (thisImage == 0) {
    ARMCI_Get(baseAddress[1], myBuffer, strlen("hello")+1, 1);
    printf("[%d] %s %s\n",thisImage, baseAddress[0], myBuffer);
  } else if (thisImage == 1) {
    ARMCI_Get(baseAddress[0], myBuffer, strlen("hello")+1, 0);
    printf("[%d] %s %s\n",thisImage, myBuffer, baseAddress[1]);
  }

  // sanity check (should segfault)
  // printf("[%d] %s %s\n",thisImage, baseAddress[0], baseAddress[1]);

  // finalize
  ARMCI_Finalize();
  return 0;
}
