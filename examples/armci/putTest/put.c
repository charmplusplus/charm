#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <armci.h>
#include <charm.h>

#define MAX_PROCESSORS 2
#define MAX_BUF_SIZE 1048576

int main(int argc, char * argv[]) {
  void *baseAddress[MAX_PROCESSORS];
  char *local;
  int thisImage;

  int iter = 100, size;
  double startTime, endTime;
  int i;

  // initialize
  ARMCI_Init();
  ARMCI_Myid(&thisImage);

  // allocate data (collective operation)
  ARMCI_Malloc(baseAddress, MAX_BUF_SIZE*sizeof(char));
  local = (char *)ARMCI_Malloc_local(MAX_BUF_SIZE*sizeof(char));

  ARMCI_Barrier();
  ARMCI_Migrate();

  if (thisImage == 0) {
    for(size = 1; size <= MAX_BUF_SIZE; size = size<<1){
      startTime = CkWallTimer();
      for(i = 0; i < iter; i++){
        ARMCI_Put(local, baseAddress[1], size, 1);
      }
      ARMCI_Fence(1);
      endTime = CkWallTimer();
      printf("%d: %f us\n", size, (endTime-startTime)*1000);
    }
    ARMCI_Barrier();
  } else if (thisImage == 1) {
    ARMCI_Barrier();
  }

  
  ARMCI_Free(baseAddress[thisImage]);
  ARMCI_Free_local(local);
  // finalize
  ARMCI_Finalize();
  return 0;
}
