#include <stdio.h>
#include <string.h>
#include <spert_ppu.h>
#include "hello_shared.h"

#define NUM_WORK_REQUESTS  12

int main(int argc, char* argv[]) {

  WRHandle wrHandle[NUM_WORK_REQUESTS];
  char msg[] __attribute__((aligned(128))) = { "Hello" };
  int msgLen = ROUNDUP_16(strlen(msg));

  // Initialize the Offload API
  InitOffloadAPI();

  // Send some work requests
  for (int i = 0; i < NUM_WORK_REQUESTS; i++)
    wrHandle[i] = sendWorkRequest(FUNC_SAYHI,
                                  NULL, 0,
                                  msg, msgLen,
                                  NULL, 0
                                 );

  // Wait for the work requets to finish
  for (int i = 0; i < NUM_WORK_REQUESTS; i++)
    waitForWRHandle(wrHandle[i]);

  // Close the Offload API
  CloseOffloadAPI();

  // All Good
  return EXIT_SUCCESS;
}
