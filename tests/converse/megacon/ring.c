#include <stdio.h>
#include <converse.h>
#include <sys/types.h>
#include "ring.cpm.h"

void Cpm_megacon_ack();

void ring_fail()
{
  CmiPrintf("data corrupted in ring_hop.\n");
  exit(1);
}

CpmInvokable ring_hop(int steps, CpmStr t1, CpmDim t2, CpmStr *t3)
{
  int thispe = CmiMyPe();
  int nextpe = (thispe+1) % CmiNumPes();
  if (strcmp(t1,"Howdy, Dude.")) ring_fail();
  if (t2 != 4) ring_fail();
  if (strcmp(t3[0],"this")) ring_fail();
  if (strcmp(t3[1],"is")) ring_fail();
  if (strcmp(t3[2],"a")) ring_fail();
  if (strcmp(t3[3],"test")) ring_fail();
  if (steps) {
    Cpm_ring_hop(CpmSend(nextpe), steps-1, t1, t2, t3);
  } else {
    Cpm_megacon_ack(CpmSend(0));
  }
}

void ring_init(void)
{
  char *data[4] = { "this", "is", "a", "test" };
  Cpm_ring_hop(CpmSend(0), 1000, "Howdy, Dude.", 4, data);
}

void ring_moduleinit()
{
  CpmInitializeThisModule();
}
