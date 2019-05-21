#include <stdio.h>
#include <converse.h>
#include <sys/types.h>
#include "ring.cpm.h"

void Cpm_megacon_ack(CpmDestination);

void ring_fail()
{
  CmiPrintf("data corrupted in ring_hop.\n");
  exit(1);
}

static char s_HowdyDude[] = "Howdy, Dude.";
static char s_this[] = "this";
static char s_is[] = "is";
static char s_a[] = "a";
static char s_test[] = "test";
static char *data[4] = { s_this, s_is, s_a, s_test };

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
  Cpm_ring_hop(CpmSend(0), 1000, s_HowdyDude, 4, data);
}

void ring_moduleinit()
{
  CpmInitializeThisModule();
}
