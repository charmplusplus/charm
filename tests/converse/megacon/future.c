#include <stdio.h>
#include <converse.h>
#include "future.cpm.h"

void Cpm_megacon_ack();

void future_fail()
{
  CmiPrintf("data corrupted in future test.\n");
  exit(1);
}

CpmInvokable future_sendback(Cfuture f)
{
  int i = CmiMyPe();
  CfutureSet(f, &i, sizeof(int));
}

void future_wait(Cfuture *futures)
{
  int i; int *val;
  for (i=0; i<CmiNumPes(); i++) {
    val = (int*)CfutureWait(futures[i]);
    if (*val != i) future_fail();
    CfutureDestroy(futures[i]);
  }
  Cpm_megacon_ack(CpmSend(0));
  CthFree(CthSelf());
  CthSuspend();
}

void future_init(void)
{
  int i; CthThread t;
  Cfuture *futures;
  futures=(Cfuture *)malloc(CmiNumPes()*sizeof(Cfuture));
  for (i=0; i<CmiNumPes(); i++) {
    futures[i] = CfutureCreate();
    Cpm_future_sendback(CpmSend(i), futures[i]);
  }
  t = CthCreate((CthVoidFn)future_wait, (void *)futures, 0);
  CthSetStrategyDefault(t);
  CthAwaken(t);
}

void future_moduleinit()
{
  CpmInitializeThisModule();
}
