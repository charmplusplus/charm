#include <stdio.h>
#include <converse.h>

void Cpm_megacon_ack();

typedef struct priotest_chare
{
  int numreceived;
  int totalexpected;
}
*priotest_chare;
CpmDeclareSimple(priotest_chare);
#define CpmPack_priotest_chare(p) (0)
#define CpmUnpack_priotest_chare(p) (0)

#include "priotest.cpm.h"

CpmInvokable priotest_bink(priotest_chare c, int n)
{
  if (n != c->numreceived) {
    CmiError("priotest: message received in wrong order.\n");
    exit(1);
  }
  c->numreceived++;
  if (c->numreceived == c->totalexpected) {
    CmiFree(c);
    Cpm_megacon_ack(CpmSend(0));
  }
}

CpmInvokable priotest_send()
{
  int me = CmiMyPe();
  priotest_chare c = (priotest_chare)CmiAlloc(sizeof(struct priotest_chare));
  c->numreceived = 0;
  c->totalexpected = 8;
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 1), c, 1);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 5), c, 5);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 2), c, 2);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 4), c, 4);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 3), c, 3);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 6), c, 6);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 0), c, 0);
  Cpm_priotest_bink(CpmEnqueueIFIFO(me, 7), c, 7);
}

void priotest_init()
{
  Cpm_priotest_send(CpmSend(CpmALL));
}

void priotest_moduleinit()
{
  CpmInitializeThisModule();
}
