#include <stdio.h>
#include <converse.h>

typedef struct node_info
{
  int pe;
  int rank;
  int host;
}
*node_info;

typedef struct nodenum_chare
{
  int countdown; CthThread pending;
  struct node_info info[64];
}
*nodenum_chare;

CpmDeclareSimple(nodenum_chare);
#define CpmPack_nodenum_chare(x) (0)
#define CpmUnpack_nodenum_chare(x) (0)
CsvDeclare(int, myhost);

void Cpm_megacon_ack();

#include "nodenum.cpm.h"

CpmInvokable nodenum_ack(nodenum_chare c)
{
  c->countdown--;
  if ((c->countdown==0)&&(c->pending))
    CthAwaken(c->pending);
}

CpmInvokable nodenum_reply(nodenum_chare c, int pe, int rank, int host)
{
  c->info[pe].pe   = pe; 
  c->info[pe].rank = rank;
  c->info[pe].host = host;
  nodenum_ack(c);
}

CpmInvokable nodenum_initialize_myhost(nodenum_chare c)
{
  if (CmiMyRank()==0) CsvAccess(myhost) = CmiMyPe();
  Cpm_nodenum_reply(CpmSend(0), c, 0, 0, 0);
}

CpmInvokable nodenum_collect_info(nodenum_chare c)
{
  if ((CmiMyRank()==0) && (CsvAccess(myhost)!=CmiMyPe())) {
    CmiPrintf("failure in nodenum-test #1234\n");
    exit(1);
  }
  Cpm_nodenum_reply(CpmSend(0),c,CmiMyPe(),CmiMyRank(),CsvAccess(myhost));
}

CpmInvokable nodenum_control()
{
  struct nodenum_chare c;
  int i, npe; node_info curr, prev;
  npe = CmiNumPes();

  /* gather the processor/rank/host table */
  Cpm_nodenum_initialize_myhost(CpmSend(CpmALL), &c);
  c.countdown = CmiNumPes(); c.pending = CthSelf(); CthSuspend();
  Cpm_nodenum_collect_info(CpmSend(CpmALL), &c);
  c.countdown = CmiNumPes(); c.pending = CthSelf(); CthSuspend();
  
  /* check that the processor/host/rank table contains reasonable values */
  if ((c.info[0].host != 0)||(c.info[0].rank != 0)||(c.info[0].pe != 0))
    goto badnum;
  for (i=1; i<npe; i++) {
    curr = &(c.info[i]);
    prev = &(c.info[i-1]);
    if (curr->host == prev->host) {
      if (curr->rank != prev->rank + 1)	goto badnum;
    } else {
      if (curr->host != curr->pe) goto badnum;
      if (curr->rank != 0) goto badnum;
    }
  }
  
  Cpm_megacon_ack(CpmSend(0));
  return;
badnum:
  CmiPrintf("nodenum: error in processor node/rank numbering system.\n");
  exit(1);
}

void nodenum_init()
{
  if (CmiNumPes() > 64) {
    CmiPrintf("skipping nodenum test (code only works on 64 PE's or less)\n");
    Cpm_megacon_ack(CpmSend(0));
  } else Cpm_nodenum_control(CpmMakeThread(0));
}

void nodenum_moduleinit()
{
  CpmInitializeThisModule();
}
