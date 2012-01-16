#include <stdio.h>
#include <converse.h>
#define entries    10
void Cpm_megacon_ack();

typedef struct
{
  char core[CmiMsgHeaderSizeBytes]; 
  int hops, ringno;
  int data[10];
} ringmsg;

CpvDeclare(int, ringsimple_hop_index);

void ringsimple_fail()
{
  CmiPrintf("data corrupted in ringsimple_hop.\n");
  exit(1);
}

void ringsimple_hop(ringmsg *msg)
{
  int thispe = CmiMyPe();
  int nextpe = (thispe+1) % CmiNumPes();
  // CmiPrintf("[%d] ringsimple #%d hop send to %d hop: %d\n", thispe, msg->ringno, nextpe, msg->hops);
  int i;
  for (i=0; i<10; i++)
    if (msg->data[i] != i) ringsimple_fail();
  if (msg->hops) {
    msg->hops--;
    CmiSyncSendAndFree(nextpe, sizeof(ringmsg), msg);
  } else {
    Cpm_megacon_ack(CpmSend(0));
    CmiFree(msg);
  }
}

void ringsimple_init(void)
{
  int i; ringmsg msg={{0},1000,0,{0}};
  for (i=0; i<10; i++) msg.data[i] = i;
  CmiSetHandler(&msg, CpvAccess(ringsimple_hop_index));
  for (i=0; i<entries; i++) {
    msg.ringno = i;
    CmiSyncSend(0, sizeof(ringmsg), &msg);
  }
}

void ringsimple_moduleinit()
{
  CpvInitialize(int, ringsimple_hop_index);
  CpvAccess(ringsimple_hop_index) = CmiRegisterHandler((CmiHandler)ringsimple_hop);
}



