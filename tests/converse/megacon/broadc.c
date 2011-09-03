#include <stdio.h>
#include <converse.h>

void Cpm_megacon_ack();

typedef struct bchare
{
  int totalsent;
  int totalreplies;
}
*bchare;

typedef struct mesg
{
  char head[CmiMsgHeaderSizeBytes];
  int reply_pe;
  bchare reply_ptr;
  int magic;
}
*mesg;

CpvDeclare(int, broadc_recv_idx);
CpvDeclare(int, broadc_reply_idx);

void broadc_recv(mesg m)
{
  if (m->magic != 0x12345678) {
    CmiPrintf("broadc failed.\n");
    exit(1);
  }
  CmiSetHandler(m, CpvAccess(broadc_reply_idx));
  CmiSyncSendAndFree(m->reply_pe, sizeof(struct mesg), m);
}

void broadc_start_cycle(bchare c)
{
  struct mesg m={{0},CmiMyPe(),c,0x12345678}; struct mesg *mp; CmiCommHandle h;
  switch (c->totalsent) {
  case 0:
    CmiSetHandler(&m, CpvAccess(broadc_recv_idx));
    CmiSyncBroadcastAll(sizeof(struct mesg),&m);
    c->totalsent++;
    break;
  case 1:
  case 2:
    mp = (mesg)CmiAlloc(sizeof(struct mesg));
    CmiSetHandler(mp, CpvAccess(broadc_recv_idx));
    mp->reply_ptr = c; mp->reply_pe = CmiMyPe(); mp->magic = 0x12345678;
    CmiSyncBroadcastAllAndFree(sizeof(struct mesg),mp);
    c->totalsent++;
    break;
  case 3:
    free(c);
    Cpm_megacon_ack(CpmSend(0));
  }
}

void broadc_reply(mesg m)
{
  bchare c;
  if (m->magic != 0x12345678) {
    CmiPrintf("broadc failed.\n");
    exit(1);
  }
  c = m->reply_ptr;
  c->totalreplies++;
  if ((c->totalreplies % CmiNumPes())==0) broadc_start_cycle(c);
  CmiFree(m);
}

void broadc_init(void)
{
  bchare c = (bchare)malloc(sizeof(struct bchare));
  c->totalsent = 0;
  c->totalreplies = 0;
  broadc_start_cycle(c);
}

void broadc_moduleinit()
{
  CpvInitialize(int, broadc_recv_idx);
  CpvInitialize(int, broadc_reply_idx);
  CpvAccess(broadc_recv_idx) = CmiRegisterHandler((CmiHandler)broadc_recv);
  CpvAccess(broadc_reply_idx) = CmiRegisterHandler((CmiHandler)broadc_reply);
}
