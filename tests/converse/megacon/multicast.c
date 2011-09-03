#include <stdio.h>
#include <converse.h>

void Cpm_megacon_ack();

typedef struct bchare
{
  CmiGroup grp;
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

CpvDeclare(int, multicast_recv_idx);
CpvDeclare(int, multicast_reply_idx);

void multicast_recv(mesg m)
{
  if (m->magic != 0x12345678) {
    CmiPrintf("multicast failed.\n");
    exit(1);
  }
  CmiSetHandler(m, CpvAccess(multicast_reply_idx));
  CmiSyncSendAndFree(m->reply_pe, sizeof(struct mesg), m);
}

void multicast_start_cycle(bchare c)
{
  struct mesg m={{0},CmiMyPe(),c,0x12345678}; struct mesg *mp;
  switch (c->totalsent) {
  case 0:
    CmiSetHandler(&m, CpvAccess(multicast_recv_idx));
    m.reply_ptr = c; m.reply_pe = CmiMyPe(); m.magic = 0x12345678;
    CmiSyncMulticast(c->grp, sizeof(struct mesg),&m);
    c->totalsent++;
    break;
  case 1:
  case 2:
    mp = (mesg)CmiAlloc(sizeof(struct mesg));
    CmiSetHandler(mp, CpvAccess(multicast_recv_idx));
    mp->reply_ptr = c; mp->reply_pe = CmiMyPe();mp->magic = 0x12345678;
    CmiSyncMulticastAndFree(c->grp, sizeof(struct mesg), mp);
    c->totalsent++;
    break;
  case 3:
    free(c);
    Cpm_megacon_ack(CpmSend(0));
  }
}

void multicast_reply(mesg m)
{
  bchare c;
  if (m->magic != 0x12345678) {
    CmiPrintf("multicast failed.\n");
    exit(1);
  }
  c = m->reply_ptr;
  c->totalreplies++;
  if ((c->totalreplies % CmiNumPes())==0) multicast_start_cycle(c);
  CmiFree(m);
}

CmiGroup multicast_all()
{
  int i, *pes, npes; CmiGroup grp;
  npes = CmiNumPes();
  pes = (int*)malloc(npes*sizeof(int));
  for (i=0; i<npes; i++) pes[i] = i;
  grp = CmiEstablishGroup(CmiNumPes(), pes);
  free(pes);
  return grp;
}

void multicast_init(void)
{
  bchare c;
  c = (bchare)malloc(sizeof(struct bchare));
  c->grp = multicast_all();
  c->totalsent = 0;
  c->totalreplies = 0;
  multicast_start_cycle(c);
}

void multicast_moduleinit()
{
  CpvInitialize(int, multicast_recv_idx);
  CpvInitialize(int, multicast_reply_idx);
  CpvAccess(multicast_recv_idx) = CmiRegisterHandler((CmiHandler)multicast_recv);
  CpvAccess(multicast_reply_idx) = CmiRegisterHandler((CmiHandler)multicast_reply);
}






