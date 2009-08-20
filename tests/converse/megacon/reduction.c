#include <converse.h>

static int reduction_debug = 0;

#define DebugPrintf  if (reduction_debug) CmiPrintf

void Cpm_megacon_ack();

typedef struct _mesg {
  char head[CmiMsgHeaderSizeBytes];
  int sum;
} *mesg;

struct twoInts {
  int positive;
  int negative;
};

void pupTwoInts(pup_er p, void *data) {
  struct twoInts *d = (struct twoInts *)data;
  DebugPrintf("[%d] called pupTwoInts on %p (%d/%d)\n",CmiMyPe(),data,d->positive,d->negative);
  pup_int(p, &d->positive);
  pup_int(p, &d->negative);
}

void deleteTwoInts(void *data) {
  DebugPrintf("[%d] called deleteTwoInts %p (%d/%d)\n",CmiMyPe(),data,
      ((struct twoInts*)data)->positive,((struct twoInts*)data)->negative);
  free(data);
}

void *mergeTwoInts(int *size, void *data, void **remote, int count) {
  int i;
  struct twoInts *local = (struct twoInts *)data;
  DebugPrintf("[%d] called mergeTwoInts with local(%d/%d), %d remote(",CmiMyPe(),
      local->positive,local->negative,count);
  for (i=0; i<count; ++i) {
    struct twoInts *d = (struct twoInts *)remote[i];
    DebugPrintf("%d/%d,",d->positive,d->negative);
    local->positive += d->positive;
    local->negative += d->negative;
  }
  DebugPrintf(")\n");
  return data;
}

CpvDeclare(int, broadcast_msg_idx);
CpvDeclare(int, reduction_msg_idx);
CpvDeclare(int, broadcast_struct_idx);
CpvDeclare(int, reduction_struct_idx);

void * addMessage(int *size, void *data, void **remote, int count) {
  mesg msg = (mesg)data;
  int i;
  DebugPrintf("[%d] called addMessage with local(%d), %d remote(",CmiMyPe(),msg->sum,count);
  for (i=0; i<count; ++i) {
    DebugPrintf("%d,",((mesg)remote[i])->sum);
    msg->sum += ((mesg)remote[i])->sum;
  }
  DebugPrintf(")\n");
  return data;
}

void reduction_msg(mesg m) {
  int i, sum=0;
  CmiAssert(CmiMyPe() == 0);
  for (i=0; i<CmiNumPes(); ++i) sum += i+1;
  if (m->sum != sum) {
    CmiPrintf("Sum not matching: received %d, expecting %d\n", m->sum, sum);
    exit(1);
  }
  CmiSetHandler(m, CpvAccess(broadcast_struct_idx));
  CmiSyncBroadcastAllAndFree(sizeof(struct _mesg),m);
}

void broadcast_msg(mesg m) {
  m->sum = CmiMyPe()+1;
  CmiSetHandler(m, CpvAccess(reduction_msg_idx));
  CmiReduce(m, sizeof(struct _mesg), addMessage);
}

void reduction_struct(void *data) {
  int i, sum=0;
  struct twoInts *two = (struct twoInts *)data; 
  CmiAssert(CmiMyPe() == 0);
  for (i=0; i<CmiNumPes(); ++i) sum += i+1;
  if (two->positive != sum || two->negative != -2*sum) {
    CmiPrintf("Sum not matching: received %d/%d, expecting %d/%d\n",
        two->positive, two->negative, sum, -2*sum);
    exit(1);
  }
  Cpm_megacon_ack(CpmSend(0));
}

void broadcast_struct(mesg m) {
  struct twoInts *two = (struct twoInts*)malloc(sizeof(struct twoInts));
  CmiFree(m);
  DebugPrintf("[%d] allocated struct %p\n",CmiMyPe(),two);
  two->positive = CmiMyPe()+1;
  two->negative = -2*(CmiMyPe()+1);
  CmiReduceStruct(two, pupTwoInts, mergeTwoInts, reduction_struct, deleteTwoInts);
}

void reduction_init(void)
{
  mesg msg = (mesg)CmiAlloc(sizeof(struct _mesg));
  CmiSetHandler(msg, CpvAccess(broadcast_msg_idx));
  CmiSyncBroadcastAllAndFree(sizeof(struct _mesg),msg);
}

void reduction_moduleinit()
{
  CpvInitialize(int, broadcast_msg_idx);
  CpvInitialize(int, reduction_msg_idx);
  CpvInitialize(int, broadcast_struct_idx);
  CpvAccess(broadcast_msg_idx) = CmiRegisterHandler((CmiHandler)broadcast_msg);
  CpvAccess(reduction_msg_idx) = CmiRegisterHandler((CmiHandler)reduction_msg);
  CpvAccess(broadcast_struct_idx) = CmiRegisterHandler((CmiHandler)broadcast_struct);
}
