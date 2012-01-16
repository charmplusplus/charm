#include <stdio.h>
#include <time.h>
#include <converse.h>

CpvDeclare(int, bigmsg_index);

#define CmiMsgHeaderSizeInts \
    ((CmiMsgHeaderSizeBytes+sizeof(int)-1)/sizeof(int))

void Cpm_megacon_ack();

void bigmsg_handler(void *vmsg)
{
  int i, next;
  int *msg = vmsg;
  if (CmiMyPe()==0) {
    for (i=CmiMsgHeaderSizeInts; i<250000; i++) {
      if (msg[i] != i) {
	CmiPrintf("Failure in bigmsg test, data corrupted.\n");
	exit(1);
      }
    }
    CmiFree(msg);
    Cpm_megacon_ack(CpmSend(0));
  } else {
    next = (CmiMyPe()+1) % CmiNumPes();
    CmiSyncSendAndFree(next, 250000*sizeof(int), msg);
  }
}

void bigmsg_init()
{
  int i, *msg;
  if (CmiNumPes()<2) {
    CmiPrintf("note: bigmsg requires at least 2 processors, skipping test.\n");
    Cpm_megacon_ack(CpmSend(0));
  } else {
    msg = CmiAlloc(250000 * sizeof(int));
    for (i=CmiMsgHeaderSizeInts; i<250000; i++) msg[i] = i;
    CmiSetHandler(msg, CpvAccess(bigmsg_index));
    CmiSyncSendAndFree(1, 250000 * sizeof(int), msg);
  }
}

void bigmsg_moduleinit()
{
  CpvInitialize(int, bigmsg_index);
  CpvAccess(bigmsg_index) = CmiRegisterHandler(bigmsg_handler);
}
