#include "converse.h"
#include "commbench.h"

CpvStaticDeclare(int, numiter);
CpvStaticDeclare(int, nextidx);
CpvStaticDeclare(int, sync_handler);
CpvStaticDeclare(int, free_handler);
CpvStaticDeclare(int, enqueue_handler);
CpvStaticDeclare(double, starttime);

static struct testdata {
  int size;
  int numiter;
  double time;
} sizes[] = {
  {0,       1024,      0.0},
  {16,      1024,      0.0},
  {256,     1024,      0.0},
  {4096,    1024,      0.0},
  {65536,   1024,      0.0},
  {1048576, 1024,      0.0},
  {-1,      -1,        0.0},
};

static char *sync_outstr =
"[overhead] (%s) %le seconds per %d bytes\n"
;

static void print_results(char *func)
{
  int i=0;

  while(sizes[i].size != (-1)) {
    CmiPrintf(sync_outstr, func, sizes[i].time/sizes[i].numiter, sizes[i].size);
    i++;
  }
}

static void fill_message(void *msg, int size)
{
  int *imsg = (int *) msg;
  int start = CmiMsgHeaderSizeBytes/sizeof(int);
  int end = start+size/sizeof(int);
  int i;

  for(i=start;i<end;i++)
    imsg[i] = i;
}

static void check_message(void *msg, int size)
{
  int *imsg = (int *) msg;
  int start = CmiMsgHeaderSizeBytes/sizeof(int);
  int end = start+size/sizeof(int);
  int i;

  for(i=start;i<end;i++)
    if (imsg[i] != i)
      CmiAbort("[overhead] Message corrupted. Run megacon again !!\n");
}

static void sync_handler(void *msg)
{
  int idx = CpvAccess(nextidx);

  CpvAccess(numiter)++;
  if(CpvAccess(numiter)<sizes[idx].numiter) {
    CmiSyncSend(CmiMyPe(),CmiMsgHeaderSizeBytes+sizes[idx].size, msg);
    CmiFree(msg);
    return;
  } else {
    sizes[idx].time = CmiWallTimer() - CpvAccess(starttime);
    check_message(msg, sizes[idx].size);
    CmiFree(msg);
    idx++;
    CpvAccess(numiter) = 0;
    if(sizes[idx].size == (-1)) {
      print_results("CmiSyncSend");
      CpvAccess(nextidx) = 0;
      msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[0].size);
      fill_message(msg, sizes[0].size);
      CmiSetHandler(msg, CpvAccess(free_handler));
      CpvAccess(starttime) = CmiWallTimer();
      CmiSyncSendAndFree(CmiMyPe(), CmiMsgHeaderSizeBytes+sizes[0].size, msg);
      return;
    } else {
      CpvAccess(nextidx) = idx;
      msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[idx].size);
      fill_message(msg, sizes[idx].size);
      CmiSetHandler(msg, CpvAccess(sync_handler));
      CpvAccess(starttime) = CmiWallTimer();
      CmiSyncSend(CmiMyPe(), CmiMsgHeaderSizeBytes+sizes[idx].size, msg);
      CmiFree(msg);
    }
  }
}

static void free_handler(void *msg)
{
  int idx = CpvAccess(nextidx);

  CpvAccess(numiter)++;
  if(CpvAccess(numiter)<sizes[idx].numiter) {
    CmiSyncSendAndFree(CmiMyPe(),CmiMsgHeaderSizeBytes+sizes[idx].size, msg);
    return;
  } else {
    sizes[idx].time = CmiWallTimer() - CpvAccess(starttime);
    check_message(msg, sizes[idx].size);
    CmiFree(msg);
    CpvAccess(numiter) = 0;
    idx++;
    if(sizes[idx].size == (-1)) {
      print_results("CmiSyncSendAndFree");
      CpvAccess(nextidx) = 0;
      msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[0].size);
      fill_message(msg, sizes[0].size);
      CmiSetHandler(msg, CpvAccess(enqueue_handler));
      CpvAccess(starttime) = CmiWallTimer();
      CsdEnqueue(msg);
      return;
    } else {
      CpvAccess(nextidx) = idx;
      msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[idx].size);
      fill_message(msg, sizes[idx].size);
      CmiSetHandler(msg, CpvAccess(free_handler));
      CpvAccess(starttime) = CmiWallTimer();
      CmiSyncSendAndFree(CmiMyPe(), CmiMsgHeaderSizeBytes+sizes[idx].size, msg);
    }
  }
}

static void enqueue_handler(void *msg)
{
  int idx = CpvAccess(nextidx);
  EmptyMsg emsg;

  CpvAccess(numiter)++;
  if(CpvAccess(numiter)<sizes[idx].numiter) {
    CsdEnqueue(msg);
    return;
  } else {
    sizes[idx].time = CmiWallTimer() - CpvAccess(starttime);
    check_message(msg, sizes[idx].size);
    CmiFree(msg);
    idx++;
    CpvAccess(numiter) = 0;
    if(sizes[idx].size == (-1)) {
      print_results("CsdEnqueue");
      CmiSetHandler(&emsg, CpvAccess(ack_handler));
      CmiSyncSend(0, sizeof(EmptyMsg), &emsg);
      return;
    } else {
      CpvAccess(nextidx) = idx;
      msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[idx].size);
      fill_message(msg, sizes[idx].size);
      CmiSetHandler(msg, CpvAccess(enqueue_handler));
      CpvAccess(starttime) = CmiWallTimer();
      CsdEnqueue(msg);
    }
  }
}

void overhead_init(void)
{
  void *msg;

  msg = CmiAlloc(CmiMsgHeaderSizeBytes+sizes[0].size);
  fill_message(msg, sizes[0].size);
  CmiSetHandler(msg, CpvAccess(sync_handler));
  CpvAccess(starttime) = CmiWallTimer();
  CmiSyncSend(CmiMyPe(), CmiMsgHeaderSizeBytes+sizes[0].size, msg);
  CmiFree(msg);
}

void overhead_moduleinit(void)
{
  CpvInitialize(int, numiter);
  CpvInitialize(int, nextidx);
  CpvInitialize(double, starttime);
  CpvInitialize(int, sync_handler);
  CpvInitialize(int, free_handler);
  CpvInitialize(int, enqueue_handler);
  CpvAccess(numiter) = 0;
  CpvAccess(nextidx) = 0;
  CpvAccess(sync_handler) = CmiRegisterHandler((CmiHandler)sync_handler);
  CpvAccess(free_handler) = CmiRegisterHandler((CmiHandler)free_handler);
  CpvAccess(enqueue_handler) = CmiRegisterHandler((CmiHandler)enqueue_handler);
}
