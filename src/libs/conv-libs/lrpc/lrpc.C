#include "lrpc.h"

#define NORMAL 1
#define QUICK  2
#define ASYNC  3
#define RETVAL 4

struct lrpc_hdr {
  char header[CmiMsgHeaderSizeBytes];
  int mype;
  int lrpc_type;
  int funcnum;
  int prio, stksize;
  CthThread me;
  int sizeout;
  int *out;
};

CpvDeclare(int, hndl);

struct threadArgs {
  int funcnum;
  int *in, *out;
  CthThread waiting;
};

void threadWrapper(struct threadArgs *args)
{
  CmiHandlerFunction(args->funcnum)(in, out);
  CthAwaken(args->waiting);
  CthFree(CthSelf());
}

void internal_handler(void *msg)
{
  struct lrpc_hdr *hdr = (struct lrpc_hdr *) msg;
  struct lrpc_hdr hdrout;
  int *out;
  int sizes[2];
  char *msgs[2];
  struct threadArgs args;
  CthThread th;

  switch(hdr->lrpc_type) {
    case NORMAL:
      args.funcnum = hdr->funcnum;
      args.in = (hdr+1);
      out = (int *) CmiAlloc(sizeof(int) + hdr->sizeout);
      *out = hdr->sizeout;
      args.out = out;
      args.waiting = CthSelf();
      th = CthCreate(threadWrapper, &args, hdr->stksize);
      CthAwaken(th);
      CthSuspend();
      CmiSetHandler(&hdrout, CpvAccess(hndl));
      hdrout.lrpc_type = RETVAL;
      hdrout.out = hdr.out;
      hdrout.me = hdr.me;
      sizes[0] = sizeof(struct lrpc_hdr); msgs[0] = &hdrout;
      sizes[1] = (*out) + sizeof(int); msgs[1] = out;
      CmiSyncVectorSend(hdr.mype, 2, sizes, msgs);
      CmiFree(msg);
      CmiFree(out);
      return;
    case QUICK:
      out = (int *) CmiAlloc(sizeof(int) + hdr->sizeout);
      *out = hdr->sizeout;
      CmiHandlerFunction(hdr->funcnum)(hdr+1, out);
      CmiSetHandler(&hdrout, CpvAccess(hndl));
      hdrout.lrpc_type = RETVAL;
      hdrout.out = hdr.out;
      hdrout.me = hdr.me;
      sizes[0] = sizeof(struct lrpc_hdr); msgs[0] = &hdrout;
      sizes[1] = (*out) + sizeof(int); msgs[1] = out;
      CmiSyncVectorSend(hdr.mype, 2, sizes, msgs);
      CmiFree(msg);
      CmiFree(out);
      return;
    case ASYNC:
      CmiHandlerFunction(hdr->funcnum)(hdr+1);
      CmiFree(msg);
      return;
    case RETVAL:
      memcpy(hdr->out, hdr+1, *(hdr->out));
      CthAwaken(hdr->me);
      CmiFree(msg);
      return;
  }
}

void lrpc_init(void)
{
  CpvInitialize(int, hndl);
  CpvAccess(hndl) = CmiRegisterHandler(internal_handler);
}

int register_lrpc(lrpc_handler func)
{
  return  CmiRegisterHandler(func);
}

void lrpc(int node, int funcnum, int prio, int stksiz, int *in, int *out)
{
  struct lrpc_hdr hdr;
  int sizes[2];
  char *msgs[2];
  CmiSetHandler(&hdr, CpvAccess(hndl));
  hdr.mype = CmiMyPe();
  hdr.lrpc_type = NORMAL;
  hdr.funcnum = funcnum;
  hdr.prio = prio;
  hdr.stksize = stksiz;
  hdr.me = CthSelf();
  hdr.out = out;
  hdr.sizeout = *out;
  sizes[0] = sizeof(struct lrpc_hdr); msgs[0] = &hdr;
  sizes[1] = (*in) + sizeof(int); msgs[1] = in;
  CmiSyncVectorSend(node, 2, sizes, msgs);
  CthSuspend(); 
}

void quick_lrpc(int node, int funcnum, void *in, void *out)
{
  struct lrpc_hdr hdr;
  int sizes[2];
  char *msgs[2];
  CmiSetHandler(&hdr, CpvAccess(hndl));
  hdr.mype = CmiMyPe();
  hdr.lrpc_type = QUICK;
  hdr.funcnum = funcnum;
  hdr.me = CthSelf();
  hdr.out = out;
  hdr.sizeout = *out;
  sizes[0] = sizeof(struct lrpc_hdr); msgs[0] = &hdr;
  sizes[1] = (*in) + sizeof(int); msgs[1] = in;
  CmiSyncVectorSend(node, 2, sizes, msgs);
  CthSuspend(); 
}

void async_lrpc(int node, int funcnum, void *in)
{
  struct lrpc_hdr hdr;
  int sizes[2];
  char *msgs[2];
  CmiSetHandler(&hdr, CpvAccess(hndl));
  hdr.lrpc_type = ASYNC;
  hdr.funcnum = funcnum;
  sizes[0] = sizeof(struct lrpc_hdr); msgs[0] = &hdr;
  sizes[1] = (*in) + sizeof(int); msgs[1] = in;
  CmiSyncVectorSend(node, 2, sizes, msgs);
}

