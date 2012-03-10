#include <converse.h>
#include "simplemsg.h"

typedef struct CsmMessageStruct *CsmMessage;

struct CsmMessageStruct
{
  char cmiheader[CmiMsgHeaderSizeBytes];
  int seqno, size, ntags;
  int tags[1];
};

CpvStaticDeclare(int, CsmHandlerIndex);
CpvStaticDeclare(int *, CsmSeqOut);
CpvStaticDeclare(int *, CsmSeqIn);
CpvStaticDeclare(CmmTable, CsmMessages);
CpvStaticDeclare(CmmTable, CsmSleepers);

void CsmHandler(m)
CsmMessage m;
{
  CthThread t;
  CmmPut(CpvAccess(CsmMessages), m->ntags, m->tags, m);
  t = (CthThread)CmmGet(CpvAccess(CsmSleepers), m->ntags, m->tags, (int *)0);
  if (t) CthAwaken(t);
}

void CsmInit(argv)
char **argv;
{
  int *seqout, *seqin; int i;

  seqout = (int *)CmiAlloc(CmiNumPes()*sizeof(int));
  seqin  = (int *)CmiAlloc(CmiNumPes()*sizeof(int));
  for (i=0; i<CmiNumPes(); i++) seqout[i] = 0;
  for (i=0; i<CmiNumPes(); i++) seqin [i] = 0;

  CpvInitialize(int, CsmHandlerIndex);
  CpvInitialize(int *, CsmSeqOut);
  CpvInitialize(int *, CsmSeqIn);
  CpvInitialize(CmmTable, CsmMessages);
  CpvInitialize(CmmTable, CsmSleepers);

  CpvAccess(CsmHandlerIndex) = CmiRegisterHandler(CsmHandler);
  CpvAccess(CsmSeqOut) = seqout;
  CpvAccess(CsmSeqIn)  = seqin;
  CpvAccess(CsmMessages) = CmmNew();
  CpvAccess(CsmSleepers) = CmmNew();
}

void CsmTVSend(pe, ntags, tags, buffer, buflen)
int pe, ntags;
int *tags;
void *buffer;
int buflen;
{
  int headsize, totsize, i; CsmMessage msg;

  headsize = sizeof(struct CsmMessageStruct) + (ntags*sizeof(int));
  headsize = ((headsize + 7) & (~7));
  totsize = headsize + buflen;
  msg = (CsmMessage)CmiAlloc(totsize);
  CmiSetHandler(msg, CpvAccess(CsmHandlerIndex));
  msg->seqno = (CpvAccess(CsmSeqOut)[pe])++;
  msg->size = buflen;
  msg->ntags = ntags;
  for (i=0; i<ntags; i++) msg->tags[i] = tags[i];
  memcpy((((char *)msg)+headsize), buffer, buflen);
  CmiSyncSend(pe, totsize, msg);
}

void CsmTVRecv(ntags, tags, buffer, buflen, rtags)
int ntags;
int *tags;
void *buffer;
int buflen;
int *rtags;
{
  CsmMessage msg; CthThread self;
  int headsize;

  while (1) {  
    msg = (CsmMessage)CmmGet(CpvAccess(CsmMessages), ntags, tags, rtags);
    if (msg) break;
    self = CthSelf();
    CmmPut(CpvAccess(CsmSleepers), ntags, tags, self);
    CthSuspend();
  }
  
  if (msg->size > buflen) buflen = msg->size;
  headsize = sizeof(struct CsmMessageStruct) + ((ntags-1)*sizeof(int));
  headsize = ((headsize + 7) & (~7));
  memcpy(buffer, ((char *)msg)+headsize, buflen);
  CmiFree(msg);
  return;
}

