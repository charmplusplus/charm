#include <converse.h>
#include "sm.h"

typedef struct SMMessageStruct *SMMessage;

struct SMMessageStruct
{
  char cmiheader[CmiMsgHeaderSizeBytes];
  int seqno, size, ntags;
  int tags[1];
};

CpvStaticDeclare(int, SMHandlerIndex);
CpvStaticDeclare(int *, SMSeqOut);
CpvStaticDeclare(int *, SMSeqIn);
CpvStaticDeclare(CmmTable, SMMessages);

void SMHandler(m)
SMMessage m;
{
  CmmPut(CpvAccess(SMMessages), m->ntags, m->tags, m);
}

void SMInit(argv)
char **argv;
{
  int *seqout, *seqin; int i;

  seqout = (int *)CmiAlloc(CmiNumPes()*sizeof(int));
  seqin  = (int *)CmiAlloc(CmiNumPes()*sizeof(int));
  for (i=0; i<CmiNumPes(); i++) seqout[i] = 0;
  for (i=0; i<CmiNumPes(); i++) seqin [i] = 0;

  CpvInitialize(int, SMHandlerIndex);
  CpvInitialize(int *, SMSeqOut);
  CpvInitialize(int *, SMSeqIn);
  CpvInitialize(CmmTable, SMMessages);

  CpvAccess(SMHandlerIndex) = CmiRegisterHandler(SMHandler);
  CpvAccess(SMSeqOut) = seqout;
  CpvAccess(SMSeqIn) = seqin;
  CpvAccess(SMMessages) = CmmNew();
}

void GeneralSend(pe, ntags, tags, buffer, buflen)
int pe, ntags;
int *tags;
void *buffer;
int buflen;
{
  int headsize, totsize, i; SMMessage msg;

  headsize = sizeof(struct SMMessageStruct) + (ntags*sizeof(int));
  headsize = ((headsize + 7) & (~7));
  totsize = headsize + buflen;
  msg = (SMMessage)CmiAlloc(totsize);
  CmiSetHandler(msg, CpvAccess(SMHandlerIndex));
  msg->seqno = (CpvAccess(SMSeqOut)[pe])++;
  msg->size = buflen;
  msg->ntags = ntags;
  for (i=0; i<ntags; i++) msg->tags[i] = tags[i];
  memcpy((((char *)msg)+headsize), buffer, buflen);
  CmiSyncSend(pe, totsize, msg);
}

int GeneralBroadcast(rootpe, ntags, tags, buffer, buflen, rtags)
int rootpe, ntags;
int *tags, *rtags;
void *buffer;
int buflen;
{
  if(CmiMyPe()==rootpe) {
    int headsize, totsize, i; SMMessage msg;

    headsize = sizeof(struct SMMessageStruct) + (ntags*sizeof(int));
    headsize = ((headsize + 7) & (~7));
    totsize = headsize + buflen;
    msg = (SMMessage)CmiAlloc(totsize);
    CmiSetHandler(msg, CpvAccess(SMHandlerIndex));
    msg->size = buflen;
    msg->ntags = ntags;
    for (i=0; i<ntags; i++) msg->tags[i] = tags[i];
    memcpy((((char *)msg)+headsize), buffer, buflen);
    CmiSyncBroadcast(totsize, msg);
    return buflen;
  } else {
    SMMessage msg;
    int headsize;
  
    while (1) {  
      msg = (SMMessage)CmmGet(CpvAccess(SMMessages), ntags, tags, rtags);
      if (msg) break;
    }
    if (msg->size > buflen) buflen = msg->size;
    headsize = sizeof(struct SMMessageStruct) + ((ntags-1)*sizeof(int));
    headsize = ((headsize + 7) & (~7));
    memcpy(buffer, ((char *)msg)+headsize, buflen);
    CmiFree(msg);
    return buflen;
  }
}

int GeneralRecv(ntags, tags, buffer, buflen, rtags)
int ntags;
int *tags;
void *buffer;
int buflen;
int *rtags;
{
  SMMessage msg;
  int headsize;

  while (1) {  
    CsdScheduler(0);
    msg = (SMMessage)CmmGet(CpvAccess(SMMessages), ntags, tags, rtags);
    if (msg) break;
  }
  
  if (msg->size > buflen) buflen = msg->size;
  headsize = sizeof(struct SMMessageStruct) + ((ntags-1)*sizeof(int));
  headsize = ((headsize + 7) & (~7));
  memcpy(buffer, ((char *)msg)+headsize, buflen);
  CmiFree(msg);
  return buflen;
}

