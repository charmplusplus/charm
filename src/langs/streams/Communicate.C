#include <string.h>
#include "Communicate.h"
#include "MStream.h"

CpvStaticDeclare(CmmTable, CsmMessages);

static void CsmHandler(void *msg)
{
  // get start of user message
  int *m = (int *) ((char *)msg+CmiMsgHeaderSizeBytes);
  // sending node  & tag act as tags
  CmmPut(CpvAccess(CsmMessages), 2, m, msg);
}

Communicate::Communicate(void) 
{
  CpvInitialize(CmmTable, CsmMessages);
  CsmHandlerIndex = CmiRegisterHandler((CmiHandler) CsmHandler);
  CpvAccess(CsmMessages) = CmmNew();
}


Communicate::~Communicate(void) 
{
  // do nothing
}

MIStream *Communicate::newInputStream(int PE, int tag)
{
  MIStream *st = new MIStream(this, PE, tag);
  return st;
}

MOStream *Communicate::newOutputStream(int PE, int tag, unsigned int bufSize)
{
  MOStream *st = new MOStream(this, PE, tag, bufSize);
  return st;
}

void *Communicate::getMessage(int PE, int tag)
{
  int itag[2], rtag[2];
  void *msg;

  itag[0] = (PE==(-1)) ? (CmmWildCard) : PE;
  itag[1] = (tag==(-1)) ? (CmmWildCard) : tag;
  while((msg=CmmGet(CpvAccess(CsmMessages),2,itag,rtag))==0) {
    CmiDeliverMsgs(0);
  }
  return msg;
}

void Communicate::sendMessage(int PE, void *msg, int size)
{
  CmiSetHandler(msg, CsmHandlerIndex);
  switch(PE) {
    case ALL:
      CmiSyncBroadcastAll(size, (char *)msg);
      break;
    case ALLBUTME:
      CmiSyncBroadcast(size, (char *)msg);
      break;
    default:
      CmiSyncSend(PE, size, (char *)msg);
      break;
  }
}
