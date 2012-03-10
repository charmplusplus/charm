#include "Communicate.h"
#include "MStream.h"
#include <string.h>

MIStream::MIStream(Communicate *c, int p, int t)
{
  cobj = c;
  PE = p;
  tag = t;
  msg = (StreamMessage *) 0;
}

MIStream::~MIStream()
{
  if(msg!=0)
    CmiFree(msg);
}

MOStream::MOStream(Communicate *c, int p, int t, unsigned int size)
{
  cobj = c;
  PE = p;
  tag = t;
  bufLen = size;
  msgBuf = (StreamMessage *)CmiAlloc(sizeof(StreamMessage)+size);
  msgBuf->PE = CmiMyPe();
  msgBuf->tag = tag;
  msgBuf->len = 0;
  msgBuf->isLast = 0;
}

MOStream::~MOStream()
{
  if(msgBuf != 0)
    end();
}

MIStream *MIStream::Get(char *buf, int len)
{
  while(len) {
    if(msg==0) {
      msg = (StreamMessage *) cobj->getMessage(PE, tag);
      currentPos = 0;
    }
    if(currentPos+len <= msg->len) {
      memcpy(buf, &(msg->data[currentPos]), len);
      currentPos += len;
      len = 0;
    } else {
      int b = msg->len-currentPos;
      memcpy(buf, &(msg->data[currentPos]), b);
      len -= b;
      buf += b;
      currentPos += b;
    }
    if(currentPos == msg->len) {
      CmiFree(msg);
      msg = 0;
    }
  }
  return this;
}

MOStream *MOStream::Put(char *buf, int len)
{
  while(len) {
    if(msgBuf->len + len <= bufLen) {
      memcpy(&(msgBuf->data[msgBuf->len]), buf, len);
      msgBuf->len += len;
      len = 0;
    } else {
      int b = bufLen - msgBuf->len;
      memcpy(&(msgBuf->data[msgBuf->len]), buf, b);
      msgBuf->len = bufLen;
      cobj->sendMessage(PE, (void *)msgBuf, bufLen+sizeof(StreamMessage));
      msgBuf->len = 0;
      msgBuf->isLast = 0;
      len -= b;
      buf += b;
    }
  }
  return this;
}

void MOStream::end(void)
{
  msgBuf->isLast = 1;
  cobj->sendMessage(PE,(void*)msgBuf,msgBuf->len+sizeof(StreamMessage));
  msgBuf->len = 0;
  msgBuf->isLast = 0;
}

