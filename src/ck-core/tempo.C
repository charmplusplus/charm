#include "charm++.h"
#include "tempo.h"

void *TempoMessage::pack(TempoMessage *in)
{
  int len = in->length + 2 * sizeof(int) + sizeof(ArrayMessage);
  void *themsg = CkAllocBuffer(in, len);
  *((int *)themsg) = in->tag;
  *((int *)themsg + 1) = in->length;
  memcpy(((int *)themsg + 2), in->data, in->length); 
  char *tempmsg = ((char*)((int *)themsg+2))+in->length;
  memcpy(tempmsg, (ArrayMessage *) in, sizeof(ArrayMessage));
  delete in;
  return(themsg);
}

TempoMessage *
TempoMessage::unpack(void *in)
{
  int tag = *((int *) in);
  int length = *(((int *) in) + 1);
  void * data = (void *) (((int *) in) + 2);
  void *buf = CkAllocBuffer(in, sizeof(TempoMessage));
  TempoMessage *msg = new (buf) TempoMessage(tag, length, data);
  char *tempmsg = (char *) (((int *) in) + 2) + length;
  memcpy((ArrayMessage *) msg, tempmsg, sizeof(ArrayMessage));
  CkFreeMsg(in);
  return msg;
}

Tempo::Tempo(void)
{
  tempoMessages = CmmNew();
  thread_id = CthSelf(); 
  sleeping = 0;
}

void Tempo::ckTempoRecv(int tag, void *buffer, int buflen)
{
  TempoMessage *msg = 0;
  while(1) {
    sleeping = 0;
    msg = (TempoMessage *) CmmGet(tempoMessages, 1, &tag, 0);
    if (msg) break;
    sleeping = 1;
    thread_id = CthSelf(); 
    CthSuspend();
  }
  if (msg->length < buflen) 
    buflen = msg->length;
  memcpy(buffer, msg->data, buflen);
  delete msg; 
}

void 
Tempo::ckTempoSend(CkChareID chareid, int tag, void *buffer, int buflen)
{
  TempoMessage *msg = new TempoMessage(tag, buflen, buffer);
  CProxy_TempoChare ptc(chareid);
  ptc.tempoGeneric(msg);
}

void Tempo::tempoGeneric(TempoMessage *themsg)
{
  CmmPut(tempoMessages, 1, &(themsg->tag), themsg); 
  if (sleeping)
    CthAwaken(thread_id);
}

int Tempo::ckTempoProbe(int tag)
{
  return (CmmProbe(tempoMessages, 1, &tag, 0)!=0);
}

void 
TempoGroup::ckTempoBcast(int sender, int bocid, int tag, 
                         void *buffer, int buflen)
{
  if (!sender) return;
  TempoMessage *msg = new TempoMessage(tag, buflen, buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg.tempoGeneric(msg);
}

void 
TempoGroup::ckTempoSendBranch(int bocid, int tag, void *buffer, int buflen, 
                              int processor)
{
  TempoMessage *msg = new TempoMessage(tag, buflen, buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg.tempoGeneric(msg, processor);
}

void 
TempoGroup::ckTempoBcast(int sender, int tag, void *buffer, int buflen)
{
  TempoGroup::ckTempoBcast(sender, thisgroup, tag, buffer, buflen);
}

void 
TempoArray::ckTempoSendElem(CkAID aid,int tag,void *buffer,int buflen,int idx)
{
  TempoMessage *msg = new TempoMessage(tag, buflen, buffer);
  CProxy_TempoArray pta(aid);
  pta[idx].tempoGeneric(msg);
}

void 
TempoArray::ckTempoSendElem(int tag, void *buffer, int buflen, int idx)
{
  TempoArray::ckTempoSendElem(thisAID, tag, buffer, buflen, idx);
}

#include "tempo.def.h"
