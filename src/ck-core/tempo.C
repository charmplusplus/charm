#include "charm++.h"
#include "tempo.h"

// static
void *
TempoMessage::pack(TempoMessage *in)
{
  int len = in->length + 3*sizeof(int) + sizeof(ArrayMessage);
  void *themsg = CkAllocBuffer(in, len);
  *((int *)themsg) = in->tag1;
  *((int *)themsg + 1) = in->tag2;
  *((int *)themsg + 2) = in->length;
  memcpy(((int *)themsg + 3), in->data, in->length); 
  char *tempmsg = ((char*)((int *)themsg+3))+in->length;
  memcpy(tempmsg, (ArrayMessage *) in, sizeof(ArrayMessage));
  delete in;
  return(themsg);
}

// static
TempoMessage *
TempoMessage::unpack(void *in)
{
  int tag1 = *((int *) in);
  int tag2 = *(((int *) in)+1);
  int length = *(((int *) in) + 2);
  void * data = (void *) (((int *) in) + 3);
  void *buf = CkAllocBuffer(in, sizeof(TempoMessage));
  TempoMessage *msg = new (buf) TempoMessage(tag1, tag2, length, data);
  char *tempmsg = (char *) (((int *) in) + 3) + length;
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

void Tempo::ckTempoRecv(int tag1, int tag2, void *buffer, int buflen)
{
  int tags[2];
  TempoMessage *msg = 0;
  while(1) {
    sleeping = 0;
    tags[0] = tag1; tags[1] = tag2;
    msg = (TempoMessage *) CmmGet(tempoMessages, 2, tags, 0);
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

void Tempo::ckTempoRecv(int tag, void *buffer, int buflen)
{
  ckTempoRecv(tag, TEMPO_ANY, buffer, buflen);
}

// static
void 
Tempo::ckTempoSend(int tag1,int tag2,void *buffer,int buflen, CkChareID cid)
{
  TempoMessage *msg = new TempoMessage(tag1, tag2, buflen, buffer);
  CProxy_TempoChare ptc(cid);
  ptc.tempoGeneric(msg);
}

// static
void 
Tempo::ckTempoSend(int tag, void *buffer, int buflen, CkChareID cid)
{
  ckTempoSend(tag, TEMPO_ANY, buffer, buflen, cid);
}

void Tempo::tempoGeneric(TempoMessage *themsg)
{
  int tags[2];
  tags[0] = themsg->tag1; tags[1] = themsg->tag2;
  CmmPut(tempoMessages, 2, tags, themsg); 
  if (sleeping)
    CthAwaken(thread_id);
}

int Tempo::ckTempoProbe(int tag1, int tag2)
{
  int tags[2];
  tags[0] = tag1; tags[1] = tag2;
  return (CmmProbe(tempoMessages, 2, tags, 0)!=0);
}

int Tempo::ckTempoProbe(int tag)
{
  return ckTempoProbe(tag, TEMPO_ANY);
}

// static
void 
TempoGroup::ckTempoBcast(int tag, void *buffer, int buflen, int bocid)
{
  TempoMessage *msg = new TempoMessage(tag, BCAST_TAG, buflen, buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg.tempoGeneric(msg);
}

// static
void 
TempoGroup::ckTempoSendBranch(int tag1, int tag2, void *buffer, int buflen, 
                              int bocid, int processor)
{
  TempoMessage *msg = new TempoMessage(tag1, tag2, buflen, buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg.tempoGeneric(msg, processor);
}

// static
void 
TempoGroup::ckTempoSendBranch(int tag, void *buffer, int buflen, 
                              int bocid, int processor)
{
  ckTempoSendBranch(tag, TEMPO_ANY, buffer, buflen, bocid, processor);
}

void 
TempoGroup::ckTempoSendBranch(int tag1, int tag2, void *buffer, 
                              int buflen, int processor)
{
  ckTempoSendBranch(tag1, tag2, buffer, buflen, thisgroup, processor);
}

void 
TempoGroup::ckTempoSendBranch(int tag, void *buffer, int buflen, int processor)
{
  ckTempoSendBranch(tag, TEMPO_ANY, buffer, buflen, processor);
}

void 
TempoGroup::ckTempoBcast(int sender, int tag, void *buffer, int buflen)
{
  if(sender)
    TempoGroup::ckTempoBcast(tag, buffer, buflen, thisgroup);
  ckTempoRecv(tag, BCAST_TAG, buffer, buflen);
}

// static
void 
TempoArray::ckTempoSendElem(int tag1, int tag2, void *buffer, int buflen,
                            CkAID aid, int idx)
{
  TempoMessage *msg = new TempoMessage(tag1, tag2, buflen, buffer);
  CProxy_TempoArray pta(aid);
  pta[idx].tempoGeneric(msg);
}

// static
void 
TempoArray::ckTempoSendElem(int tag,void *buffer,int buflen,CkAID aid, int idx)
{
  ckTempoSendElem(tag, TEMPO_ANY, buffer, buflen, aid, idx);
}

void 
TempoArray::ckTempoSendElem(int tag1, int tag2, void *buffer, int buflen, 
                            int idx)
{
  ckTempoSendElem(tag1, tag2, buffer, buflen, thisAID, idx);
}

void 
TempoArray::ckTempoSendElem(int tag, void *buffer, int buflen, int idx)
{
  ckTempoSendElem(tag, TEMPO_ANY, buffer, buflen, idx);
}

#include "tempo.def.h"
