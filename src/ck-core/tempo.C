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

void
TempoArray::ckTempoBarrier(void)
{
  if(thisIndex) {
    ckTempoSendElem(BARR_TAG, thisIndex, (void*) 0, 0, 0);
    ckTempoRecv(BARR_TAG, 0, (void*) 0, 0);
  } else {
     int i;
     for(i=1;i<numElements;i++)
       ckTempoRecv(BARR_TAG, (void *) 0, 0);
     for(i=1;i<numElements;i++)
       ckTempoSendElem(BARR_TAG, 0, (void *) 0, 0, i);
  }
}

void
TempoArray::ckTempoBcast(int sender, int tag, void *buffer, int buflen)
{
  if(sender) {
    int i;
    for(i=1;i<numElements;i++)
      ckTempoSendElem(tag, BCAST_TAG, buffer, buflen, i);
  } else
    ckTempoRecv(tag, BCAST_TAG, buffer, buflen);
}

static void doOp(int op, int type, int count, void *inbuf, void *outbuf)
{
  switch(type) {
    case TEMPO_FLOAT :
    {
      float *a, *b;
      a = (float *) inbuf;
      b = (float *) outbuf;
      for(int i=0; i<count; i++) {
        switch(op) {
          case TEMPO_MIN : if(b[i]>a[i]) b[i]=a[i]; break;
          case TEMPO_MAX : if(b[i]<a[i]) b[i]=a[i]; break;
          case TEMPO_SUM : b[i] += a[i]; break;
          case TEMPO_PROD :b[i] *= a[i]; break;
        }
      }
    }
    break;
    case TEMPO_INT   :
    {
      int *a, *b;
      a = (int *) inbuf;
      b = (int *) outbuf;
      for(int i=0; i<count; i++) {
        switch(op) {
          case TEMPO_MIN : if(b[i]>a[i]) b[i]=a[i]; break;
          case TEMPO_MAX : if(b[i]<a[i]) b[i]=a[i]; break;
          case TEMPO_SUM : b[i] += a[i]; break;
          case TEMPO_PROD :b[i] *= a[i]; break;
        }
      }
    }
    break;
    case TEMPO_DOUBLE:
    {
      double *a, *b;
      a = (double *) inbuf;
      b = (double *) outbuf;
      for(int i=0; i<count; i++) {
        switch(op) {
          case TEMPO_MIN : if(b[i]>a[i]) b[i]=a[i]; break;
          case TEMPO_MAX : if(b[i]<a[i]) b[i]=a[i]; break;
          case TEMPO_SUM : b[i] += a[i]; break;
          case TEMPO_PROD :b[i] *= a[i]; break;
        }
      }
    }
    break;
  }
}

void 
TempoArray::ckTempoReduce(int root, int op, void *inbuf, void *outbuf, 
                          int count, int type)
{
  int size = count;
  switch(type) {
    case TEMPO_FLOAT : size *= sizeof(float); break;
    case TEMPO_INT : size *= sizeof(int); break;
    case TEMPO_DOUBLE : size *= sizeof(double); break;
  }
  if(thisIndex==root) {
    memcpy(outbuf, inbuf, size);
    void *tbuf = malloc(size);
    for(int i=0; i<numElements-1; i++) {
      ckTempoRecv(REDUCE_TAG, tbuf, size);
      doOp(op, type, count, tbuf, outbuf);
    }
    free(tbuf);
  } else {
    ckTempoSendElem(REDUCE_TAG, inbuf, size, root);
  }
}

void 
TempoArray::ckTempoAllReduce(int op, void *inbuf, void *outbuf, 
                             int count, int type)
{
  ckTempoReduce(0, op, inbuf, outbuf, count, type);
  int size = count;
  switch(type) {
    case TEMPO_FLOAT : size *= sizeof(float); break;
    case TEMPO_INT : size *= sizeof(int); break;
    case TEMPO_DOUBLE : size *= sizeof(double); break;
  }
  ckTempoBcast(thisIndex==0, REDUCE_TAG, outbuf, size);
}

#include "tempo.def.h"
