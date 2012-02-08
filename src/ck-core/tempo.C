#include "charm++.h"
#include "tempo.h"

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
  TempoMessage *msg = new (&buflen, 0) TempoMessage(tag1, tag2, buflen, buffer);
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
  if (sleeping) {
    sleeping = 0;
    CthAwaken(thread_id);
  }
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
TempoGroup::ckTempoBcast(int tag, void *buffer, int buflen, CkGroupID bocid)
{
  TempoMessage *msg = new (&buflen,0) TempoMessage(tag,BCAST_TAG,buflen,buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg.tempoGeneric(msg);
}

// static
void 
TempoGroup::ckTempoSendBranch(int tag1, int tag2, void *buffer, int buflen, 
                              CkGroupID bocid, int processor)
{
  TempoMessage *msg = new (&buflen, 0) TempoMessage(tag1, tag2, buflen, buffer);
  CProxy_TempoGroup ptg(bocid);
  ptg[processor].tempoGeneric(msg);
}

// static
void 
TempoGroup::ckTempoSendBranch(int tag, void *buffer, int buflen, 
                              CkGroupID bocid, int processor)
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
                            CkArrayID aid, int idx)
{
  TempoMessage *msg = new (&buflen, 0) TempoMessage(tag1, tag2, buflen, buffer);
  CProxy_TempoArray pta(aid);
  pta[idx].tempoGeneric(msg);
}

// static
void 
TempoArray::ckTempoSendElem(int tag,void *buffer,int buflen,CkArrayID aid, int idx)
{
  ckTempoSendElem(tag, TEMPO_ANY, buffer, buflen, aid, idx);
}

void 
TempoArray::ckTempoSendElem(int tag1, int tag2, void *buffer, int buflen, 
                            int idx)
{
  ckTempoSendElem(tag1, tag2, buffer, buflen, thisArrayID, idx);
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
    ckTempoSendElem(BARR_TAG, nGOps, (void*) 0, 0, 0);
    ckTempoRecv(BARR_TAG, nGOps, (void*) 0, 0);
  } else {
     int i;
     for(i=1;i<ckGetArraySize();i++)
       ckTempoRecv(BARR_TAG, nGOps, (void *) 0, 0);
     for(i=1;i<ckGetArraySize();i++)
       ckTempoSendElem(BARR_TAG, nGOps, (void *) 0, 0, i);
  }
  nGOps++;
}

void
TempoArray::ckTempoBcast(int sender, int tag, void *buffer, int buflen)
{
  if(sender) {
    int i;
    for(i=0;i<ckGetArraySize();i++)
      ckTempoSendElem(tag, BCAST_TAG+nGOps, buffer, buflen, i);
  }
  ckTempoRecv(tag, BCAST_TAG+nGOps, buffer, buflen);
  nGOps++;
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
    _MEMCHECK(tbuf);
    for(int i=0; i<ckGetArraySize()-1; i++) {
      ckTempoRecv(REDUCE_TAG, nGOps, tbuf, size);
      doOp(op, type, count, tbuf, outbuf);
    }
    free(tbuf);
  } else {
    ckTempoSendElem(REDUCE_TAG, nGOps, inbuf, size, root);
  }
  nGOps++;
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
