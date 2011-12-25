#ifndef TEMPO_H
#define TEMPO_H

#include "charm++.h"
#include <stdlib.h>
#include "tempo.decl.h"

#define TEMPO_ANY  CmmWildCard
#define BCAST_TAG  1025
#define BARR_TAG   1026
#define REDUCE_TAG 1027

#define TempoAlign(x) (((x)+7)&~7)

class TempoMessage : public CMessage_TempoMessage
{
  public:
    int tag1, tag2, length;
    void *data;
  
    TempoMessage(void) { data = (char *)this + sizeof(TempoMessage); }
    TempoMessage(int t1, int t2, int l, void *d):tag1(t1),tag2(t2),length(l) {
      data = (char *)this + sizeof(TempoMessage);
      memcpy(data, d, l);
    }
    static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
      return CkAllocMsg(msgnum, (int)size+sizes[0], pbits);
    }
    static void *pack(TempoMessage *in) { return (void *) in; }
    static TempoMessage *unpack(void *in) { return new (in) TempoMessage; }
};

class Tempo 
{
  protected:
    CmmTable tempoMessages;
    int sleeping;
    CthThread thread_id;
    
  public :
    Tempo();
    void ckTempoRecv(int tag, void *buffer, int buflen);
    void ckTempoRecv(int tag1, int tag2, void *buffer, int buflen);
    static void ckTempoSend(int tag1, int tag2, void *buffer,int buflen, 
                            CkChareID cid);
    static void ckTempoSend(int tag, void *buffer,int buflen, CkChareID cid);
    void tempoGeneric(TempoMessage *themsg);
    int ckTempoProbe(int tag1, int tag2);
    int ckTempoProbe(int tag);
};

class TempoChare : public Chare, public Tempo
{
  public:
    TempoChare(void) {}
    TempoChare(CkMigrateMessage *m):Chare(m) {}
};

class TempoGroup : public IrrGroup, public Tempo
{
  public :
    TempoGroup(void) {};
    TempoGroup(CkMigrateMessage *m):IrrGroup(m) {};
    static void ckTempoBcast(int tag, void *buffer, int buflen, CkGroupID bocid);
    static void ckTempoSendBranch(int tag1, int tag2, void *buffer, int buflen,
                                  CkGroupID bocid, int processor);
    static void ckTempoSendBranch(int tag, void *buffer, int buflen,
                                  CkGroupID bocid, int processor);
    void ckTempoBcast(int sender, int tag, void *buffer, int buflen);
    void ckTempoSendBranch(int tag1, int tag2, void *buffer, int buflen, 
                           int processor);
    void ckTempoSendBranch(int tag, void *buffer, int buflen, int processor);
};

#define TEMPO_MAX  1
#define TEMPO_MIN  2
#define TEMPO_SUM  3
#define TEMPO_PROD 4

#define TEMPO_FLOAT  1
#define TEMPO_INT    2
#define TEMPO_DOUBLE 3

class TempoArray : public ArrayElement1D, public Tempo
{
  int nGOps;
  public:
    TempoArray(void) { nGOps=0; }
    TempoArray(CkMigrateMessage *msg):ArrayElement1D(msg) { }
    static void ckTempoSendElem(int tag1, int tag2, void *buffer, int buflen,
                                CkArrayID aid, int idx);
    static void ckTempoSendElem(int tag, void *buffer, int buflen,
                                CkArrayID aid, int idx);
    void ckTempoSendElem(int tag1, int tag2, void *buffer, int buflen, int idx);
    void ckTempoSendElem(int tag, void *buffer, int buflen, int idx);
    void ckTempoBarrier(void);
    void ckTempoBcast(int sender, int tag, void *buffer, int buflen);
    void ckTempoReduce(int root, int op, void *inbuf, void *outbuf, int count,
                       int type);
    void ckTempoAllReduce(int op,void *inbuf,void *outbuf,int count,int type);
    void pup(PUP::er& p)
    {
      ArrayElement1D::pup(p);
      p(nGOps);
    }
};

#endif
