/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include "ampi.h"
#include "ampi.decl.h"
#include "ampimain.decl.h"
#include "ddt.h"

class ampimain : public Chare
{
  int nblocks;
  int numDone;
  public:
    static CkChareID handle;
    static CkArrayID ampiAid;
    ampimain(CkArgMsg *);
    ampimain(CkMigrateMessage *m) {}
    void done(void);
};

static inline void 
itersDone(void) 
{ 
  CProxy_ampimain pm(ampimain::handle); 
  pm.done(); 
}

#define AMPI_BCAST_TAG  1025
#define AMPI_BARR_TAG   1026
#define AMPI_REDUCE_TAG 1027
#define AMPI_GATHER_TAG 1028

extern int _redntype;
extern int _rednroot;

#if 0
// This is currently not used.
class BlockMap : public CkArrayMap {
 public:
  BlockMap(void) {}
  BlockMap(CkMigrateMessage *m) {}
  int registerArray(CkArrayMapRegisterMessage *m) {
    delete m;
    return 0;
  }
  int procNum(int /*arrayHdl*/,const CkArrayIndex &idx) {
    int elem=*(int *)idx.data();
    int penum =  (elem/(32/CkNumPes()));
    CkPrintf("%d mapped to %d proc\n", elem, penum);
    return penum;
  }
};
#endif

#define MyAlign8(x) (((x)+7)&(~7))

class PersReq {
  public:
    int sndrcv; // 1 if send , 2 if recv
    void *buf;
    int count;
    int type;
    int proc;
    int tag;
    int nextfree, prevfree;
};

class ArgsInfo : public CMessage_ArgsInfo {
  public:
    int argc;
    char **argv;
    ArgsInfo(void) { argc = 0; }
    ArgsInfo(int c, char **v) { argc = c; argv = v; }
    static void* pack(ArgsInfo*);
    static ArgsInfo* unpack(void*);
};

class AmpiMsg : public CMessage_AmpiMsg {
 public:
  int tag1, tag2, length;
  void *data;

  AmpiMsg(void) { data = (char *)this + sizeof(AmpiMsg); }
  AmpiMsg(int t1, int t2, int l):tag1(t1),tag2(t2),length(l) {
    data = (char *)this + sizeof(AmpiMsg);
  }
  static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
    return CkAllocMsg(msgnum, size+sizes[0], pbits);
  }
  static void *pack(AmpiMsg *in) { return (void *) in; }
  static AmpiMsg *unpack(void *in) { return new (in) AmpiMsg; }
};

class ampi : public ArrayElement1D {
  public: // entry methods
    ampi(void);
    ampi(CkMigrateMessage *msg) {}
    void run(ArgsInfo *);
    void run(void);
    void generic(AmpiMsg *);
    void migrate(void)
    {
      AtSync();
    }
    void start_running(void)
    {
      thisArray->the_lbdb->ObjectStart(ldHandle);
    }
    void stop_running(void)
    {
      thisArray->the_lbdb->ObjectStop(ldHandle);
    }

  public: // to be used by AMPI_* functions
    void send(int t1, int t2, void* buf, int count, int type, int idx);
    static void sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, 
                        int idx);
    void recv(int t1, int t2, void* buf, int count, int type);
    void barrier(void);
    void bcast(int root, void* buf, int count, int type);
    static void bcastraw(void* buf, int len, CkArrayID aid);
  public:
    CmmTable msgs;
    int msize;
    CthThread thread_id;
    int tsize;
    int nbcasts;
    PersReq requests[100];
    int nrequests;
    PersReq irequests[100];
    int nirequests;
    int firstfree;
    DDT *myDDT ;

    virtual void pup(PUP::er &p);
    virtual void start(void); // should be overloaded in derived class
    void ResumeFromSync(void)
    {
      CthAwaken(thread_id);
      thread_id = 0;
    }
};

#endif
