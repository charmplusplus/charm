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
#include "ddt.h"

extern CkChareID mainhandle;

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

#define MyAlign8(x) (((x)+7)&(~7))

class MigrateInfo : public CMessage_MigrateInfo {
  public:
    ArrayElement1D *elem;
    int where;
    MigrateInfo(ArrayElement1D *e, int w) : elem(e), where(w) {}
};

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

// FIXME: Make this a packed message.
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
  private:
    CmmTable msgs;
    CthThread thread_id;
    int nbcasts;
  public: // entry methods
    ampi(void);
    ampi(CkMigrateMessage *msg); 
    void run(ArgsInfo *);
    void run(void);
    void generic(AmpiMsg *);
  public: // to be used by AMPI_* functions
    void send(int t1, int t2, void* buf, int count, int type, int idx);
    static void sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, 
                        int idx);
    void recv(int t1, int t2, void* buf, int count, int type);
    void barrier(void);
    void bcast(int root, void* buf, int count, int type);
    static void bcastraw(void* buf, int len, CkArrayID aid);
    void reduce(int root, int op, void* inb, void *outb, int count, int type);
  public:
    int csize, isize, rsize, fsize;
    int totsize;
    PersReq requests[100];
    int nrequests;
    int types[100]; // currently just gives the size
    int ntypes;
    PersReq irequests[100];
    int nirequests;
    int firstfree;
    void *packedBlock;
    int nReductions;
    int nAllReductions;
    int niRecvs, niSends, biRecv, biSend;
    DDT *myDDT ;

    virtual void pup(PUP::er &p);
    virtual void start(void); // should be overloaded in derived class
};

extern int migHandle;

class migrator : public Group {
  public:
    migrator(void) { migHandle = thisgroup; }
    migrator(CkMigrateMessage *m) {}
    void migrateElement(MigrateInfo *msg) {
      msg->elem->migrateMe(msg->where);
      delete msg;
    }
};

#endif
