#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include "ampi.h"
#include "ampi.decl.h"

extern CkChareID mainhandle;

class main : public Chare
{
  int nblocks;
  int numDone;
  CkArrayID arr;
  public:
    main(CkArgMsg *);
    void done(void);
    void qd(void);
};

static inline void itersDone(void) { CProxy_main pm(mainhandle); pm.done(); }

class BlockMap : public ArrayMap {
 public:
  BlockMap(void) {
  }
  void registerArray(ArrayMapRegisterMessage *m) {
    Array1D *array = (Array1D *) CkLocalBranch(m->groupID);
    array->RecvMapID(this, 0);
    delete m;
  }
  int procNum(int /*arrayHdl*/, int elem) {
    int penum =  (elem/(32/CkNumPes()));
    CkPrintf("%d mapped to %d proc\n", elem, penum);
    return penum;
  }
};

#define MyAlign8(x) (((x)+7)&(~7))

class MigrateInfo : public CMessage_MigrateInfo {
  public:
    ArrayElement *elem;
    int where;
    MigrateInfo(ArrayElement *e, int w) : elem(e), where(w) {}
};

class PersReq {
  public:
    int sndrcv; // 1 if send , 2 if recv
    void *buf;
    int size;
    int proc;
    int tag;
    int nextfree, prevfree;
};

class ampi : public TempoArray {
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
    int nbcasts; // to keep bcasts from mixing up
    void *packedBlock;

    ampi(ArrayElementCreateMessage *msg);
    ampi(ArrayElementMigrateMessage *msg);
    int packsize(void);
    void pack(void *buf);
    void run(void);
};

extern int migHandle;

class migrator : public Group {
  public:
    migrator(void) { migHandle = thisgroup; }
    void migrateElement(MigrateInfo *msg) {
      msg->elem->migrate(msg->where);
      delete msg;
    }
};
#endif
