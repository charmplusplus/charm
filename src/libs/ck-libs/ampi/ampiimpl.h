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

/*
class ampimain : public Chare
{
  int nblocks;
  int numDone;
  CkArrayID arr;
  public:
    ampimain(CkArgMsg *);
    ampimain(CkMigrateMessage *m) {}
    void done(void);
    void qd(void);
};

static inline void itersDone(void) { CProxy_ampimain pm(mainhandle); pm.done(); }
*/

class BlockMap : public CkArrayMap {
 public:
  BlockMap(void) {
  }
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
    int size;
    int proc;
    int tag;
    int nextfree, prevfree;
};

class ArgsInfo : public CMessage_ArgsInfo {
  public:
    int argc;
    char **argv;
    ArgsInfo(int c, char **v) { argc = c; argv = v; }
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
    int nReductions;
    int nAllReductions;
    int niRecvs, niSends, biRecv, biSend;
	DDT	*myDDT ;

    ampi(void);
    ampi(CkMigrateMessage *msg); 
    
    virtual void pup(PUP::er &p);
    
    virtual void start(void);
    
    void run(ArgsInfo *);
    void run(void);		// for C++ inheirt
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
