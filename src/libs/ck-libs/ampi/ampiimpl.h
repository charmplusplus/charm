/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include "ampi.h"
#include "charm++.h"

#define AMPI_MAX_COMM 8

struct ampi_redn_spec
{
  int type;
  int root;
};

struct ampi_comm_struct
{
  CkArrayID aid;
  void (*mainfunc)(int, char **);
  char *name;
  int nobj;
  ampi_redn_spec rspec;
};
class ampi_comm_structs {
	ampi_comm_struct s[AMPI_MAX_COMM];
public:
	ampi_comm_struct &operator[](int i) {return s[i];}
};

#include "tcharm.h"
#include "tcharmc.h"
#include "ampi.decl.h"
#include "ddt.h"
#include "charm-api.h"
#include <sys/stat.h> // for mkdir

extern int ampi_ncomms;

#define AMPI_BCAST_TAG  1025
#define AMPI_BARR_TAG   1026
#define AMPI_REDUCE_TAG 1027
#define AMPI_GATHER_TAG 1028

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
    int comm;
    int nextfree, prevfree;
};

class argvPupable {
	bool isSeparate;//Separately allocated strings
	char **argv;
 public:
	char **getArgv(void) {return argv;}
	int getArgc(void) const {return CmiGetArgc(argv);}
	argvPupable() {argv=NULL;isSeparate=false;}
	argvPupable(char **argv_) {argv=argv_; isSeparate=false;}
	argvPupable(const argvPupable &p);
	~argvPupable();
	void pup(PUP::er &p);
};
PUPmarshall(argvPupable);

//A simple destructive-copy memory buffer
class memBuf {
	int bufSize;
	char *buf;
	void make(int size=0) {
		clear();
		bufSize=size;
		if (bufSize>0) buf=new char[bufSize];
		else buf=NULL;
	}
	void steal(memBuf &b) {
		bufSize=b.bufSize;
		buf=b.buf;
		b.bufSize=-1;
		b.buf=NULL;
	}
	void clear(void) { if (buf!=NULL) {delete[] buf; buf=NULL;} }
	//No copy semantics:
	memBuf(memBuf &b);
	memBuf &operator=(memBuf &b);
 public:
	memBuf() {buf=NULL; bufSize=0;}
	memBuf(int size) {buf=NULL; make(size);}
	~memBuf() {clear();}
	void setSize(int s) {make(s);}
	int getSize(void) const {return bufSize;}
	const void *getData(void) const {return (const void *)buf;}
	void *getData(void) {return (void *)buf;}
};

template <class T>
void pupIntoBuf(memBuf &b,T &t) {
	PUP::sizer ps;ps|t;
	b.setSize(ps.size());
	PUP::toMem pm(b.getData()); pm|t;	
}

template <class T>
void pupFromBuf(const void *data,T &t) {
	PUP::fromMem p(data); p|t;
}

class AmpiMsg : public CMessage_AmpiMsg {
 public:
  int tag1, tag2, comm, length;
  void *data;

  AmpiMsg(void) { data = (char *)this + sizeof(AmpiMsg); }
  AmpiMsg(int t1, int t2, int l, int c):tag1(t1),tag2(t2),length(l),comm(c) {
    data = (char *)this + sizeof(AmpiMsg);
  }
  static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
    return CkAllocMsg(msgnum, size+sizes[0], pbits);
  }
  static void *pack(AmpiMsg *in) { return (void *) in; }
  static AmpiMsg *unpack(void *in) { return new (in) AmpiMsg; }
  static AmpiMsg* pup(PUP::er &p, AmpiMsg *m)
  {
    int length, tag1, tag2, comm;
    if(p.isPacking() || p.isSizing()) {
      tag1 = m->tag1;
      tag2 = m->tag2;
      comm = m->comm;
      length = m->length;
    }
    p(tag1); p(tag2); p(comm); p(length);
    if(p.isUnpacking()) {
      m = new (&length, 0) AmpiMsg(tag1, tag2, length, comm);
    }
    p(m->data, length);
    if(p.isDeleting()) {
      delete m;
      m = 0;
    }
    return m;
  }
};

class ampi : public ArrayElement1D {
	//char str[128];    
    CProxy_TCharm threads;
    TCharm *thread;
    int ampiBlockedThread;
    void prepareCtv(void);
  public: // entry methods
    ampi(int commidx_,CProxy_TCharm threads_);
    ampi(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    ~ampi();
    
    virtual void pup(PUP::er &p);
    void generic(AmpiMsg *);
    
  public: // to be used by AMPI_* functions
    void send(int t1, int t2, void* buf, int count, int type, int idx, int comm);
    static void sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, 
                        int idx);
    void recv(int t1,int t2,void* buf,int count,int type,int comm,int *sts=0);
    void probe(int t1,int t2,int comm,int *sts);
    int iprobe(int t1,int t2,int comm,int *sts);
    void barrier(void);
    void bcast(int root, void* buf, int count, int type);
    static void bcastraw(void* buf, int len, CkArrayID aid);

    inline int getIndex(void) const {return thisIndex;}
    inline int getArraySize(void) const {return numElements;}
  public:
    //These are directly used by , which is hideous
    int commidx;
    CmmTable msgs;
    int nbcasts;
    PersReq requests[100];
    int nrequests;
    PersReq irequests[100];
    int nirequests;
    int firstfree;
    DDT *myDDT ;
};

#endif
