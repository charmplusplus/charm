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

#define MPI_MAX_COMM 8

struct mpi_comm_struct
{
  CkArrayID aid;
  void (*mainfunc)(int, char **);
  char *name;
  int nobj;
};
class mpi_comm_structs {
	mpi_comm_struct s[MPI_MAX_COMM];
public:
	mpi_comm_struct &operator[](int i) {return s[i];}
};

#include "tcharm.h"
#include "tcharmc.h"
#include "ampi.decl.h"
#include "ddt.h"
#include "charm-api.h"
#include <sys/stat.h> // for mkdir

extern int mpi_ncomms;

#define MPI_BCAST_TAG  1025
#define MPI_BARR_TAG   1026
#define MPI_REDUCE_TAG 1027
#define MPI_GATHER_TAG 1028

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
inline void pupIntoBuf(memBuf &b,T &t) {
	PUP::sizer ps;ps|t;
	b.setSize(ps.size());
	PUP::toMem pm(b.getData()); pm|t;	
}

template <class T>
inline void pupFromBuf(const void *data,T &t) {
	PUP::fromMem p(data); p|t;
}

class AmpiMsg : public CMessage_AmpiMsg {
 public:
  int seq, tag, src, comm, length;
  void *data;

  AmpiMsg(void) { data = (char *)this + sizeof(AmpiMsg); }
  AmpiMsg(int _s, int t, int s, int l, int c) : 
    seq(_s), tag(t),src(s),length(l),comm(c) {
    data = (char *)this + sizeof(AmpiMsg);
  }
  static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
    return CkAllocMsg(msgnum, size+sizes[0], pbits);
  }
  static void *pack(AmpiMsg *in) { return (void *) in; }
  static AmpiMsg *unpack(void *in) { return new (in) AmpiMsg; }
  static AmpiMsg* pup(PUP::er &p, AmpiMsg *m)
  {
    int seq, length, tag, src, comm;
    if(p.isPacking() || p.isSizing()) {
      seq = m->seq;
      tag = m->tag;
      src = m->src;
      comm = m->comm;
      length = m->length;
    }
    p(seq); p(tag); p(src); p(comm); p(length);
    if(p.isUnpacking()) {
      m = new (&length, 0) AmpiMsg(seq, tag, src, length, comm);
    }
    p(m->data, length);
    if(p.isDeleting()) {
      delete m;
      m = 0;
    }
    return m;
  }
};

class AmpiSeqQ : private CkNoncopyable {
  int next;
  CkQ<AmpiMsg*> q;
 public:
  AmpiSeqQ() { init(); }
  void init(void) { next = 0; }
  AmpiMsg *get(void)
  {
    if(q.isEmpty() || (q[0]->seq != next)) {
      return 0;
    }
    next++;
    return q.deq();
  }
  void put(int seq, AmpiMsg *elt)
  {
    int i, len;
    len = q.length();
    for(i=0;i<len;i++) {
      if(q[i]->seq > seq)
        break;
    }
    q.insert(i, elt);
  }
  void pup(PUP::er &p) {
    p(next);
    int len = q.length();
    p(len);
    for(int i=0;i<len;i++) {
     if(p.isUnpacking())
       q.enq(AmpiMsg::pup(p,0));
     else
       AmpiMsg::pup(p, q[i]);
    }
  }
};

PUPmarshall(AmpiSeqQ);

class ampi : public ArrayElement1D {
	//char str[128];    
    CProxy_TCharm threads;
    TCharm *thread;
    int ampiBlockedThread;
    void prepareCtv(void);
    void inorder(AmpiMsg *msg);
  public: // entry methods
    ampi(int commidx_,CProxy_TCharm threads_);
    ampi(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    ~ampi();
    
    virtual void pup(PUP::er &p);
    void generic(AmpiMsg *);
    void reduceResult(CkReductionMsg *m);
    
  public: // to be used by MPI_* functions
    void send(int t, int s, void* buf, int count, int type, int idx, int comm);
    static void sendraw(int t, int s, void* buf, int len, CkArrayID aid, 
                        int idx);
    void recv(int t,int s,void* buf,int count,int type,int comm,int *sts=0);
    void probe(int t,int s,int comm,int *sts);
    int iprobe(int t,int s,int comm,int *sts);
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
    int *nextseq;
    AmpiSeqQ *oorder;
};

//Use this to mark the start of AMPI interface routines:
#define AMPIAPI(routineName) TCHARM_API_TRACE(routineName,"ampi")

#endif
