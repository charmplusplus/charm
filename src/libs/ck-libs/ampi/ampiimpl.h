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
#include "EachToManyMulticastStrategy.h" /* for ComlibManager Strategy*/
#include <string.h> /* for strlen */

class CProxy_ampi;
class CProxyElement_ampi;

//Describes an AMPI communicator
class ampiCommStruct {
	MPI_Comm comm; //Communicator
	CkArrayID ampiID; //ID of corresponding ampi array
	int size; //Number of processes in communicator
	int isWorld; //1 if ranks are 0..size-1?
	CkPupBasicVec<int> indices;  //indices[r] gives the array index for rank r
public:
	ampiCommStruct(int ignored=0) {size=-1;isWorld=-1;}
	ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,int size_)
		:comm(comm_), ampiID(id_),size(size_), isWorld(1) {}
	ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,
		int size_,const CkPupBasicVec<int> &indices_)
		:comm(comm_), ampiID(id_),size(size_),
		 isWorld(0), indices(indices_) {}

	void setArrayID(const CkArrayID &nID) {ampiID=nID;}

	MPI_Comm getComm(void) const {return comm;}
	CkPupBasicVec<int> getIndices(void) const {return indices;}

	//Get the proxy for the entire array
	CProxy_ampi getProxy(void) const;

	//Get the array index for rank r in this communicator
	int getIndexForRank(int r) const {
#ifndef CMK_OPTIMIZE
		if (r>=size) CkAbort("AMPI> You passed in an out-of-bounds process rank!");
#endif
		if (isWorld) return r;
		else return indices[r];
	}
	//Get the rank for this array index (Warning: linear time)
	int getRankForIndex(int i) const {
		if (isWorld) return i;
		else {
		  for (int r=0;r<indices.size();r++)
		    if (indices[r]==i) return r;
		  return -1; /*That index isn't in this communicator*/
		}
	}

	int getSize(void) const {return size;}

	void pup(PUP::er &p) {
		p|comm;
		p|ampiID;
		p|size;
		p|isWorld;
		p|indices;
	}
};
PUPmarshall(ampiCommStruct);

struct mpi_comm_world
{
  mpi_comm_world(const mpi_comm_world &m); //DO NOT USE
  void operator=(const mpi_comm_world &m);
  char *name; //new'd human-readable zero-terminated string name, or NULL
public:
  ampiCommStruct comm;
  mpi_comm_world() {
    name=NULL;
  }
  ~mpi_comm_world() {
	  if (name) { delete[] name; name=0; }
  }
  void setName(const char *src) {
    setName(src,strlen(src));
  }
  void setName(const char *src,int len) {
	name=new char[len+1];
	memcpy(name,src,len);
	name[len] = '\0';
  }
  const char *getName(void) const { return name; }
  void pup(PUP::er &p) {
    p|comm;
    int len=0;
    if (name!=NULL) len=strlen(name)+1;
    p|len;
    if (p.isUnpacking()) name=new char[len];
    p(name,len);
  }
};
class mpi_comm_worlds {
	mpi_comm_world s[MPI_MAX_COMM_WORLDS];
public:
	mpi_comm_world &operator[](int i) {return s[i];}
	void pup(PUP::er &p) {
		for (int i=0;i<MPI_MAX_COMM_WORLDS;i++)
			s[i].pup(p);
	}
};

typedef CkPupBasicVec<int> groupStruct;
// groupStructure operations
inline void outputOp(groupStruct vec){
  CkPrintf("output vector: size=%d  {",vec.size());
  for(int i=0;i<vec.size();i++)
    CkPrintf(" %d ",vec[i]);
  CkPrintf("}\n");
}
inline int getPosOp(int idx, groupStruct vec){
  for (int r=0;r<vec.size();r++)
    if (vec[r]==idx) return r;
  return MPI_UNDEFINED;
}
inline groupStruct unionOp(groupStruct vec1, groupStruct vec2){
  groupStruct newvec(vec1);
  for(int i=0;i<vec2.size();i++){
    if(getPosOp(vec2[i],vec1)==MPI_UNDEFINED)
      newvec.push_back(vec2[i]);
  }
  return newvec;
}
inline groupStruct intersectOp(groupStruct vec1, groupStruct vec2){
  groupStruct newvec;
  for(int i=0;i<vec1.size();i++){
    if(getPosOp(vec1[i],vec2)!=MPI_UNDEFINED)
      newvec.push_back(vec1[i]);
  }
  return newvec;
}
inline groupStruct diffOp(groupStruct vec1, groupStruct vec2){
  groupStruct newvec;
  for(int i=0;i<vec1.size();i++){
    if(getPosOp(vec1[i],vec2)==MPI_UNDEFINED)
      newvec.push_back(vec1[i]);
  }
  return newvec;
}
inline int* translateRanksOp(int n,groupStruct vec1,int* ranks1,groupStruct vec2){
  int* ret = new int[n];
  for(int i=0;i<n;i++){
    ret[i] = getPosOp(vec1[ranks1[i]],vec2);
  }
  return ret;
}
inline int compareVecOp(groupStruct vec1,groupStruct vec2){
  int i,pos,ret = MPI_IDENT;
  if(vec1.size() != vec2.size()) return MPI_UNEQUAL;
  for(i=0;i<vec1.size();i++){
    pos = getPosOp(vec1[i],vec2);
    if(pos == MPI_UNDEFINED) return MPI_UNEQUAL;
    if(pos != i)   ret = MPI_SIMILAR;
  }
  return ret;
}
inline groupStruct inclOp(int n,int* ranks,groupStruct vec){
  groupStruct retvec;
  for(int i=0;i<n;i++){
    retvec.push_back(vec[ranks[i]]);
  }
  return retvec;
}
inline groupStruct exclOp(int n,int* ranks,groupStruct vec){
  groupStruct retvec;
  int add=1;
  for(int j=0;j<vec.size();j++){
    for(int i=0;i<n;i++)
      if(j==ranks[i]){ add=0; break; }
    if(add==1)  retvec.push_back(vec[j]);
    else add=1;
  }
  return retvec;
}
inline groupStruct rangeInclOp(int n, int ranges[][3], groupStruct vec){
  groupStruct retvec;
  int first,last,stride;
  for(int i=0;i<n;i++){
    first = ranges[i][0];
    last = ranges[i][1];
    stride = ranges[i][2];
    for(int j=0;j<=(last-first)/stride;j++)
      retvec.push_back(vec[first+stride*j]);
  }
  return retvec;
}
inline groupStruct rangeExclOp(int n, int ranges[][3], groupStruct vec){
  groupStruct retvec;
  CkPupBasicVec<int> ranksvec;
  int first,last,stride;
  int *ranks,cnt;
  int i,j;
  for(i=0;i<n;i++){
    first = ranges[i][0];
    last = ranges[i][1];
    stride = ranges[i][2];
    for(j=0;j<=(last-first)/stride;j++)
      ranksvec.push_back(first+stride*j);
  }
  cnt=ranksvec.size();
  ranks=new int[cnt];
  for(i=0;i<cnt;i++)
    ranks[i]=ranksvec[i];
  return exclOp(cnt,ranks,vec);
}

#include "tcharm.h"
#include "tcharmc.h"

#include "ampi.decl.h"
#include "ddt.h"
#include "charm-api.h"
#include <sys/stat.h> // for mkdir

extern int mpi_nworlds;

#define MPI_BCAST_TAG   MPI_TAG_UB+10
#define MPI_BARR_TAG    MPI_TAG_UB+11
#define MPI_REDUCE_TAG  MPI_TAG_UB+12
#define MPI_GATHER_TAG  MPI_TAG_UB+13
#define MPI_SCATTER_TAG MPI_TAG_UB+14

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

/**
Represents an MPI request that has been initiated
using Isend, Irecv, Ialltoall, Send_init, etc.
*/
class AmpiRequest {
protected:
	bool isvalid;
public:
	AmpiRequest(){ }
	/// Close this request (used by free and cancel)
	virtual ~AmpiRequest(){ }

	/// Activate this persistent request.
	///  Only meaningful for persistent requests,
	///  other requests just abort.
	virtual int start(void){ return -1; }

	/// Return true if this request is finished (progress).
	virtual CmiBool test(MPI_Status *sts) =0;

	/// Completes the operation hanging on the request
	virtual void complete(MPI_Status *sts) =0;

	/// Block until this request is finished,
	///  returning a valid MPI error code.
	virtual int wait(MPI_Status *sts) =0;

	/// Frees up the request: invalidate it
	virtual inline void free(void){ isvalid=false; }
	inline bool isValid(void){ return isvalid; }

	/// Returns the type of request: 1-PersReq, 2-IReq, 3-ATAReq
	virtual int getType(void) =0;
};

class PersReq : public AmpiRequest {
	void *buf;
	int count;
	int type;
	int src;
	int tag;
	int comm;

	int sndrcv; // 1 if send , 2 if recv
public:
	PersReq(void *buf_, int count_, int type_, int src_, int tag_,
		MPI_Comm comm_, int sndrcv_) :
		buf(buf_), count(count_), type(type_), src(src_), tag(tag_),
		comm(comm_), sndrcv(sndrcv_)
		{ isvalid=true; }
	~PersReq(){ }
	int start();
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	inline int getType(void){ return 1; }
};

class IReq : public AmpiRequest {
	void *buf;
	int count;
	int type;
	int src;
	int tag;
	int comm;
public:
	IReq(void *buf_, int count_, int type_, int src_, int tag_, MPI_Comm comm_):
	     buf(buf_), count(count_), type(type_), src(src_), tag(tag_), comm(comm_)
	     { isvalid=true; }
	~IReq(){ }
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	inline int getType(void){ return 2; }
};

class ATAReq : public AmpiRequest {
	class Request {
	protected:
		void *buf;
		int count;
		int type;
		int src;
		int tag;
		int comm;
	friend class ATAReq;
	};
	Request *myreqs;
	int count;
	int idx;
public:
	ATAReq(int c_):count(c_),idx(0) { myreqs = new Request [c_]; isvalid=true; }
	~ATAReq(void) { if(myreqs) delete [] myreqs; }
	int addReq(void *buf_, int count_, int type_, int src_, int tag_, MPI_Comm comm_){
		myreqs[idx].buf=buf_;	myreqs[idx].count=count_;
		myreqs[idx].type=type_;	myreqs[idx].src=src_;
		myreqs[idx].tag=tag_;	myreqs[idx].comm=comm_;
		return (++idx);
	}
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	inline int getCount(void){ return count; }
	inline int getType(void){ return 3; }
	inline void free(void){ isvalid=false; delete [] myreqs; }
};

/// Special CkVec<AmpiRequest*> for AMPI. Most code copied from cklist.h
class AmpiRequestList : private CkSTLHelper<AmpiRequest *> {
    AmpiRequest** block; //Elements of vector
    int blklen; //Allocated size of block
    int len; //Number of used elements in block
    void makeBlock(int blklen_,int len_) {
       block=new AmpiRequest* [blklen_];
       blklen=blklen_; len=len_;
    }
    void freeBlock(void) {
       len=0; blklen=0;
       delete[] block; block=NULL;
    }
    void copyFrom(const AmpiRequestList &src) {
       makeBlock(src.blklen, src.len);
       elementCopy(block,src.block,blklen);
    }
  public:
    AmpiRequestList() {block=NULL;blklen=len=0;}
    ~AmpiRequestList() { freeBlock(); }
    AmpiRequestList(const AmpiRequestList &src) {copyFrom(src);}
    AmpiRequestList(int size) { makeBlock(size,size); }
    AmpiRequestList &operator=(const AmpiRequestList &src) {
      freeBlock();
      copyFrom(src);
      return *this;
    }

    AmpiRequest* operator[](size_t n) { return block[n]; }

    int size(void) const {return len;}
    void setSize(int blklen_) {
      AmpiRequest **oldBlock=block;
      makeBlock(blklen_,len);
      elementCopy(block,oldBlock,len);
      delete[] oldBlock; //WARNING: leaks if element copy throws exception
    }
    //Grow to contain at least this position:
    void growAtLeast(int pos) {
      if (pos>=blklen) setSize(pos*2+16);
    }
    void insertAt(int pos, AmpiRequest* elt) {
      if (pos>=len) {
        growAtLeast(pos);
        len=pos+1;
      }
      block[pos] = elt;
    }
    void push_back(AmpiRequest* elt) {insertAt(len,elt);}
    int insert(AmpiRequest* elt){
      //search for invalidated slot
      for(int i=0;i<len;i++)
        if(!block[i]->isValid()){
	  block[i] = elt;
	  return i;
        }
      push_back(elt);
      return len-1;
    }

    /// PUPer is empty because there shouldn't be any
    /// outstanding requests at migration.
    void pup(PUP::er &p) { /* empty */ }
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
  int seq; //Sequence number (for message ordering)
  int tag; //MPI tag
  int srcIdx; //Array index of source
  int srcRank; //Communicator rank for source
  MPI_Comm comm; //Communicator for source
  int length; //Number of bytes in message (for pup)
  void *data;

  AmpiMsg(void) { data = (char *)this + sizeof(AmpiMsg); }
  AmpiMsg(int _s, int t, int sIdx,int sRank, int l, int c) :
    seq(_s), tag(t),srcIdx(sIdx), srcRank(sRank), comm(c), length(l) {
    data = (char *)this + sizeof(AmpiMsg);
  }
  static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
    return CkAllocMsg(msgnum, size+sizes[0], pbits);
  }
  static void *pack(AmpiMsg *in) { return (void *) in; }
  static AmpiMsg *unpack(void *in) { return new (in) AmpiMsg; }
  static AmpiMsg* pup(PUP::er &p, AmpiMsg *m)
  {
    int seq, length, tag, srcIdx, srcRank, comm;
    if(p.isPacking() || p.isSizing()) {
      seq = m->seq;
      tag = m->tag;
      srcIdx = m->srcIdx;
      srcRank = m->srcRank;
      comm = m->comm;
      length = m->length;
    }
    p(seq); p(tag); p(srcIdx); p(srcRank); p(comm); p(length);
    if(p.isUnpacking()) {
      m = new (&length, 0) AmpiMsg(seq, tag, srcIdx, srcRank, length, comm);
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
    if (i>1000) CkAbort("Logic error in AmpiSeqQ::put");
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

inline CProxy_ampi ampiCommStruct::getProxy(void) const {return ampiID;}
const ampiCommStruct &universeComm2proxy(MPI_Comm universeNo);

/*
An ampiParent holds all the communicators and the TCharm thread
for its children, which are bound to it.
*/
class ampiParent : public CBase_ampiParent {
    CProxy_TCharm threads;
    ComlibInstanceHandle comlib;
    TCharm *thread;
    void prepareCtv(void);

    MPI_Comm worldNo; //My MPI_COMM_WORLD
    ampi *worldPtr; //AMPI element corresponding to MPI_COMM_WORLD
    ampiCommStruct worldStruct;
    ampiCommStruct selfStruct;

    CkPupPtrVec<ampiCommStruct> splitComm; //Communicators from MPI_Comm_split
    CkPupPtrVec<ampiCommStruct> groupComm; //Communicators from MPI_Comm_group
    CkPupPtrVec<groupStruct> groups; // "Wild" groups that don't have a communicator

    inline int isSplit(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_SPLIT && comm<MPI_COMM_FIRST_GROUP);
    }
    const ampiCommStruct &getSplit(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_SPLIT;
      if (idx>=splitComm.size()) CkAbort("Bad split communicator used");
      return *splitComm[idx];
    }
    void splitChildRegister(const ampiCommStruct &s);

    inline int isGroup(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_GROUP && comm<MPI_COMM_FIRST_RESVD);
    }
    const ampiCommStruct &getGroup(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_GROUP;
      if (idx>=groupComm.size()) CkAbort("Bad group communicator used");
      return *groupComm[idx];
    }
    void groupChildRegister(const ampiCommStruct &s);

    inline int isInGroups(MPI_Group group) const {
      return (group>=0 && group<groups.size());
    }

public:
    ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_, ComlibInstanceHandle comlib_);
    ampiParent(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    ~ampiParent();

    ampi *lookupComm(MPI_Comm comm) {
       if (comm!=worldStruct.getComm())
          CkAbort("ampiParent::lookupComm> Bad communicator!");
       return worldPtr;
    }

    //Children call this when they are first created, or just migrated
    TCharm *registerAmpi(ampi *ptr,ampiCommStruct s,bool forMigration);

    //Grab the next available split/group communicator
    MPI_Comm getNextSplit(void) const {return MPI_COMM_FIRST_SPLIT+splitComm.size();}
    MPI_Comm getNextGroup(void) const {return MPI_COMM_FIRST_GROUP+groupComm.size();}

    void pup(PUP::er &p);

    void startCheckpoint(char* dname);
    void Checkpoint(int len, char* dname);
    void ResumeThread(void);

    inline const ampiCommStruct &comm2proxy(MPI_Comm comm) {
      if (comm==MPI_COMM_WORLD) return worldStruct;
      if (comm==MPI_COMM_SELF) return selfStruct;
      if (comm==worldNo) return worldStruct;
      if (isSplit(comm)) return getSplit(comm);
      if (isGroup(comm)) return getGroup(comm);
      return universeComm2proxy(comm);
    }
    inline ampi *comm2ampi(MPI_Comm comm) {
      if (comm==MPI_COMM_WORLD) return worldPtr;
      if (comm==MPI_COMM_SELF) return worldPtr;
      if (comm==worldNo) return worldPtr;
      if (isSplit(comm)) {
         const ampiCommStruct &st=getSplit(comm);
	 return st.getProxy()[thisIndex].ckLocal();
      }
      if (isGroup(comm)) {
         const ampiCommStruct &st=getGroup(comm);
	 return st.getProxy()[thisIndex].ckLocal();
      }
      if (comm>MPI_COMM_WORLD) return worldPtr; //Use MPI_WORLD ampi for cross-world messages:
      CkAbort("Invalid communicator used!");
    }

    inline int hasComm(const MPI_Group group){
      MPI_Comm comm = (MPI_Comm)group;
      return (comm==MPI_COMM_WORLD || comm==worldNo || isSplit(comm) || isGroup(comm));
    }
    inline const groupStruct group2vec(MPI_Group group){
      if(hasComm(group))
        return comm2proxy((MPI_Comm)group).getIndices();
      if(isInGroups(group))
        return *groups[group];
      CkAbort("ampiParent::group2vec: Invalid group id!");
    }
    inline MPI_Group saveGroupStruct(groupStruct vec){
      int idx = groups.size();
      groups.setSize(idx+1);
      groups.length()=idx+1;
      groups[idx]=new groupStruct(vec);
      return (MPI_Group)idx;
    }
    inline int getRank(const MPI_Group group){
      groupStruct vec = group2vec(group);
      return getPosOp(thisIndex,vec);
    }
    
    ComlibInstanceHandle getComlib(void) { 
    	return comlib; 
    }
    
    int hasWorld(void) const {
        return worldPtr!=NULL;
    }

    /// this is assuming no inter-communicator
    inline MPI_Group comm2group(const MPI_Comm comm){
      ampiCommStruct s = comm2proxy(comm);
      if(comm!=MPI_COMM_WORLD && comm!=s.getComm()) CkAbort("Error in ampiParent::comm2group()");
      return (MPI_Group)(s.getComm());
    }

    CkDDT myDDTsto;
    CkDDT *myDDT;
    AmpiRequestList ampiReqs;
};


/*
An ampi manages the communication of one thread over
one MPI communicator.
*/
class ampi : public CBase_ampi {
    CProxy_ampiParent parentProxy;
    void findParent(bool forMigration);
    ampiParent *parent;
    TCharm *thread;
    int waitingForGeneric;

    ampiCommStruct myComm;
    int myRank;
    groupStruct tmpVec; // stores the group info temporarily

    int seqEntries; //Number of elements in below arrays
    int *nextseq;
    AmpiSeqQ *oorder;
    void inorder(AmpiMsg *msg);

  public: // entry methods
    ampi();
    ampi(CkArrayID parent_,const ampiCommStruct &s);
    ampi(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    ~ampi();

    virtual void pup(PUP::er &p);
    void generic(AmpiMsg *);
    void reduceResult(CkReductionMsg *m);
    void splitPhase1(CkReductionMsg *msg);
    void commCreatePhase1(CkReductionMsg *msg);

  public: // to be used by MPI_* functions

    inline const ampiCommStruct &comm2proxy(MPI_Comm comm) {
      return parent->comm2proxy(comm);
    }

    AmpiMsg *makeAmpiMsg(int destIdx,
    	int t,int sRank,const void *buf,int count,int type,MPI_Comm destcomm);

    void send(int t, int s, const void* buf, int count, int type,  int rank, MPI_Comm destcomm);
    static void sendraw(int t, int s, void* buf, int len, CkArrayID aid,
                        int idx);
    void delesend(int t, int s, const void* buf, int count, int type,  int rank, MPI_Comm destcomm, CProxy_ampi arrproxy);
    void recv(int t,int s,void* buf,int count,int type,int comm,int *sts=0);
    void probe(int t,int s,int comm,int *sts);
    int iprobe(int t,int s,int comm,int *sts);
    void bcast(int root, void* buf, int count, int type,MPI_Comm comm);
    static void bcastraw(void* buf, int len, CkArrayID aid);
    void split(int color,int key,MPI_Comm *dest);
    void commCreate(const groupStruct vec,MPI_Comm *newcomm);

    inline int getRank(void) const {return myRank;}
    inline int getSize(void) const {return myComm.getSize();}
    inline MPI_Comm getComm(void) const {return myComm.getComm();}
    inline CProxy_ampi getProxy(void) const {return thisArrayID;}
    ComlibInstanceHandle getComlib(void) { return parent->getComlib(); }

    CkDDT *getDDT(void) {return parent->myDDT;}
  public:
    //These are directly used by API routines, which is hideous
    /*
    FIXME: CmmTable is only indexed by the tag, sender, and communicator.
    It should also be indexed by the source data type and length (if any).
    */
    CmmTable msgs;
    int nbcasts;
};

//Use this to mark the start of AMPI interface routines:
#define AMPIAPI(routineName) TCHARM_API_TRACE(routineName,"ampi")

#endif

