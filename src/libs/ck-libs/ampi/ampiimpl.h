#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include <string.h> /* for strlen */

#include "ampi.h"
#include "charm++.h"
#include "ckliststring.h"

#if AMPI_COMLIB
//#warning COMPILING IN UNTESTED AMPI COMLIB SUPPORT
#include "StreamingStrategy.h"
#include "EachToManyMulticastStrategy.h" /* for ComlibManager Strategy*/
#include "BroadcastStrategy.h"
#else
#define ComlibInstanceHandle int
#endif

#if 0
#define AMPI_DEBUG CkPrintf
#else
#define AMPI_DEBUG /* empty */
#endif

#if AMPIMSGLOG

//static int msgLogRank;
static CkListString msgLogRanks;
static int msgLogWrite;
static int msgLogRead;
static char *msgLogFilename;

#if CMK_PROJECTIONS_USE_ZLIB && 0
#include <zlib.h>
namespace PUP{
class zdisk : public er {
 protected:
  gzFile F;//Disk file to read from/write to
  zdisk(unsigned int type,gzFile f):er(type),F(f) {}
  zdisk(const zdisk &p);			//You don't want to copy
  void operator=(const zdisk &p);	// You don't want to copy

  //For seeking (pack/unpack in different orders)
  virtual void impl_startSeek(seekBlock &s); /*Begin a seeking block*/
  virtual int impl_tell(seekBlock &s); /*Give the current offset*/
  virtual void impl_seek(seekBlock &s,int off); /*Seek to the given offset*/
};

//For packing to a disk file
class tozDisk : public zdisk {
 protected:
  //Generic bottleneck: pack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given file pointer
  // (must be opened for binary write)
  // You must close the file yourself when done.
  tozDisk(gzFile f):zdisk(IS_PACKING,f) {}
};

//For unpacking from a disk file
class fromzDisk : public zdisk {
 protected:
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given file pointer 
  // (must be opened for binary read)
  // You must close the file yourself when done.
  fromzDisk(gzFile f):zdisk(IS_UNPACKING,f) {}
};
}; // namespace PUP
#endif
#endif // AMPIMSGLOG

#define AMPI_COUNTER 0

#define AMPI_ALLTOALL_SHORT_MSG   32
#if CMK_CONVERSE_LAPI ||  CMK_BLUEGENE_CHARM
#define AMPI_ALLTOALL_MEDIUM_MSG   4194304
#else
#define AMPI_ALLTOALL_MEDIUM_MSG   32768
#endif

#if AMPI_COUNTER
class AmpiCounters{
public:
	int send,recv,isend,irecv,barrier,bcast,gather,scatter,allgather,alltoall,reduce,allreduce,scan;
	AmpiCounters(){
		send=0;recv=0;isend=0;irecv=0;barrier=0;bcast=0;gather=0;scatter=0;allgather=0;alltoall=0;reduce=0;allreduce=0;scan=0;
	}
	void output(int idx){
		printf("[%d]send=%d;recv=%d;isend=%d;irecv=%d;barrier=%d;bcast=%d;gather=%d;scatter=%d;allgather=%d;alltoall=%d;reduce=%d;allreduce=%d;scan=%d\n",idx,send,recv,isend,irecv,barrier,bcast,gather,scatter,allgather,alltoall,reduce,allreduce,scan);
	}
};
#endif



void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* invec, void* inoutvec);
PUPfunctionpointer(MPI_Op)
class AmpiOpHeader {
public:
  MPI_User_function* func;
  MPI_Datatype dtype;  
  int len;
  int szdata;
  AmpiOpHeader(MPI_User_function* f,MPI_Datatype d,int l,int szd):
    func(f),dtype(d),len(l),szdata(szd) { }
};

//------------------- added by YAN for one-sided communication -----------
/* the index is unique within a communicator */
class WinStruct{
public:
  MPI_Comm comm;
  int index;
  WinStruct(void):comm(MPI_COMM_NULL),index(-1){ }
  WinStruct(MPI_Comm comm_, int index_):comm(comm_),index(index_){ }
  void pup(PUP::er &p){ p|comm; p|index; }
};

class ampi;
class lockQueueEntry {
public:
  int requestRank;
  int pe_src; 
  int ftHandle; 
  int lock_type;
  lockQueueEntry (int _requestRank, int _pe_src, int _ftHandle, int _lock_type) 
  	: requestRank(_requestRank), pe_src(_pe_src), ftHandle(_ftHandle),  lock_type(_lock_type) {}
  lockQueueEntry () {}
};

typedef CkQ<lockQueueEntry *> LockQueue;

class win_obj {
 public:
  char* winName;
  int winNameLeng;
  int initflag; 
  
  void *baseAddr;
  MPI_Aint winSize;
  int disp_unit;
  MPI_Comm comm;

  int owner;   // Rank of owner of the lock, -1 if not locked
  LockQueue lockQueue;  // queue of waiting processors for the lock
                     // top of queue is the one holding the lock
                     // queue is empty if lock is not applied
		         
  void setName(const char *src,int len);
  void getName(char *src,int *len);
  
 public:
  void pup(PUP::er &p); 

  win_obj();
  win_obj(char *name, void *base, MPI_Aint size, int disp_unit, MPI_Comm comm);
  ~win_obj();
  
  int create(char *name, void *base, MPI_Aint size, int disp_unit, 
	     MPI_Comm comm);
  int free();
  
  int put(void *orgaddr, int orgcnt, int orgunit, 
	  MPI_Aint targdisp, int targcnt, int targunit);

  int get(void *orgaddr, int orgcnt, int orgunit, 
	  MPI_Aint targdisp, int targcnt, int targunit);
  int accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp, int targcnt, 
  		  MPI_Datatype targtype, MPI_Op op);

  int iget(int orgcnt, MPI_Datatype orgtype,
          MPI_Aint targdisp, int targcnt, MPI_Datatype targtype);
  int igetWait(MPI_Request *req, MPI_Status *status);
  int igetFree(MPI_Request *req, MPI_Status *status);

  int fence();
	  
  int lock(int requestRank, int pe_src, int ftHandle, int lock_type);
  int unlock(int requestRank, int pe_src, int ftHandle);
 
  int wait();
  int post();
  int start();
  int complete();

  void lockTopQueue();
  void enqueue(int requestRank, int pe_src, int ftHandle, int lock_type);	
  void dequeue();
  bool emptyQueue();
};
//-----------------------End of code by YAN ----------------------

class KeyvalPair{
protected:
  int klen, vlen;
  char* key;
  char* val;
public:
  KeyvalPair(void){ }
  KeyvalPair(char* k, char* v);
  ~KeyvalPair(void);
  void pup(PUP::er& p){
    p|klen; p|vlen;
    if(p.isUnpacking()){
      key=new char[klen];
      val=new char[vlen];
    }
    p(key,klen);
    p(val,vlen);
  }
friend class InfoStruct;
};

class InfoStruct{
  CkPupPtrVec<KeyvalPair> nodes;
  bool valid;
public:
  InfoStruct(void):valid(true) { }
  void setvalid(bool valid_){ valid = valid_; }
  bool getvalid(void){ return valid; }
  void set(char* k, char* v);
  void dup(InfoStruct& src);
  int get(char* k, int vl, char*& v); // return flag
  int deletek(char* k); // return -1 when not found
  int get_valuelen(char* k, int* vl); // return flag
  int get_nkeys(void) { return nodes.size(); }
  int get_nthkey(int n,char* k);
  void myfree(void);
  void pup(PUP::er& p);
};

class CProxy_ampi;
class CProxyElement_ampi;

//Describes an AMPI communicator
class ampiCommStruct {
	MPI_Comm comm; //Communicator
	CkArrayID ampiID; //ID of corresponding ampi array
	int size; //Number of processes in communicator
	int isWorld; //1 if ranks are 0..size-1?
	int isInter; // 0: intra-communicator; 1: inter-communicator
	CkVec<int> indices;  //indices[r] gives the array index for rank r
	CkVec<int> remoteIndices;  // remote group for inter-communicator
	// cartesian virtual topology parameters
	int ndims;
	CkVec<int> dims;
	CkVec<int> periods;
	
	// For communicator attributes (MPI_Attr_get): indexed by keyval
	CkVec<void *> keyvals;

	// graph virtual topology parameters
	int nvertices;
	CkVec<int> index;
	CkVec<int> edges;

	// Lazily fill world communicator indices
	void makeWorldIndices(void) const {
		// cast away constness of "index" list
	  CkVec<int> *ind=(CkVec<int> *)&indices;  // changed by Isaac (as a guess to fix a bug). Was "index" not "indices"
		for (int i=0;i<size;i++) ind->push_back(i);
	}
public:
	ampiCommStruct(int ignored=0) {size=-1;isWorld=-1;isInter=0;}
	ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,int size_)
		:comm(comm_), ampiID(id_),size(size_), isWorld(1), isInter(0) {}
	ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,
		int size_,const CkVec<int> &indices_)
		:comm(comm_), ampiID(id_),size(size_),isInter(0),
		 isWorld(0), indices(indices_) {}
	ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,
		int size_,const CkVec<int> &indices_,
		const CkVec<int> &remoteIndices_)
		:comm(comm_),ampiID(id_),size(size_),isWorld(0),isInter(1),
		indices(indices_),remoteIndices(remoteIndices_) {}
	void setArrayID(const CkArrayID &nID) {ampiID=nID;}

	MPI_Comm getComm(void) const {return comm;}
	const CkVec<int> &getIndices(void) const {
		if (isWorld && indices.size()!=size) makeWorldIndices();
		return indices;
	}
	const CkVec<int> &getRemoteIndices(void) const {return remoteIndices;}
	CkVec<void *> &getKeyvals(void) {return keyvals;}

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
	int getIndexForRemoteRank(int r) const {
#ifndef CMK_OPTIMIZE
		if (r>=remoteIndices.size()) CkAbort("AMPI> You passed in an out-of-bounds process rank!");
#endif
		if (isWorld) return r;
		else return remoteIndices[r];
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

	inline int isinter(void) const { return isInter; }
	inline const CkVec<int> &getindices() const {
		if (isWorld && indices.size()!=size) makeWorldIndices();
		return indices;
	}
	inline const CkVec<int> &getdims() const {return dims;}
	inline const CkVec<int> &getperiods() const {return periods;}

	inline int getndims() {return ndims;}
	inline void setndims(int ndims_) {ndims = ndims_; }
	inline void setdims(const CkVec<int> &dims_) { dims = dims_; }
	inline void setperiods(const CkVec<int> &periods_) { periods = periods_; }

	/* Similar hack for graph vt */
	inline int getnvertices() {return nvertices;}
	inline const CkVec<int> &getindex() const {return index;}
	inline const CkVec<int> &getedges() const {return edges;}

	inline void setnvertices(int nvertices_) {nvertices = nvertices_; }
	inline void setindex(const CkVec<int> &index_) { index = index_; }
	inline void setedges(const CkVec<int> &edges_) { edges = edges_; }

	void pup(PUP::er &p) {
		p|comm;
		p|ampiID;
		p|size;
		p|isWorld;
		p|isInter;
		p|indices;
		p|remoteIndices;
		p|ndims;
		p|dims;
		p|periods;
		p|nvertices;
		p|index;
		p|edges;
	}
};
PUPmarshall(ampiCommStruct)

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

typedef CkVec<int> groupStruct;
// groupStructure operations
inline void outputOp(groupStruct vec){
  if(vec.size()>50){
    CkPrintf("vector too large to output!\n");
    return;
  }
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
inline int* translateRanksOp(int n,groupStruct vec1,int* ranks1,groupStruct
vec2, int *ret){
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
  CkVec<int> ranksvec;
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

extern int _mpi_nworlds;

#define MPI_BCAST_TAG   MPI_TAG_UB_VALUE+10
#define MPI_BARR_TAG    MPI_TAG_UB_VALUE+11
#define MPI_REDUCE_TAG  MPI_TAG_UB_VALUE+12
#define MPI_GATHER_TAG  MPI_TAG_UB_VALUE+13
#define MPI_SCATTER_TAG MPI_TAG_UB_VALUE+14
#define MPI_SCAN_TAG 	MPI_TAG_UB_VALUE+15
#define MPI_ATA_TAG	MPI_TAG_UB_VALUE+16

#define MyAlign8(x) (((x)+7)&(~7))

/**
Represents an MPI request that has been initiated
using Isend, Irecv, Ialltoall, Send_init, etc.
*/
class AmpiRequest {
public:
	void *buf;
	int count;
	int type;
	int tag;            // the order must match MPI_Status
	int src;
	int comm;

#if CMK_BLUEGENE_CHARM
public:
	void *event;	// the event point that corresponding to this message
#endif
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

	virtual void receive(ampi *ptr, AmpiMsg *msg) = 0; 

	/// Frees up the request: invalidate it
	virtual void free(void){ isvalid=false; }
	inline bool isValid(void){ return isvalid; }

	/// Returns the type of request: 1-PersReq, 2-IReq, 3-ATAReq,
	/// 4-SReq, 5-GPUReq
	virtual int getType(void) =0;

	virtual void pup(PUP::er &p) {
		p((char *)&buf,sizeof(void *));  //supposed to work only with isomalloc
		p(count);
		p(type);
		p(src);
		p(tag);
		p(comm);
		p(isvalid);
#if CMK_BLUEGENE_CHARM
		//needed for bigsim out-of-core emulation
		//as the "log" is not moved from memory, this pointer is safe
		//to be reused
		p((char *)&event, sizeof(void *));
#endif
	}
	
	//added due to BIGSIM_OOC DEBUGGING
	virtual void print();
};

class PersReq : public AmpiRequest {
	int sndrcv; // 1 if send , 2 if recv, 3 if ssend
public:
	PersReq(void *buf_, int count_, int type_, int src_, int tag_, 
		MPI_Comm comm_, int sndrcv_) 
	{
		buf=buf_;  count=count_;  type=type_;  src=src_;  tag=tag_; 
		comm=comm_;  sndrcv=sndrcv_;  isvalid=true; 
	}
	PersReq(){};
	~PersReq(){ }
	int start();
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	void receive(ampi *ptr, AmpiMsg *msg) {}
	inline int getType(void){ return 1; }
	virtual void pup(PUP::er &p){
		AmpiRequest::pup(p);
		p(sndrcv);
	}
	//added due to BIGSIM_OOC DEBUGGING
	virtual void print();
};

class IReq : public AmpiRequest {
public:
	bool statusIreq;
	int length;     // recv'ed length
	IReq(void *buf_, int count_, int type_, int src_, int tag_, MPI_Comm comm_)
	{
		buf=buf_;  count=count_;  type=type_;  src=src_;  tag=tag_; 
		comm=comm_;  isvalid=true; statusIreq=false; length=0;
	}
	IReq(): statusIreq(false){};
	~IReq(){ }
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	inline int getType(void){ return 2; }
	void receive(ampi *ptr, AmpiMsg *msg); 
	virtual void pup(PUP::er &p){
		AmpiRequest::pup(p);
		p|statusIreq;  p|length;
	}
	//added due to BIGSIM_OOC DEBUGGING
	virtual void print();
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
#if CMK_BLUEGENE_CHARM
		void *event;             // event buffered for the request
#endif
		virtual void pup(PUP::er &p){
			p((char *)&buf,sizeof(void *));  //supposed to work only with isomalloc
			p(count);
			p(type);
			p(src);p(tag);p(comm);
#if CMK_BLUEGENE_CHARM
		//needed for bigsim out-of-core emulation
		//as the "log" is not moved from memory, this pointer is safe
		//to be reused
			p((char *)&event, sizeof(void *));
#endif
		}
	friend class ATAReq;
	};
	Request *myreqs;
	int elmcount;
	int idx;
public:
	ATAReq(int c_):elmcount(c_),idx(0) { myreqs = new Request [c_]; isvalid=true; }
	ATAReq(){};
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
	void receive(ampi *ptr, AmpiMsg *msg) {}
	inline int getCount(void){ return elmcount; }
	inline int getType(void){ return 3; }
// 	inline void free(void){ isvalid=false; delete [] myreqs; }
	virtual void pup(PUP::er &p){
		AmpiRequest::pup(p);
		p(elmcount);
		p(idx);
		if(p.isUnpacking()){
			myreqs = new Request[elmcount];
		}
		for(int i=0;i<idx;i++){
			myreqs[i].pup(p);
		}
		if(p.isDeleting()){
			delete []myreqs;
		}
	}
	//added due to BIGSIM_OOC DEBUGGING
	virtual void print();
};

class SReq : public AmpiRequest {
public:
	bool statusIreq;
	SReq(MPI_Comm comm_): statusIreq(false) {
		comm = comm_; isvalid=true;
	}
	SReq(): statusIreq(false) {}
	~SReq(){ }
	CmiBool test(MPI_Status *sts);
	void complete(MPI_Status *sts);
	int wait(MPI_Status *sts);
	void receive(ampi *ptr, AmpiMsg *msg) {}
	inline int getType(void){ return 4; }
	virtual void pup(PUP::er &p){
		AmpiRequest::pup(p);
		p|statusIreq;
	}
	//added due to BIGSIM_OOC DEBUGGING
	virtual void print();
};

class GPUReq : public AmpiRequest {
    bool isComplete;

public:
    GPUReq();
    int getType() { return 5; }
    CmiBool test(MPI_Status *sts);
    void complete(MPI_Status *sts);
    int wait(MPI_Status *sts);
    void receive(ampi *ptr, AmpiMsg *msg);
    void setComplete();
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
    void free(int pos) {
      if (pos<0 || pos>=len) return;
      block[pos]->free();
      delete block[pos];
      block[pos]=NULL;
    }
    void push_back(AmpiRequest* elt) {insertAt(len,elt);}
    int insert(AmpiRequest* elt){
      //search for invalidated slot
      // disabled to make requests monotonously ascending 
      // for multiple completion calls like MPI_Waitany
       for(int i=0;i<len;i++){
        if(block[i]==NULL){
	  block[i] = elt;
	  return i;
	}
      } 
      push_back(elt);
      return len-1;
    }

    inline void checkRequest(MPI_Request idx){
      if(!(idx==-1 || (idx < this->len && (block[idx])->isValid())))
        CkAbort("Invalid MPI_Request\n");
    }

    //find an AmpiRequest by its pointer value
    //return -1 if not found!
    int findRequestIndex(AmpiRequest *req){
	for(int i=0; i<len; i++){
	    if(block[i]==req) return i;
	}
	return -1;
    }

    void pup(PUP::er &p);
    
    //BIGSIM_OOC DEBUGGING
    void print(){
	for(int i=0; i<len; i++){
	    if(block[i]==NULL) continue;
	    CmiPrintf("AmpiRequestList Element %d [%p]: \n", i+1, block[i]);
	    block[i]->print();
	}
    }
};

//A simple memory buffer
class memBuf {
	CkVec<char> buf;
public:
	memBuf() { }
	memBuf(int size) :buf(size) {}
	void setSize(int s) {buf.resize(s);}
	int getSize(void) const {return buf.size();}
	const void *getData(void) const {return (const void *)&buf[0];}
	void *getData(void) {return (void *)&buf[0];}
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
  int length; //Number of bytes in this message 
#if CMK_BLUEGENE_CHARM
  void *event;
  int  eventPe;	 // the PE that the event is located
#endif
  char *data;

  AmpiMsg(void) { data = NULL; }
  AmpiMsg(int _s, int t, int sIdx,int sRank, int l, int c) :
    seq(_s), tag(t),srcIdx(sIdx), srcRank(sRank), comm(c), length(l) {
  }  
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
      m = new (length, 0) AmpiMsg(seq, tag, srcIdx, srcRank, length, comm);
    }
    p(m->data, length);
    if(p.isDeleting()) {
      delete m;
      m = 0;
    }
    return m;
  }
};

/**
  Our local representation of another AMPI
 array element.  Used to keep track of incoming
 and outgoing message sequence numbers, and 
 the out-of-order message list.
*/
class AmpiOtherElement {
public:
/// Next incoming and outgoing message sequence number
  int seqIncoming, seqOutgoing;
  
/// Number of elements in out-of-order queue. (normally 0)
  int nOut;

  AmpiOtherElement(void) {
    seqIncoming=0; seqOutgoing=0;
    nOut=0;
  }
  
  void pup(PUP::er &p) {
    p|seqIncoming; p|seqOutgoing;
    p|nOut;
  }
};

class AmpiSeqQ : private CkNoncopyable {
  CkMsgQ<AmpiMsg> out;        // all out of order messages
  CkPagedVector<AmpiOtherElement>  elements;   // element info

  void putOutOfOrder(int srcIdx, AmpiMsg *msg);
  
public:
  AmpiSeqQ() {}
  void init(int numP);
  ~AmpiSeqQ ();
  void pup(PUP::er &p);
  
  /// Insert this message in the table.  Returns the number 
  ///  of messages now available for the element.
  ///   If 0, the message was out-of-order and is buffered.
  ///   If 1, this message can be immediately processed.
  ///   If >1, this message can be immediately processed,
  ///    and you should call "getOutOfOrder" repeatedly.
  inline int put(int srcIdx, AmpiMsg *msg) {
     AmpiOtherElement &el=elements[srcIdx];
     if (msg->seq==el.seqIncoming) { // In order:
       el.seqIncoming++;
       return 1+el.nOut;
     }
     else { // Out of order: stash message
       putOutOfOrder(srcIdx, msg);
       return 0;
     }
  }
  
  /// Get an out-of-order message from the table.
  ///  (in-order messages never go into the table)
  AmpiMsg *getOutOfOrder(int p);
  
  /// Return the next outgoing sequence number, and increment it.
  int nextOutgoing(int p) {
     return elements[p].seqOutgoing++;
  }
};
PUPmarshall(AmpiSeqQ)


inline CProxy_ampi ampiCommStruct::getProxy(void) const {return ampiID;}
const ampiCommStruct &universeComm2CommStruct(MPI_Comm universeNo);

/* KeyValue class for caching */
class KeyvalNode {
public:
	MPI_Copy_function *copy_fn;
	MPI_Delete_function *delete_fn;
	void *extra_state;
	/* value is associated with getKeyvals of communicator */

	KeyvalNode(void): copy_fn(NULL), delete_fn(NULL), extra_state(NULL)
	{ }
	KeyvalNode(MPI_Copy_function *cf, MPI_Delete_function *df, void* es):
		copy_fn(cf), delete_fn(df), extra_state(es)
	{ }
	// KeyvalNode is not supposed to be pup'ed
	void pup(PUP::er& p){ /* empty */ }
};

/*
An ampiParent holds all the communicators and the TCharm thread
for its children, which are bound to it.
*/
class ampiParent : public CBase_ampiParent {
    CProxy_TCharm threads;
    TCharm *thread;
    void prepareCtv(void);

    MPI_Comm worldNo; //My MPI_COMM_WORLD
    ampi *worldPtr; //AMPI element corresponding to MPI_COMM_WORLD
    ampiCommStruct worldStruct;
    ampiCommStruct selfStruct;

    CkPupPtrVec<ampiCommStruct> splitComm; //Communicators from MPI_Comm_split
    CkPupPtrVec<ampiCommStruct> groupComm; //Communicators from MPI_Comm_group
    CkPupPtrVec<ampiCommStruct> cartComm;  //Communicators from MPI_Cart_create
    CkPupPtrVec<ampiCommStruct> graphComm; //Communicators from MPI_Graph_create
    CkPupPtrVec<ampiCommStruct> interComm; //Communicators from MPI_Intercomm_create
    CkPupPtrVec<ampiCommStruct> intraComm; //Communicators from MPI_Intercomm_merge

    CkPupPtrVec<groupStruct> groups; // "Wild" groups that don't have a communicator
    CkPupPtrVec<WinStruct> winStructList;   //List of windows for one-sided communication
    CkPupPtrVec<InfoStruct> infos; // list of all MPI_Infos

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
      return (comm>=MPI_COMM_FIRST_GROUP && comm<MPI_COMM_FIRST_CART);
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

    void cartChildRegister(const ampiCommStruct &s);
    void graphChildRegister(const ampiCommStruct &s);
    void interChildRegister(const ampiCommStruct &s);

    inline int isIntra(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_INTRA && comm<MPI_COMM_FIRST_RESVD);
    }
    const ampiCommStruct &getIntra(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_INTRA;
      if (idx>=intraComm.size()) CkAbort("Bad intra-communicator used");
      return *intraComm[idx];
    }
    void intraChildRegister(const ampiCommStruct &s);

    // MPI MPI_Attr_get C binding returns a *pointer* to an integer,
    //  so there needs to be some storage somewhere to point to.
    int* kv_builtin_storage;
    int kv_is_builtin(int keyval);
    CkPupPtrVec<KeyvalNode> kvlist;

    int RProxyCnt;
    CProxy_ampi tmpRProxy;

public:
    int ampiInitCallDone;

public:
    ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_);
    ampiParent(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    void ckJustRestored(void);
    ~ampiParent();

    ampi *lookupComm(MPI_Comm comm) {
       if (comm!=worldStruct.getComm())
          CkAbort("ampiParent::lookupComm> Bad communicator!");
       return worldPtr;
    }

    //Children call this when they are first created, or just migrated
    TCharm *registerAmpi(ampi *ptr,ampiCommStruct s,bool forMigration);

    // exchange proxy info between two ampi proxies
    void ExchangeProxy(CProxy_ampi rproxy){
      if(RProxyCnt==0){ tmpRProxy=rproxy; RProxyCnt=1; }
      else if(RProxyCnt==1) { tmpRProxy.setRemoteProxy(rproxy); rproxy.setRemoteProxy(tmpRProxy); RProxyCnt=0; }
      else CkAbort("ExchangeProxy: RProxyCnt>1");
    }

    //Grab the next available split/group communicator
    MPI_Comm getNextSplit(void) const {return MPI_COMM_FIRST_SPLIT+splitComm.size();}
    MPI_Comm getNextGroup(void) const {return MPI_COMM_FIRST_GROUP+groupComm.size();}
    MPI_Comm getNextCart(void) const {return MPI_COMM_FIRST_CART+cartComm.size();}
    MPI_Comm getNextGraph(void) const {return MPI_COMM_FIRST_GRAPH+graphComm.size();}
    MPI_Comm getNextInter(void) const {return MPI_COMM_FIRST_INTER+interComm.size();}
    MPI_Comm getNextIntra(void) const {return MPI_COMM_FIRST_INTRA+intraComm.size();}

    inline int isCart(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_CART && comm<MPI_COMM_FIRST_GRAPH);
    }
    ampiCommStruct &getCart(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_CART;
      if (idx>=cartComm.size()) CkAbort("Bad cartesian communicator used");
      return *cartComm[idx];
    }
    inline int isGraph(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_GRAPH && comm<MPI_COMM_FIRST_INTER);
    }
    ampiCommStruct &getGraph(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_GRAPH;
      if (idx>=graphComm.size()) CkAbort("Bad graph communicator used");
      return *graphComm[idx];
    }
    inline int isInter(MPI_Comm comm) const {
      return (comm>=MPI_COMM_FIRST_INTER && comm<MPI_COMM_FIRST_INTRA);
    }
    const ampiCommStruct &getInter(MPI_Comm comm) {
      int idx=comm-MPI_COMM_FIRST_INTER;
      if (idx>=interComm.size()) CkAbort("Bad inter-communicator used");
      return *interComm[idx];
    }

    void pup(PUP::er &p);

    inline void start_measure() {
      usesAutoMeasure = CmiFalse;
    }
    inline void stop_measure() {
      usesAutoMeasure = CmiTrue;
    }
    virtual void UserSetLBLoad(void) {
      // empty
    }

    void startCheckpoint(const char* dname);
    void Checkpoint(int len, const char* dname);
    void ResumeThread(void);
    TCharm* getTCharmThread() {return thread;}

    inline const ampiCommStruct &comm2CommStruct(MPI_Comm comm) {
      if (comm==MPI_COMM_WORLD) return worldStruct;
      if (comm==MPI_COMM_SELF) return selfStruct;
      if (comm==worldNo) return worldStruct;
      if (isSplit(comm)) return getSplit(comm);
      if (isGroup(comm)) return getGroup(comm);
      if (isCart(comm)) return getCart(comm);
      if (isGraph(comm)) return getGraph(comm);
      if (isInter(comm)) return getInter(comm);
      if (isIntra(comm)) return getIntra(comm);
      return universeComm2CommStruct(comm);
    }
    
     //ampi *comm2ampi(MPI_Comm comm);
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
      if (isCart(comm)) {
	const ampiCommStruct &st = getCart(comm);
	return st.getProxy()[thisIndex].ckLocal();
      }
      if (isGraph(comm)) {
	const ampiCommStruct &st = getGraph(comm);
	return st.getProxy()[thisIndex].ckLocal();
      }
      if (isInter(comm)) {
         const ampiCommStruct &st=getInter(comm);
	 return st.getProxy()[thisIndex].ckLocal();
      }
      if (isIntra(comm)) {
         const ampiCommStruct &st=getIntra(comm);
	 return st.getProxy()[thisIndex].ckLocal();
      }
      if (comm>MPI_COMM_WORLD) return worldPtr; //Use MPI_WORLD ampi for cross-world messages:
      CkAbort("Invalid communicator used!");
      return NULL;
    }

    inline int hasComm(const MPI_Group group){
      MPI_Comm comm = (MPI_Comm)group;
      return ( comm==MPI_COMM_WORLD || comm==worldNo || isSplit(comm) || isGroup(comm) || isCart(comm) || isGraph(comm) || isIntra(comm) ); //isInter omitted because its comm number != its group number
    }
    inline const groupStruct group2vec(MPI_Group group){
      if(hasComm(group))
        return comm2CommStruct((MPI_Comm)group).getIndices();
      if(isInGroups(group))
        return *groups[group];
      CkAbort("ampiParent::group2vec: Invalid group id!");
      return *groups[0]; //meaningless return
    }
    inline MPI_Group saveGroupStruct(groupStruct vec){
      int idx = groups.size();
      groups.resize(idx+1);
      groups[idx]=new groupStruct(vec);
      return (MPI_Group)idx;
    }
    inline int getRank(const MPI_Group group){
      groupStruct vec = group2vec(group);
      return getPosOp(thisIndex,vec);
    }
    
    inline int getMyPe(void){
      return CkMyPe();
    }
    inline int hasWorld(void) const {
      return worldPtr!=NULL;
    }
    
    inline void checkComm(MPI_Comm comm){
      if ((comm > MPI_COMM_FIRST_RESVD && comm != MPI_COMM_SELF && comm != MPI_COMM_WORLD) 
       || (isSplit(comm) && comm-MPI_COMM_FIRST_SPLIT >= splitComm.size())
       || (isGroup(comm) && comm-MPI_COMM_FIRST_GROUP >= groupComm.size())
       || (isCart(comm)  && comm-MPI_COMM_FIRST_CART  >=  cartComm.size())
       || (isGraph(comm) && comm-MPI_COMM_FIRST_GRAPH >= graphComm.size())
       || (isInter(comm) && comm-MPI_COMM_FIRST_INTER >= interComm.size())
       || (isIntra(comm) && comm-MPI_COMM_FIRST_INTRA >= intraComm.size()) )
          CkAbort("Invalide MPI_Comm\n");
    }

    /// if intra-communicator, return comm, otherwise return null group
    inline MPI_Group comm2group(const MPI_Comm comm){
      if(isInter(comm)) return MPI_GROUP_NULL;   // we don't support inter-communicator in such functions
      ampiCommStruct s = comm2CommStruct(comm);
      if(comm!=MPI_COMM_WORLD && comm!=s.getComm()) CkAbort("Error in ampiParent::comm2group()");
      return (MPI_Group)(s.getComm());
    }

    inline int getRemoteSize(const MPI_Comm comm){
      if(isInter(comm)) return getInter(comm).getRemoteIndices().size();
      else return -1;
    }
    inline MPI_Group getRemoteGroup(const MPI_Comm comm){
      if(isInter(comm)) return saveGroupStruct(getInter(comm).getRemoteIndices());
      else return MPI_GROUP_NULL;
    }

    int createKeyval(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                     int *keyval, void* extra_state);
    int freeKeyval(int *keyval);
    int putAttr(MPI_Comm comm, int keyval, void* attribute_val);
    int getAttr(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
    int deleteAttr(MPI_Comm comm, int keyval);

    CkDDT myDDTsto;
    CkDDT *myDDT;
    AmpiRequestList ampiReqs;

    //added to make sure post_ireqs in ampi class share the same pointers
    //with those in ampiReqs after pupping routines.
    //AmpiRequestList oldAmpiReqs;
    
    int addWinStruct(WinStruct* win);
    WinStruct getWinStruct(MPI_Win win);
    void removeWinStruct(WinStruct win);

#if AMPI_COUNTER
public:
    AmpiCounters counters;
#endif    
    
public:
    MPI_Info createInfo(void);
    MPI_Info dupInfo(MPI_Info info);
    void setInfo(MPI_Info info, char *key, char *value);
    int deleteInfo(MPI_Info info, char *key);    
    int getInfo(MPI_Info info, char *key, int valuelen, char *value);
    int getInfoValuelen(MPI_Info info, char *key, int *valuelen);
    int getInfoNkeys(MPI_Info info);
    int getInfoNthkey(MPI_Info info, int n, char *key);
    void freeInfo(MPI_Info info);

 public:
#if AMPIMSGLOG
    /* message logging */
    int pupBytes;
#if CMK_PROJECTIONS_USE_ZLIB && 0
    gzFile fMsgLog;
    PUP::tozDisk *toPUPer;
    PUP::fromzDisk *fromPUPer;
#else
    FILE* fMsgLog;
    PUP::toDisk *toPUPer;
    PUP::fromDisk *fromPUPer;
#endif
#endif
    void init();
    void finalize();
};

/*
An ampi manages the communication of one thread over
one MPI communicator.
*/
class ampi : public CBase_ampi {
friend class IReq;
friend class SReq;
    CProxy_ampiParent parentProxy;
    void findParent(bool forMigration);
    ampiParent *parent;
    TCharm *thread;
    bool resumeOnRecv;

    ampiCommStruct myComm;
    int myRank;
    groupStruct tmpVec; // stores temp group info
    CProxy_ampi remoteProxy; // valid only for intercommunicator

#if AMPI_COMLIB
    /// A proxy used when delegating message sends to comlib
    CProxy_ampi comlibProxy;
    
    /// References to the comlib instance handles(currently just integers)
    ComlibInstanceHandle ciStreaming;
    ComlibInstanceHandle ciBcast;
    ComlibInstanceHandle ciAllgather;
    ComlibInstanceHandle ciAlltoall;
#endif
    
    int seqEntries; //Number of elements in below arrays
    AmpiSeqQ oorder;
    void inorder(AmpiMsg *msg);

    void init(void);

  public: // entry methods

    ampi();
    ampi(CkArrayID parent_,const ampiCommStruct &s);
    ampi(CkArrayID parent_,const ampiCommStruct &s,ComlibInstanceHandle ciStreaming_,
    	ComlibInstanceHandle ciBcast_,ComlibInstanceHandle ciAllgather_,ComlibInstanceHandle ciAlltoall_);
    ampi(CkMigrateMessage *msg);
    void ckJustMigrated(void);
    void ckJustRestored(void);
    ~ampi();

    virtual void pup(PUP::er &p);

    void allInitDone(CkReductionMsg *m);
    void setInitDoneFlag();
  
    void block(void);
    void unblock(void);
    void yield(void);
    void generic(AmpiMsg *);
    void ssend_ack(int sreq);
    void reduceResult(CkReductionMsg *m);
    void splitPhase1(CkReductionMsg *msg);
    void commCreatePhase1(CkReductionMsg *msg);
    void cartCreatePhase1(CkReductionMsg *m);
    void graphCreatePhase1(CkReductionMsg *m);
    void intercommCreatePhase1(CkReductionMsg *m);
    void intercommMergePhase1(CkReductionMsg *msg);

  public: // to be used by MPI_* functions

    inline const ampiCommStruct &comm2CommStruct(MPI_Comm comm) {
      return parent->comm2CommStruct(comm);
    }

    AmpiMsg *makeAmpiMsg(int destIdx,int t,int sRank,const void *buf,int count,
                         int type,MPI_Comm destcomm, int sync=0);

#if AMPI_COMLIB
    inline void comlibsend(int t, int s, const void* buf, int count, int type, int rank, MPI_Comm destcomm);
#endif
    inline void send(int t, int s, const void* buf, int count, int type, int rank, MPI_Comm destcomm, int sync=0);
    static void sendraw(int t, int s, void* buf, int len, CkArrayID aid,
                        int idx);
    void delesend(int t, int s, const void* buf, int count, int type,  
                  int rank, MPI_Comm destcomm, CProxy_ampi arrproxy, int sync=0);
    inline int processMessage(AmpiMsg *msg, int t, int s, void* buf, int count, int type);
    inline AmpiMsg * getMessage(int t, int s, int comm, int *sts);
    int recv(int t,int s,void* buf,int count,int type,int comm,int *sts=0);
    void probe(int t,int s,int comm,int *sts);
    int iprobe(int t,int s,int comm,int *sts);
    void bcast(int root, void* buf, int count, int type,MPI_Comm comm);
    static void bcastraw(void* buf, int len, CkArrayID aid);
    void split(int color,int key,MPI_Comm *dest, int type);
    void commCreate(const groupStruct vec,MPI_Comm *newcomm);
    void cartCreate(const groupStruct vec, MPI_Comm *newcomm);
    void graphCreate(const groupStruct vec, MPI_Comm *newcomm);
    void intercommCreate(const groupStruct rvec, int root, MPI_Comm *ncomm);

    inline int isInter(void) { return myComm.isinter(); }
    void intercommMerge(int first, MPI_Comm *ncomm);

    inline int getWorldRank(void) const {return parent->thisIndex;}
    /// Return our rank in this communicator
    inline int getRank(MPI_Comm comm) const {
    	if (comm==MPI_COMM_SELF) return 0;
	else return myRank;
    }
    inline int getSize(MPI_Comm comm) const {
    	if (comm==MPI_COMM_SELF) return 1;
        else return myComm.getSize();
    }
    inline MPI_Comm getComm(void) const {return myComm.getComm();}
    inline CkVec<int> getIndices(void) const { return myComm.getindices(); }
    inline const CProxy_ampi &getProxy(void) const {return thisProxy;}
    inline const CProxy_ampi &getRemoteProxy(void) const {return remoteProxy;}
    inline void setRemoteProxy(CProxy_ampi rproxy) { remoteProxy = rproxy; thread->resume(); }
    inline int getIndexForRank(int r) const {return myComm.getIndexForRank(r);}
    inline int getIndexForRemoteRank(int r) const {return myComm.getIndexForRemoteRank(r);}
#if AMPI_COMLIB
    inline const CProxy_ampi &getComlibProxy(void) const { return comlibProxy; }
    inline ComlibInstanceHandle getStreaming(void) { return ciStreaming; }
    inline ComlibInstanceHandle getBcast(void) { return ciBcast; }
    inline ComlibInstanceHandle getAllgather(void) { return ciAllgather; }
    inline ComlibInstanceHandle getAlltoall(void) { return ciAlltoall; }

    inline Strategy* getStreamingStrategy(void) { return CkpvAccess(conv_com_object).getStrategy(ciStreaming); }
    inline Strategy* getBcastStrategy(void) { return CkpvAccess(conv_com_object).getStrategy(ciBcast); }
    inline Strategy* getAllgatherStrategy(void) { return CkpvAccess(conv_com_object).getStrategy(ciAllgather); }
    inline Strategy* getAlltoallStrategy(void) { return CkpvAccess(conv_com_object).getStrategy(ciAlltoall); }
#endif
    
    CkDDT *getDDT(void) {return parent->myDDT;}
    CthThread getThread() { return thread->getThread(); }
#if CMK_LBDB_ON
    void setMigratable(int mig) { 
      if(mig) thread->setMigratable(CmiTrue); 
      else thread->setMigratable(CmiFalse);
    }
#endif
  public:
    //These are directly used by API routines, which is hideous
    /*
    FIXME: CmmTable is only indexed by the tag, sender, and communicator.
    It should also be indexed by the source data type and length (if any).
    */
    CmmTable msgs;
    CmmTable posted_ireqs;         // posted irecv req
    int nbcasts;
    //------------------------ Added by YAN ---------------------
 private:
    CkPupPtrVec<win_obj> winObjects;
 public:
    MPI_Win createWinInstance(void *base, MPI_Aint size, int disp_unit, MPI_Info info); 
    int deleteWinInstance(MPI_Win win);
    int winGetGroup(WinStruct win, MPI_Group *group); 
    int winPut(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	       MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win);
    int winGet(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	       MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win);
    int winIGet(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, int rank,
               MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win, 
	       MPI_Request *req);
    int winIGetWait(MPI_Request *request, MPI_Status *status);
    int winIGetFree(MPI_Request *request, MPI_Status *status);
    void winRemotePut(int orgtotalsize, char* orgaddr, int orgcnt, MPI_Datatype orgtype,
		      MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
		      int winIndex, CkFutureID ftHandle, int pe_src);
    void winRemoteGet(int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp, 
		      int targcnt, MPI_Datatype targtype, 
    		      int winIndex, CkFutureID ftHandle, int pe_src);
    AmpiMsg* winRemoteIGet(int orgdisp, int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp,
			   int targcnt, MPI_Datatype targtype, int winIndex);
    int winLock(int lock_type, int rank, WinStruct win);
    int winUnlock(int rank, WinStruct win);
    void winRemoteLock(int lock_type, int winIndex, CkFutureID ftHandle, int pe_src, int requestRank);
    void winRemoteUnlock(int winIndex, CkFutureID ftHandle, int pe_src, int requestRank);
    int winAccumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
	    	      MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
		      MPI_Op op, WinStruct win);
    void winRemoteAccumulate(int orgtotalsize, char* orgaddr, int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp, 
	  		    int targcnt, MPI_Datatype targtype, 
		            MPI_Op op, int winIndex, CkFutureID ftHandle, 
			    int pe_src);
    void winSetName(WinStruct win, char *name);
    void winGetName(WinStruct win, char *name, int *length);
    win_obj* getWinObjInstance(WinStruct win); 
    int getNewSemaId(); 

    AmpiMsg* Alltoall_RemoteIGet(int disp, int targcnt, MPI_Datatype targtype, int tag);
private:
    int AlltoallGetFlag;
    void *Alltoallbuff;
public:
    void setA2AIGetFlag(void* ptr) {AlltoallGetFlag=1;Alltoallbuff=ptr;}
    void resetA2AIGetFlag() {AlltoallGetFlag=0;Alltoallbuff=NULL;} 
    //------------------------ End of code by YAN ---------------------
};

ampiParent *getAmpiParent(void);
ampi *getAmpiInstance(MPI_Comm comm);
void checkComm(MPI_Comm comm);
void checkRequest(MPI_Request req);

//Use this to mark the start of AMPI interface routines:
#define AMPIAPI(routineName) TCHARM_API_TRACE(routineName,"ampi")

#endif

