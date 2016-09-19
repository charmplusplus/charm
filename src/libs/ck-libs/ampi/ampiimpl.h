#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include <string.h> /* for strlen */

#include "ampi.h"
#include "ddt.h"
#include "charm++.h"

using std::vector;

//Uncomment for debug print statements
#define AMPI_DEBUG(...) //CkPrintf(__VA_ARGS__)

#if AMPIMSGLOG
#include "ckliststring.h"
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

#define AMPI_ALLTOALL_SHORT_MSG   32
#if CMK_CONVERSE_LAPI ||  CMK_BIGSIM_CHARM
#define AMPI_ALLTOALL_MEDIUM_MSG   4194304
#else
#define AMPI_ALLTOALL_MEDIUM_MSG   32768
#endif

typedef void (*MPI_MigrateFn)(void);

void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* invec, void* inoutvec);

PUPfunctionpointer(MPI_User_function*)

/*
 * OpStruct's are used to lookup an MPI_User_function* and check its commutativity.
 * They are also used to create AmpiOpHeader's, which are transmitted in reductions
 * that are user-defined or else lack an equivalent Charm++ reducer type.
 */
class OpStruct {
 public:
  MPI_User_function* func;
  bool isCommutative;
  OpStruct(void) {}
  OpStruct(MPI_User_function* f) : func(f), isCommutative(true) {}
  OpStruct(MPI_User_function* f, bool c) : func(f), isCommutative(c) {}
  void pup(PUP::er &p) {
    p|func;  p|isCommutative;
  }
};

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
  int lock_type;
  lockQueueEntry (int _requestRank, int _lock_type)
    : requestRank(_requestRank), lock_type(_lock_type) {}
  lockQueueEntry () {}
};

typedef CkQ<lockQueueEntry *> LockQueue;

class win_obj {
 public:
  char winName[MPI_MAX_OBJECT_NAME];
  int winNameLen;
  int initflag;

  void *baseAddr;
  MPI_Aint winSize;
  int disp_unit;
  MPI_Comm comm;

  int owner; // Rank of owner of the lock, -1 if not locked
  LockQueue lockQueue; // queue of waiting processors for the lock
                       // top of queue is the one holding the lock
                       // queue is empty if lock is not applied

  void setName(const char *src);
  void getName(char *src,int *len);

 public:
  void pup(PUP::er &p);

  win_obj();
  win_obj(const char *name, void *base, MPI_Aint size, int disp_unit, MPI_Comm comm);
  ~win_obj();

  int create(const char *name, void *base, MPI_Aint size, int disp_unit,
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

  int lock(int requestRank, int lock_type);
  int unlock(int requestRank);

  int wait();
  int post();
  int start();
  int complete();

  void lockTopQueue();
  void enqueue(int requestRank, int lock_type);
  void dequeue();
  bool emptyQueue();
};
//-----------------------End of code by YAN ----------------------

class KeyvalPair{
 protected:
  int klen, vlen;
  const char* key;
  const char* val;
 public:
  KeyvalPair(void){ }
  KeyvalPair(const char* k, const char* v);
  ~KeyvalPair(void);
  void pup(PUP::er& p){
    p|klen;
    p|vlen;
    if(p.isUnpacking()){
      if(klen>0)
        key = new char[klen+1];
      if(vlen>0)
        val = new char[vlen+1];
    }
    if(klen>0)
      p((char*)key, klen+1);
    if(vlen>0)
      p((char*)val, vlen+1);
  }
  friend class InfoStruct;
};

class InfoStruct{
  CkPupPtrVec<KeyvalPair> nodes;
  bool valid;
 public:
  InfoStruct(void):valid(true) { }
  void setvalid(bool valid_){ valid = valid_; }
  bool getvalid(void) const { return valid; }
  int set(const char* k, const char* v);
  int dup(InfoStruct& src);
  int get(const char* k, int vl, char*& v, int *flag) const;
  int deletek(const char* k);
  int get_valuelen(const char* k, int* vl, int *flag) const;
  int get_nkeys(int *nkeys) const;
  int get_nthkey(int n,char* k) const;
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
  vector<int> indices;  //indices[r] gives the array index for rank r
  vector<int> remoteIndices;  // remote group for inter-communicator

  // cartesian virtual topology parameters
  int ndims;
  vector<int> dims;
  vector<int> periods;

  // graph virtual topology parameters
  int nvertices;
  vector<int> index;
  vector<int> edges;

  // For virtual topology neighbors
  vector<int> nbors;

  // For communicator attributes (MPI_*_get_attr): indexed by keyval
  vector<void *> keyvals;

  // For communicator names
  char commName[MPI_MAX_OBJECT_NAME];
  int commNameLen;

  // Lazily fill world communicator indices
  void makeWorldIndices(void) const {
    vector<int> *ind=const_cast<vector<int> *>(&indices);
    for (int i=0;i<size;i++) ind->push_back(i);
  }
 public:
  ampiCommStruct(int ignored=0) {size=-1;isWorld=-1;isInter=0;commNameLen=0;}
  ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,int size_)
    :comm(comm_), ampiID(id_),size(size_), isWorld(1), isInter(0), commNameLen(0) {}
  ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,
                 int size_,const vector<int> &indices_)
                :comm(comm_), ampiID(id_),size(size_),isWorld(0),
                 isInter(0), indices(indices_), commNameLen(0) {}
  ampiCommStruct(MPI_Comm comm_,const CkArrayID &id_,
                 int size_,const vector<int> &indices_,
                 const vector<int> &remoteIndices_)
                :comm(comm_),ampiID(id_),size(size_),isWorld(0),isInter(1),
                 indices(indices_),remoteIndices(remoteIndices_),commNameLen(0) {}
  void setArrayID(const CkArrayID &nID) {ampiID=nID;}

  MPI_Comm getComm(void) const {return comm;}
  inline const vector<int> &getIndices(void) const {
    if (isWorld && indices.size()!=size) makeWorldIndices();
    return indices;
  }
  const vector<int> &getRemoteIndices(void) const {return remoteIndices;}
  vector<void *> &getKeyvals(void) {return keyvals;}

  void setName(const char *src) {
    CkDDT_SetName(commName, src, &commNameLen);
  }

  void getName(char *name, int *len) const {
    *len = commNameLen;
    memcpy(name, commName, *len+1);
  }

  //Get the proxy for the entire array
  CProxy_ampi getProxy(void) const;

  //Get the array index for rank r in this communicator
  int getIndexForRank(int r) const {
#if CMK_ERROR_CHECKING
    if (r>=size) CkAbort("AMPI> You passed in an out-of-bounds process rank!");
#endif
    if (isWorld) return r;
    else return indices[r];
  }
  int getIndexForRemoteRank(int r) const {
#if CMK_ERROR_CHECKING
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
  inline const vector<int> &getdims() const {return dims;}
  inline const vector<int> &getperiods() const {return periods;}
  inline int getndims() const {return ndims;}

  inline void setdims(const vector<int> &dims_) { dims = dims_; }
  inline void setperiods(const vector<int> &periods_) { periods = periods_; }
  inline void setndims(int ndims_) {ndims = ndims_; }

  /* Similar hack for graph vt */
  inline int getnvertices() const {return nvertices;}
  inline const vector<int> &getindex() const {return index;}
  inline const vector<int> &getedges() const {return edges;}

  inline void setnvertices(int nvertices_) {nvertices = nvertices_; }
  inline void setindex(const vector<int> &index_) { index = index_; }
  inline void setedges(const vector<int> &edges_) { edges = edges_; }

  inline const vector<int> &getnbors() const {return nbors;}
  inline void setnbors(const vector<int> &nbors_) { nbors = nbors_; }

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
    p|nbors;
    p|commNameLen;
    p(commName,MPI_MAX_OBJECT_NAME);
  }
};
PUPmarshall(ampiCommStruct)

class mpi_comm_worlds{
  ampiCommStruct comms[MPI_MAX_COMM_WORLDS];
 public:
  ampiCommStruct &operator[](int i) {return comms[i];}
  void pup(PUP::er &p) {
    for (int i=0;i<MPI_MAX_COMM_WORLDS;i++)
      comms[i].pup(p);
  }
};

typedef vector<int> groupStruct;
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

inline int* translateRanksOp(int n,groupStruct vec1,int* ranks1,groupStruct vec2, int *ret){
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

inline groupStruct rangeInclOp(int n, int ranges[][3], groupStruct vec, int *flag){
  groupStruct retvec;
  int first,last,stride;
  for(int i=0;i<n;i++){
    first = ranges[i][0];
    last = ranges[i][1];
    stride = ranges[i][2];
    if(stride!=0){
      for(int j=0;j<=(last-first)/stride;j++)
        retvec.push_back(vec[first+stride*j]);
    }else{
      *flag = MPI_ERR_ARG;
      return retvec;
    }
  }
  *flag = MPI_SUCCESS;
  return retvec;
}

inline groupStruct rangeExclOp(int n, int ranges[][3], groupStruct vec, int *flag){
  groupStruct retvec;
  vector<int> ranksvec;
  int first,last,stride;
  int *ranks,cnt;
  int i,j;
  for(i=0;i<n;i++){
    first = ranges[i][0];
    last = ranges[i][1];
    stride = ranges[i][2];
    if(stride!=0){
      for(j=0;j<=(last-first)/stride;j++)
        ranksvec.push_back(first+stride*j);
    }else{
      *flag = MPI_ERR_ARG;
      return retvec;
    }
  }
  cnt=ranksvec.size();
  ranks=new int[cnt];
  for(i=0;i<cnt;i++)
    ranks[i]=ranksvec[i];
  *flag = MPI_SUCCESS;
  return exclOp(cnt,ranks,vec);
}

#include "tcharm.h"
#include "tcharmc.h"

#include "ampi.decl.h"
#include "charm-api.h"
#include <sys/stat.h> // for mkdir

extern int _mpi_nworlds;

#define MPI_ATA_SEQ_TAG MPI_TAG_UB_VALUE+1
#define MPI_BCAST_TAG   MPI_TAG_UB_VALUE+10
#define MPI_REDN_TAG    MPI_TAG_UB_VALUE+11
#define MPI_SCATTER_TAG MPI_TAG_UB_VALUE+12
#define MPI_SCAN_TAG    MPI_TAG_UB_VALUE+13
#define MPI_EXSCAN_TAG  MPI_TAG_UB_VALUE+14
#define MPI_ATA_TAG     MPI_TAG_UB_VALUE+15
#define MPI_NBOR_TAG    MPI_TAG_UB_VALUE+16

#define AMPI_COLL_SOURCE 0
#define AMPI_COLL_COMM   MPI_COMM_WORLD

#define MPI_PERS_REQ    1
#define MPI_I_REQ       2
#define MPI_IATA_REQ    3
#define MPI_SEND_REQ    4
#define MPI_SSEND_REQ   5
#define MPI_REDN_REQ    6
#define MPI_GATHER_REQ  7
#define MPI_GATHERV_REQ 8
#define MPI_GPU_REQ     9

enum AmpiReqSts : bool {
  AMPI_REQ_PENDING = false,
  AMPI_REQ_COMPLETED = true
};

#define MyAlign8(x) (((x)+7)&(~7))

/**
Represents an MPI request that has been initiated
using Isend, Irecv, Ialltoall, Send_init, etc.
*/
class AmpiRequest {
 public:
  void *buf;
  int count;
  MPI_Datatype type;
  int tag; // the order must match MPI_Status
  int src;
  MPI_Comm comm;
  bool statusIreq;

#if CMK_BIGSIM_CHARM
 public:
  void *event; // the event point that corresponds to this message
  int eventPe; // the PE that the event is located on
#endif
 protected:
  bool isvalid;
 public:
  AmpiRequest(){ statusIreq=false; }
  /// Close this request (used by free and cancel)
  virtual ~AmpiRequest(){ }

  /// Activate this persistent request.
  ///  Only meaningful for persistent requests,
  ///  other requests just abort.
  virtual int start(void){ return -1; }

  /// Return true if this request is finished (progress):
  ///  test always yields before returning false.
  ///  itest does not yield before returning false for
  //   IReq's, RednReq's, Gather(v)Req's, SendReq's, and SsendReq's.
  virtual bool test(MPI_Status *sts) =0;
  virtual bool itest(MPI_Status *sts) =0;

  /// Completes the operation hanging on the request
  virtual void complete(MPI_Status *sts) =0;

  /// Block until this request is finished,
  ///  returning a valid MPI error code.
  virtual int wait(MPI_Status *sts) =0;

  /// Receive an AmpiMsg
  virtual void receive(ampi *ptr, AmpiMsg *msg) = 0;

  /// Receive a CkReductionMsg
  virtual void receive(ampi *ptr, CkReductionMsg *msg) = 0;

  /// Frees up the request: invalidate it
  virtual void free(void){ isvalid=false; }
  inline bool isValid(void) const { return isvalid; }

  /// Returns the type of request:
  ///  MPI_PERS_REQ, MPI_I_REQ, MPI_IATA_REQ, MPI_SEND_REQ, MPI_SSEND_REQ,
  //   MPI_REDN_REQ, MPI_GATHER_REQ, MPI_GATHERV_REQ, MPI_GPU_REQ
  virtual int getType(void) const =0;

  virtual void pup(PUP::er &p) {
    p((char *)&buf,sizeof(void *)); //supposed to work only with isomalloc
    p(count);
    p(type);
    p(src);
    p(tag);
    p(comm);
    p(isvalid);
    p(statusIreq);
#if CMK_BIGSIM_CHARM
    //needed for bigsim out-of-core emulation
    //as the "log" is not moved from memory, this pointer is safe
    //to be reused
    p((char *)&event, sizeof(void *));
    p(eventPe);
#endif
  }

  virtual void print();
};

class PersReq : public AmpiRequest {
  int sndrcv; // 1 if send , 2 if recv, 3 if ssend
 public:
  PersReq(void *buf_, int count_, MPI_Datatype type_, int src_, int tag_, MPI_Comm comm_, int sndrcv_){
    buf=buf_;  count=count_;  type=type_;  src=src_;  tag=tag_;
    comm=comm_;  sndrcv=sndrcv_;  isvalid=true;
  }
  PersReq(){};
  ~PersReq(){}
  int start();
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  inline int getType(void) const { return MPI_PERS_REQ; }
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
    p(sndrcv);
  }
  virtual void print();
};

class IReq : public AmpiRequest {
 public:
  int length; // recv'ed length
  IReq(void *buf_, int count_, MPI_Datatype type_, int src_, int tag_, MPI_Comm comm_){
    buf=buf_;  count=count_;  type=type_;  src=src_;  tag=tag_;
    comm=comm_;  isvalid=true; length=0;
  }
  IReq(){}
  ~IReq(){}
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  inline int getType(void) const { return MPI_I_REQ; }
  void receive(ampi *ptr, AmpiMsg *msg);
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
    p|length;
  }
  virtual void print();
};

class RednReq : public AmpiRequest {
 public:
  MPI_Op op;
  RednReq(void *buf_, int count_, MPI_Datatype type_, MPI_Comm comm_, MPI_Op op_){
    buf=buf_;  count=count_;  type=type_;  src=AMPI_COLL_SOURCE;  tag=MPI_REDN_TAG;
    comm=comm_;  op=op_;  isvalid=true;
  }
  RednReq(){};
  ~RednReq(){}
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  inline int getType(void) const { return MPI_REDN_REQ; }
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg);
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
    p|op;
  }
  virtual void print();
};

class GatherReq : public AmpiRequest {
 public:
  GatherReq(void *buf_, int count_, MPI_Datatype type_, MPI_Comm comm_){
    buf=buf_;  count=count_;  type=type_;  src=AMPI_COLL_SOURCE;  tag=MPI_REDN_TAG;
    comm=comm_;  isvalid=true;
  }
  GatherReq(){}
  ~GatherReq(){}
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  inline int getType(void) const { return MPI_GATHER_REQ; }
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg);
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
  }
  virtual void print();
};

class GathervReq : public AmpiRequest {
 public:
  vector<int> recvCounts;
  vector<int> displs;
  GathervReq(void *buf_, int count_, MPI_Datatype type_, MPI_Comm comm_, int *rc, int *d){
    buf=buf_;  count=count_;  type=type_;  src=AMPI_COLL_SOURCE;  tag=MPI_REDN_TAG;
    comm=comm_;  isvalid=true;
    recvCounts.resize(count);
    for(int i=0; i<count; i++) recvCounts[i]=rc[i];
    displs.resize(count);
    for(int i=0; i<count; i++) displs[i]=d[i];
  }
  GathervReq(){}
  ~GathervReq(){}
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  inline int getType(void) const { return MPI_GATHERV_REQ; }
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg);
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
    p|recvCounts;  p|displs;
  }
  virtual void print();
};

class SendReq : public AmpiRequest {
 public:
  SendReq(MPI_Comm comm_) {
    comm = comm_; isvalid=true;
  }
  SendReq(){}
  ~SendReq(){ }
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  inline int getType(void) const { return MPI_SEND_REQ; }
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
  }
  virtual void print();
};

class SsendReq : public AmpiRequest {
 public:
  SsendReq(MPI_Comm comm_) {
    comm = comm_; isvalid=true;
  }
  SsendReq() {}
  ~SsendReq(){ }
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  inline int getType(void) const { return MPI_SSEND_REQ; }
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
  }
  virtual void print();
};

class GPUReq : public AmpiRequest {
 public:
  GPUReq();
  inline int getType(void) const { return MPI_GPU_REQ; }
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  void receive(ampi *ptr, AmpiMsg *msg);
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  void setComplete();
};

class IATAReq : public AmpiRequest {
  vector<IReq> myreqs;
  int elmcount;
  int idx;
 public:
  IATAReq(int c_):elmcount(c_),idx(0){ myreqs.resize(c_); isvalid=true; }
  IATAReq(){};
  ~IATAReq(void) { }
  int addReq(void *buf_, int count_, MPI_Datatype type_, int src_, int tag_, MPI_Comm comm_){
    myreqs[idx].buf=buf_;   myreqs[idx].count=count_;
    myreqs[idx].type=type_; myreqs[idx].src=src_;
    myreqs[idx].tag=tag_;   myreqs[idx].comm=comm_;
    return (++idx);
  }
  bool test(MPI_Status *sts);
  bool itest(MPI_Status *sts);
  void complete(MPI_Status *sts);
  int wait(MPI_Status *sts);
  void receive(ampi *ptr, AmpiMsg *msg) {}
  void receive(ampi *ptr, CkReductionMsg *msg) {}
  inline int getCount(void) const { return elmcount; }
  inline int getType(void) const { return MPI_IATA_REQ; }
  virtual void pup(PUP::er &p){
    AmpiRequest::pup(p);
    p(elmcount);
    p(idx);
    p|myreqs;
  }
  virtual void print();
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

  inline void checkRequest(MPI_Request idx) const {
    if(!(idx==-1 || (idx < this->len && (block[idx])->isValid())))
      CkAbort("Invalid MPI_Request\n");
  }

  //find an AmpiRequest by its pointer value
  //return -1 if not found!
  int findRequestIndex(AmpiRequest *req) const {
    for(int i=0; i<len; i++){
      if(block[i]==req) return i;
    }
    return -1;
  }

  void pup(PUP::er &p);

  void print(){
    for(int i=0; i<len; i++){
      if(block[i]==NULL) continue;
      CkPrintf("AmpiRequestList Element %d [%p]: \n", i+1, block[i]);
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
#if CMK_BIGSIM_CHARM
  void *event;
  int  eventPe; // the PE that the event is located
#endif
  char *data;

  AmpiMsg(void) { data = NULL; }
  AmpiMsg(int _s, int t, int sIdx,int sRank, int l, int c) :
    seq(_s), tag(t),srcIdx(sIdx), srcRank(sRank), comm(c), length(l) {}
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
  CkMsgQ<AmpiMsg> out; // all out of order messages
  CkPagedVector<AmpiOtherElement>  elements; // element info

  void putOutOfOrder(int srcIdx, AmpiMsg *msg);

public:
  AmpiSeqQ() {}
  void init(int numP);
  ~AmpiSeqQ ();
  void pup(PUP::er &p);

  /// Insert this message in the table.  Returns the number
  /// of messages now available for the element.
  ///   If 0, the message was out-of-order and is buffered.
  ///   If 1, this message can be immediately processed.
  ///   If >1, this message can be immediately processed,
  ///     and you should call "getOutOfOrder" repeatedly.
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
  /// (in-order messages never go into the table)
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
  KeyvalNode(void): copy_fn(NULL), delete_fn(NULL), extra_state(NULL) { }
  KeyvalNode(MPI_Copy_function *cf, MPI_Delete_function *df, void* es):
             copy_fn(cf), delete_fn(df), extra_state(es) { }
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
  CkPupPtrVec<WinStruct> winStructList; //List of windows for one-sided communication
  CkPupPtrVec<InfoStruct> infos; // list of all MPI_Infos
  vector<OpStruct> ops; // list of all MPI_Ops

  inline int isSplit(MPI_Comm comm) const {
    return (comm>=MPI_COMM_FIRST_SPLIT && comm<MPI_COMM_FIRST_GROUP);
  }
  const ampiCommStruct &getSplit(MPI_Comm comm) const {
    int idx=comm-MPI_COMM_FIRST_SPLIT;
    if (idx>=splitComm.size()) CkAbort("Bad split communicator used");
    return *splitComm[idx];
  }
  void splitChildRegister(const ampiCommStruct &s);

  inline int isGroup(MPI_Comm comm) const {
    return (comm>=MPI_COMM_FIRST_GROUP && comm<MPI_COMM_FIRST_CART);
  }
  const ampiCommStruct &getGroup(MPI_Comm comm) const {
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
  const ampiCommStruct &getIntra(MPI_Comm comm) const {
    int idx=comm-MPI_COMM_FIRST_INTRA;
    if (idx>=intraComm.size()) CkAbort("Bad intra-communicator used");
    return *intraComm[idx];
  }
  void intraChildRegister(const ampiCommStruct &s);

  /* MPI_*_get_attr C binding returns a *pointer* to an integer,
   *  so there needs to be some storage somewhere to point to.
   * All builtin keyvals are ints, except for MPI_WIN_BASE, which
   *  is a pointer, and MPI_WIN_SIZE, which is an MPI_Aint. */
  int* kv_builtin_storage;
  MPI_Aint* win_size_storage;
  void** win_base_storage;
  bool kv_set_builtin(int keyval, void* attribute_val);
  bool kv_get_builtin(int keyval);
  CkPupPtrVec<KeyvalNode> kvlist;

  int RProxyCnt;
  CProxy_ampi tmpRProxy;

  MPI_MigrateFn userAboutToMigrateFn, userJustMigratedFn;

 public:
  int ampiInitCallDone;

 public:
  ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_);
  ampiParent(CkMigrateMessage *msg);
  void ckAboutToMigrate(void);
  void ckJustMigrated(void);
  void ckJustRestored(void);
  void setUserAboutToMigrateFn(MPI_MigrateFn f);
  void setUserJustMigratedFn(MPI_MigrateFn f);
  ~ampiParent();

  ampi *lookupComm(MPI_Comm comm) const {
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
  ampiCommStruct &getCart(MPI_Comm comm) const {
    int idx=comm-MPI_COMM_FIRST_CART;
    if (idx>=cartComm.size()) CkAbort("AMPI> Bad cartesian communicator used!\n");
    return *cartComm[idx];
  }
  inline int isGraph(MPI_Comm comm) const {
    return (comm>=MPI_COMM_FIRST_GRAPH && comm<MPI_COMM_FIRST_INTER);
  }
  ampiCommStruct &getGraph(MPI_Comm comm) const {
    int idx=comm-MPI_COMM_FIRST_GRAPH;
    if (idx>=graphComm.size()) CkAbort("AMPI> Bad graph communicator used!\n");
    return *graphComm[idx];
  }
  inline int isInter(MPI_Comm comm) const {
    return (comm>=MPI_COMM_FIRST_INTER && comm<MPI_COMM_FIRST_INTRA);
  }
  const ampiCommStruct &getInter(MPI_Comm comm) const {
    int idx=comm-MPI_COMM_FIRST_INTER;
    if (idx>=interComm.size()) CkAbort("AMPI> Bad inter-communicator used!\n");
    return *interComm[idx];
  }

  void pup(PUP::er &p);

  inline void start_measure() {
    usesAutoMeasure = false;
  }
  inline void stop_measure() {
    usesAutoMeasure = true;
  }
  virtual void UserSetLBLoad(void) {
    // empty
  }

  void startCheckpoint(const char* dname);
  void Checkpoint(int len, const char* dname);
  void ResumeThread(void);
  TCharm* getTCharmThread() const {return thread;}

  inline const ampiCommStruct &comm2CommStruct(MPI_Comm comm) const {
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
  inline ampi *comm2ampi(MPI_Comm comm) const {
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

  inline int hasComm(const MPI_Group group) const {
    MPI_Comm comm = (MPI_Comm)group;
    return ( comm==MPI_COMM_WORLD || comm==worldNo || isSplit(comm) || isGroup(comm) ||
             isCart(comm) || isGraph(comm) || isIntra(comm) );
    //isInter omitted because its comm number != its group number
  }
  inline const groupStruct group2vec(MPI_Group group) const {
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
  inline int getRank(const MPI_Group group) const {
    groupStruct vec = group2vec(group);
    return getPosOp(thisIndex,vec);
  }

  inline int getMyPe(void) const {
    return CkMyPe();
  }
  inline int hasWorld(void) const {
    return worldPtr!=NULL;
  }

  inline void checkComm(MPI_Comm comm) const {
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
  inline MPI_Group comm2group(const MPI_Comm comm) const {
    if(isInter(comm)) return MPI_GROUP_NULL;   // we don't support inter-communicator in such functions
    ampiCommStruct s = comm2CommStruct(comm);
    if(comm!=MPI_COMM_WORLD && comm!=s.getComm()) CkAbort("Error in ampiParent::comm2group()");
    return (MPI_Group)(s.getComm());
  }

  inline int getRemoteSize(const MPI_Comm comm) const {
    if(isInter(comm)) return getInter(comm).getRemoteIndices().size();
    else return -1;
  }
  inline MPI_Group getRemoteGroup(const MPI_Comm comm) {
    if(isInter(comm)) return saveGroupStruct(getInter(comm).getRemoteIndices());
    else return MPI_GROUP_NULL;
  }

  int createKeyval(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                  int *keyval, void* extra_state);
  int freeKeyval(int *keyval);
  bool getBuiltinKeyval(int keyval, void *attribute_val);
  int setUserKeyval(MPI_Comm comm, int keyval, void *attribute_val);
  bool getUserKeyval(MPI_Comm comm, int keyval, void *attribute_val, int *flag);

  int setCommAttr(MPI_Comm comm, int keyval, void *attribute_val);
  int getCommAttr(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
  int deleteCommAttr(MPI_Comm comm, int keyval);

  int setWinAttr(MPI_Win win, int keyval, void *attribute_val);
  int getWinAttr(MPI_Win win, int keyval, void *attribute_val, int *flag);
  int deleteWinAttr(MPI_Win win, int keyval);

  CkDDT myDDTsto;
  CkDDT *myDDT;
  AmpiRequestList ampiReqs;

  int addWinStruct(WinStruct* win);
  WinStruct getWinStruct(MPI_Win win) const;
  void removeWinStruct(WinStruct win);

 public:
  int createInfo(MPI_Info *newinfo);
  int dupInfo(MPI_Info info, MPI_Info *newinfo);
  int setInfo(MPI_Info info, const char *key, const char *value);
  int deleteInfo(MPI_Info info, const char *key);
  int getInfo(MPI_Info info, const char *key, int valuelen, char *value, int *flag) const;
  int getInfoValuelen(MPI_Info info, const char *key, int *valuelen, int *flag) const;
  int getInfoNkeys(MPI_Info info, int *nkeys) const;
  int getInfoNthkey(MPI_Info info, int n, char *key) const;
  int freeInfo(MPI_Info info);

  void initOps(void);
  inline int createOp(MPI_User_function *fn, int isCommutative) {
    OpStruct newop = OpStruct(fn, isCommutative);
    ops.push_back(newop);
    return ops.size()-1;
  }
  inline bool opIsPredefined(MPI_Op op) const {
    return (op>=MPI_INFO_NULL && op<=MPI_NO_OP);
  }
  inline bool opIsCommutative(MPI_Op op) const {
    CkAssert(op>MPI_OP_NULL && op<ops.size());
    return ops[op].isCommutative;
  }
  inline MPI_User_function* op2User_function(MPI_Op op) const {
    CkAssert(op>MPI_OP_NULL && op<ops.size());
    return ops[op].func;
  }
  inline AmpiOpHeader op2AmpiOpHeader(MPI_Op op, MPI_Datatype type, int count) const {
    CkAssert(op>MPI_OP_NULL && op<ops.size());
    int size = myDDT->getType(type)->getSize(count);
    return AmpiOpHeader(ops[op].func, type, count, size);
  }

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
  friend class IReq; // for checking resumeOnRecv
  friend class SendReq;
  friend class SsendReq;
  friend class RednReq;
  friend class GatherReq;
  friend class GathervReq;
  CProxy_ampiParent parentProxy;
  void findParent(bool forMigration);
  ampiParent *parent;
  TCharm *thread;
  bool resumeOnRecv;
  bool resumeOnColl;
  AmpiRequest *blockingReq;

  ampiCommStruct myComm;
  int myRank;
  groupStruct tmpVec; // stores temp group info
  CProxy_ampi remoteProxy; // valid only for intercommunicator

  int seqEntries; //Number of elements in below arrays
  AmpiSeqQ oorder;
  void inorder(AmpiMsg *msg);

  void init(void);

 public: // entry methods

  ampi();
  ampi(CkArrayID parent_,const ampiCommStruct &s);
  ampi(CkMigrateMessage *msg);
  void ckJustMigrated(void);
  void ckJustRestored(void);
  ~ampi();

  void pup(PUP::er &p);

  void allInitDone();
  void setInitDoneFlag();

  void block(void);
  void unblock(void);
  void yield(void);
  void generic(AmpiMsg *);
  void ssend_ack(int sreq);
  void barrierResult(void);
  void ibarrierResult(void);
  void rednResult(CkReductionMsg *msg);
  void irednResult(CkReductionMsg *msg);

  void splitPhase1(CkReductionMsg *msg);
  void commCreatePhase1(MPI_Comm nextGroupComm);
  void intercommCreatePhase1(MPI_Comm nextInterComm);
  void intercommMergePhase1(MPI_Comm nextIntraComm);

 private: // Used by the above entry methods that create new MPI_Comm objects
  CProxy_ampi createNewChildAmpiSync();
  void insertNewChildAmpiElements(MPI_Comm newComm, CProxy_ampi newAmpi);

 public: // to be used by MPI_* functions

  inline const ampiCommStruct &comm2CommStruct(MPI_Comm comm) const {
    return parent->comm2CommStruct(comm);
  }

  inline ampi* blockOnRecv(void);
  inline ampi* blockOnColl(void);
  inline ampi* blockOnRedn(AmpiRequest *req);
  inline MPI_Request postReq(AmpiRequest* newreq, AmpiReqSts status=AMPI_REQ_PENDING);

  AmpiMsg *makeAmpiMsg(int destIdx,int t,int sRank,const void *buf,int count,
                       MPI_Datatype type,MPI_Comm destcomm, int sync=0);

  void send(int t, int s, const void* buf, int count, MPI_Datatype type, int rank,
            MPI_Comm destcomm, int sync=0);
  static void sendraw(int t, int s, void* buf, int len, CkArrayID aid,
                      int idx);
  void delesend(int t, int s, const void* buf, int count, MPI_Datatype type,
                int rank, MPI_Comm destcomm, CProxy_ampi arrproxy, int sync=0);
  inline void processAmpiMsg(AmpiMsg *msg, void* buf, MPI_Datatype type, int count);
  inline void processRednMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int count);
  inline void processNoncommutativeRednMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int count,
                                           MPI_User_function* func);
  inline void processGatherMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int recvCount);
  inline void processGathervMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type,
                               int* recvCounts, int* displs);
  inline AmpiMsg * getMessage(int t, int s, MPI_Comm comm, int *sts) const;
  int recv(int t,int s,void* buf,int count,MPI_Datatype type,MPI_Comm comm,MPI_Status *sts=NULL);
  void irecv(void *buf, int count, MPI_Datatype type, int src,
             int tag, MPI_Comm comm, MPI_Request *request);
  void sendrecv(void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                void *rbuf, int rcount, MPI_Datatype rtype, int src, int rtag,
                MPI_Comm comm, MPI_Status *sts);
  void probe(int t,int s,MPI_Comm comm,MPI_Status *sts);
  int iprobe(int t,int s,MPI_Comm comm,MPI_Status *sts);
  void barrier(void);
  void ibarrier(MPI_Request *request);
  void bcast(int root, void* buf, int count, MPI_Datatype type, MPI_Comm comm);
  void ibcast(int root, void* buf, int count, MPI_Datatype type, MPI_Comm comm, MPI_Request* request);
  static void bcastraw(void* buf, int len, CkArrayID aid);
  void split(int color,int key,MPI_Comm *dest, int type);
  void commCreate(const groupStruct vec,MPI_Comm *newcomm);
  void cartCreate(const groupStruct vec, MPI_Comm *newcomm);
  void graphCreate(const groupStruct vec, MPI_Comm *newcomm);
  void intercommCreate(const groupStruct rvec, int root, MPI_Comm *ncomm);

  inline int isInter(void) const { return myComm.isinter(); }
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
  inline void setCommName(const char *name){myComm.setName(name);}
  inline void getCommName(char *name, int *len) const {myComm.getName(name,len);}
  inline vector<int> getIndices(void) const { return myComm.getIndices(); }
  inline const CProxy_ampi &getProxy(void) const {return thisProxy;}
  inline const CProxy_ampi &getRemoteProxy(void) const {return remoteProxy;}
  inline void setRemoteProxy(CProxy_ampi rproxy) { remoteProxy = rproxy; thread->resume(); }
  inline int getIndexForRank(int r) const {return myComm.getIndexForRank(r);}
  inline int getIndexForRemoteRank(int r) const {return myComm.getIndexForRemoteRank(r);}
  void findNeighbors(MPI_Comm comm, int rank, vector<int>& neighbors) const;
  inline const vector<int>& getNeighbors() const { return myComm.getnbors(); }
  inline bool opIsCommutative(MPI_Op op) const { return parent->opIsCommutative(op); }
  inline MPI_User_function* op2User_function(MPI_Op op) const { return parent->op2User_function(op); }

  CkDDT *getDDT(void) const {return parent->myDDT;}
  CthThread getThread() const { return thread->getThread(); }
#if CMK_LBDB_ON
  void setMigratable(int mig) {
    if(mig) thread->setMigratable(true);
    else thread->setMigratable(false);
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
  //------------------------ Added by YAN ---------------------
 private:
  CkPupPtrVec<win_obj> winObjects;
 public:
  MPI_Win createWinInstance(void *base, MPI_Aint size, int disp_unit, MPI_Info info);
  int deleteWinInstance(MPI_Win win);
  int winGetGroup(WinStruct win, MPI_Group *group) const;
  int winPut(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win);
  int winGet(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win);
  int winIget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, int rank,
              MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, WinStruct win,
              MPI_Request *req);
  int winIgetWait(MPI_Request *request, MPI_Status *status);
  int winIgetFree(MPI_Request *request, MPI_Status *status);
  void winRemotePut(int orgtotalsize, char* orgaddr, int orgcnt, MPI_Datatype orgtype,
                    MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, int winIndex);
  AmpiMsg* winRemoteGet(int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp,
                    int targcnt, MPI_Datatype targtype, int winIndex);
  AmpiMsg* winRemoteIget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, MPI_Aint targdisp,
                         int targcnt, MPI_Datatype targtype, int winIndex);
  int winLock(int lock_type, int rank, WinStruct win);
  int winUnlock(int rank, WinStruct win);
  void winRemoteLock(int lock_type, int winIndex, int requestRank);
  void winRemoteUnlock(int winIndex, int requestRank);
  int winAccumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                    MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                    MPI_Op op, WinStruct win);
  void winRemoteAccumulate(int orgtotalsize, char* orgaddr, int orgcnt, MPI_Datatype orgtype,
                           MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                           MPI_Op op, int winIndex);
  int winGetAccumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, void *resaddr,
                       int rescnt, MPI_Datatype restype, int rank, MPI_Aint targdisp,
                       int targcnt, MPI_Datatype targtype, MPI_Op op, WinStruct win);
  int winCompareAndSwap(void *orgaddr, void *compaddr, void *resaddr, MPI_Datatype type,
                        int rank, MPI_Aint targdisp, WinStruct win);
  AmpiMsg* winRemoteCompareAndSwap(int size, char *sorgaddr, char *compaddr, MPI_Datatype type,
                                   MPI_Aint targdisp, int winIndex);
  void winSetName(WinStruct win, const char *name);
  void winGetName(WinStruct win, char *name, int *length) const;
  win_obj* getWinObjInstance(WinStruct win) const;
  int getNewSemaId();

  AmpiMsg* Alltoall_RemoteIget(MPI_Aint disp, int targcnt, MPI_Datatype targtype, int tag);
 private:
  int AlltoallGetFlag;
  void *Alltoallbuff;
 public:
  void setA2AIgetFlag(void* ptr) {AlltoallGetFlag=1;Alltoallbuff=ptr;}
  void resetA2AIgetFlag() {AlltoallGetFlag=0;Alltoallbuff=NULL;}
  //------------------------ End of code by YAN ---------------------
};

ampiParent *getAmpiParent(void);
ampi *getAmpiInstance(MPI_Comm comm);
void checkComm(MPI_Comm comm);
void checkRequest(MPI_Request req);
void handle_MPI_BOTTOM(void* &buf, MPI_Datatype type);
void handle_MPI_BOTTOM(void* &buf1, MPI_Datatype type1, void* &buf2, MPI_Datatype type2);

#if AMPI_ERROR_CHECKING
int ampiErrhandler(const char* func, int errcode);
#else
#define ampiErrhandler(func, errcode) (errcode)
#endif

//Use this to mark the start of AMPI interface routines:
#define AMPIAPI(routineName) TCHARM_API_TRACE(routineName,"ampi")

#endif

