#ifndef __AMR_H
#define __AMR_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "bitvec.h"
#include "amr.decl.h"
#include "statcoll.h"
#include "fifo.h"
#include <math.h>

#define NEG_X 0
#define POS_X 1
#define NEG_Y 2
#define POS_Y 3
#define NEG_Z 4
#define POS_Z 5
#define DEBUGA(x) /*CkPrintf x*/
#define DEBUGR(x) /*CkPrintf x */
#define DEBUGT(x) /*CkPrintf x*/
#define DEBUGS(x) /*CkPrintf x*/
#define DEBUGRC(x) /*CkPrintf x*/
#define DEBUGJ(x) /*CkPrintf x*/
#define DEBUG(x) /*CkPrintf x*/
#define DEBUGN(x) /*CkPrintf x*/

class NeighborMsg;
class ChildInitMsg;

class AmrUserData:public PUP::able {
  int dimension;
 public:
  BitVec myIndex;
  AmrUserData(){}
  AmrUserData(CkMigrateMessage *m): PUP::able(m){}
  virtual void init()
    {}
  static AmrUserData *createDataWrapper(BitVec idx, int dim) {
    // init(idx,dim);
    AmrUserData* ptr = createData();
    ptr->myIndex = idx;
    ptr->dimension = dim;
    ptr->init();
    return ptr;
  }
  static AmrUserData *createDataWrapper(BitVec idx, int dim, void *data, int dataSize)
    {
      //      init(idx,dim);
      AmrUserData* ptr = createData(data, dataSize);
      ptr->myIndex = idx;
      ptr->dimension = dim;
      ptr->init();
      return ptr;
    }

  static AmrUserData *createData();
  static AmrUserData *createData(void *data, int dataSize);
  
  static void deleteNborData(void * data);
  static void deleteChildData(void * data);
  
  NeighborMsg **fragment(NeighborMsg* msg,int nMsg);
  void combineAndStore(NeighborMsg *msg1,NeighborMsg *msg2);

  void combineAndStore(NeighborMsg* msg1, NeighborMsg *msg2,NeighborMsg *msg3, NeighborMsg *msg4);
  void store(NeighborMsg *msg);
 
  bool isOnNegXBoundary() {
    return (myIndex.vec[0] == 0)? true:false;
  }
  bool isOnPosXBoundary() {
    int i = myIndex.numbits/dimension;
    int mask=0, mult = 1;
    for(int k=i; k>0; k--) {
      mask += mult;
      mult = mult << 1;
    }
    return ((myIndex.vec[0] & mask) ==mask)? true:false;
  }

  bool isOnNegYBoundary() {
    return (myIndex.vec[1] == 0)? true:false;
  }

  bool isOnPosYBoundary() {
    int i = myIndex.numbits/dimension;
    int mask=0, mult = 1;
    for(int k=i; k>0; k--) {
      mask += mult;
      mult = mult << 1;
    }
    return ((myIndex.vec[1] & mask) ==mask)? true:false;
  }

  bool isOnNegZBoundary() {
    return (myIndex.vec[2] == 0)? true:false;
  }

  bool isOnPosZBoundary() {
    int i = myIndex.numbits/dimension;
    int mask=0, mult = 1;
    for(int k=i; k>0; k--) {
      mask += mult;
      mult = mult << 1;
    }
    return ((myIndex.vec[2] & mask) ==mask)? true:false;
  }

  virtual void ** fragmentNborData(void* data, int *sizePtr){ return NULL;}
  virtual void ** getNborMsgArray(int* sizePtr){return NULL;}
  
  virtual void store(void* data , int dataSize, int neighborSide){}
  virtual void combineAndStore(void **dataArray, int dataSize,int neighborSide){}
  virtual bool refineCriterion(void){return false;}
  //returns an array of userdefined data to be given to the children
  //gets an array of 4 for quad tree
  virtual void **fragmentForRefine(int *sizePtr){ return NULL;}
  virtual void doComputation(void){}
  virtual void pup(PUP::er &p) {
    PUP::able::pup(p);
    myIndex.pup(p);
    p(dimension);
  }
  PUPable_decl(AmrUserData);
  virtual ~AmrUserData(){}  
};

class NeighborMsg : public CMessage_NeighborMsg 
{
 public:
  int which_neighbor;
  int run_until;
  int numbits;
  int dataSize;
  BitVec nborIdx;
  void *data;
  NeighborMsg() {
    data = NULL;
  }
  static void* pack(NeighborMsg *msg);
  static NeighborMsg* unpack(void *inbuf);
  void pup(PUP::er &p);


  ~NeighborMsg(){ 
    AmrUserData::deleteNborData(data);
  }
};

class ChildInitMsg : public CMessage_ChildInitMsg 
{
 public:
  int run_until;
  int num_neighbors;
  int synchstep;
  int dataSize;
  void *data;
  static void* pack(ChildInitMsg *msg);
  static ChildInitMsg* unpack(void *inbuf);
 
  ~ChildInitMsg() { //delete (double*) data;
    AmrUserData::deleteChildData(data);
  }
};

class _DMsg : public CMessage__DMsg {
 public:
  BitVec sender;
  int from;
  _DMsg(){}
  _DMsg(BitVec sendVec, int pos){
    sender = sendVec;
    from = pos;
  }
};

class _RefineChkMsg : public CMessage__RefineChkMsg {
 public:
  BitVec index;
  int run_until;
  _RefineChkMsg() {}
  _RefineChkMsg(BitVec idx,int run) {
    index = idx;
    run_until = run;
  }
};

class _RefineMsg : public CMessage__RefineMsg {
 public:
  // autorefine = 0 for regular refine messages
  //            = 1 for automatic refinement messages
  int autorefine; 
  // index has the return address to which the response is to be sent to
  // for automatic refinement messages
  BitVec index;

  _RefineMsg() { autorefine = 0;}
  _RefineMsg(int reftype) {
    autorefine = reftype;
    if(autorefine == 1) {
      CkError("Automatic refinement message without a return address\n");
    }
  }
  _RefineMsg(int reftype,BitVec idx) {
    index = idx;
    autorefine = reftype;
  }
};


class _RedMsg : public CMessage__RedMsg {
 public:
  _RedMsg() { type  = 0;}
  _RedMsg(int x) {type = x;}
  int type;
  /*type = 0 synchronise before refinement
         = 1 synchronise after refinement
         = 2 synchronise at the end of program
   */
};

class _ArrInitMsg : public CMessage__ArrInitMsg 
{
 public:
  BitVec parent; //bitvector of the parent
  char type; //type of the node that is being made ie r (root), n (node)
             // v (virtual leaf) 
  int interval;//synchronisation interval
  int depth; //depth of the tree tobe created
  int totalIterations; //total iterations for the simulation
  CkChareID coordHandle;
  bool statCollection;
  // CProxy_StatCollector grpProxy;
  CkGroupID gid;
  //int dimension;
};

class StartUpMsg :public CMessage_StartUpMsg
{
 public:
  int depth;
  int synchInterval;
  int dimension;
  int totalIterations;
  int statCollection;
  StartUpMsg () {
    depth = 2;
    synchInterval = 30;
    dimension = 2;
    statCollection = 1;
  }
  StartUpMsg(int dep,int synchInt, int dim,int totIter) {
    depth = dep;
    synchInterval = synchInt;
    dimension = dim;
    totalIterations = totIter;
    statCollection = 0;
  }

  StartUpMsg(int dep,int synchInt, int dim,int totIter, bool statcoll) {
    depth = dep;
    synchInterval = synchInt;
    dimension = dim;
    totalIterations = totIter;
    statCollection = statcoll;
  }
  
};

class AmrCoordinator: public Chare {
  private:
  CProxy_Cell arrayProxy;
  int synchInterval;
  int depth;
  int dimension;
  int totalIterations;
  CkChareID myHandle;

  int statCollection;
  CkGroupID gid;

  int leaves;
  int refine;
  int arefine;
  int migrations;
  int statMsgs;

  double startTime;
  int phase; 
  //simulation phase to determine if load balancing or refine is required
  int phaseStep;
 public:
  //constructors
  AmrCoordinator(){}
  AmrCoordinator(_DMsg* msg);
  AmrCoordinator(StartUpMsg *msg);
  AmrCoordinator(CkMigrateMessage *msg){}

  void synchronise(_RedMsg *msg);
  void create_tree();
  void reportStats(_StatCollMsg *m);
  void resetClock();
};




class Cell : public ArrayElementT <BitVec> {
 protected:
  int dimension;
  AmrUserData *userData;
  CProxy_Cell arrayProxy; 
  char type;	 //node(n) or a leaf(l) or root(r)..
  
  BitVec parent; //parent index
  
  //children of this node 2 dimensional array ... 2 
  //in each dimension
  BitVec **children;

  BitVec myIndex; //my bitvector of the cell
  
  //member variables used by neighbor data to determine how may neighbors
  //have communicated their data to me and how many intotal are there
  int num_neighbors, neighbors_reported;
  
  int run_until; // count of iterations
  int run_done;  // maximum limit of iterations to run
  
  int *neighbors;// array of neighbors 2 * DIMENSION
  
  //variables too keep track of the messages from the neighbors
  //int low_count, up_count, right_count, left_count;
  int *nborRecvMsgCount;
  NeighborMsg ***nborRecvMsgBuff;
  // for queuing messages which arrive out of order...
  FIFO_QUEUE *msg_queue;
  FIFO_QUEUE *temp_queue;
  char* start_ptr;
  int msg_count;
  
  //for autoreifinement code
  int refined;
  int autorefine;
  BitVec retidx;

  //for reduction step
  int synchleavrep;
  int synchinterval,synchstep;
  int justRefined;
  
  //Handle to get the proxy for the coordinator
  CkChareID coordHandle;
  
  int statCollection;
  // CProxy_StatCollector grpProxy;
  CkGroupID gid;
  
  void init_cell(_ArrInitMsg *msg);
  void treeSetup(_ArrInitMsg *msg);
  
/*  void frag_msg(NeighborMsg *,int,int,int,int);
    void jacobi_relax(void);
    int isodd(int);
    int iseven(int);
*/
  virtual void reg_nbor_msg(int neighbor_side, NeighborMsg *msg){}
  void check_queue(void);
  /*void check_queue_node(void);*/
  friend void FIFO_EnQueue(FIFO_QUEUE *queue, void *elt);
  friend int FIFO_Empty(FIFO_QUEUE *);
  friend void FIFO_DeQueue(FIFO_QUEUE *queue, void **element);
  friend void FIFO_Destroy(FIFO_QUEUE *queue);
  friend int FIFO_Fill(FIFO_QUEUE *queue);
  
  int sendInDimension(int dim,int side,NeighborMsg* msg);
  int sendInDimension(int dim,int side);
  int powerOfTwo(int);
  
 public:
  Cell(){}
  //Cell(ArrInitMsg*);
  Cell(CkMigrateMessage *msg) {}
  
  void refine(_RefineMsg* msg);
  void change_to_leaf(ChildInitMsg* msg);
  void neighbor_data(NeighborMsg *msg);

  void cpyNborMsg(NeighborMsg* dest,NeighborMsg * src);  
  void refine_confirmed(_DMsg *msg);
  void resume(_DMsg *msg);
  void synchronise(_RedMsg *msg);
  void refineExec(_DMsg *msg);
  void checkRefine(_RefineChkMsg* msg);
  void refineReady(BitVec retidx,int pos);
  virtual void create_children(_ArrInitMsg** cmsg){}
  virtual void doIterations(void);
  virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side) {}
  virtual void pup(PUP::er &p);
  virtual void ResumeFromSync() {
    // ArrayElement::ResumeFromSync();
    //    resume(new _DMsg);
    synchstep += synchinterval;
    if (type == 'l') {
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(new _RedMsg(1));
      DEBUGN(("Synchronising: x %d y %d z %d bits %d\n", myIndex.vec[0],
	      myIndex.vec[1],myIndex.vec[2], myIndex.numbits));
    }
  }
  
  void goToAtSync(_DMsg* msg) {
    delete msg;
    ArrayElement::AtSync();
  }

  virtual ~Cell(){
    if(type == 'l') {
      if(userData)
	delete userData;
    }
    
    for(int i=0;i<dimension;i++)
      delete[] children[i];
    delete[] children;
    
    delete[] neighbors;
    
    delete[] nborRecvMsgCount;
    int size = powerOfTwo(dimension-1);
    
    if(nborRecvMsgBuff) {
      for(int i=0; i<2*dimension; i++) {
	for(int j=0; j<size; j++) {
	  /*if(nborRecvMsgBuff[i][j])
	    delete nborRecvMsgBuff[i][j];*/
	}
	delete[] nborRecvMsgBuff[i];
      }
      delete[] nborRecvMsgBuff;
    }
    
    if(msg_queue) {
      while(!FIFO_Empty(msg_queue)) {
	NeighborMsg* tempMsg;
	FIFO_DeQueue(msg_queue,(void**) &tempMsg);
	delete tempMsg;
      }
      FIFO_Destroy(msg_queue);
    }
  } 
};

class Cell2D: public Cell {
 private:
  void frag_msg(NeighborMsg *,int,int,int,int);

 public:
  Cell2D() {}
  Cell2D(_ArrInitMsg *);
  Cell2D(CkMigrateMessage *msg) {}
  void reg_nbor_msg(int neighbor_side, NeighborMsg *msg);
  virtual void create_children(_ArrInitMsg** cmsg);
  virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side);
  
  virtual void pup(PUP::er &p){
    Cell::pup(p);
    if (p.isUnpacking()){
      CProxy_Cell2D aProxy(thisArrayID);
      arrayProxy = *(CProxy_Cell*)&aProxy;
    } 
  }
};

class Cell1D: public Cell {
 private:
  
 public:
  Cell1D() {}
  Cell1D(_ArrInitMsg *);
  Cell1D(CkMigrateMessage *msg) {}
  
  void reg_nbor_msg(int neighbor_side, NeighborMsg *msg);
  virtual void create_children(_ArrInitMsg** cmsg);
  virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side);
  virtual void pup(PUP::er &p){
    Cell::pup(p);
    if (p.isUnpacking()){
      CProxy_Cell1D aProxy(thisArrayID);
      arrayProxy = *(CProxy_Cell*)&aProxy;
    } 
  }
};


class Cell3D: public Cell {
 private:
  void frag_msg(NeighborMsg *,int,int,int,int,int,int,int,int);
 public:
  Cell3D() {}
  Cell3D(_ArrInitMsg *);
  Cell3D(CkMigrateMessage *msg) {}
  void reg_nbor_msg(int neighbor_side, NeighborMsg *msg);
  virtual void create_children(_ArrInitMsg** cmsg);
  virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side);
  virtual void pup(PUP::er &p){
    Cell::pup(p);
    if (p.isUnpacking()){
      CProxy_Cell3D aProxy(thisArrayID);
      arrayProxy = *(CProxy_Cell*)&aProxy;
    }
  }
};


#endif





