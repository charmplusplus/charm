#ifndef _AMR_H
#define _AMR_H

#include "charm++.h"
#include "bitvec.h"
#include "amr.decl.h"
#include "fifo.h"
#include <math.h>
#define NEG_X 0
#define POS_X 1
#define NEG_Y 2
#define POS_Y 3

#define DEBUGR(x) CkPrintf x
#define DEBUGT(x) /*CkPrintf x*/
#define DEBUGS(x) CkPrintf x
#define DEBUGRC(x) /*CkPrintf x*/
#define DEBUGJ(x) /*CkPrintf x*/

class NeighborMsg;
class ChildInitMsg;

class AmrUserData {
 public:
  AmrUserData(){}
  static AmrUserData *createData();
  static AmrUserData *createData(void *data, int dataSize);
  
  static void deleteNborData(void * data);
  static void deleteChildData(void * data);
  
  NeighborMsg **fragment(NeighborMsg* msg);
  void combineAndStore(NeighborMsg *msg1,NeighborMsg *msg2);
  void store(NeighborMsg *msg);
  
  
  virtual void ** fragmentNborData(void* data, int *sizePtr){ return NULL;}
  virtual void ** getNborMsgArray(int* sizePtr){return NULL;}
  
  virtual void store(void* data , int dataSize, int neighborSide){}
  virtual void combineAndStore(void *data1, void *data2, int dataSize,int neighborSide){}
  virtual bool refineCriterion(void){return false;}
  //returns an array of userdefined data to be given to the children
  //gets an array of 4 for quad tree
  virtual void **fragmentForRefine(int *sizePtr){ return NULL;}
  virtual void doComputation(void){}
  virtual ~AmrUserData(){}  
};

class NeighborMsg :public CMessage_NeighborMsg 
{
 public:
  int which_neighbor;
  int run_until;
  int numbits;
  int dataSize;
  void *data;

  static void* pack(NeighborMsg *msg);
  static NeighborMsg* unpack(void *inbuf);


  ~NeighborMsg(){ 
    AmrUserData::deleteNborData(data);
  }
};

class ChildInitMsg : public CMessage_ChildInitMsg 
{
 public:
  //double data[CELLSIZE*CELLSIZE];
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
  bitvec sender;
  int from;
  _DMsg(){}
  _DMsg(bitvec sendVec, int pos){
    sender = sendVec;
    from = pos;
  }
};

class _RefineChkMsg : public CMessage__RefineChkMsg {
 public:
  bitvec index;
  _RefineChkMsg() {}
  _RefineChkMsg(bitvec idx) {
    index = idx;
  }
};

class _RefineMsg : public CMessage__RefineMsg {
 public:
  // autorefine = 0 for regular refine messages
  //            = 1 for automatic refinement messages
  int autorefine; 
  // index has the return address to which the response is to be sent to
  // for automatic refinement messages
  bitvec index;

  _RefineMsg() { autorefine = 0;}
  _RefineMsg(int reftype) {
    autorefine = reftype;
    if(autorefine == 1) {
      CkError("Automatic refinement message without a return address\n");
    }
  }
  _RefineMsg(int reftype,bitvec idx) {
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
  bitvec parent; //bitvector of the parent
  char type; //type of the node that is being made ie r (root), n (node)
             // v (virtual leaf) 
  int interval;//synchronisation interval
  int depth; //depth of the tree tobe created
  int totalIterations; //total iterations for the simulation
  CkChareID coordHandle;
  //int dimension;
};

class StartUpMsg :public CMessage_StartUpMsg
{
 public:
  int depth;
  int synchInterval;
  int dimension;
  int totalIterations;

  StartUpMsg () {
    depth = 2;
    synchInterval = 30;
    dimension = 2;
  }
  StartUpMsg(int dep,int synchInt, int dim,int totIter) {
    depth = dep;
    synchInterval = synchInt;
    dimension = dim;
    totalIterations = totIter;
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
 public:
  //constructors
  AmrCoordinator(_DMsg* msg);
  AmrCoordinator(StartUpMsg *msg);
  AmrCoordinator(CkMigrateMessage *msg){}

  void synchronise(_RedMsg *msg);
  void create_tree();
};




class Cell : public ArrayElementT <bitvec> {
 protected:
  int dimension;
  AmrUserData *userData;
  CProxy_Cell arrayProxy; 
  char type;	 //node(n) or a leaf(l) or root(r)..
  
  bitvec parent; //parent index

  //children of this node 2 dimensional array ... 2 
  //in each dimension
  bitvec **children;

  bitvec myIndex; //my bitvector of the cell
  
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
  //int msg_count;

  //for autoreifinement code
  int refined;
  int autorefine;
  bitvec retidx;

  //for reduction step
  int synchleavrep;
  int synchinterval,synchstep;

  //Handle to get the proxy for the coordinator
  CkChareID coordHandle;
  void init_cell(_ArrInitMsg *msg);
  void treeSetup(_ArrInitMsg *msg);

/*  void frag_msg(NeighborMsg *,int,int,int,int);
  void jacobi_relax(void);
  int isodd(int);
  int iseven(int);
*/
  void reg_nbor_msg(int neighbor_side, NeighborMsg *msg);
  void check_queue(void);
  /*void check_queue_node(void);*/
  friend void FIFO_EnQueue(FIFO_QUEUE *queue, void *elt);
  friend int FIFO_Empty(FIFO_QUEUE *);
  friend void FIFO_DeQueue(FIFO_QUEUE *queue, void **element);
  friend void FIFO_Destroy(FIFO_QUEUE *queue);
  friend int FIFO_Fill(FIFO_QUEUE *queue);

  int sendInDimension(int dim,int side,NeighborMsg* msg);
  int sendInDimension(int dim,int side);
 public:
  Cell(){}
  //Cell(ArrInitMsg*);
  Cell(CkMigrateMessage *msg) {}

  void refine(_RefineMsg* msg);
  void change_to_leaf(ChildInitMsg* msg);
  void neighbor_data(NeighborMsg *msg);
  //void boundary_condition(void);
  
  void refine_confirmed(_DMsg *msg);
  void resume(_DMsg *msg);
  void synchronise(_RedMsg *msg);
  void refineExec(_DMsg *msg);
  void checkRefine(_RefineChkMsg* msg);
  void refineReady(bitvec retidx,int pos);
  //int refineCriterion(void);
  virtual void create_children(_ArrInitMsg** cmsg){}
  virtual void doIterations(void) {}
  virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side) {}
  //virtual void pup(PUP::er &p);
};

class Cell2D: public Cell {
  private:
   void frag_msg(NeighborMsg *,int,int,int,int);

  public:
   Cell2D() {}
   Cell2D(_ArrInitMsg *);
   Cell2D(CkMigrateMessage *msg) {}
   
   virtual void create_children(_ArrInitMsg** cmsg);
   virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side);
   virtual void doIterations(void);
};

class Cell1D: public Cell {
  private:

  public:
   Cell1D() {}
   Cell1D(_ArrInitMsg *);
   Cell1D(CkMigrateMessage *msg) {}
   
   virtual void create_children(_ArrInitMsg** cmsg);
   virtual void forwardSplitMsg(NeighborMsg *msg ,int neighbor_side);
   virtual void doIterations(void);
};

#endif
