/*
Charm++ File: Chare Arrays
Array Reduction Library section
added 11/11/1999 by Orion Sky Lawlor, olawlor@acm.org
*/
#ifndef _CKARRAY_H
#define _CKARRAY_H

#include "charm++.h"

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif

extern void _registerCkArray(void);

class PtrQ;
class PtrVec;

#define ALIGN8(x)       (int)((~7)&((x)+7))

#define MessageIndex(mt)        CMessage_##mt##::__idx
#define ChareIndex(ct)          CProxy_##ct##::__idx
#define EntryIndex(ct,ep,mt)    CProxy_##ct##::ckIdx_##ep##((##mt##*)0)
#define ConstructorIndex(ct,mt) EntryIndex(ct,ct,mt)

typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;

extern CkGroupID _RRMapID;

#if CMK_LBDB_ON
class LBDatabase;
#endif

class Array1D;
class ArrayMapRegisterMessage;
class ArrayElementCreateMessage;
class ArrayElementMigrateMessage;
class ArrayElementExitMessage;


class ArrayMap : public Group
{
public:
  virtual int procNum(int arrayHdl, int element) = 0;
  virtual void registerArray(ArrayMapRegisterMessage *) = 0;
};

//////////////////////// Array Reduction Library //////////////
#define CK_ARRAY_REDUCTIONS 1
#ifdef CK_ARRAY_REDUCTIONS

class ArrayReductionMessage;//See definition at end of file

//An ArrayReductionFn is used to combine the contributions
//of several array elements into a single summed contribution:
//  nMsg gives the number of messages to reduce.
//  msgs[i] contains a contribution from a local element or remote branch.
typedef ArrayReductionMessage *(*ArrayReductionFn)(int nMsg,ArrayReductionMessage **msgs);

//An ArrayReductionClientFn is called on PE 0 when the contributions
// from all array elements have been received and reduced.
//  param can be ignored, or used to pass any client-specific data you wish
//  dataSize gives the size (in bytes) of the data array
//  data gives the reduced contributions of all array elements.  
//       It will be disposed of by the Array BOC when this procedure returns.
typedef void (*ArrayReductionClientFn)(void *param,int dataSize,void *data);
#endif //CK_ARRAY_REDUCTIONS


class ArrayElement : public Chare
{
friend class Array1D;
public:
  ArrayElement(ArrayElementCreateMessage *msg);
  ArrayElement(ArrayElementMigrateMessage *msg);

private:
  ArrayElement(void) {};

protected:
	//Call contribute to add your contribution to a new global reduction.
	// The array BOC will keep a copy the data. reducer must be the same on all PEs.
	void contribute(int dataSize,void *data,ArrayReductionFn reducer);
	//This value is used by Array1D to keep track of which ArrayElements have contribute()'d.
	// It is simply the number of times contribute has been called.
	int nContributions;
	

  // For Backward compatibility:
  void finishConstruction(void) { finishConstruction(CmiFalse); };

  void finishConstruction(CmiBool use_local_barrier);
  void finishMigration(void);

  virtual int packsize(void) { return 0; }
  virtual void pack(void *) { return; }
  void AtSync();
  virtual void ResumeFromSync(void) {
    CkPrintf("No ResumeFromSync() defined for this element!\n");
  };

  int thisIndex;
  CkArrayID thisAID;     // thisArrayID is preferred
  CkArrayID thisArrayID; // A duplicate of thisAID
  int numElements;

public:
  void migrate(int where);
  void exit(ArrayElementExitMessage *msg);
  int getIndex(void) { return thisIndex; }
  int getSize(void)  { return numElements; }

private:
  CkChareID arrayChareID;
  CkGroupID arrayGroupID;

protected:
  Array1D *thisArray;
};

enum {unknownPe = -1};

class ArrayCreateMessage;
class ArrayMessage;
class ArrayMigrateMessage;
class ArrayElementAckMessage;
class ArrayElementUpdateMessage;

class Array1D : public Group {
friend class ArrayElement;

public:
  static  CkGroupID CreateArray(int numElements,
				CkGroupID mapID,
				ChareIndexType elementChare,
				EntryIndexType elementConstructor,
				EntryIndexType elementMigrator);

  Array1D(ArrayCreateMessage *);
  void send(ArrayMessage *msg, int index, EntryIndexType ei);
  void broadcast(ArrayMessage *msg, EntryIndexType ei);
  void RecvMapID(ArrayMap *mapPtr,int mapHandle);
  void RecvElementID(int index, ArrayElement *elem, CkChareID handle,
		     CmiBool uses_barrier);
  void RecvForElement(ArrayMessage *msg);
  void RecvMigratedElement(ArrayMigrateMessage *msg);
  void RecvMigratedElementID(int index, ArrayElement *elem, CkChareID handle);
  void AckMigratedElement(ArrayElementAckMessage *msg);
  void UpdateLocation(ArrayElementUpdateMessage *msg);
  int array_size(void) { return numElements; };
  int num_local(void) { return numLocalElements; };
  int ckGetGroupId(void) { return thisgroup; }
  ArrayElement *getElement(int idx) { return elementIDs[idx].element; }
  void DummyAtSync(void);

#ifdef CK_ARRAY_REDUCTIONS
  //Register a function to be called once the reduction is complete--
  //  need only be called on PE 0 (but is harmless otherwise).
#define CkRegisterArrayReductionHandler(aid,handler,param) \
      (aid)._array->registerReductionHandler(handler,param)
  void registerReductionHandler(ArrayReductionClientFn handler,void *param);
  void RecvReductionMessage(ArrayReductionMessage *msg);
#endif

#if CMK_LBDB_ON
  static void staticMigrate(LDObjHandle _h, int _dest);
  static void staticSetStats(LDOMHandle _h, int _state);
  static void staticQueryLoad(LDOMHandle _h);
  static void staticResumeFromSync(void* data);
  static void staticRecvAtSync(void* data);
  static void staticDummyResumeFromSync(void* data);
#endif

  typedef enum {creating, here, moving_to, arriving, at} ElementState;

private:
  void migrateMe(int index, int where);

#if CMK_LBDB_ON
  void Migrate(LDObjHandle _h, int _dest);
  void SetStats(LDOMHandle _h, int _state);
  void QueryLoad(LDOMHandle _h);

  void RegisterElementForSync(int index);
  void AtSync(int index);
  void ResumeFromSync(int index);
  void DummyResumeFromSync();
  void RecvAtSync();
#endif

  struct ElementIDs {
    struct BarrierClientData {
      Array1D *me;            
      int index;
    };

    ElementState state;
    int originalPE;
    int pe;
    ArrayElement *element;
    CkChareID elementHandle;
    int cameFrom;
    int curHop;
    ArrayMigrateMessage *migrateMsg;
#if CMK_LBDB_ON
    LDObjHandle ldHandle;
    CmiBool uses_barrier;
    LDBarrierClient barrierHandle;
    BarrierClientData barrierData;
#endif
  };

  int numElements;
  int mapHandle;
  CkGroupID mapGroup;
  ArrayMap *map;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  ElementIDs *elementIDs;
  int elementIDsReported;
  int numLocalElements;

#if CMK_LBDB_ON
  LDOMHandle myHandle;
  LBDatabase *the_lbdb;
  LDBarrierClient dummyBarrierHandle;
#endif
  PtrQ *bufferedForElement;
  PtrQ *bufferedMigrated; 
 
#ifdef CK_ARRAY_REDUCTIONS
// Array Reduction Implementation:
	ArrayReductionClientFn reductionClient;//Will be called when reduction is complete
	void *reductionClientParam;//Parameter to pass to reduction client

#define ARRAY_RED_TREE_LOG 2 //Log-base-2 of fan-out of reduction tree (for binary tree, 1)
#define ARRAY_RED_TREE (1<<ARRAY_RED_TREE_LOG) //Number of kids of each tree node

//This is used to hold messages that arrive for reductions we haven't started yet
#define ARRAY_RED_FUTURE_MAX 250 //The length of the out-of-order-reduction-message buffer
	int nFuture;//Number of messages waiting in queue below
	ArrayReductionMessage *futureBuffer[ARRAY_RED_FUTURE_MAX];
	
	int reductionNo;//The number of the current reduction (starts at -1)
	int reductionFinished;//Flag: is the current reduction (above) complete? (as far as we are concerned)
	ArrayReductionFn curReducer;//Current reduction function (or NULL)
	ArrayReductionMessage **curMsgs;//Buffered message array for the current reduction
	int curMax;//Dimentions of above array
	int nCur;//Number of reduction messages we have received so far.
	int nComposite;//Number of messages recieved up the reduction tree
	int expectedComposite;//Number of messages we expect to receive from our kids

//This is called by ArrayElement::contribute() and RcvReductionMessage.
// reducer may be NULL. The given message is kept by Array1D.
	void addReductionContribution(ArrayReductionMessage *m,ArrayReductionFn reducer);

//These two are called by addReductionContribution, above
	void beginReduction(int extraLocal);//Allocate msgs array above, increment reductionNo
	int expectedLocalMessages(void);//How many messages do we still need from locals?
	void tryEndReduction(void);//Check if we're done, and if so, finish.
	void endReduction(void);//Combine msgs array and send off, set finished flag
#endif //CK_ARRAY_REDUCTIONS
};

#include "CkArray.decl.h"

class ArrayCreateMessage : public CMessage_ArrayCreateMessage
{
public:
  int numElements;
  CkGroupID mapID;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  CkGroupID loadbalancer;
};

class ArrayMessage
{
public:
  int from_pe;
  int destIndex;
  EntryIndexType entryIndex;
  int hopCount;
  int serial_num;
};

class ArrayElementAckMessage : public CMessage_ArrayElementAckMessage
{
public:
  int index;
  int arrivedAt;
  int deleteElement;
  CkChareID handle;
  int hopCount;
};

class ArrayElementUpdateMessage : public CMessage_ArrayElementUpdateMessage
{
public:
  int index;
  int hopCount;
  int pe;
};

class ArrayMigrateMessage : public CMessage_ArrayMigrateMessage
{
public:
  int from;
  int index,nContributions;
  int elementSize;
  void *elementData;
  int hopCount;
  CmiBool uses_barrier;

  static void *alloc(int msgnum, int size, int *array, int priobits);
  static void *pack(ArrayMigrateMessage *);
  static ArrayMigrateMessage *unpack(void *in);
};

class RRMap : public ArrayMap
{
private:
  PtrVec *arrayVec;
public:
  RRMap(void);
  void registerArray(ArrayMapRegisterMessage *);
  int procNum(int arrayHdl, int element);
};

class ArrayInit : public Chare 
{
public:
  ArrayInit(CkArgMsg *msg) {
    _RRMapID = CProxy_RRMap::ckNew();
    delete msg;
  }
};

class ArrayMapRegisterMessage : public CMessage_ArrayMapRegisterMessage
{
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
};

class ArrayElementCreateMessage : public CMessage_ArrayElementCreateMessage {
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementMigrateMessage : public CMessage_ArrayElementMigrateMessage {
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index,nContributions;
  void* packData;
};

class ArrayElementExitMessage : public CMessage_ArrayElementExitMessage
{
public:
  int dummy;
};

#ifdef CK_ARRAY_REDUCTIONS
//An ArrayReductionMessage is sent up the reduction tree-- it
// carries the contribution of one 
// (or reduced contributions of several) array elements.
class ArrayReductionMessage : public CMessage_ArrayReductionMessage
{
private:
  //Default constructor is private-- use "buildNew", below
  ArrayReductionMessage();
public:
//External fields
  //Length of array below, in bytes
  int dataSize;
  //Reduction data
  void *data;
  //Index of array element which made this contribution,
  //  or -n, where n is the number of contributing elements
  int source;
  
  //Return the number of array elements from which this message's data came
  int getSources();
  
  //"Constructor"-- builds and returns a new ArrayReductionMessage.
  //  the "srcData" array you specify will be copied into this object (unless NULL).
  static ArrayReductionMessage *buildNew(int NdataSize,void *srcData);


//Internal fields
  //The number of this reduction (0, 1, ...)
  int reductionNo;
  //(non-packed field) Used only if this message needs to be buffered in the future buffer.
  ArrayReductionFn futureReducer;
 
  //Message runtime support
  static void *alloc(int msgnum, int size, int *reqSize, int priobits);
  static void *pack(ArrayReductionMessage *);
  static ArrayReductionMessage *unpack(void *in);
};

//Reduction Library:
/*
A small library of oft-used reductions for use with the 
Array Reduction Manager.

Parallel Programming Lab, University of Illinois at Urbana-Champaign
Orion Sky Lawlor, 11/13/1999, olawlor@acm.org

*/

//Compute the sum the numbers passed by each element.
ArrayReductionMessage *CkReduction_sum_int(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_sum_float(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_sum_double(int nMsg,ArrayReductionMessage **msg);

//Compute the product the numbers passed by each element.
ArrayReductionMessage *CkReduction_product_int(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_product_float(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_product_double(int nMsg,ArrayReductionMessage **msg);

//Compute the largest number passed by any element.
ArrayReductionMessage *CkReduction_max_int(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_max_float(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_max_double(int nMsg,ArrayReductionMessage **msg);

//Compute the smallest number passed by any element.
ArrayReductionMessage *CkReduction_min_int(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_min_float(int nMsg,ArrayReductionMessage **msg);
ArrayReductionMessage *CkReduction_min_double(int nMsg,ArrayReductionMessage **msg);


//Compute the logical AND of the integers passed by each element.
// The resulting integer will be zero if any source integer is zero.
ArrayReductionMessage *CkReduction_and(int nMsg,ArrayReductionMessage **msg);

//Compute the logical OR of the integers passed by each element.
// The resulting integer will be 1 if any source integer is nonzero.
ArrayReductionMessage *CkReduction_or(int nMsg,ArrayReductionMessage **msg);


//This structure contains the contribution of one array element.
typedef struct {
	int sourceElement;//The element number from which this contribution came
	int dataSize;//The length of the data array below
	char data[1];//The (dataSize-long) array of data
} CkReduction_set_element;

//Combine the data passed by each element into an list of reduction_set_elements.
// Each element may contribute arbitrary data (with arbitrary length).
ArrayReductionMessage *CkReduction_set(int nMsg,ArrayReductionMessage **msg);

//Utility routine: get the next reduction_set_element in the list
// if there is one, or return NULL if there are none.
//To get all the elements, just keep feeding this procedure's output back to
// its input until it returns NULL.
CkReduction_set_element *CkReduction_set_element_next(CkReduction_set_element *cur);

#endif //CK_ARRAY_REDUCTIONS


#endif
