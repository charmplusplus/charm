/* Generalized Chare Arrays

These classes implement Chare Arrays.  
These are dynamic (i.e. allowing insertion
and deletion) collections of ordinary Chares 
indexed by arbitrary runs of bytes.

The general structure is:

CkArray is the "array manager" Group, or BOC-- 
it creates, keeps track of, and cares for all the
array elements on this PE (i.e.. "local" elements).  
It does so using a hashtable of CkArrayRec objects--
there's an entry for each local, home-here, and recently
communicated remote array elements.

CkArrayElement is the type of the array 
elements (a subclass of Chare).

CkArrayIndex is an arbitrary run of bytes,
used to index into the CkArray hashtable.


Converted from 1-D arrays 2/27/2000 by
Orion Sky Lawlor, olawlor@acm.org

*/
#ifndef _CKARRAY_H
#define _CKARRAY_H

/*******************************************************
Array Index class.  An array index is just a HashKey-- 
a run of bytes used to look up an object in a hash table.
An Array Index cannot be modified once it is created.
 */

#include "ckhashtable.h"

#ifndef CK_ARRAYINDEX_MAXLEN 
#define CK_ARRAYINDEX_MAXLEN 3 /*Max. # of integers in an array index*/
#endif

class CkArrayIndex
{
public:
	//Length of index in *integers*
	int nInts;
	
	//Index data immediately follows...
	
	int *data(void) {return (&nInts)+1;}
	const int *data(void) const {return (&nInts)+1;}
	
	void pup(PUP::er &p);
	CkHashCode hash(void) const;
};

//Simple ArrayIndex classes: the key is just integer indices.
class CkArrayIndex1D : public CkArrayIndex {
public: int index;
	CkArrayIndex1D(int i0) {index=i0;nInts=1;}
};
class CkArrayIndex2D : public CkArrayIndex {
public: int index[2];
	CkArrayIndex2D(int i0,int i1) {index[0]=i0;index[1]=i1;
		nInts=2;}
};
class CkArrayIndex3D : public CkArrayIndex {
public: int index[3];
	CkArrayIndex3D(int i0,int i1,int i2) {index[0]=i0;index[1]=i1;index[2]=i2;
		nInts=3;}
};

//A slightly more complex array index: the key is an object
// whose size is fixed at compile time.
template <class object> //Key object
class CkArrayIndexT : public CkArrayIndex {
public:
	object obj;
	CkArrayIndexT(const object &srcObj) {obj=srcObj; 
		nInts=sizeof(obj)/sizeof(int);}
};

//This class is as large as any CkArrayIndex
class CkArrayIndexMax : public CkArrayIndex {
	struct {
		int data[CK_ARRAYINDEX_MAXLEN];
	} index;
	void copyFrom(const CkArrayIndex &that)
	{
		nInts=that.nInts;
		index=((const CkArrayIndexMax *)&that)->index;
		//for (int i=0;i<nInts;i++) index[i]=that.data()[i];
	}
public:
	CkArrayIndexMax(void) { }
	CkArrayIndexMax(const CkArrayIndex &that) 
		{copyFrom(that);}
	CkArrayIndexMax &operator=(const CkArrayIndex &that) 
		{copyFrom(that); return *this;}
};

class CkArrayIndexStruct {
public:
	int nInts;
	int index[CK_ARRAYINDEX_MAXLEN];
};

/*********************** Array Messages ************************/
class CkArrayMessage : public Message {
public:
  //These routines are implementation utilities
  inline CkArrayIndexMax &array_index(void);
  unsigned short &array_ep(void);
  unsigned char &array_hops(void);
  unsigned int array_getSrcPe(void);
  unsigned int array_ifNotThere(void);
  void array_setIfNotThere(unsigned int);
  
  //This allows us to delete bare CkArrayMessages
  void operator delete(void *p){CkFreeMsg(p);}
  static void* alloc(int idx, size_t sz, int *szs, int pb) {
    return CkAllocMsg(idx, sz, pb);
  }
  static void *pack(CkArrayMessage *m) { return (void *) m; }
  static CkArrayMessage *unpack(void *buf) { return (CkArrayMessage*) buf; }
};

#include "ckreduction.h"

/***********************************************************
	Utility defines, includes, etc.
*/

//#undef CMK_LBDB_ON  //FOR TESTING:  DISABLE LOAD BALANCER
//#define CMK_LBDB_ON 0

#if CMK_LBDB_ON
#include "LBDatabase.h"
class LBDatabase;
#endif

extern void _registerCkArray(void);
extern CkGroupID _RRMapID;

#define ALIGN8(x)       (int)((~7)&((x)+7))

#define MessageIndex(mt)        CMessage_##mt##::__idx
#define ChareIndex(ct)          CkIndex_##ct##::__idx
#define EntryIndex(ct,ep,mt)    CkIndex_##ct##::ep##((mt *)0)
#define ConstructorIndex(ct,mt) EntryIndex(ct,ckNew,mt)

typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;

//Forward declarations
class CkArray;
class ArrayElement;
class CkArrayMessage;
class CkArrayElementMigrateMessage;
class CkArrayCreateMsg;
class CkArrayRemoveMsg;
class CkArrayUpdateMsg;

//What to do if an entry method is invoked on
// an array element that does not (yet) exist:
typedef enum {
	CkArray_IfNotThere_buffer=0, //Wait for it to be created
	CkArray_IfNotThere_createhere=1, //Make it on sending PE
	CkArray_IfNotThere_createhome=2 //Make it on (a) home PE
} CkArray_IfNotThere;

//This class is a wrapper around a CkArrayIndex and ArrayID,
// used by array element proxies.  This makes the translator's
// job simpler, and the translated code smaller. 
class CProxy_ArrayBase :public CProxyBase_Delegatable {
private:
	CkArrayID _aid;
public:
	CProxy_ArrayBase() { }
	CProxy_ArrayBase(const CkArrayID &aid,CkGroupID dTo=-1) 
		:CProxyBase_Delegatable(dTo), _aid(aid) { }

	void ckInsertIdx(CkArrayMessage *m,int ctor,int onPE,const CkArrayIndex &idx);	
	void ckInsert1D(CkArrayMessage *m,int ctor,int numElements);
	void ckBroadcast(CkArrayMessage *m, int ep) const;
	CkArrayID ckGetArrayID(void) const { return _aid; }
	CkArray *ckLocalBranch(void) const { return _aid.ckLocalBranch(); }	
	void doneInserting(void);
	void setReductionClient(CkReductionMgr::clientFn fn,void *param=NULL);

	void pup(PUP::er &p);
};
PUPmarshall(CProxy_ArrayBase);
#define CK_DISAMBIG_ARRAY(super) \
	CK_DISAMBIG_DELEGATABLE(super) \
	void ckInsertIdx(CkArrayMessage *m,int ctor,int onPE,const CkArrayIndex &idx) \
	  { super::ckInsertIdx(m,ctor,onPE,idx); }\
	void ckInsert1D(CkArrayMessage *m,int ctor,int n) \
	  { super::ckInsert1D(m,ctor,n); } \
	void ckBroadcast(CkArrayMessage *m, int ep) const \
	  { super::ckBroadcast(m,ep); } \
	CkArrayID ckGetArrayID(void) const \
	  { return super::ckGetArrayID();} \
	CkArray *ckLocalBranch(void) const \
	  { return super::ckLocalBranch(); } \
	void doneInserting(void) { super::doneInserting(); }\
	void setReductionClient(CkReductionMgr::clientFn fn,void *param=NULL)\
	  { super::setReductionClient(fn,param); }\


class CProxyElement_ArrayBase:public CProxy_ArrayBase {
private:
	CkArrayIndexMax _idx;//<- our element's array index
public:
	CProxyElement_ArrayBase() { }
	CProxyElement_ArrayBase(const CkArrayID &aid,
		const CkArrayIndex &idx,CkGroupID dTo=-1)
		:CProxy_ArrayBase(aid,dTo), _idx(idx) { }
	
	void ckInsert(CkArrayMessage *m,int ctor,int onPE);
	void ckSend(CkArrayMessage *m, int ep) const;
	const CkArrayIndex &ckGetIndex() const {return _idx;}

	ArrayElement *ckLocal(void) const;
	void pup(PUP::er &p);
};
PUPmarshall(CProxyElement_ArrayBase);
#define CK_DISAMBIG_ARRAY_ELEMENT(super) \
	CK_DISAMBIG_ARRAY(super) \
	void ckInsert(CkArrayMessage *m,int ctor,int onPE) \
	  { super::ckInsert(m,ctor,onPE); }\
	void ckSend(CkArrayMessage *m, int ep) const \
	  { super::ckSend(m,ep); }\
	const CkArrayIndex &ckGetIndex() const \
	  { return super::ckGetIndex(); }\


/************************* Array Map  ************************
An array map tells which PE to put each array element on.
*/
class CkArrayMapRegisterMessage
{
public:
  int numElements;
  CkArray *array;
};

class CkArrayMap : public CkGroupInitCallback
{
public:
  CkArrayMap(void);
  CkArrayMap(CkMigrateMessage *m) {}
  virtual int registerArray(CkArrayMapRegisterMessage *);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element);
  virtual void pup(PUP::er &p) { CkGroupInitCallback::pup(p); }
};

/************************ Array Element *********************/

class ArrayElement : public Chare
{
	friend class CkArray;
  void initBasics(void);
public:
  ArrayElement(void);
  ArrayElement(CkMigrateMessage *m);
  virtual ~ArrayElement();
  
  CkArrayIndexMax thisIndexMax;//Index of this element
  
  void destroy(void);
  
//Contribute to the given reduction type.  Data is copied, not deleted.
  void contribute(int dataSize,void *data,CkReduction::reducerType type);

//Migrate to the given processor number
  void migrateMe(int toPE);
//Pack/unpack routine (called before and after migration)
  virtual void pup(PUP::er &p);

protected:
  CkArray *thisArray;//My source array

  CkArrayID thisArrayID;//My source array's ID
  int thisChareType;//My chare type index

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(void);
  virtual void ResumeFromSync(void);
  CmiBool usesAtSync;//You must set this in the constructor to use AtSync().
  LDObjHandle ldHandle;//Transient (not migrated)
private: //Load balancer state:
  CmiBool barrierRegistered;//True iff handle below is set
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
  static void staticResumeFromSync(void* data);
  static void staticMigrate(LDObjHandle h, int dest);
#endif
  void lbRegister(void);//Connect to load balancer
  void lbUnregister(void);//Disconnect from load balancer

//Array implementation methods: 
private:
  int bcastNo;//Number of broadcasts received (also serial number)
  CkReductionMgr::contributorInfo reductionInfo;//My reduction information
};



//An ArrayElement1D is a utility class where you are 
// constrained to a 1D "thisIndex" and 1D "numElements".
class ArrayElement1D : public ArrayElement
{
public:
  ArrayElement1D(void);
  ArrayElement1D(CkMigrateMessage *m);
  int getIndex(void) {return thisIndex;}
  int getArraySize(void)  {return numElements;}
  
//Pack/unpack routine (called before and after migration)
  virtual void pup(PUP::er &p);
 
  int numElements;//Initial array size
  int thisIndex;//1-D array index
};

//An ArrayElementT is a utility class where you are 
// constrained to a "thisIndex" of some fixed-sized type T.
template <class T>
class ArrayElementT : public ArrayElement
{
public:
  ArrayElementT(void) {thisIndex=*(T *)thisIndexMax.data();}
  ArrayElementT(CkMigrateMessage *msg) {thisIndex=*(T *)thisIndexMax.data();}
  
  T thisIndex;//Object array index
};

typedef struct {int x,y;} CkArray_index2D;
void operator|(PUP::er &p,CkArray_index2D &i);
typedef ArrayElementT<CkArray_index2D> ArrayElement2D;

typedef struct {int x,y,z;} CkArray_index3D;
void operator|(PUP::er &p,CkArray_index3D &i);
typedef ArrayElementT<CkArray_index3D> ArrayElement3D;

/*********************** Array Manager BOC *******************/

class CkArrayRec;//An array element record

class CkArray : public CkReductionMgr {
	friend class ArrayElement;
	friend class ArrayElement1D;
	friend class CProxy_ArrayBase;
	friend class CkArrayRec;
	friend class CkArrayRec_aging;
	friend class CkArrayRec_local;
	friend class CkArrayRec_remote;
	friend class CkArrayRec_buffering;

//A fixed value, to detect bad GroupID/heap corruption
#define CkArray_MAGIC 0xf1a5D00D
  int CkArray_magic;
#ifdef CMK_OPTIMIZE
# define CkArray_MAGIC_CHECK() /*empty, for speed*/
#else
# define CkArray_MAGIC_CHECK() do{ \
	if (CkArray_magic != CkArray_MAGIC) \
		CkAbort("Charm++ array manager trashed-- indicates "\
        "a bad GroupID, corrupted message, or heap corruption");\
  } while(0)
#endif

public:
//Array Creation:
  static  CkGroupID CreateArray(CkGroupID mapID,int numInitial=0);

  CkArray(CkArrayCreateMsg *);
  CkArray(CkMigrateMessage *);
  CkGroupID &getGroupID(void) {return thisgroup;}

//Element creation/destruction:
  void InsertElement(CkArrayMessage *m);
  void DoneInserting(void);
  void ElementDying(CkArrayRemoveMsg *m);
  
  //Fetch a local element via its index (return NULL if not local)
  ArrayElement *getElement(const CkArrayIndex &index);

//Messaging:
  //Called by proxy to deliver message to any array index
  // After send, the array owns msg.
  void Send(CkArrayMessage *msg);
  //Called by send to deliver message to an element.
  inline void RecvForElement(CkArrayMessage *msg);
  //Called by CkArrayRec for a local message
  void deliverLocal(CkArrayMessage *msg,ArrayElement *el);
   //Called by CkArrayRec for a remote message
  inline void deliverRemote(CkArrayMessage *msg,int onPE);
   //Called by CkArrayRec for a remote message
  void deliverUnknown(CkArrayMessage *msg);

//Migration:
  void migrateMe(ArrayElement *elem, int where);
  
  //Internal:
  void RecvMigratedElement(CkArrayElementMigrateMessage *msg);
  void UpdateLocation(CkArrayUpdateMsg *msg);

//Load balancing:
  void DummyAtSync(void);
  
//Housecleaning: called periodically from node zero
  void SpringCleaning(void);

//Broadcast:
  void SendBroadcast(CkArrayMessage *msg);
  void RecvBroadcast(CkArrayMessage *msg);
  
#if CMK_LBDB_ON
  LBDatabase *the_lbdb;
#endif
private:
#if CMK_LBDB_ON
  LDBarrierClient dummyBarrierHandle;
  static void staticDummyResumeFromSync(void* data);
  void dummyResumeFromSync(void);
  static void staticRecvAtSync(void* data);
  void recvAtSync(void);
  
  //Load balancing callbacks:
  static void staticSetStats(LDOMHandle _h, int _state);
  void setStats(LDOMHandle _h, int _state);
  static void staticQueryLoad(LDOMHandle _h);
  void queryLoad(LDOMHandle _h);
  
  LDOMHandle myLBHandle;
  void initLB(LBDatabase *Nlbdb);
#endif

  //This flag lets us detect element suicide, so we can stop timing
  CmiBool curElementIsDead;
  void localElementDying(ArrayElement *e);

  //Maps array index to array element records (CkArrayRec)
  CkHashtableT<CkArrayIndexMax,CkArrayRec *> hash;
  
  //Add given element array record (which then owns it) at idx.
  void insertRec(CkArrayRec *rec,const CkArrayIndex &idx);
  //Look up array element in hash table.  Index out-of-bounds if not found.
  CkArrayRec *elementRec(const CkArrayIndex &idx);
  //Look up array element in hash table.  Return NULL if not there.
  CkArrayRec *elementNrec(const CkArrayIndex &idx);
  
  //This structure keeps counts of numbers of array elements:
  struct {
  	int local;//Living here
  	int migrating;//Just left
  } num;//<- array element counts
  
  int numInitial;//Number of 1D initial array elements (backward-compatability)
  CmiBool isInserting;//Are we currently inserting elements?

  //Deliver this message to this array element
  void invokeEntry(ArrayElement *el,int entryIdx,void *msg);

  //Create, but do not register an element
  ArrayElement *buildElement(int type,int ctor,void *msg,
		const CkArrayIndex &ind,bool fromMigration);

  //Create, initialize, and register this new element
  ArrayElement *newElement(int ctor,void *msg,const CkArrayIndex &ind);

//Broadcast support
  int bcastNo;//Number of broadcasts received (also serial number)
  int oldBcastNo;//Above value last spring cleaning
  //This queue stores old broadcasts (in case a migrant arrives
  // and needs to be brought up to date)
  CkQ<CkArrayMessage *> oldBcasts;
  void bringBroadcastUpToDate(ArrayElement *el);
  void deliverBroadcast(CkArrayMessage *bcast);
  void deliverBroadcast(CkArrayMessage *bcast,ArrayElement *el);

//For houscleaning:
  int nSprings;//Number of times "SpringCleaning" has been broadcast

///// Map support:
  CkGroupID mapID;
  int mapHandle;
  CkArrayMap *map;
//Return the home PE of the given array index
  int homePE(const CkArrayIndex &idx) const
  	{return map->procNum(mapHandle,idx);}
//Return 1 if this is the home PE of the given array index
  CmiBool isHome(const CkArrayIndex &idx) const
  	{return (CmiBool)(homePE(idx)==CkMyPe());}

//Initialization support:
  static void static_initAfterMap(void *dis);
  void initAfterMap(void);
  CkArrayRec* pupArrayRec(PUP::er &p, CkArrayRec *rec, CkArrayIndex *idx);
  void pupHashTable(PUP::er &p);
public:
  int array_size(void) { return numInitial; } // required for historic reasons
  virtual void pup(PUP::er &p); //pack-unpack method
  static void pupArrayMsgQ(CkQ<CkArrayMessage *> &q, PUP::er &p);
};

/************************** Array Messages ****************************/

#include "CkArray.decl.h"

/*This really doesn't belong here: need a marshall.h */
class CkMarshallMsg : public CMessage_CkMarshallMsg {
public: char *msgBuf;
};

//This is the default creation message sent to a new array element
class CkArrayElementCreateMsg:public CMessage_CkArrayElementCreateMsg {};

class CkArrayElementMigrateMessage : public CMessage_CkArrayElementMigrateMessage {
protected:
	friend class CkArray;
	void* packData;
	~CkArrayElementMigrateMessage() {}
public:
  static void *alloc(int msgnum, size_t size, int *array, int priobits);
  static void *pack(CkArrayElementMigrateMessage *);
  static CkArrayElementMigrateMessage *unpack(void *in);
};

//Message: Remove the array element at the given index.
class CkArrayRemoveMsg : public CMessage_CkArrayRemoveMsg 
{public:
	CkArrayRemoveMsg(const CkArrayIndex &idx);	// move to ckarray.C
};

//Message: Direct future messages for this array element to this PE.
class CkArrayUpdateMsg : public CMessage_CkArrayUpdateMsg 
{public:
	CkArrayUpdateMsg(const CkArrayIndex &idx);
};

#endif
