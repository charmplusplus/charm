/* Generalized Chare Arrays

These classes implement Chare Arrays.  
These are dynamic (i.e. allowing insertion
and deletion) collections of ordinary Chares 
indexed by arbitrary runs of bytes.

The general structure is:

CkArray is the "array manager" Group, or BOC-- 
it creates, keeps track of, and cares for all the
array elements on this PE (i.e.. "local" elements).  
It does so using a hashtable.

CkArrayElement is the type of the array 
elements (a subclass of Chare).

CkArrayIndex is an arbitrary run of bytes,
used to index into the CkArray hashtable.


Converted from 1-D arrays 2/27/2000 by
Orion Sky Lawlor, olawlor@acm.org

*/
#ifndef _CKARRAY_H
#define _CKARRAY_H

#include "ckreduction.h"

/*******************************************************
Array Index class.  An array index is just a HashKey-- 
a run of bytes used to look up an object in a hash table.
An Array Index cannot be modified once it is created.
 */

#include "ckhashtable.h"

class CkArrayIndex : public HashKey 
{
public:
	// This method returns the length of and a pointer to the key data.
	//The returned pointer must be aligned to at least an integer boundary.
	virtual const unsigned char *getKey(/*out*/ int &len) const =0;
	
	//These utility routines call the routine above
	// (they're slightly less efficient, but easier to use)
	int len(void) const {int len;getKey(len);return len;}
	const unsigned char *data(void) const {int len;return getKey(len);}
};

//Simple ArrayIndex classes: the key is just integer indices.
class CkArrayIndex1D : public CkArrayIndex {
public: int index;
	CkArrayIndex1D(int i0) {index=i0;}
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};
class CkArrayIndex2D : public CkArrayIndex {
public: int index[2];
	CkArrayIndex2D(int i0,int i1) {index[0]=i0;index[1]=i1;}
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};
class CkArrayIndex3D : public CkArrayIndex {
public: int index[3];
	CkArrayIndex3D(int i0,int i1,int i2) {index[0]=i0;index[1]=i1;index[2]=i2;}
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};
class CkArrayIndex4D : public CkArrayIndex {
public: int index[4];
	CkArrayIndex4D(int i0,int i1,int i2,int i3) 
	  {index[0]=i0;index[1]=i1;index[2]=i2;index[3]=i3;}
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};

//A slightly more complex array index: the key is an object
// whose size is fixed at compile time.
template <class object> //Key object
class CkArrayIndexObject : public CkArrayIndex {
public:
	object obj;
	CkArrayIndexObject(const object &srcObj) {obj=srcObj;}
	virtual const unsigned char *getKey(/*out*/ int &len) const 
	  {len=sizeof(object);return (const unsigned char *)&obj;}
};

//Here the key is a run of bytes whose length can vary at run time;
// the data does not belong to us, and is not deleted when we are.
class CkArrayIndexConst : public CkArrayIndex {
protected:
	int nBytes;//Length of key in bytes
	const unsigned char *constData;//Data array, which we do not own
public:
	CkArrayIndexConst(int len,const void *srcData);//Copy given data
	CkArrayIndexConst(const CkArrayIndex &that); //Copy given index's data
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};

//Finally, here the key is a run of bytes whose length can vary at run time.  
// Is generic because it can contain the data of *any* kind of ArrayIndex.
class CkArrayIndexGeneric : public CkArrayIndex {
protected:
	int nBytes;//Length of key in bytes
	unsigned char *heapData;//Heap-allocated data array
	void copyFrom(int len,const void *srcData);
public:
	CkArrayIndexGeneric(int len,const void *srcData);//Copy given data
	CkArrayIndexGeneric(const CkArrayIndex &that); //Copy given index's data
	virtual ~CkArrayIndexGeneric();//Deletes allocated data
	virtual const unsigned char *getKey(/*out*/ int &len) const;
};


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
#define ChareIndex(ct)          CProxy_##ct##::__idx
#define EntryIndex(ct,ep,mt)    CProxy_##ct##::ckIdx_##ep##((##mt##*)0)
#define ConstructorIndex(ct,mt) EntryIndex(ct,ct,mt)

typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;
typedef struct {
	ChareIndexType chareType;
	EntryIndexType constructorType;
	EntryIndexType migrateType;
} CkArrayElementType;

//Forward declarations
class CkArray;
class ArrayElement;
class ArrayMessage;
class CkArrayCreateMsg;
class CkArrayInsertMsg;
class CkArrayRemoveMsg;
class CkArrayUpdateMsg;

//This class is a wrapper around a CkArrayIndex and ArrayID,
// used by array proxies. 
class CkArrayProxyBase :public CkArrayID {
protected:
	CkArrayIndex *_idx;//<- specialized array index, or NULL
	CkArrayProxyBase() {}
	CkArrayProxyBase(const CkArrayID &aid) {_aid=aid._aid;_idx=NULL;}
	CkArrayProxyBase(const CkArrayID &aid,CkArrayIndex *idx)
	  {_aid=aid._aid;_idx=idx;}
public:
	CkGroupID ckGetGroupID(void) { return _aid; }
//Messaging:
	void send(ArrayMessage *msg, int entryIndex);
	void broadcast(ArrayMessage *msg, int entryIndex);
		
//Array element insertion
	void insert(int onPE=-1);
	void doneInserting(void);//Call on PE 0 after inserts (for load balancer)
	
//Register the given reduction client:
	void reductionClient(CkReductionMgr::clientFn fn,void *param=NULL);
};

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
  virtual int registerArray(CkArrayMapRegisterMessage *);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element);
};

/************************ Array Element *********************/
class ArrayElementCreateMessage;
class ArrayElementMigrateMessage;

class ArrayElement : public Chare
{
	friend class CkArray;
public:
  ArrayElement(ArrayElementCreateMessage *msg);
  ArrayElement(ArrayElementMigrateMessage *msg);

  virtual ~ArrayElement();//Deletes heap-allocated array index
  
  CkArrayIndexGeneric *thisindex;//Array index (allocated on heap)

//Remote method: deletes this array element
  void destroy(void);
  
//Contribute to the given reduction type.  Data is copied, not deleted.
  void contribute(int dataSize,void *data,CkReduction::reducerType type);

//Migrate to the given processor number
  void migrateMe(int toPE);
  
private:
  ArrayElement(void) {} /*Forces us to use the odd constructors above*/

protected:
  CkArray *thisArray;//My source array
  
  //For migration, overload these:
  virtual int packsize(void) const;//Returns number of bytes I need
  virtual void *pack(void *intoBuf);//Write me to given buffer
  virtual const void *unpack(const void *fromBuf);//Extract me from given buffer

  CkArrayID thisArrayID;//My source array
  CkReductionMgr::contributorInfo reductionInfo;//My reduction information

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(void);
  virtual void ResumeFromSync(void);
protected:
    CmiBool usesAtSync;//You must set this in the constructor to use AtSync().
private: //Load balancer state:
    LDObjHandle ldHandle;//Transient (not migrated)
    LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
    static void staticResumeFromSync(void* data);
    static void staticMigrate(LDObjHandle h, int dest);
#endif
    void lbRegister(void);//Connect to load balancer
    void lbUnregister(void);//Disconnect from load balancer

//Array implementation methods: 
private:
  int bcastNo;//Number of broadcasts received (also serial number)
  void private_startConstruction(CkGroupID agID,const CkArrayIndex &idx);
  void private_startMigration(CkGroupID agID,const CkArrayIndex &idx);
public:
//these are called by the translator-generated constructors.
  void private_finishConstruction(void);
  void private_finishMigration(void);
};

//An ArrayElement1D is a utility class where you are 
// constrained to a 1D "thisIndex" and 1D "numElements".
class ArrayElement1D : public ArrayElement
{
public:
  ArrayElement1D(ArrayElementCreateMessage *msg);
  ArrayElement1D(ArrayElementMigrateMessage *msg);
  int getIndex(void) {return thisIndex;}
  int getSize(void)  {return numElements;}

protected:
  //For migration, overload these:
  virtual int packsize(void) const;//Returns number of bytes I need
  virtual void *pack(void *intoBuf);//Write me to given buffer
  virtual const void *unpack(const void *fromBuf);//Extract me from given buffer
  
  int numElements;//Initial array size
  int thisIndex;//1-D array index
};

//An ArrayElementT is a utility class where you are 
// constrained to a "thisIndex" of some fixed-sized type T.
template <class T>
class ArrayElementT : public ArrayElement
{
public:
  ArrayElementT(ArrayElementCreateMessage *msg):ArrayElement(msg)
  {thisIndex=*(T *)thisindex->data();}
  ArrayElementT(ArrayElementMigrateMessage *msg):ArrayElement(msg) {}

protected:
  //For migration, overload these:
  virtual int packsize(void) const//Returns number of bytes I need
  {return ArrayElement::packsize()+sizeof(T);}
  virtual void *pack(void *intoBuf)//Write me to given buffer
  {
  	char *buf=(char *)ArrayElement::pack(intoBuf);
  	*(T *)buf=thisIndex; buf+=sizeof(T);
  	return buf;
  }
  virtual const void *unpack(const void *fromBuf)//Extract me from given buffer
  {
  	const char *buf=(const char *)ArrayElement::unpack(fromBuf);
  	thisIndex=*(T *)buf; buf+=sizeof(T);
  	return buf;
  }
  
  T thisIndex;//Object array index
};

/*********************** Array Manager BOC *******************/

class CkArrayRec;//An array element record

class CkArray : public CkReductionMgr {
	friend class ArrayElement;
	friend class CkArrayProxyBase;
	friend class CkArrayRec;
	friend class CkArrayRec_aging;
	friend class CkArrayRec_local;
	friend class CkArrayRec_remote;
	friend class CkArrayRec_buffering;
	friend class CkArrayRec_buffering_migrated;
public:
//Array Creation:
  static  CkGroupID CreateArray(int numInitialElements,
				CkGroupID mapID,
				ChareIndexType elementChare,
				EntryIndexType elementConstructor,
				EntryIndexType elementMigrator);

  CkArray(CkArrayCreateMsg *);
  CkGroupID &getGroupID(void) {return thisgroup;}

//Element creation/destruction:
  void InsertElement(CkArrayInsertMsg *m);
  void DoneInserting(void);
  void ElementDying(CkArrayRemoveMsg *m);
  
  //Fetch a local element via its index (return NULL if not local)
  ArrayElement *getElement(const CkArrayIndex &index);
  
  //Internal creation calls (called only by ArrayElement)
  void recvElementID(const CkArrayIndex &index, ArrayElement *elem,bool fromMigration);

//Messaging:
  //Called by proxy to deliver message to any array index
  // After send, the array owns msg.
  void Send(ArrayMessage *msg);
  //Called by send to deliver message to an element.
  void RecvForElement(ArrayMessage *msg);
  //Called by CkArrayRec for a local message
  void deliverLocal(ArrayMessage *msg,ArrayElement *el);
   //Called by CkArrayRec for a remote message
  void deliverRemote(ArrayMessage *msg,int onPE);

//Migration:
  void migrateMe(ArrayElement *elem, int where);
  
  //Internal:
  void RecvMigratedElement(ArrayElementMigrateMessage *msg);
  void UpdateLocation(CkArrayUpdateMsg *msg);

//Load balancing:
  void DummyAtSync(void);
  
//Housecleaning: called periodically from node zero
  void SpringCleaning(void);

//Broadcast:
  void SendBroadcast(ArrayMessage *msg);
  void RecvBroadcast(ArrayMessage *msg);
  
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
  LBDatabase *the_lbdb;
#endif
  //This flag lets us detect element suicide, so we can stop timing
  bool curElementIsDead;


  Hashtable hash;//Maps array index to array element records (CkArrayRec)
  //Add given element array record (which then owns it) at idx.
  // If replaceOld, old record is discarded and replaced.
  void insertRec(CkArrayRec *rec,const CkArrayIndex &idx,int replaceOld=1);
  //Look up array element in hash table.  Index out-of-bounds if not found.
  CkArrayRec *elementRec(const CkArrayIndex &idx);
  //Look up array element in hash table.  Return NULL if not there.
  CkArrayRec *elementNrec(const CkArrayIndex &idx) 
  	{return (CkArrayRec *)hash.get(idx);}
  
  //This structure keeps counts of numbers of array elements:
  struct {
  	int local;//Living here
  	int migrating;//Just left
  	int arriving;//Just arrived
  	int creating;//Just created
  } num;//<- array element counts
  
  CkArrayElementType type;
  int numInitial;//Initial array size (used only for 1D case)

//Broadcast support
  int bcastNo;//Number of broadcasts received (also serial number)
  int oldBcastNo;//Above value last spring cleaning
  //This queue stores old broadcasts (in case a migrant arrives
  // and needs to be brought up to date)
  CkQ<ArrayMessage *> oldBcasts;
  void bringBroadcastUpToDate(ArrayElement *el);
  void deliverBroadcast(ArrayMessage *bcast,ArrayElement *el);

//For houscleaning:
  int nSprings;//Number of times "SpringCleaning" has been broadcast
  double lastCleaning;//Wall time that last springcleaning was broadcast

///// Map support:
  CkGroupID mapID;
  int mapHandle;
  CkArrayMap *map;
//Return the home PE of the given array index
  int homePE(const CkArrayIndex &idx) const
  	{return map->procNum(mapHandle,idx);}
//Return 1 if this is the home PE of the given array index
  bool isHome(const CkArrayIndex &idx) const
  	{return homePE(idx)==CkMyPe();}

//Initialization support:
  static void static_initAfterMap(void *dis);
  void initAfterMap(void);
};

/*********************** Array Messages ************************/
//This is a superclass of all the messages which contain array indices.
class CkArrayIndexMsg
{
private:
  //This array keeps the array index of the destination if it fits,
  //  otherwise the array index gets appended at the end of the message.
#define CKARRAYINDEX_STORELEN_INTS 3 //Store 3 ints in indexStore
#define CKARRAYINDEX_STORELEN sizeof(int)*CKARRAYINDEX_STORELEN_INTS
  int indexStore[CKARRAYINDEX_STORELEN_INTS];
  int indexLength;//Bytes in destination array index

public:
  //Allocate and return a new CkArrayIndexGeneric with this message's index
  CkArrayIndexGeneric *copyIndex(void);
  
  //Return a CkArrayIndexConst with this message's index
  const CkArrayIndexConst index(void) const;

  //Writes the given index into this array message,
  // reallocating and copying if needed-- it may not be possible
  // to do this "in place".
  CkArrayIndexMsg *insertArrayIndex(const CkArrayIndex &idx);
};

//This class is used to avoid the cast on the insert index call above--
// It's just "syntactic sugar".
template <class M>
class CkArrayIndexMsgT : public CkArrayIndexMsg {
public:
	M *insert(const CkArrayIndex &idx) {return (M *)insertArrayIndex(idx);}
};

class ArrayMessage:public CkArrayIndexMsgT<ArrayMessage>  {
public:
  int from_pe,hopCount;//Original sender, number of hops since
  EntryIndexType entryIndex;//Destination entry method
};

#include "CkArray.decl.h"

class ArrayElementCreateMessage : public CkArrayIndexMsgT<ArrayElementCreateMessage>, 
public CMessage_ArrayElementCreateMessage {
public:
	CkGroupID agID;//Array's group ID
	int numInitial;//Initial array size (used only for 1D case)
};
class ArrayElementMigrateMessage : public CkArrayIndexMsgT<ArrayElementMigrateMessage>, 
public CMessage_ArrayElementMigrateMessage {
public:
	static void *alloc(int msgnum, int size, int *array, int priobits);
	static void *pack(ArrayElementMigrateMessage *);
	static ArrayElementMigrateMessage *unpack(void *in);

	CkGroupID agID;//Array's group ID
	int numInitial;//Initial array size (used only for 1D case)
	int from_pe;//Source PE
	void* packData;
};

//Message: Add an array element at this index.
class CkArrayInsertMsg : public CkArrayIndexMsgT<CkArrayInsertMsg>,
public CMessage_CkArrayInsertMsg 
{public:
	int onPE;//PE to create on, or -1 if not known yet.
	int chareType;//Kind of chare to create, or -1 for array's default type.
	int constructorIndex;//Chare constructor index, or -1 for array's default type.
	CkArrayInsertMsg(int NonPE=-1,
		int NchareType=-1,int NconstructorIndex=-1);
};

//Message: Remove the array element at the given index.
class CkArrayRemoveMsg : public CkArrayIndexMsgT<CkArrayRemoveMsg>,
public CMessage_CkArrayRemoveMsg 
{/*The array index is enough*/};

//Message: Direct future messages for this array element to this PE.
class CkArrayUpdateMsg : public CkArrayIndexMsgT<CkArrayUpdateMsg>, 
public CMessage_CkArrayUpdateMsg 
{public:
	int onPE;//Indicates given array index lives on this PE
	CkArrayUpdateMsg(void);
};

#endif
