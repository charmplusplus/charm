/*
Location manager: keeps track of an indexed set of migratable
objects.  Used by the array manager to locate array elements,
interact with the load balancer, and perform migrations.

Does not handle reductions (see ckreduction.h), broadcasts,
array proxies, or the details of element creation (see ckarray.h).
*/
#ifndef __CKLOCATION_H
#define __CKLOCATION_H

/*******************************************************
Array Index class.  An array index is just a hash key-- 
a run of integers used to look up an object in a hash table.
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

    //These routines allow CkArrayIndex to be used in
    //  a CkHashtableT
	CkHashCode hash(void) const;
	static CkHashCode staticHash(const void *a,size_t);
	int compare(const CkArrayIndex &ind) const;
	static int staticCompare(const void *a,const void *b,size_t);
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
	CkArrayIndexMax(int i) {copyFrom(CkArrayIndex3D(i,i,i)); }
	CkArrayIndexMax(const CkArrayIndex &that) 
		{copyFrom(that);}
	CkArrayIndexMax &operator=(const CkArrayIndex &that) 
		{copyFrom(that); return *this;}
        void print() { CmiPrintf("%d: %d %d %d\n", nInts,index.data[0], index.data[1], index.data[2]); }
};

class CkArrayIndexStruct {
public:
	int nInts;
	int index[CK_ARRAYINDEX_MAXLEN];
};

/*********************** Array Messages ************************/
class CkArrayMessage : public CkMessage {
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
};

/* Utility */
#if CMK_LBDB_ON
#include "LBDatabase.h"
class LBDatabase;
#endif

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
//What to do if an entry method is invoked on
// an array element that does not (yet) exist:
typedef enum {
	CkArray_IfNotThere_buffer=0, //Wait for it to be created
	CkArray_IfNotThere_createhere=1, //Make it on sending Pe
	CkArray_IfNotThere_createhome=2 //Make it on (a) home Pe
} CkArray_IfNotThere;

#include "CkLocation.decl.h"

/************************** Array Messages ****************************/
//This is the default creation message sent to a new array element

class CkArrayElementMigrateMessage : public CMessage_CkArrayElementMigrateMessage {
public:
	int length;//Size in bytes of the packed data
	double* packData;
};

/******************* Map object ******************/

extern CkGroupID _RRMapID;
class CkLocMgr;
class CkArrMgr;

class CkArrayMap : public IrrGroup // : public CkGroupReadyCallback
{
public:
  CkArrayMap(void);
  CkArrayMap(CkMigrateMessage *m) {}
  virtual ~CkArrayMap();
  virtual int registerArray(int numElements,CkArrayID aid);
  virtual void populateInitial(int arrayHdl,int numElements,void *ctorMsg,CkArrMgr *mgr);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element) =0;
//  virtual void pup(PUP::er &p) { CkGroupReadyCallback::pup(p); }
};

static inline CkGroupID CkCreatePropMap(void)
{
  return CProxy_PropMap::ckNew();
}

extern void _propMapInit(void);
extern void _CkMigratable_initInfoInit(void);


/*************************** CkLocRec ******************************/
class CkArray;//Array manager
class CkLocMgr;//Location manager
class CkMigratable;//Migratable object

class CkLocRec {
protected:
  CkLocMgr *myLocMgr;
  int lastAccess;//Age when last accessed
  //Called when we discover we are obsolete before we delete ourselves
  virtual void weAreObsolete(const CkArrayIndex &idx);
public:
  CkLocRec(CkLocMgr *mgr) :myLocMgr(mgr) { }
  virtual ~CkLocRec();
  inline CkLocMgr *getLocMgr(void) const {return myLocMgr;}
  
  //Return the type of this ArrayRec:
  typedef enum {
    base=0,//Base class (invalid type)
    local,//Array element that lives on this Pe
    remote,//Array element that lives on some other Pe
    buffering,//Array element that was just created
    dead//Deleted element (for debugging)
  } RecType;
  virtual RecType type(void)=0;
  
  //Accept a message for this element
  virtual bool deliver(CkArrayMessage *m,bool viaScheduler)=0;
  
  //This is called when this ArrayRec is about to be replaced.
  // It is only used to deliver buffered element messages.
  virtual void beenReplaced(void);
  
  //Return if this rec is now obsolete
  virtual bool isObsolete(int nSprings,const CkArrayIndex &idx)=0; 

  //Return the represented array element; or NULL if there is none
  virtual CkMigratable *lookupElement(CkArrayID aid);
  //Return the last known processor; or -1 if none
  virtual int lookupProcessor(void);
};

class CkLocRec_local : public CkLocRec {
  CkArrayIndexMax idx;//Array index
  int localIdx; //Local index
  bool running; //Inside a startTiming/stopTiming pair
  bool *deletedMarker; //Set this if we're deleted during processing
  CkQ<CkArrayMessage *> halfCreated; //Messages for nonexistent siblings of existing elements
public:
  //Creation and Destruction:
  CkLocRec_local(CkLocMgr *mgr,bool fromMigration,
  	const CkArrayIndex &idx_,int localIdx_);
  void migrateMe(int toPe); //Leave this processor
  void destroy(void); //User called destructor
  virtual ~CkLocRec_local();
  
  //A new element has been added to this index
  void addedElement(void);

  //Accept a message for this element
  // Returns false if the element died in transit
  virtual bool deliver(CkArrayMessage *m,bool viaScheduler);
  
  //Invoke the given entry method on this element
  // Returns false if the element died in transit
  bool invokeEntry(CkMigratable *obj,void *msg,int idx);

  virtual RecType type(void);
  virtual bool isObsolete(int nSprings,const CkArrayIndex &idx);
  
#if CMK_LBDB_ON  //For load balancing:
  //Load balancer
  void startTiming(void);
  void stopTiming(void);
#else
  inline void startTiming(void) {  }
  inline void stopTiming(void) { }
#endif
  inline int getLocalIndex(void) const {return localIdx;}
  inline const CkArrayIndex &getIndex(void) const {return idx;}
  virtual CkMigratable *lookupElement(CkArrayID aid);
  virtual int lookupProcessor(void);
  
#if CMK_LBDB_ON
public:
  inline LBDatabase *getLBDB(void) const {return the_lbdb;}
  static void staticMigrate(LDObjHandle h, int dest);
private:
  LBDatabase *the_lbdb;
  LDObjHandle ldHandle;
#endif
};
class CkLocRec_remote;

/*********************** CkMigratable ******************************/
//This is the superclass of all migratable parallel objects
class CkMigratable : public Chare {
private:
  CkLocRec_local *myRec;
  int thisChareType;//My chare type
  void commonInit(void);
public:
  CkArrayIndexMax thisIndexMax;

  CkMigratable(void);
  CkMigratable(CkMigrateMessage *m);
  virtual ~CkMigratable();
  virtual void pup(PUP::er &p);

  inline int ckGetChareType(void) const {return thisChareType;}
  const CkArrayIndex &ckGetArrayIndex(void) const {return myRec->getIndex();}

#if CMK_LBDB_ON  //For load balancing:
  //Suspend load balancer measurements (e.g., before CthSuspend)
  inline void ckStopTiming(void) {myRec->stopTiming();}
  //Begin load balancer measurements again (e.g., after CthSuspend)
  inline void ckStartTiming(void) {myRec->startTiming();}
#else
  inline void ckStopTiming(void) { }
  inline void ckStartTiming(void) { }
#endif
  //Initiate a migration to the given processor
  inline void ckMigrate(int toPe) {myRec->migrateMe(toPe);}
  
  //Called by the system just before and after migration to another processor:  
  virtual void ckAboutToMigrate(void); /*default is empty*/
  virtual void ckJustMigrated(void); /*default is empty*/

  //Delete this object
  virtual void ckDestroy(void);

  //Execute the given entry method.  Returns false if the element 
  // deleted itself or migrated away during execution.
  inline bool ckInvokeEntry(int epIdx,void *msg) 
	  {return myRec->invokeEntry(this,msg,epIdx);}

protected:
  CmiBool usesAtSync;//You must set this in the constructor to use AtSync().
  virtual void ResumeFromSync(void);
  CmiBool barrierRegistered;//True iff barrier handle below is set

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(void);
private: //Load balancer state:
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
  static void staticResumeFromSync(void* data);
public:
  void ckFinishConstruction(void);
#else
  void AtSync(void) { ResumeFromSync();}
public:
  void ckFinishConstruction(void) { }
#endif
};

//Stores a list of array elements
class CkMigratableList {
	CkVec<CkMigratable *> el;
 public:
	CkMigratableList();
	~CkMigratableList();
	
	void setSize(int s);
	inline int length(void) const {return el.length();}

	//Add an element at the given location
	void put(CkMigratable *v,int atIdx);

	//Return the element at the given location
	inline CkMigratable *get(int localIdx) {return el[localIdx];}

	//Return the next non-empty element starting from the given index,
	// or NULL if there is none.  Updates from to point past the returned index.
	CkMigratable *next(int &from) {
		while (from<length()) {
			CkMigratable *ret=el[from];
			from++;
			if (ret!=NULL) return ret;
		}
		return NULL;
	}

	//Remove the element at the given location
	inline void empty(int localIdx) {el[localIdx]=NULL;}
};

//A typed version of the above
template <class T>
class CkMigratableListT : public CkMigratableList {
	typedef CkMigratableList super;
public:
	inline void put(T *v,int atIdx) {super::put((void *)v,atIdx);}
	inline T *get(int localIdx) {return (T *)super::get(localIdx);}
	inline T *next(int &from) {return (T *)super::next(from);}
};


/********************** CkLocMgr ********************/
//A tiny class for detecting heap corruption
class CkMagicNumber_impl {
 protected:
	int magic;
	void badMagicNumber(int expected) const;
	CkMagicNumber_impl(int m);
};
template<class T>
class CkMagicNumber : public CkMagicNumber_impl {
	enum {good=sizeof(T)^0x7EDC0000};
 public:
	CkMagicNumber(void) :CkMagicNumber_impl(good) {}
#ifdef CMK_OPTIMIZE 
	inline void check(void) const { /*Empty, for speed*/ }
#else
	inline void check(void) const {
		if (magic!=good) badMagicNumber(good);
	}
#endif
};

//Abstract superclass of all array manager objects 
class CkArrMgr {
public:
	//Insert this initial element on this processor
	virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg)=0;
	
	//Done with initial insertions
	virtual void doneInserting(void)=0;
	
	//Create an uninitialized element after migration
	//  The element's constructor will be called immediately after.
	virtual CkMigratable *allocateMigrated(int elChareType,const CkArrayIndex &idx);

	//Demand-create an element at this index on this processor
	// Returns true if the element was successfully added;
	// false if the element migrated away or deleted itself.
	virtual bool demandCreateElement(const CkArrayIndex &idx,int onPe,int ctor) =0;
};

//A group which manages the location of an indexed set of
// migratable objects.
class CkLocMgr : public IrrGroup {
	CkMagicNumber<CkMigratable> magic; //To detect heap corruption
public:
	CkLocMgr(CkGroupID map,CkGroupID lbdb,int numInitial);

	CkGroupID &getGroupID(void) {return thisgroup;}
	inline CProxy_CkLocMgr &getProxy(void) 
		{return thisProxy;}
	inline CProxyElement_CkLocMgr &getLocalProxy(void) 
		{return thislocalproxy;}

//Interface used by array manager and proxies
	//Add a new local array manager to our list.  Array managers
	// must be registered in the same order on all processors.
	//Returns a list which will contain that array's local elements
	CkMigratableList *addManager(CkArrayID aid,CkArrMgr *mgr);

	//Populate this array with initial elements
	void populateInitial(int numElements,void *initMsg,CkArrMgr *mgr) 
		{map->populateInitial(mapHandle,numElements,initMsg,mgr);}
	
	//Add a new local array element, calling element's constructor
	// Returns true if the element was successfully added;
	// false if the element migrated away or deleted itself.
	bool addElement(CkArrayID aid,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);
	
	//Deliver message to this element, going via the scheduler if local
	void deliverViaQueue(CkMessage *m);

	//Done inserting elements for now
	void doneInserting(void);

//Advisories:
	//This index now lives on the given processor-- update local records
	void inform(const CkArrayIndex &idx,int nowOnPe);
	
	//This index now lives on the given processor-- tell the home processor
	void informHome(const CkArrayIndex &idx,int nowOnPe);

	//This message took several hops to reach us-- fix it
	void multiHop(CkArrayMessage *m);

//Interface used by CkLocRec_local
	//Look up the object with this local index
	inline CkMigratable *lookupLocal(int localIdx,CkArrayID arrayID) {
#ifndef CMK_OPTIMIZE
		if (managers.find(arrayID).mgr==NULL)
			CkAbort("CkLocMgr::lookupLocal called for unknown array!\n");
#endif
		return managers.find(arrayID).elts.get(localIdx);
	}
	
	//Migrate us to another processor
	void migrate(CkLocRec_local *rec,int toPe);
	
#if CMK_LBDB_ON
	LBDatabase *getLBDB(void) const { return the_lbdb; }
	const LDOMHandle &getOMHandle(void) const { return myLBHandle; }
#endif
	
	//This index will no longer be used-- delete the associated elements
	void reclaim(const CkArrayIndex &idx,int localIdx);

	int getSpringCount(void) const { return nSprings; }

	bool demandCreateElement(CkArrayMessage *msg,int onPe);
	
//Interface used by external users:
	//Home mapping
	inline int homePe(const CkArrayIndex &idx) const
		{return map->procNum(mapHandle,idx);}	
	inline bool isHome(const CkArrayIndex &idx) const
		{return homePe(idx)==CkMyPe();}
	//Look up the object with this array index, or return NULL
	CkMigratable *lookup(const CkArrayIndex &idx,CkArrayID aid);
	//"Last-known" location (returns a processor number)
	int lastKnown(const CkArrayIndex &idx) const;

//Communication:
	bool deliver(CkMessage *m);
	void migrateIncoming(CkArrayElementMigrateMessage *msg);
	void updateLocation(const CkArrayIndexMax &idx,int nowOnPe);
	void reclaimRemote(const CkArrayIndexMax &idx,int deletedOnPe);
	void dummyAtSync(void);

private:
//Internal interface:
	//Add given element array record at idx, replacing the existing record
	void insertRec(CkLocRec *rec,const CkArrayIndex &idx);
	//Add given record, when there is guarenteed to be no prior record
	void insertRecN(CkLocRec *rec,const CkArrayIndex &idx);
	//Insert a remote record at the given index
	CkLocRec_remote *insertRemote(const CkArrayIndex &idx,int nowOnPe);

	//Look up array element in hash table.  Index out-of-bounds if not found.
	CkLocRec *elementRec(const CkArrayIndex &idx);
	//Look up array element in hash table.  Return NULL if not there.
	CkLocRec *elementNrec(const CkArrayIndex &idx);

	void pupElementsFor(PUP::er &p,CkLocRec_local *rec);
	bool deliverUnknown(CkArrayMessage *msg);

//Data Members:
	//Map array ID to manager and elements
	class ManagerRec {
	public:
		ManagerRec *next; //next non-null array manager
		CkArrMgr *mgr;
		CkMigratableList elts;
		ManagerRec() {
			next=NULL;
			mgr=NULL;
		}
		void init(void) { next=NULL; mgr=NULL; }
		CkMigratable *element(int localIdx) {
			return elts.get(localIdx);
		}
	};
	GroupIdxArray<ManagerRec> managers;
	int nManagers;
	ManagerRec *firstManager; //First non-null array manager

	bool addElementToRec(CkLocRec_local *rec,ManagerRec *m,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);
	
	//For keeping track of free local indices
	CkVec<int> freeList;//Linked list of free local indices
	int firstFree;//First free local index
	int localLen;//Last allocated local index plus one
	int nextFree(void);
	
	CProxy_CkLocMgr thisProxy;
	CProxyElement_CkLocMgr thislocalproxy;
	CkHashtableT<CkArrayIndexMax,CkLocRec *> hash;
	
	//This flag is set while we delete an old copy of a migrator
	bool duringMigration;
	
	//Occasionally clear out stale remote pointers
	static void staticSpringCleaning(void *mgr);
	void springCleaning(void);
	int nSprings;
	
	//Map object
	CkGroupID mapID;
	int mapHandle;
	CkArrayMap *map;
  	
#if CMK_LBDB_ON
	LBDatabase *the_lbdb;
	LDBarrierClient dummyBarrierHandle;
	static void staticDummyResumeFromSync(void* data);
	void dummyResumeFromSync(void);
	static void staticRecvAtSync(void* data);
	void recvAtSync(void);
	LDOMHandle myLBHandle;
#endif
	void initLB(CkGroupID lbdbID);
};


#endif /*def(thisHeader)*/
