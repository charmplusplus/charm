/*
Location manager: keeps track of an indexed set of migratable
objects.  Used by the array manager to locate array elements,
interact with the load balancer, and perform migrations.

Does not handle reductions (see ckreduction.h), broadcasts,
array proxies, or the details of element creation (see ckarray.h).
*/
#ifndef __CKLOCATION_H
#define __CKLOCATION_H

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
  unsigned int array_isImmediate(void);
  void array_setImmediate(unsigned int);
  
  //This allows us to delete bare CkArrayMessages
  void operator delete(void *p){CkFreeMsg(p);}
};

/* Utility */
#if CMK_LBDB_ON
#include "LBDatabase.h"
class LBDatabase;
#endif

#define IMMEDIATE               CmiTrue
#define NOT_IMMEDIATE		CmiFalse

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

/// How to do a message delivery:
typedef enum {
	CkDeliver_queue=0, //Deliver via the scheduler's queue
	CkDeliver_inline=1, //Deliver via a regular call
	CkDeliver_immediate=2 //Deliver immediate message
} CkDeliver_t;

#include "CkLocation.decl.h"

/************************** Array Messages ****************************/
/**
 *  This is the message type used to actually send a migrating array element.
 */

class CkArrayElementMigrateMessage : public CMessage_CkArrayElementMigrateMessage {
public:
	int length;//Size in bytes of the packed data
	double* packData;
};

/******************* Map object ******************/

extern CkGroupID _RRMapID;
class CkLocMgr;
class CkArrMgr;

/** The "map" is used by the array manager to map an array index to 
 * a home processor number.
 */
class CkArrayMap : public IrrGroup // : public CkGroupReadyCallback
{
public:
  CkArrayMap(void);
  CkArrayMap(CkMigrateMessage *m): IrrGroup(m) {}
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

/**
 * A CkLocRec is our local representation of an array element.
 * The location manager's main hashtable maps array indices to
 * CkLocRec *'s.
 */
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

  /// Return the type of this ArrayRec:
  typedef enum {
    base=0,//Base class (invalid type)
    local,//Array element that lives on this Pe
    remote,//Array element that lives on some other Pe
    buffering,//Array element that was just created
    dead//Deleted element (for debugging)
  } RecType;
  virtual RecType type(void)=0;
  
  /// Accept a message for this element
  virtual CmiBool deliver(CkArrayMessage *m,CkDeliver_t type)=0;
  
  /// This is called when this ArrayRec is about to be replaced.
  /// It is only used to deliver buffered element messages.
  virtual void beenReplaced(void);
  
  /// Return if this rec is now obsolete
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0; 

  /// Return the represented array element; or NULL if there is none
  virtual CkMigratable *lookupElement(CkArrayID aid);

  /// Return the last known processor; or -1 if none
  virtual int lookupProcessor(void);
};

/**
 * Represents a local array element.
 */
class CkLocRec_local : public CkLocRec {
  CkArrayIndexMax idx;/// Element's array index
  int localIdx; /// Local index (into array manager's element lists)
  CmiBool running; /// True when inside a startTiming/stopTiming pair
  CmiBool *deletedMarker; /// Set this if we're deleted during processing
  CkQ<CkArrayMessage *> halfCreated; /// Stores messages for nonexistent siblings of existing elements
public:
  //Creation and Destruction:
  CkLocRec_local(CkLocMgr *mgr,CmiBool fromMigration,
  	const CkArrayIndex &idx_,int localIdx_);
  void migrateMe(int toPe); //Leave this processor
  void destroy(void); //User called destructor
  virtual ~CkLocRec_local();

  /// A new element has been added to this index
  void addedElement(void);

  /**
   *  Accept a message for this element.
   *  Returns false if the element died during the receive.
   */
  virtual CmiBool deliver(CkArrayMessage *m,CkDeliver_t type);

  /** Invoke the given entry method on this element.
   *   Returns false if the element died during the receive.
   *   If doFree is true, the message is freed after send;
   *    if false, the message can be reused.
   */
  CmiBool invokeEntry(CkMigratable *obj,void *msg,int idx,bool doFree);

  virtual RecType type(void);
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx);

#if CMK_LBDB_ON  //For load balancing:
  /// Control the load balancer:
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
  void setMigratable(int migratable);
private:
  LBDatabase *the_lbdb;
  LDObjHandle ldHandle;
#endif
};
class CkLocRec_remote;

/*********************** CkMigratable ******************************/
/** This is the superclass of all migratable parallel objects.
 *  Currently, that's just array elements.
 */
#if CMK_OUT_OF_CORE
#  include "conv-ooc.h"
extern CooPrefetchManager CkArrayElementPrefetcher;
// If this flag is set, this creation/deletion is just 
//   a "fake" constructor/destructor call for prefetching.
CkpvExtern(int,CkSaveRestorePrefetch);
#endif

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
  inline LBDatabase *getLBDB(void) const {return myRec->getLBDB();}
#else
  inline void ckStopTiming(void) { }
  inline void ckStartTiming(void) { }
#endif
  //Initiate a migration to the given processor
  inline void ckMigrate(int toPe) {myRec->migrateMe(toPe);}
  
  /// Called by the system just before and after migration to another processor:  
  virtual void ckAboutToMigrate(void); /*default is empty*/
  virtual void ckJustMigrated(void); /*default is empty*/

  /// Delete this object
  virtual void ckDestroy(void);

  /// Execute the given entry method.  Returns false if the element 
  /// deleted itself or migrated away during execution.
  inline CmiBool ckInvokeEntry(int epIdx,void *msg,bool doFree) 
	  {return myRec->invokeEntry(this,msg,epIdx,doFree);}

protected:
  /// A more verbose form of abort
  virtual void CkAbort(const char *str) const;

  CmiBool usesAtSync;//You must set this in the constructor to use AtSync().
  virtual void ResumeFromSync(void);
  CmiBool barrierRegistered;//True iff barrier handle below is set

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(void);
private: //Load balancer state:
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
  LDBarrierReceiver ldBarrierRecvHandle;//Transient (not migrated)  
  static void staticResumeFromSync(void* data);
public:
  void ckFinishConstruction(void);
  void setMigratable(int migratable);
#else
  void AtSync(void) { ResumeFromSync();}
public:
  void ckFinishConstruction(void) { }
#endif
#if CMK_OUT_OF_CORE
private:
  friend class CkLocMgr;
  friend int CkArrayPrefetch_msg2ObjId(void *msg);
  friend void CkArrayPrefetch_writeToSwap(FILE *swapfile,void *objptr);
  friend void CkArrayPrefetch_readFromSwap(FILE *swapfile,void *objptr);
  int prefetchObjID; //From CooRegisterObject
  CmiBool isInCore; //If true, the object is present in memory
#endif
};

/** 
 * Stores a list of array elements.  These lists are 
 * kept by the array managers. 
 */
class CkMigratableList {
	CkVec< CkZeroPtr<CkMigratable> > el;
 public:
	CkMigratableList();
	~CkMigratableList();
	
	void setSize(int s);
	inline int length(void) const {return el.length();}

	/// Add an element at the given location
	void put(CkMigratable *v,int atIdx);

	/// Return the element at the given location
	inline CkMigratable *get(int localIdx) {return el[localIdx];}

	/**
	 * Return the next non-empty element starting from the given index,
	 * or NULL if there is none.  Updates from to point past the returned index.
	*/
	CkMigratable *next(int &from) {
		while (from<length()) {
			CkMigratable *ret=el[from];
			from++;
			if (ret!=NULL) return ret;
		}
		return NULL;
	}

	/// Remove the element at the given location
	inline void empty(int localIdx) {el[localIdx]=NULL;}
};

/**
 *A typed version of the above.
 */
template <class T>
class CkMigratableListT : public CkMigratableList {
	typedef CkMigratableList super;
public:
	inline void put(T *v,int atIdx) {super::put((void *)v,atIdx);}
	inline T *get(int localIdx) {return (T *)super::get(localIdx);}
	inline T *next(int &from) {return (T *)super::next(from);}
};


/********************** CkLocMgr ********************/
/// A tiny class for detecting heap corruption
class CkMagicNumber_impl {
 protected:
	int magic;
	void badMagicNumber(int expected,const char *file,int line,void *obj) const;
	CkMagicNumber_impl(int m);
};
template<class T>
class CkMagicNumber : public CkMagicNumber_impl {
	enum {good=sizeof(T)^0x7EDC0000};
 public:
	CkMagicNumber(void) :CkMagicNumber_impl(good) {}
	inline void check(const char *file,int line,void *obj) const {
		if (magic!=good) badMagicNumber(good,file,line,obj);
	}
#ifndef CMK_OPTIMIZE
#   define CK_MAGICNUMBER_CHECK magic.check(__FILE__,__LINE__,this);
#else
#   define CK_MAGICNUMBER_CHECK /*empty, for speed*/
#endif
};

/**
 * The "data" class passed to a CkLocIterator, which refers to a bound
 * glob of array elements.
 * This is a transient class-- do not attempt to store it or send 
 * it across processors.
 */
class CkLocation {
	CkLocMgr *mgr;
	CkLocRec_local *rec;
public:
	CkLocation(CkLocMgr *mgr_, CkLocRec_local *rec_);
	
	/// Find our location manager
	inline CkLocMgr *getManager(void) const {return mgr;}
	
	/// Look up and return the array index of this location.
	const CkArrayIndex &getIndex(void) const;
	
	/// Pup all the array elements at this location.
	void pup(PUP::er &p);
};

/**
 * This interface describes the destination for an iterator over
 * the locations in an array.
 */
class CkLocIterator {
public:
	virtual ~CkLocIterator();
	
	/// This location is part of the calling location manager.
	virtual void addLocation(CkLocation &loc) =0;
};

enum CkElementCreation_t {
  CkElementCreation_migrate=2, // Create object for normal migration arrival
  CkElementCreation_resume=3, // Create object after checkpoint
};
/// Abstract superclass of all array manager objects 
class CkArrMgr {
public:
	/// Insert this initial element on this processor
	virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg)=0;
	
	/// Done with initial insertions
	virtual void doneInserting(void)=0;
	
	/// Create an uninitialized element after migration
	///  The element's constructor will be called immediately after.
	virtual CkMigratable *allocateMigrated(int elChareType,
		const CkArrayIndex &idx,CkElementCreation_t type) =0;

	/// Demand-create an element at this index on this processor
	///  Returns true if the element was successfully added;
	///  false if the element migrated away or deleted itself.
	virtual CmiBool demandCreateElement(const CkArrayIndex &idx,
		int onPe,int ctor,CkDeliver_t type) =0;
};

/**
 * A group which manages the location of an indexed set of
 * migratable objects.  Knows about insertions, deletions,
 * home processors, migration, and message forwarding.
 */
class CkLocMgr : public IrrGroup {
	CkMagicNumber<CkMigratable> magic; //To detect heap corruption
public:
	CkLocMgr(CkGroupID map,CkGroupID lbdb,int numInitial);
	CkLocMgr(CkMigrateMessage *m);
	inline bool isLocMgr(void) { return true; }
	CkGroupID &getGroupID(void) {return thisgroup;}
	inline CProxy_CkLocMgr &getProxy(void)
		{return thisProxy;}
	inline CProxyElement_CkLocMgr &getLocalProxy(void)
		{return thislocalproxy;}

//Interface used by array manager and proxies
	/// Add a new local array manager to our list.  Array managers
	///  must be registered in the same order on all processors.
	/// Returns a list which will contain that array's local elements
	CkMigratableList *addManager(CkArrayID aid,CkArrMgr *mgr);

	/// Populate this array with initial elements
	void populateInitial(int numElements,void *initMsg,CkArrMgr *mgr)
		{map->populateInitial(mapHandle,numElements,initMsg,mgr);}

	/// Add a new local array element, calling element's constructor
	///  Returns true if the element was successfully added;
	///  false if the element migrated away or deleted itself.
	CmiBool addElement(CkArrayID aid,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);

	///Deliver message to this element:
	inline void deliverViaQueue(CkMessage *m) {deliver(m,CkDeliver_queue);}
	inline void deliverInline(CkMessage *m) {deliver(m,CkDeliver_inline);}
	inline void deliverImmediate(CkMessage *m) {deliver(m,CkDeliver_immediate);}
	void deliver(CkMessage *m, CkDeliver_t type);

	///Done inserting elements for now
	void doneInserting(void);

//Advisories:
	///This index now lives on the given processor-- update local records
	void inform(const CkArrayIndex &idx,int nowOnPe);

	///This index now lives on the given processor-- tell the home processor
	void informHome(const CkArrayIndex &idx,int nowOnPe);

	///This message took several hops to reach us-- fix it
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

	CmiBool demandCreateElement(CkArrayMessage *msg,int onPe,CkDeliver_t type);

//Interface used by external users:
	/// Home mapping
	inline int homePe(const CkArrayIndex &idx) const
		{return map->procNum(mapHandle,idx);}
	inline CmiBool isHome(const CkArrayIndex &idx) const
		{return (CmiBool)(homePe(idx)==CkMyPe());}

	/// Look up the object with this array index, or return NULL
	CkMigratable *lookup(const CkArrayIndex &idx,CkArrayID aid);

	/// Return the "last-known" location (returns a processor number)
	int lastKnown(const CkArrayIndex &idx) const;

	/// Pass each of our locations (each separate array index) to this destination.
	void iterate(CkLocIterator &dest);

	/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
	void resume(const CkArrayIndex &idx, PUP::er &p);

//Communication:
	void migrateIncoming(CkArrayElementMigrateMessage *msg);
	void updateLocation(const CkArrayIndexMax &idx,int nowOnPe);
	void reclaimRemote(const CkArrayIndexMax &idx,int deletedOnPe);
	void dummyAtSync(void);

	void pup(PUP::er &p);
	
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
	//Remove this entry from the table (does not delete record)
	void removeFromTable(const CkArrayIndex &idx);

	friend class CkLocation; //so it can call pupElementsFor
	void pupElementsFor(PUP::er &p,CkLocRec_local *rec,
		CkElementCreation_t type);

	/// Call this member function on each element of this location:
	typedef void (CkMigratable::* CkMigratable_voidfn_t)(void);
	void callMethod(CkLocRec_local *rec,CkMigratable_voidfn_t fn);

	CmiBool deliverUnknown(CkArrayMessage *msg,CkDeliver_t type);

	/// Create a new local record at this array index.
	CkLocRec_local *createLocal(const CkArrayIndex &idx, CmiBool forMigration,
		CmiBool notifyHome);

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

	CmiBool addElementToRec(CkLocRec_local *rec,ManagerRec *m,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);

	//For keeping track of free local indices
	CkVec<int> freeList;//Linked list of free local indices
	int firstFree;//First free local index
	int localLen;//Last allocated local index plus one
	int nextFree(void);

	CProxy_CkLocMgr thisProxy;
	CProxyElement_CkLocMgr thislocalproxy;
	/// The core of the location manager: map array index to element representative
	CkHashtableT<CkArrayIndexMax,CkLocRec *> hash;

	/// This flag is set while we delete an old copy of a migrator
	CmiBool duringMigration;

	//Occasionally clear out stale remote pointers
	static void staticSpringCleaning(void *mgr);
	void springCleaning(void);
	int nSprings;

	//Map object
	CkGroupID mapID;
	int mapHandle;
	CkArrayMap *map;

	CkGroupID lbdbID;
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
