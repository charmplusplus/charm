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
  CkArrayIndex &array_index(void);
  unsigned short &array_ep(void);
  unsigned short &array_ep_bcast(void);
  unsigned char &array_hops(void);
  unsigned int array_getSrcPe(void);
  unsigned int array_ifNotThere(void);
  void array_setIfNotThere(unsigned int);
  
  //This allows us to delete bare CkArrayMessages
  void operator delete(void *p){CkFreeMsg(p);}
};

/* Utility */
//#if CMK_LBDB_ON
#include "LBDatabase.h"
class LBDatabase;
//#endif

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
	CkDeliver_inline=1  //Deliver via a regular call
} CkDeliver_t;

#include "CkLocation.decl.h"

/************************** Array Messages ****************************/
/**
 *  This is the message type used to actually send a migrating array element.
 */

class CkArrayElementMigrateMessage : public CMessage_CkArrayElementMigrateMessage {
public:
	CkArrayIndex idx; // Array index that is migrating
	int ignoreArrival;   // if to inform LB of arrival
	int length;//Size in bytes of the packed data
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CkGroupID gid; //gid of location manager
#endif
	CmiBool bounced;
	double* packData;
};

/******************* Map object ******************/

extern CkGroupID _defaultArrayMapID;
extern CkGroupID _fastArrayMapID;
class CkLocMgr;
class CkArrMgr;

/**
\addtogroup CkArray
*/
/*@{*/

/** The "map" is used by the array manager to map an array index to 
 * a home processor number.
 */
class CkArrayMap : public IrrGroup // : public CkGroupReadyCallback
{
public:
  CkArrayMap(void);
  CkArrayMap(CkMigrateMessage *m): IrrGroup(m) {}
  virtual ~CkArrayMap();
  virtual int registerArray(CkArrayIndex& numElements,CkArrayID aid);
  virtual void populateInitial(int arrayHdl,CkArrayIndex& numElements,void *ctorMsg,CkArrMgr *mgr);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element) =0;
  virtual int homePe(int arrayHdl,const CkArrayIndex &element)
             { return procNum(arrayHdl, element); }
//  virtual void pup(PUP::er &p) { CkGroupReadyCallback::pup(p); }
};
/*@}*/

/**
\addtogroup CkArrayImpl
\brief Migratable Chare Arrays: Implementation classes.
*/
/*@{*/
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
  //int lastAccess;//Age when last accessed. Removed since unused and colliding with a inheriting class, Filippo
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
  virtual CmiBool deliver(CkArrayMessage *m,CkDeliver_t type,int opts=0)=0;
  
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
  CkArrayIndex idx;/// Element's array index
  int localIdx; /// Local index (into array manager's element lists)
  CmiBool running; /// True when inside a startTiming/stopTiming pair
  CmiBool *deletedMarker; /// Set this if we're deleted during processing
  CkQ<CkArrayMessage *> halfCreated; /// Stores messages for nonexistent siblings of existing elements
public:
  //Creation and Destruction:
  CkLocRec_local(CkLocMgr *mgr,CmiBool fromMigration,CmiBool ignoreArrival,
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
  virtual CmiBool deliver(CkArrayMessage *m,CkDeliver_t type,int opts=0);

  /** Invoke the given entry method on this element.
   *   Returns false if the element died during the receive.
   *   If doFree is true, the message is freed after send;
   *    if false, the message can be reused.
   */
  CmiBool invokeEntry(CkMigratable *obj,void *msg,int idx,CmiBool doFree);

  virtual RecType type(void);
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx);

#if CMK_LBDB_ON  //For load balancing:
  /// Control the load balancer:
  void startTiming(int ignore_running=0);
  void stopTiming(int ignore_running=0);
  void setObjTime(double cputime);
  double getObjTime();
#else
  inline void startTiming(int ignore_running=0) {  }
  inline void stopTiming(int ignore_running=0) { }
#endif
  inline int getLocalIndex(void) const {return localIdx;}
  inline const CkArrayIndex &getIndex(void) const {return idx;}
  virtual CkMigratable *lookupElement(CkArrayID aid);
  virtual int lookupProcessor(void);

#if CMK_LBDB_ON
public:
  inline LBDatabase *getLBDB(void) const {return the_lbdb;}
  inline LDObjHandle getLdHandle() const{return ldHandle;}
  static void staticMigrate(LDObjHandle h, int dest);
  void recvMigrate(int dest);
  void setMigratable(int migratable);	/// set migratable
  void AsyncMigrate(CmiBool use);
  CmiBool isAsyncMigrate()   { return asyncMigrate; }
  void ReadyMigrate(CmiBool ready) { readyMigrate = ready; } ///called from user
  int  isReadyMigrate()	{ return readyMigrate; }
  CmiBool checkBufferedMigration();	// check and execute pending migration
  int   MigrateToPe();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        void Migrated();
#endif
  inline void setMeasure(CmiBool status) { enable_measure = status; }
private:
  LBDatabase *the_lbdb;
  LDObjHandle ldHandle;
  CmiBool  asyncMigrate;  /// if readyMove is inited
  CmiBool  readyMigrate;    /// status whether it is ready to migrate
  CmiBool  enable_measure;
  int  nextPe;              /// next migration dest processor
#else
  void AsyncMigrate(CmiBool use){};
#endif
/**FAULT_EVAC*/
private:
	CmiBool asyncEvacuate; //can the element be evacuated anytime, false for tcharm 
	CmiBool bounced; //did this element try to immigrate into a processor which was evacuating
											// and was bounced away to some other processor. This is assumed to happen
											//only if this object was migrated by a load balancer, but the processor
											// started crashing soon after
public:	
	CmiBool isAsyncEvacuate(){return asyncEvacuate;}
	void AsyncEvacuate(CmiBool set){asyncEvacuate = set;}
	CmiBool isBounced(){return bounced;}
	void Bounced(CmiBool set){bounced = set;}
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
protected:
  CkLocRec_local *myRec;
private:
  int thisChareType;//My chare type
  void commonInit(void);
  CmiBool asyncEvacuate;
public:
  CkArrayIndex thisIndexMax;

  CkMigratable(void);
  CkMigratable(CkMigrateMessage *m);
  virtual ~CkMigratable();
  virtual void pup(PUP::er &p);
  virtual void CkAddThreadListeners(CthThread tid, void *msg);

  virtual int ckGetChareType(void) const;// {return thisChareType;}
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

  /// for inline call
  LDObjHandle timingBeforeCall(int *objstopped);
  void timingAfterCall(LDObjHandle objHandle,int *objstopped);

  //Initiate a migration to the given processor
  inline void ckMigrate(int toPe) {myRec->migrateMe(toPe);}
  
  /// Called by the system just before and after migration to another processor:  
  virtual void ckAboutToMigrate(void); /*default is empty*/
  virtual void ckJustMigrated(void); /*default is empty*/

  //used for out-of-core emulation
  virtual void ckJustRestored(void); /*default is empty*/

  /// Delete this object
  virtual void ckDestroy(void);

  /// Execute the given entry method.  Returns false if the element 
  /// deleted itself or migrated away during execution.
  inline CmiBool ckInvokeEntry(int epIdx,void *msg,CmiBool doFree) 
	  {return myRec->invokeEntry(this,msg,epIdx,doFree);}

protected:
  /// A more verbose form of abort
  virtual void CkAbort(const char *str) const;

  CmiBool usesAtSync;//You must set this in the constructor to use AtSync().
  CmiBool usesAutoMeasure; //You must set this to use auto lb instrumentation.
  CmiBool barrierRegistered;//True iff barrier handle below is set

public:
  virtual void ResumeFromSync(void);
  virtual void UserSetLBLoad(void);  /// user define this when setLBLoad is true
  void setObjTime(double cputime);
  double getObjTime();

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(int waitForMigration=1);
  int MigrateToPe()  { return myRec->MigrateToPe(); }

private: //Load balancer state:
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
  LDBarrierReceiver ldBarrierRecvHandle;//Transient (not migrated)  
  static void staticResumeFromSync(void* data);
public:
  void ReadyMigrate(CmiBool ready);
  void ckFinishConstruction(void);
  void setMigratable(int migratable);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  void setMigratable(int migratable)  { }
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

  // FAULT_EVAC
  void AsyncEvacuate(CmiBool set){myRec->AsyncEvacuate(set);asyncEvacuate = set;};
public:
  CmiBool isAsyncEvacuate(){return asyncEvacuate;};
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
	inline void put(T *v,int atIdx) {super::put((CkMigratable *)v,atIdx);}
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
#if CMK_ERROR_CHECKING
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
	
	/// Find the local record that refers to this element
	inline CkLocRec_local *getLocalRecord(void) const {return rec;}
	
	/// Look up and return the array index of this location.
	const CkArrayIndex &getIndex(void) const;
	
        void destroyAll();

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
  CkElementCreation_resume=3,  // Create object after checkpoint
  CkElementCreation_restore=4  // Create object after checkpoint, skip listeners
};
/// Abstract superclass of all array manager objects 
class CkArrMgr {
public:
	virtual ~CkArrMgr() {}
	/// Insert this initial element on this processor
	virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg, int local=1)=0;
	
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


#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
typedef void (*CkLocFn)(CkArray *,void *,CkLocRec *,CkArrayIndex *);
#endif


/**
 * A group which manages the location of an indexed set of
 * migratable objects.  Knows about insertions, deletions,
 * home processors, migration, and message forwarding.
 */
class CkLocMgr : public IrrGroup {
	CkMagicNumber<CkMigratable> magic; //To detect heap corruption
public:
	CkLocMgr(CkGroupID map,CkGroupID _lbdb,CkArrayIndex& numInitial);
	CkLocMgr(CkMigrateMessage *m);
	inline CmiBool isLocMgr(void) { return CmiTrue; }
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
	void populateInitial(CkArrayIndex& numElements,void *initMsg,CkArrMgr *mgr)
		{map->populateInitial(mapHandle,numElements,initMsg,mgr);}

	/// Add a new local array element, calling element's constructor
	///  Returns true if the element was successfully added;
	///  false if the element migrated away or deleted itself.
	CmiBool addElement(CkArrayID aid,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);

	///Deliver message to this element:
	inline void deliverViaQueue(CkMessage *m) {deliver(m,CkDeliver_queue);}
	inline void deliverInline(CkMessage *m) {deliver(m,CkDeliver_inline);}
	int deliver(CkMessage *m, CkDeliver_t type, int opts=0);

	///Done inserting elements for now
	void doneInserting(void);
	void startInserting(void);

	// How many elements of each associated array are local to this PE?
	// If this returns n, and there are k associated arrays, that
	// means k*n elements are living here
	unsigned int numLocalElements();

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
#if CMK_ERROR_CHECKING
		if (managers.find(arrayID)->mgr==NULL)
			CkAbort("CkLocMgr::lookupLocal called for unknown array!\n");
#endif
		return managers.find(arrayID)->elts.get(localIdx);
	}

	//Migrate us to another processor
	void emigrate(CkLocRec_local *rec,int toPe);

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
		{return map->homePe(mapHandle,idx);}
	inline int procNum(const CkArrayIndex &idx) const
		{return map->procNum(mapHandle,idx);}
	inline CmiBool isHome(const CkArrayIndex &idx) const
		{return (CmiBool)(homePe(idx)==CkMyPe());}

	/// Look up the object with this array index, or return NULL
	CkMigratable *lookup(const CkArrayIndex &idx,CkArrayID aid);

	/// Return the "last-known" location (returns a processor number)
	int lastKnown(const CkArrayIndex &idx);

	/// Return true if this array element lives on another processor
	bool isRemote(const CkArrayIndex &idx,int *onPe) const;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	//mark the duringMigration variable .. used for parallel restart
	void setDuringMigration(CmiBool _duringMigration);
#endif

	/// Pass each of our locations (each separate array index) to this destination.
	void iterate(CkLocIterator &dest);

	/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup), skip listeners
	void restore(const CkArrayIndex &idx, PUP::er &p);
	/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void resume(const CkArrayIndex &idx, PUP::er &p, CmiBool create, int dummy=0);
#else
	void resume(const CkArrayIndex &idx, PUP::er &p, CmiBool notify=CmiTrue);
#endif

//Communication:
	void immigrate(CkArrayElementMigrateMessage *msg);
	void updateLocation(const CkArrayIndex &idx,int nowOnPe);
	void reclaimRemote(const CkArrayIndex &idx,int deletedOnPe);
	void dummyAtSync(void);

	/// return a list of migratables in this local record
	void migratableList(CkLocRec_local *rec, CkVec<CkMigratable *> &list);

	void flushAllRecs(void);
	void pup(PUP::er &p);
	
	//Look up array element in hash table.  Index out-of-bounds if not found.
	CkLocRec *elementRec(const CkArrayIndex &idx);
	//Look up array element in hash table.  Return NULL if not there.
	CkLocRec *elementNrec(const CkArrayIndex &idx);


private:
//Internal interface:
	//Add given element array record at idx, replacing the existing record
	void insertRec(CkLocRec *rec,const CkArrayIndex &idx);
	//Add given record, when there is guarenteed to be no prior record
	void insertRecN(CkLocRec *rec,const CkArrayIndex &idx);
	//Insert a remote record at the given index
	CkLocRec_remote *insertRemote(const CkArrayIndex &idx,int nowOnPe);

	//Remove this entry from the table (does not delete record)
	void removeFromTable(const CkArrayIndex &idx);

	friend class CkLocation; //so it can call pupElementsFor
	friend class ArrayElement;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void pupElementsFor(PUP::er &p,CkLocRec_local *rec,
        CkElementCreation_t type, CmiBool create=CmiTrue, int dummy=0);
#else
	void pupElementsFor(PUP::er &p,CkLocRec_local *rec,
		CkElementCreation_t type);
#endif

	/// Call this member function on each element of this location:
	typedef void (CkMigratable::* CkMigratable_voidfn_t)(void);

	CmiBool deliverUnknown(CkArrayMessage *msg,CkDeliver_t type,int opts);

	/// Create a new local record at this array index.
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
CkLocRec_local *createLocal(const CkArrayIndex &idx,
        CmiBool forMigration, CmiBool ignoreArrival,
        CmiBool notifyHome,int dummy=0);
#else
	CkLocRec_local *createLocal(const CkArrayIndex &idx, 
		CmiBool forMigration, CmiBool ignoreArrival,
		CmiBool notifyHome);
#endif

public:
	void callMethod(CkLocRec_local *rec,CkMigratable_voidfn_t fn);

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
	GroupIdxArray<ManagerRec *> managers;
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
	CkHashtableT<CkArrayIndex,CkLocRec *> hash;
	CmiImmediateLockType hashImmLock;

	/// This flag is set while we delete an old copy of a migrator
	CmiBool duringMigration;

	//Occasionally clear out stale remote pointers
	static void staticSpringCleaning(void *mgr,double curWallTime);
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

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
public:
	void callForAllRecords(CkLocFn,CkArray *,void *);
	int homeElementCount;
#endif

};





/// check the command line arguments to determine if we can use ConfigurableRRMap
bool haveConfigurableRRMap();




/*@}*/

#endif /*def(thisHeader)*/
