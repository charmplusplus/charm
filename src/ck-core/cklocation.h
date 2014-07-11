/*
Location manager: keeps track of an indexed set of migratable
objects.  Used by the array manager to locate array elements,
interact with the load balancer, and perform migrations.

Does not handle reductions (see ckreduction.h), broadcasts,
array proxies, or the details of element creation (see ckarray.h).
*/
#ifndef __CKLOCATION_H
#define __CKLOCATION_H

#include <map>

/*********************** Array Messages ************************/
class CkArrayMessage : public CkMessage {
public:
  //These routines are implementation utilities
  CmiUInt8 array_element_id(void);
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
#include "MetaBalancer.h"
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

class CkArrayOptions;
#include "CkLocation.decl.h"

/************************** Array Messages ****************************/
/**
 *  This is the message type used to actually send a migrating array element.
 */

class CkArrayElementMigrateMessage : public CMessage_CkArrayElementMigrateMessage {
public:
  CkArrayElementMigrateMessage(CkArrayIndex idx_, CmiUInt8 id_, bool ignoreArrival_, int length_,
                               int nManagers_, bool bounced_)
    : idx(idx_), id(id_), ignoreArrival(ignoreArrival_), length(length_), nManagers(nManagers_), bounced(bounced_)
  { }

	CkArrayIndex idx; // Array index that is migrating
        CmiUInt8 id; // ID of the elements with this index in this collection
	bool ignoreArrival;   // if to inform LB of arrival
	int length;//Size in bytes of the packed data
	int nManagers; // Number of associated array managers
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CkGroupID gid; //gid of location manager
#endif
	bool bounced; // Fault evac related?
	char* packData;
};

/******************* Map object ******************/

extern CkGroupID _defaultArrayMapID;
extern CkGroupID _fastArrayMapID;
class CkLocMgr;
class CkArray;

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
  virtual int registerArray(const CkArrayIndex& numElements, CkArrayID aid);
  virtual void unregisterArray(int idx);
  virtual void populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element) =0;
  virtual int homePe(int arrayHdl,const CkArrayIndex &element)
             { return procNum(arrayHdl, element); }
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

#include "cklocrec.h"

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

#include "ckmigratable.h"



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
	CkLocRec *rec;
public:
	CkLocation(CkLocMgr *mgr_, CkLocRec *rec_);
	
	/// Find our location manager
	inline CkLocMgr *getManager(void) const {return mgr;}
	
	/// Find the local record that refers to this element
	inline CkLocRec *getLocalRecord(void) const {return rec;}
	
	/// Look up and return the array index of this location.
	const CkArrayIndex &getIndex(void) const;
	CmiUInt8 getID() const;

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
	CkLocMgr(CkArrayOptions opts);
	CkLocMgr(CkMigrateMessage *m);
        ~CkLocMgr();

	inline bool isLocMgr(void) { return true; }
	CkGroupID &getGroupID(void) {return thisgroup;}
	inline CProxy_CkLocMgr &getProxy(void) {return thisProxy;}
	inline CProxyElement_CkLocMgr &getLocalProxy(void) {return thislocalproxy;}

//Interface used by external users:
	/// Home mapping
	inline int homePe(const CkArrayIndex &idx) const {return map->homePe(mapHandle,idx);}
        inline int homePe(const CmiUInt8 id) const {
          if (compressor)
            return homePe(compressor->decompress(id));

          return id >> 24;
        }
	inline int procNum(const CkArrayIndex &idx) const {return map->procNum(mapHandle,idx);}
	inline bool isHome (const CkArrayIndex &idx) const {return (bool)(homePe(idx)==CkMyPe());}
  int whichPE(const CkArrayIndex &idx) const;
  int whichPE(const CmiUInt8 id) const;
	/// Return the "last-known" location (returns a processor number)
	int lastKnown(const CkArrayIndex &idx);
	int lastKnown(CmiUInt8 id);

        inline CmiUInt8 lookupID(const CkArrayIndex &idx) const {
          if (compressor) {
            return compressor->compress(idx);
          } else {
            CkLocMgr::IdxIdMap::const_iterator itr = idx2id.find(idx);
            CkAssert(itr != idx2id.end());
            return itr->second;
          }
        }

        inline bool lookupID(const CkArrayIndex &idx, CmiUInt8& id) const {
          if (compressor) {
            id = compressor->compress(idx);
            return true;
          } else {
            CkLocMgr::IdxIdMap::const_iterator itr = idx2id.find(idx);
            if (itr == idx2id.end()) {
              return false;
            } else {
              id = itr->second;
              return true;
            }
          }
        }

	//Look up array element in hash table.  Index out-of-bounds if not found.
	CkLocRec *elementRec(const CkArrayIndex &idx);
	//Look up array element in hash table.  Return NULL if not there.
	CkLocRec *elementNrec(const CmiUInt8 id);

	/// Return true if this array element lives on another processor
	bool isRemote(const CkArrayIndex &idx,int *onPe) const;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	//mark the duringMigration variable .. used for parallel restart
	void setDuringMigration(bool _duringMigration);
#endif

	void setDuringDestruction(bool _duringDestruction);

	/// Pass each of our locations (each separate array index) to this destination.
	void iterate(CkLocIterator &dest);

	/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup), skip listeners
	void restore(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p);
	/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void resume(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p, bool create, int dummy=0);
#else
	void resume(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p, bool notify=true,bool=false);
#endif

//Interface used by array manager and proxies
	/// Add a new local array manager to our list.
	void addManager(CkArrayID aid,CkArray *mgr);
        void deleteManager(CkArrayID aid, CkArray *mgr);

	/// Populate this array with initial elements
	void populateInitial(CkArrayOptions& options,void *initMsg,CkArray *mgr)
    { map->populateInitial(mapHandle,options,initMsg,mgr); }

	/// Add a new local array element, calling element's constructor
	///  Returns true if the element was successfully added; false if the element migrated away or deleted itself.
	bool addElement(CkArrayID aid,const CkArrayIndex &idx, CkMigratable *elt,int ctorIdx,void *ctorMsg);

	///Done inserting elements for now
	void doneInserting(void);
	void startInserting(void);

	// How many elements of each associated array are local to this PE?
	// If this returns n, and there are k associated arrays, that
	// means k*n elements are living here
	unsigned int numLocalElements();

	///Deliver message to this element:
	//int deliverMsg(CkMessage *m, CkArrayID mgr, const CkArrayIndex &idx, CkDeliver_t type, int opts = 0);
	int deliverMsg(CkArrayMessage *m, CkArrayID mgr, CmiUInt8 id, const CkArrayIndex* idx, CkDeliver_t type, int opts = 0);

        void sendMsg(CkArrayMessage *msg, CkArrayID mgr, const CkArrayIndex &idx, CkDeliver_t type, int opts);

//Advisories:
	///This index now lives on the given processor-- update local records
	void inform(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe);
	void inform(CmiUInt8 id, int nowOnPe);

	///This index now lives on the given processor-- tell the home processor
	void informHome(const CkArrayIndex &idx,int nowOnPe);

	///This message took several hops to reach us-- fix it
	void multiHop(CkArrayMessage *m);

//Interface used by CkLocRec

	//Migrate us to another processor
	void emigrate(CkLocRec *rec,int toPe);
    void informLBPeriod(CkLocRec *rec, int lb_ideal_period);
    void metaLBCallLB(CkLocRec *rec);

#if CMK_LBDB_ON
	LBDatabase *getLBDB(void) const { return the_lbdb; }
    MetaBalancer *getMetaBalancer(void) const { return the_metalb;}
	const LDOMHandle &getOMHandle(void) const { return myLBHandle; }
#endif

	//This index will no longer be used-- delete the associated elements
	void reclaim(const CkArrayIndex &idx);

	bool demandCreateElement(CkArrayMessage *msg, const CkArrayIndex &idx, int onPe, CkDeliver_t type);
        void demandCreateElement(const CkArrayIndex &idx, int chareType, int onPe, CkArrayID mgr);

//Communication:
	void immigrate(CkArrayElementMigrateMessage *msg);
        void requestLocation(const CkArrayIndex &idx, int peToTell, bool suppressIfHere, int ifNonExistent, int chareType, CkArrayID mgr);
        void requestLocation(CmiUInt8 id, int peToTell, bool suppressIfHere);
        void updateLocation(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe);
        void updateLocation(CmiUInt8 id, int nowOnPe);
	void reclaimRemote(const CkArrayIndex &idx,int deletedOnPe);
	void dummyAtSync(void);

	/// return a list of migratables in this local record
	void migratableList(CkLocRec *rec, CkVec<CkMigratable *> &list);

	void flushAllRecs(void);
	void flushLocalRecs(void);
	void pup(PUP::er &p);
	

private:
//Internal interface:
	//Add given element array record at idx, replacing the existing record
	void insertRec(CkLocRec *rec,const CmiUInt8 &id);

	//Remove this entry from the table (does not delete record)
	void removeFromTable(const CmiUInt8 id);

	friend class CkLocation; //so it can call pupElementsFor
	friend class ArrayElement;
	friend class MemElementPacker;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void pupElementsFor(PUP::er &p,CkLocRec *rec,
        CkElementCreation_t type, bool create=true, int dummy=0);
#else
	void pupElementsFor(PUP::er &p,CkLocRec *rec,
		CkElementCreation_t type,bool rebuild = false);
#endif

	/// Call this member function on each element of this location:
	typedef void (CkMigratable::* CkMigratable_voidfn_t)(void);

	typedef void (CkMigratable::* CkMigratable_voidfn_arg_t)(void*);
	void callMethod(CkLocRec *rec,CkMigratable_voidfn_arg_t fn, void*);

	void deliverUnknown(CkArrayMessage *msg, const CkArrayIndex* idx, CkDeliver_t type, int opts);
	typedef std::map<CmiUInt8, std::vector<CkArrayMessage*> > MsgBuffer;
	/// Deliver any buffered msgs to a newly created array element
	void deliverAnyBufferedMsgs(CmiUInt8, MsgBuffer &buffer);

	/// Create a new local record at this array index.
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
CkLocRec *createLocal(const CkArrayIndex &idx,
        bool forMigration, bool ignoreArrival,
        bool notifyHome,int dummy=0);
#else
	CkLocRec *createLocal(const CkArrayIndex &idx, 
		bool forMigration, bool ignoreArrival,
		bool notifyHome);
#endif

        std::map<CkArrayIndex, std::vector<std::pair<int, bool> > > bufferedLocationRequests;

public:
	void callMethod(CkLocRec *rec,CkMigratable_voidfn_t fn);

//Data Members:
	//Map array ID to manager and elements
    std::map<CkArrayID, CkArray*> managers;
    // Map object ID to location
    std::map<CmiUInt8, int> id2pe;

    typedef std::map<CkArrayIndex, CmiUInt8> IdxIdMap;

    // Map array element index to object ID
    IdxIdMap idx2id;
    // Next ID to assign newly constructed array elements
    CmiUInt8 idCounter;
    CmiUInt8 getNewObjectID(const CkArrayIndex &idx);

    /// Map idx to undelivered msgs
    /// @todo: We should not buffer msgs for uncreated array elements forever.
    /// After some timeout or other policy, we should throw errors or warnings
    /// or at least report and discard any msgs addressed to uncreated array elements
    MsgBuffer bufferedMsgs;
    MsgBuffer bufferedRemoteMsgs;
    MsgBuffer bufferedShadowElemMsgs;

    std::map<CkArrayIndex, std::vector<CkArrayMessage *> > bufferedIndexMsgs;

	bool addElementToRec(CkLocRec *rec,CkArray *m,
		CkMigratable *elt,int ctorIdx,void *ctorMsg);

	CProxy_CkLocMgr thisProxy;
	CProxyElement_CkLocMgr thislocalproxy;

	/// This flag is set while we delete an old copy of a migrator
	bool duringMigration;
	/// This flag is set while we are deleting location manager
	bool duringDestruction;

private:
	/// The core of the location manager: map array index to element representative
	CkHashtableT<CkHashtableAdaptorT<CmiUInt8>, CkLocRec *> hash;
	CmiImmediateLockType hashImmLock;

	//Map object
	CkGroupID mapID;
	int mapHandle;
	CkArrayMap *map;

	CkGroupID lbdbID;
	CkGroupID metalbID;

	ck::ArrayIndexCompressor *compressor;
	CkArrayIndex bounds;
	void checkInBounds(const CkArrayIndex &idx);

#if CMK_LBDB_ON
	LBDatabase *the_lbdb;
  MetaBalancer *the_metalb;
	LDBarrierClient dummyBarrierHandle;
	static void staticDummyResumeFromSync(void* data);
	void dummyResumeFromSync(void);
	static void staticRecvAtSync(void* data);
	void recvAtSync(void);
	LDOMHandle myLBHandle;
        LDBarrierReceiver lbBarrierReceiver;
#endif
private:
	void initLB(CkGroupID lbdbID, CkGroupID metalbID);

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
