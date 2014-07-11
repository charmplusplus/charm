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
#ifndef __CKARRAY_H
#define __CKARRAY_H

#include "cklocation.h"
#include "ckmemcheckpoint.h" // for CkArrayCheckPTReqMessage
#include "ckarrayindex.h"

/***********************************************************
	Utility defines, includes, etc.
*/
extern void _registerCkArray(void);
CpvExtern (int ,serializer);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
#define _MLOG_BCAST_TREE_ 1
#define _MLOG_BCAST_BFACTOR_ 8
#endif

/** This flag is true when in the system there is anytime migration, false when
 *  the user code guarantees that no migration happens except during load balancing
 *  (in which case it can only happen between AtSync and ResumeFromSync). */
extern bool _isAnytimeMigration;

/**
  Array elements are only inserted at construction
 */
extern bool _isStaticInsertion;

/** This flag is true when users are sure there is at least one charm array element
 *  per processor. In such case, when doing reduction on the array, the children
 *  don't need to be notified that reduction starts
 */
extern bool _isNotifyChildInRed;

/**
\addtogroup CkArray
\brief Migratable Chare Arrays: user-visible classes.

All these classes are defined in ckarray.C.
*/
/*@{*/

/********************* CkArrayListener ****************/
///An arrayListener is an object that gets informed whenever
/// an array element is created, migrated, or destroyed.
///This abstract superclass just ignores everything sent to it.
class ArrayElement;
class CkArrayListener : public PUP::able {
  int nInts; //Number of ints of data to store per element
  int dataOffset; //Int offset of our data within the element
 public:
  CkArrayListener(int nInts_);
  CkArrayListener(CkMigrateMessage *m);
  virtual void pup(PUP::er &p);
  PUPable_abstract(CkArrayListener)

  ///Register this array type.  Our data is stored in the element at dataOffset
  virtual void ckRegister(CkArray *arrMgr,int dataOffset_);

  ///Return the number of ints of data to store per element
  inline int ckGetLen(void) const {return nInts;}
  ///Return the offset of our data into the element
  inline int ckGetOffset(void) const {return dataOffset;}
  ///Return our data associated with this array element
  inline int *ckGetData(ArrayElement *el) const;

  ///Elements may be being created
  virtual void ckBeginInserting(void) {}
  ///No more elements will be created (for now)
  virtual void ckEndInserting(void) {}

//The stamp/creating/created/died sequence happens, in order, exactly
// once per array element.  Migrations don't show up here.
  ///Element creation message is about to be sent
  virtual void ckElementStamp(int *eltInfo) { (void)eltInfo; }
  ///Element is about to be created on this processor
  virtual void ckElementCreating(ArrayElement *elt) { (void)elt; }
  ///Element was just created on this processor
  /// Return false if the element migrated away or deleted itself.
  virtual bool ckElementCreated(ArrayElement *elt) {
    (void)elt;
    return true;
  }

  ///Element is about to be destroyed
  virtual void ckElementDied(ArrayElement *elt) { (void)elt; }

//The leaving/arriving seqeunce happens once per migration.
  ///Element is about to leave this processor (so about to call pup)
  virtual void ckElementLeaving(ArrayElement *elt) { (void)elt; }

  ///Element just arrived on this processor (so just called pup)
  /// Return false if the element migrated away or deleted itself.
  virtual bool ckElementArriving(ArrayElement *elt) {
    (void)elt;
    return true;
  }

  /// used by checkpointing to reset the states
  virtual void flushState()  {}
};

/*@}*/

//This simple arrayListener just prints each event to stdout:
class CkVerboseListener : public CkArrayListener {
 public:
  CkVerboseListener(void);
  CkVerboseListener(CkMigrateMessage *m):CkArrayListener(m) {}
  PUPable_decl(CkVerboseListener);

  virtual void ckRegister(CkArray *arrMgr,int dataOffset_);
  virtual void ckBeginInserting(void);
  virtual void ckEndInserting(void);

  virtual void ckElementStamp(int *eltInfo);
  virtual void ckElementCreating(ArrayElement *elt);
  virtual bool ckElementCreated(ArrayElement *elt);
  virtual void ckElementDied(ArrayElement *elt);

  virtual void ckElementLeaving(ArrayElement *elt);
  virtual bool ckElementArriving(ArrayElement *elt);
};

/**
\addtogroup CkArray
*/
/*@{*/
/*********************** CkArrayOptions *******************************/
/// Arguments for array creation:
class CkArrayOptions {
	friend class CkArray;

	CkArrayIndex start, end, step;
	CkArrayIndex numInitial;///< Number of elements to create
	/// Limits of element counts in each dimension of this and all bound arrays
	CkArrayIndex bounds;
	CkGroupID map;///< Array location map object
	CkGroupID locMgr;///< Location manager to bind to
	CkPupAblePtrVec<CkArrayListener> arrayListeners; //CkArrayListeners for this array
	CkCallback reductionClient; // Default target of reductions
	bool anytimeMigration; // Elements are allowed to move freely
	bool disableNotifyChildInRed; //Child elements are not notified when reduction starts
	bool staticInsertion; // Elements are only inserted at construction
	bool broadcastViaScheduler;     // broadcast inline or through scheduler

	/// Set various safe defaults for all the constructors
	void init();

	/// Helper functions to keep numInitial and start/step/end consistent
	void updateIndices();
	void updateNumInitial();

 public:
 //Used by external world:
	CkArrayOptions(void); ///< Default: empty array
	CkArrayOptions(int ni1_); ///< With initial elements 1D
	CkArrayOptions(int ni1_, int ni2_); ///< With initial elements 2D 
	CkArrayOptions(int ni1_, int ni2_, int ni3); ///< With initial elements 3D
	CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_); ///< With initial elements 4D
	CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_, short ni5_); ///< With initial elements 5D
	CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_, short ni5_, short ni6_); ///< With initial elements 6D
	CkArrayOptions(CkArrayIndex s, CkArrayIndex e, CkArrayIndex step); ///< Initialize the start, end, and step

	/**
	 * These functions return "this" so you can string them together, e.g.:
	 *   foo(CkArrayOptions().setMap(mid).bindTo(aid));
	 */

	/// Set the start, end, and step for the initial elements to populate
	CkArrayOptions &setStart(CkArrayIndex s)
		{ start = s; updateNumInitial(); return *this; }
	CkArrayOptions &setEnd(CkArrayIndex e)
		{ end = e; updateNumInitial(); return *this; }
	CkArrayOptions &setStep(CkArrayIndex s)
		{ step = s; updateNumInitial(); return *this; }

	/// Create this many initial elements 1D
	CkArrayOptions &setNumInitial(int ni)
		{numInitial=CkArrayIndex1D(ni); updateIndices(); return *this;}
	/// Create this many initial elements 2D
	CkArrayOptions &setNumInitial(int ni1, int ni2)
		{numInitial=CkArrayIndex2D(ni1, ni2); updateIndices(); return *this;}
	/// Create this many initial elements 3D
	CkArrayOptions &setNumInitial(int ni1, int ni2, int ni3)
		{numInitial=CkArrayIndex3D(ni1, ni2, ni3); updateIndices(); return *this;}
	/// Create this many initial elements 4D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4)
		{numInitial=CkArrayIndex4D(ni1, ni2, ni3, ni4); updateIndices(); return *this;}
	/// Create this many initial elements 5D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4, short ni5)
		{numInitial=CkArrayIndex5D(ni1, ni2, ni3, ni4, ni5); updateIndices(); return *this;}
	/// Create this many initial elements 6D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4, short ni5, short ni6)
		{numInitial=CkArrayIndex6D(ni1, ni2, ni3, ni4, ni5, ni6); updateIndices(); return *this;}

	/// Allow up to this many elements in 1D
	CkArrayOptions &setBounds(int ni)
		{bounds=CkArrayIndex1D(ni); return *this;}
	/// Allow up to this many elements in 2D
	CkArrayOptions &setBounds(int ni1, int ni2)
		{bounds=CkArrayIndex2D(ni1, ni2); return *this;}
	/// Allow up to this many elements in 3D
	CkArrayOptions &setBounds(int ni1, int ni2, int ni3)
		{bounds=CkArrayIndex3D(ni1 ,ni2, ni3); return *this;}
	/// Allow up to this many elements in 4D
	CkArrayOptions &setBounds(short ni1, short ni2, short ni3, short ni4)
		{bounds=CkArrayIndex4D(ni1, ni2, ni3, ni4); return *this;}
	/// Allow up to this many elements in 5D
	CkArrayOptions &setBounds(short ni1, short ni2, short ni3, short ni4, short ni5)
		{bounds=CkArrayIndex5D(ni1, ni2, ni3, ni4, ni5); return *this;}
	/// Allow up to this many elements in 6D
	CkArrayOptions &setBounds(short ni1, short ni2, short ni3, short ni4, short ni5, short ni6)
		{bounds=CkArrayIndex6D(ni1, ni2, ni3, ni4, ni5, ni6); return *this;}

	/// Use this location map
	CkArrayOptions &setMap(const CkGroupID &m)
		{map=m; return *this;}

	/// Bind our elements to this array
	CkArrayOptions &bindTo(const CkArrayID &b);

	/// Use this location manager
	CkArrayOptions &setLocationManager(const CkGroupID &l)
		{locMgr=l; return *this;}

	/// Add an array listener component to this array (keeps the new'd listener)
	CkArrayOptions &addListener(CkArrayListener *listener);

	CkArrayOptions &setAnytimeMigration(bool b) { anytimeMigration = b; return *this; }
	CkArrayOptions &setStaticInsertion(bool b);
	CkArrayOptions &setBroadcastViaScheduler(bool b) { broadcastViaScheduler = b; return *this; }
	CkArrayOptions &setReductionClient(CkCallback cb)
	{ reductionClient = cb; return *this; }

  //Used by the array manager:
	const CkArrayIndex &getStart(void) const {return start;}
	const CkArrayIndex &getEnd(void) const {return end;}
	const CkArrayIndex &getStep(void) const {return step;}
	const CkArrayIndex &getNumInitial(void) const {return numInitial;}
	const CkArrayIndex &getBounds(void) const {return bounds;}
	const CkGroupID &getMap(void) const {return map;}
	const CkGroupID &getLocationManager(void) const {return locMgr;}
	int getListeners(void) const {return arrayListeners.size();}
	CkArrayListener *getListener(int listenerNum) {
		CkArrayListener *ret=arrayListeners[listenerNum];
		arrayListeners[listenerNum]=NULL; //Don't throw away this listener
		return ret;
	}

	void pup(PUP::er &p);
};
PUPmarshall(CkArrayOptions)


/*********************** Proxy Support ************************/
//Needed by CBase_ArrayElement
class ArrayBase { /*empty*/ };
/*forward*/ class ArrayElement;

/**
 * This class is a wrapper around a CkArrayIndex and ArrayID,
 *  used by array element proxies.  This makes the translator's
 *  job simpler, and the translated code smaller.
 */
class CProxy_ArrayBase :public CProxy {
private:
	CkArrayID _aid;
public:
	CProxy_ArrayBase() {
#if CMK_ERROR_CHECKING
		_aid.setZero();
#endif
	}
	CProxy_ArrayBase(const CkArrayID &aid,CK_DELCTOR_PARAM)
		:CProxy(CK_DELCTOR_ARGS), _aid(aid) { }
        CProxy_ArrayBase(const CkArrayID &aid)
                :CProxy(), _aid(aid) { }
	CProxy_ArrayBase(const ArrayElement *e);

#if CMK_ERROR_CHECKING
	inline void ckCheck(void) const{  //Make sure this proxy has a value
	  if (_aid.isZero())
		CkAbort("Error! This array proxy has not been initialized!");
        }
#else
	inline void ckCheck(void) const {}
#endif

	static CkArrayID ckCreateEmptyArray(CkArrayOptions opts);
        static void ckCreateEmptyArrayAsync(CkCallback cb, CkArrayOptions opts);
	static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts);

	void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx);
	void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const;
	CkArrayID ckGetArrayID(void) const { return _aid; }
	CkArray *ckLocalBranch(void) const { return _aid.ckLocalBranch(); }
	CkLocMgr *ckLocMgr(void) const;
	inline operator CkArrayID () const {return ckGetArrayID();}
	unsigned int numLocalElements() const { return ckLocMgr()->numLocalElements(); }

	void doneInserting(void);
	void beginInserting(void);

	CK_REDUCTION_CLIENT_DECL

	void pup(PUP::er &p);
};
PUPmarshall(CProxy_ArrayBase)

class CProxyElement_ArrayBase:public CProxy_ArrayBase {
private:
	CkArrayIndex _idx;//<- our element's array index
public:
	CProxyElement_ArrayBase() { }
	CProxyElement_ArrayBase(const CkArrayID &aid,
		const CkArrayIndex &idx,CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(aid,CK_DELCTOR_ARGS), _idx(idx) { }
        CProxyElement_ArrayBase(const CkArrayID &aid, const CkArrayIndex &idx)
                :CProxy_ArrayBase(aid), _idx(idx) { }
	CProxyElement_ArrayBase(const ArrayElement *e);

	void ckInsert(CkArrayMessage *m,int ctor,int onPe);
	void ckSend(CkArrayMessage *m, int ep, int opts = 0) const;
//      static void ckSendWrapper(void *me, void *m, int ep, int opts = 0);
      static void ckSendWrapper(CkArrayID _aid, CkArrayIndex _idx, void *m, int ep, int opts);
	void *ckSendSync(CkArrayMessage *m, int ep) const;
	const CkArrayIndex &ckGetIndex() const {return _idx;}

	ArrayElement *ckLocal(void) const;
	void pup(PUP::er &p);
};
PUPmarshall(CProxyElement_ArrayBase)

class CProxySection_ArrayBase:public CProxy_ArrayBase {
private:
	int _nsid;
	CkSectionID *_sid;
public:
	CProxySection_ArrayBase(): _nsid(0), _sid(NULL) {}
	CProxySection_ArrayBase(const CkArrayID &aid,
		const CkArrayIndex *elems, const int nElems)
		:CProxy_ArrayBase(aid), _nsid(1) { _sid = new CkSectionID(aid, elems, nElems); }
	CProxySection_ArrayBase(const CkArrayID &aid,
		const CkArrayIndex *elems, const int nElems, CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(aid,CK_DELCTOR_ARGS), _nsid(1) { _sid = new CkSectionID(aid, elems, nElems); }
	CProxySection_ArrayBase(const CkSectionID &sid)
		:CProxy_ArrayBase(sid._cookie.get_aid()), _nsid(1) { _sid = new CkSectionID(sid); }
	CProxySection_ArrayBase(const CkSectionID &sid, CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(sid._cookie.get_aid(), CK_DELCTOR_ARGS), _nsid(1) { _sid = new CkSectionID(sid); }
        CProxySection_ArrayBase(const CProxySection_ArrayBase &cs)
		:CProxy_ArrayBase(cs.ckGetArrayID()), _nsid(cs._nsid) {
      if (_nsid == 1) _sid = new CkSectionID(cs.ckGetArrayID(), cs.ckGetArrayElements(), cs.ckGetNumElements());
      else if (_nsid > 1) {
        _sid = new CkSectionID[_nsid];
        for (int i=0; i<_nsid; ++i) _sid[i] = cs._sid[i];
      } else _sid = NULL;
    }
        CProxySection_ArrayBase(const CProxySection_ArrayBase &cs, CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(cs.ckGetArrayID(),CK_DELCTOR_ARGS), _nsid(cs._nsid) {
      if (_nsid == 1) _sid = new CkSectionID(cs.ckGetArrayID(), cs.ckGetArrayElements(), cs.ckGetNumElements());
      else if (_nsid > 1) {
        _sid = new CkSectionID[_nsid];
        for (int i=0; i<_nsid; ++i) _sid[i] = cs._sid[i];
      } else _sid = NULL;
    }
    CProxySection_ArrayBase(const int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems)
        :CProxy_ArrayBase(aid[0]), _nsid(n) {
      if (_nsid == 1) _sid = new CkSectionID(aid[0], elems[0], nElems[0]);
      else if (_nsid > 1) {
      _sid = new CkSectionID[n];
      for (int i=0; i<n; ++i) _sid[i] = CkSectionID(aid[i], elems[i], nElems[i]);
      } else _sid = NULL;
    }
    CProxySection_ArrayBase(const int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems,CK_DELCTOR_PARAM)
        :CProxy_ArrayBase(aid[0],CK_DELCTOR_ARGS), _nsid(n) {
      if (_nsid == 1) _sid = new CkSectionID(aid[0], elems[0], nElems[0]);
      else if (_nsid > 1) {
      _sid = new CkSectionID[n];
      for (int i=0; i<n; ++i) _sid[i] = CkSectionID(aid[i], elems[i], nElems[i]);
      } else _sid = NULL;
    }

    ~CProxySection_ArrayBase() {
      if (_nsid == 1) delete _sid;
      else if (_nsid > 1) delete[] _sid;
    }

    CProxySection_ArrayBase &operator=(const CProxySection_ArrayBase &cs) {
      CProxy_ArrayBase::operator=(cs);
      _nsid = cs._nsid;
      if (_nsid == 1) _sid = new CkSectionID(*cs._sid);
      else if (_nsid > 1) {
        _sid = new CkSectionID[_nsid];
        for (int i=0; i<_nsid; ++i) _sid[i] = cs._sid[i];
      } else _sid = NULL;
      return *this;
    }
    
	void ckSectionDelegate(CkDelegateMgr *d) 
		{ ckDelegate(d); d->initDelegateMgr(this); }
//	void ckInsert(CkArrayMessage *m,int ctor,int onPe);
	void ckSend(CkArrayMessage *m, int ep, int opts = 0) ;

//	ArrayElement *ckLocal(void) const;
    inline int ckGetNumSubSections() const { return _nsid; }
	inline CkSectionInfo &ckGetSectionInfo() {return _sid->_cookie;}
	inline CkSectionID *ckGetSectionIDs() {return _sid;}
	inline CkSectionID &ckGetSectionID() {return _sid[0];}
	inline CkSectionID &ckGetSectionID(int i) {return _sid[i];}
	inline CkArrayID ckGetArrayIDn(int i) const {return _sid[i]._cookie.get_aid();}
    inline CkArrayIndex *ckGetArrayElements() const {return _sid[0]._elems;}
    inline CkArrayIndex *ckGetArrayElements(int i) const {return _sid[i]._elems;}
    inline int ckGetNumElements() const { return _sid[0]._nElems; }
	inline int ckGetNumElements(int i) const { return _sid[i]._nElems; }
	void pup(PUP::er &p);
};
PUPmarshall(CProxySection_ArrayBase)

//Simple C-like API:
void CkSetMsgArrayIfNotThere(void *msg);
void CkSendMsgArray(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts=0);
void CkSendMsgArrayInline(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts=0);
void CkBroadcastMsgArray(int entryIndex, void *msg, CkArrayID aID, int opts=0);
void CkBroadcastMsgSection(int entryIndex, void *msg, CkSectionID sID, int opts=    0);
/************************ Array Element *********************/
/**
 *An array element is a chare that lives inside the array.
 *Unlike regular chares, array elements can migrate from one
 *Pe to another.  Each element has a unique index.
 */

class ArrayElement : public CkMigratable
{
  friend class CkArray;

  friend class CkArrayListener;
  int numInitialElements; // Number of elements created by ckNew(numElements)
  void initBasics(void);
#ifdef _PIPELINED_ALLREDUCE_
AllreduceMgr * allredMgr; // for allreduce
#endif
public:
  ArrayElement(void);
  ArrayElement(CkMigrateMessage *m);
  virtual ~ArrayElement();

/// Pack/unpack routine (called before and after migration)
  void pup(PUP::er &p);

//Overridden functions:
  /// Called by the system just before and after migration to another processor:
  virtual void ckAboutToMigrate(void);
  virtual void ckJustMigrated(void);
  
  virtual void ckJustRestored(void);
  
  virtual void ckDestroy(void);
  virtual char *ckDebugChareName(void);
  virtual int ckDebugChareID(char*, int);

  /// Synonym for ckMigrate
  inline void migrateMe(int toPe) {ckMigrate(toPe);}

#ifdef _PIPELINED_ALLREDUCE_
	void contribute2(CkArrayIndex myIndex, int dataSize,const void *data,CkReduction::reducerType type,
			   const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1);
	void contribute2(int dataSize,const void *data,CkReduction::reducerType type, 
					CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1); 
	void contribute2(int dataSize,const void *data,CkReduction::reducerType type, 
					const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1); 
	void contribute2(CkReductionMsg *msg); 
	void contribute2(const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1);
	void contribute2(CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1);
#else
  CK_REDUCTION_CONTRIBUTE_METHODS_DECL
#endif
	// for _PIPELINED_ALLREDUCE_, assembler entry method
	inline void defrag(CkReductionMsg* msg);
  inline const CkArrayID &ckGetArrayID(void) const {return thisArrayID;}
  inline ck::ObjID ckGetID(void) const { return ck::ObjID(thisArrayID, myRec->getID()); }

  inline int ckGetArraySize(void) const { return numInitialElements; }
protected:
  CkArray *thisArray;//My source array
  CkArrayID thisArrayID;//My source array's ID

  //More verbose form of abort
  virtual void CkAbort(const char *str) const;

private:
//Array implementation methods:
  int listenerData[CK_ARRAYLISTENER_MAXLEN];

#if CMK_MEM_CHECKPOINT
friend class CkMemCheckPT;
friend class CkLocMgr;
protected:
  int budPEs[2];
private:
  void init_checkpt();
#endif
public:
	void inmem_checkpoint(CkArrayCheckPTReqMessage *m);
	void recvBroadcast(CkMessage *);

#if CMK_GRID_QUEUE_AVAILABLE
public:
  int grid_queue_interval;
  int grid_queue_threshold;
  int msg_count;
  int msg_count_grid;
  int border_flag;
#endif
};
inline int *CkArrayListener::ckGetData(ArrayElement *el) const
  {return &el->listenerData[dataOffset];}

/**An ArrayElementT is a utility class where you are
 * constrained to a "thisIndex" of some fixed-sized type T.
 */
template <class T>
class ArrayElementT : public ArrayElement
{
public:
  ArrayElementT(void): thisIndex(*(const T *)thisIndexMax.data()) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
        mlogData->objID.data.array.idx=thisIndexMax;
#endif
}
#ifdef _PIPELINED_ALLREDUCE_
	void contribute(int dataSize,const void *data,CkReduction::reducerType type,
						CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1)
	{
		contribute2( dataSize,data, type, userFlag);

	}
	void contribute(int dataSize,const void *data,CkReduction::reducerType type,
						const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1)
	{
		contribute2((CkArrayIndex)(thisIndex) ,dataSize, data, type,   cb, userFlag);
	}
	void contribute(CkReductionMsg *msg) 
	{
		contribute2(msg);
	}
	void contribute(const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1)
	{
		contribute2(cb ,userFlag);
	}
	void contribute(CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1)
	{
		contribute2(userFlag);
	}
#endif
  ArrayElementT(CkMigrateMessage *msg)
	:ArrayElement(msg),
	thisIndex(*(const T *)thisIndexMax.data()) {}

  const T thisIndex;/// Object array index
};

typedef ArrayElementT<CkIndex1D> ArrayElement1D;
typedef ArrayElementT<CkIndex2D> ArrayElement2D;
typedef ArrayElementT<CkIndex3D> ArrayElement3D;
typedef ArrayElementT<CkIndex4D> ArrayElement4D;
typedef ArrayElementT<CkIndex5D> ArrayElement5D;
typedef ArrayElementT<CkIndex6D> ArrayElement6D;
typedef ArrayElementT<CkIndexMax> ArrayElementMax;

/*@}*/


/*********************** Array Manager BOC *******************/
/**
\addtogroup CkArrayImpl
*/
/*@{*/

#include "CkArray.decl.h"
#include "CkArrayReductionMgr.decl.h"

void CkSendAsyncCreateArray(int ctor, CkCallback cb, CkArrayOptions opts, void *ctorMsg);

struct CkArrayCreatedMsg : public CMessage_CkArrayCreatedMsg {
  CkArrayID aid;
};

class CkArrayBroadcaster;
class CkArrayReducer;

void _ckArrayInit(void);

class CkArray : public CkReductionMgr {
  friend class ArrayElement;
  friend class CProxy_ArrayBase;
  friend class CProxyElement_ArrayBase;

  CkMagicNumber<ArrayElement> magic; //To detect heap corruption
  CkLocMgr *locMgr;
  CkGroupID locMgrID;
  CProxy_CkArray thisProxy;
  // Separate mapping and storing the element pointers to speed iteration in broadcast
  std::map<CmiUInt8, unsigned int> localElems;
  std::vector<CkMigratable *> localElemVec;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    int *children;
    int numChildren;
#endif
private:
  bool stableLocations;

public:
//Array Creation:
  CkArray(CkArrayOptions &c,CkMarshalledMessage &initMsg,CkNodeGroupID nodereductionProxy);
  CkArray(CkMigrateMessage *m);
  ~CkArray();
  CkGroupID &getGroupID(void) {return thisgroup;}

//Access & information routines
  inline CkLocMgr *getLocMgr(void) {return locMgr;}
  inline const CkArrayIndex &getNumInitial(void) const {return numInitial;}
  inline int homePe(const CkArrayIndex &idx) const {return locMgr->homePe(idx);}
  inline int procNum(const CkArrayIndex &idx) const {return locMgr->procNum(idx);}

  /// Return the last known processor for this array index.
  /// Valid for any possible array index.
  inline int lastKnown(const CkArrayIndex &idx) const
	  {return locMgr->lastKnown(idx);}
  /// Deliver message to this element (directly if local)
  /// doFree if is local
  inline void deliver(CkMessage *m, const CkArrayIndex &idx, CkDeliver_t type,int opts=0)
  { locMgr->sendMsg((CkArrayMessage*)m, thisgroup, idx, type, opts); }
  inline int deliver(CkArrayMessage *m, CkDeliver_t type)
  { return locMgr->deliverMsg(m, thisgroup, m->array_element_id(), NULL, type); }
  /// Fetch a local element via its ID (return NULL if not local)
  inline ArrayElement *lookup(const CmiUInt8 id) { return (ArrayElement*) getEltFromArrMgr(id); }
  /// Fetch a local element via its index (return NULL if not local)
  inline ArrayElement *lookup(const CkArrayIndex &idx) { 
    CkLocMgr::IdxIdMap::iterator itr = locMgr->idx2id.find(idx);
    if (itr == locMgr->idx2id.end())
      return NULL;
    else
      return (ArrayElement*) getEltFromArrMgr(itr->second);
  }

  virtual CkMigratable* getEltFromArrMgr(const CmiUInt8 id) {
    std::map<CmiUInt8, unsigned int>::iterator itr = localElems.find(id);
    return ( itr == localElems.end() ? NULL : localElemVec[itr->second] );
  }
  virtual void putEltInArrMgr(const CmiUInt8 id, CkMigratable* elt)
  {
    localElems[id] = localElemVec.size();
    localElemVec.push_back(elt);
  }
  virtual void eraseEltFromArrMgr(const CmiUInt8 id)
  { localElems.erase(id); }

  void deleteElt(const CmiUInt8 id) {
    std::map<CmiUInt8, unsigned int>::iterator itr = localElems.find(id);
    if (itr != localElems.end()) {
      unsigned int offset = itr->second;
      delete localElemVec[offset];
      localElems.erase(itr);

      if (offset != localElemVec.size() - 1) {
        CkMigratable *moved = localElemVec[localElemVec.size()-1];
        localElemVec[offset] = moved;
        localElems[moved->ckGetID()] = offset;
      }

      localElemVec.pop_back();
    }
  }

//Creation:
  /// Create-after-migrate:
  /// Create an uninitialized element after migration
  ///  The element's constructor will be called immediately after.
  virtual CkMigratable *allocateMigrated(int elChareType, CkElementCreation_t type);

  /// Prepare creation message:
  void prepareCtorMsg(CkMessage *m, int listenerData[CK_ARRAYLISTENER_MAXLEN]);

  int findInitialHostPe(const CkArrayIndex &idx, int proposedPe);

  /// Create initial array elements:
  virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg);
  virtual void doneInserting(void);
  virtual void beginInserting(void);
  void remoteDoneInserting(void);
  void remoteBeginInserting(void);

  /// Create manually:
  bool insertElement(CkArrayMessage *, const CkArrayIndex &idx, int listenerData[CK_ARRAYLISTENER_MAXLEN]);
  void insertElement(CkMarshalledMessage &, const CkArrayIndex &idx, int listenerData[CK_ARRAYLISTENER_MAXLEN]);

/// Demand-creation:
  /// Demand-create an element at this index on this processor
  ///  Returns true if the element was successfully added;
  ///  false if the element migrated away or deleted itself.
  bool demandCreateElement(const CkArrayIndex &idx,
  	int onPe,int ctor,CkDeliver_t type);

/// Broadcast communication:
  void sendBroadcast(CkMessage *msg);
  void recvBroadcast(CkMessage *msg);
  void sendExpeditedBroadcast(CkMessage *msg);
  void recvExpeditedBroadcast(CkMessage *msg) { recvBroadcast(msg); }
  void recvBroadcastViaTree(CkMessage *msg);

  /// Whole array destruction, including all elements and the group itself
  void ckDestroy();

  void pup(PUP::er &p);
  void ckJustMigrated(void){ doneInserting(); }

  virtual bool isArrMgr(void) {return true;}

private:
  CkArrayIndex numInitial;/// Number of initial array elements
  bool isInserting;/// Are we currently inserting elements?

/// Allocate space for a new array element
  ArrayElement *allocate(int elChareType, CkMessage *msg, bool fromMigration, int *listenerData);

//Spring cleaning
  void springCleaning(void);
  static void staticSpringCleaning(void *forWhom,double curWallTime);
  void setupSpringCleaning();
  int springCleaningCcd;

//ArrayListeners:
  CkPupAblePtrVec<CkArrayListener> listeners;
  int listenerDataOffset;
 public:
  void addListener(CkArrayListener *l) {
    l->ckRegister(this,listenerDataOffset);
    listenerDataOffset+=l->ckGetLen();
    listeners.push_back(l);
    if (listenerDataOffset>CK_ARRAYLISTENER_MAXLEN)
      CkAbort("Too much array listener data!\n"
	      "You'll have to either use fewer array listeners, or increase the compile-time\n"
	      "constant CK_ARRAYLISTENER_MAXLEN!\n");
  }
 private:

  CkArrayReducer *reducer; //Read-only copy of default reducer
  CkArrayBroadcaster *broadcaster; //Read-only copy of default broadcaster
public:
  void flushStates();

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	// the mlogft only support 1D arrays, then returning the number of elements in the first dimension
	virtual int numberReductionMessages(){CkAssert(CkMyPe() == 0);return numInitial.data()[0];}
	void broadcastHomeElements(void *data,CkLocRec *rec,CkArrayIndex *index);
	static void staticBroadcastHomeElements(CkArray *arr,void *data,CkLocRec *rec,CkArrayIndex *index);
#endif

        static int isIrreducible() { return 1; }
};

// Maintain old name of former parent class for backwards API compatibility
// with usage in maps' populateInitial()
typedef CkArray CkArrMgr;

/*@}*/

#endif
