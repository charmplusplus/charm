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

/***********************************************************
	Utility defines, includes, etc.
*/
extern void _registerCkArray(void);
CpvExtern (int ,serializer);


/** This flag is true when in the system there is anytime migration, false when
 *  the user code guarantees that no migration happens except during load balancing
 *  (in which case it can only happen between AtSync and ResumeFromSync). */
extern bool _isAnytimeMigration;

/**
\addtogroup CkArray
\brief Migratable Chare Arrays: user-visible classes.

All these classes are defined in ckarray.C.
*/
/*@{*/

/// Simple ArrayIndex classes: the key is just integer indices.
class CkArrayIndex1D : public CkArrayIndex {
public:
	CkArrayIndex1D() { nInts=1; dimension=1; }
	CkArrayIndex1D(int i0) {
		index[0]=i0;nInts=1;dimension=1;
	}
	CkArrayIndex1D &operator=(const CkArrayIndex1D &that)  {
		CmiAssert(that.nInts == 1);
		CmiAssert(that.dimension == 1);
		nInts = that.nInts;
		dimension = that.dimension;
		index[0] = that.index[0];
		return *this;
	}
};
class CkArrayIndex2D : public CkArrayIndex {
public:
	CkArrayIndex2D() { nInts=2; dimension=2; }
	CkArrayIndex2D(int i0,int i1) {
		index[0]=i0;index[1]=i1;nInts=2;dimension=2;
	}
	CkArrayIndex2D &operator=(const CkArrayIndex2D &that)  {
		CmiAssert(that.nInts == 2);
		CmiAssert(that.dimension == 2);
		nInts = that.nInts;
		dimension = that.dimension;
		memcpy(index, that.index, sizeof(int)*2);
		return *this;
	}
};
class CkArrayIndex3D : public CkArrayIndex {
public:
	CkArrayIndex3D() { nInts=3; dimension=3; }
	CkArrayIndex3D(int i0,int i1,int i2) {
		index[0]=i0;index[1]=i1;index[2]=i2;nInts=3;dimension=3;
	}
	CkArrayIndex3D &operator=(const CkArrayIndex3D &that)  {
		CmiAssert(that.nInts == 3);
		CmiAssert(that.dimension == 3);
		nInts = that.nInts;
		dimension = that.dimension;
		memcpy(index, that.index, sizeof(int)*3);
		return *this;
	}
};
class CkArrayIndex4D : public CkArrayIndex {
public:
	CkArrayIndex4D(){ nInts=2; dimension=4; }
	CkArrayIndex4D(short int i0,short int i1,short int i2,short int i3) {
		indexShorts[0]=i0;indexShorts[1]=i1;indexShorts[2]=i2;indexShorts[3]=i3;nInts=2;dimension=4;
	}
	CkArrayIndex4D &operator=(const CkArrayIndex4D &that)  {
		CmiAssert(that.nInts == 2);
		CmiAssert(that.dimension == 4);
		nInts = that.nInts;
		dimension = that.dimension;
		memcpy(indexShorts, that.indexShorts, sizeof(short int)*4);
		return *this;
	}
};
class CkArrayIndex5D : public CkArrayIndex {
public:
	CkArrayIndex5D(){ nInts=3; dimension=5; }
	CkArrayIndex5D(short int i0,short int i1,short int i2,short int i3,short int i4) {
		indexShorts[0]=i0;indexShorts[1]=i1;indexShorts[2]=i2;indexShorts[3]=i3;indexShorts[4]=i4;nInts=3;dimension=5;
        }
	CkArrayIndex5D &operator=(const CkArrayIndex5D &that)  {
		CmiAssert(that.nInts == 3);
		CmiAssert(that.dimension == 5);
		nInts = that.nInts;
		dimension = that.dimension;
		memcpy(indexShorts, that.indexShorts, sizeof(short int)*5);;
		return *this;
	}
};
class CkArrayIndex6D : public CkArrayIndex {
public:
	CkArrayIndex6D(){ nInts=3; dimension=6; }
	CkArrayIndex6D(short int i0,short int i1,short int i2,short int i3,short int i4,short int i5) {
		indexShorts[0]=i0;indexShorts[1]=i1;indexShorts[2]=i2;indexShorts[3]=i3;indexShorts[4]=i4;indexShorts[5]=i5;nInts=3;dimension=6;
	}
	CkArrayIndex6D &operator=(const CkArrayIndex6D &that)  {
		CmiAssert(that.nInts == 3);
		CmiAssert(that.dimension == 6);
		nInts = that.nInts;
		dimension = that.dimension;
		memcpy(indexShorts, that.indexShorts, sizeof(short int)*6);;
		return *this;
	}
};

/** A slightly more complex array index: the key is an object
 *  whose size is fixed at compile time.
 */
template <class object> //Key object
class CkArrayIndexT : public CkArrayIndex {
public:
	object obj;
	CkArrayIndexT(const object &srcObj) {obj=srcObj;
		nInts=sizeof(obj)/sizeof(int);
		dimension=0; }
};

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
  virtual void ckElementStamp(int *eltInfo) {}
  ///Element is about to be created on this processor
  virtual void ckElementCreating(ArrayElement *elt) {}
  ///Element was just created on this processor
  /// Return false if the element migrated away or deleted itself.
  virtual CmiBool ckElementCreated(ArrayElement *elt)
    { return CmiTrue; }

  ///Element is about to be destroyed
  virtual void ckElementDied(ArrayElement *elt) {}

//The leaving/arriving seqeunce happens once per migration.
  ///Element is about to leave this processor (so about to call pup)
  virtual void ckElementLeaving(ArrayElement *elt) {}

  ///Element just arrived on this processor (so just called pup)
  /// Return false if the element migrated away or deleted itself.
  virtual CmiBool ckElementArriving(ArrayElement *elt)
    { return CmiTrue; }

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
  virtual CmiBool ckElementCreated(ArrayElement *elt);
  virtual void ckElementDied(ArrayElement *elt);

  virtual void ckElementLeaving(ArrayElement *elt);
  virtual CmiBool ckElementArriving(ArrayElement *elt);
};

/**
\addtogroup CkArray
*/
/*@{*/
/*********************** CkArrayOptions *******************************/
/// Arguments for array creation:
class CkArrayOptions {
	friend class CkArray;

	CkArrayIndexMax numInitial;///< Number of elements to create
	CkGroupID map;///< Array location map object
	CkGroupID locMgr;///< Location manager to bind to
	CkPupAblePtrVec<CkArrayListener> arrayListeners; //CkArrayListeners for this array
	CkCallback reductionClient; // Default target of reductions
	bool anytimeMigration; // Elements are allowed to move freely
	bool staticInsertion; // Elements are only inserted at construction

	/// Set various safe defaults for all the constructors
	void init();

 public:
 //Used by external world:
	CkArrayOptions(void); ///< Default: empty array
	CkArrayOptions(int ni1_); ///< With initial elements 1D
	CkArrayOptions(int ni1_, int ni2_); ///< With initial elements 2D 
	CkArrayOptions(int ni1_, int ni2_, int ni3); ///< With initial elements 3D
	//CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_); ///< With initial elements 4D
	//CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_, short ni5_); ///< With initial elements 5D
	//CkArrayOptions(short ni1_, short ni2_, short ni3, short ni4_, short ni5_, short ni6_); ///< With initial elements 6D

	/**
	 * These functions return "this" so you can string them together, e.g.:
	 *   foo(CkArrayOptions().setMap(mid).bindTo(aid));
	 */

	/// Create this many initial elements 1D
	CkArrayOptions &setNumInitial(int ni)
		{numInitial=CkArrayIndex1D(ni); return *this;}
	/// Create this many initial elements 2D
	CkArrayOptions &setNumInitial(int ni1, int ni2)
		{numInitial=CkArrayIndex2D(ni1, ni2); return *this;}
	/// Create this many initial elements 3D
	CkArrayOptions &setNumInitial(int ni1, int ni2, int ni3)
		{numInitial=CkArrayIndex3D(ni1 ,ni2, ni3); return *this;}
        /*
	/// Create this many initial elements 4D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4)
		{numInitial=CkArrayIndex4D(ni1, ni2, ni3, ni4); return *this;}
	/// Create this many initial elements 5D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4, short ni5)
		{numInitial=CkArrayIndex5D(ni1, ni2, ni3, ni4, ni5); return *this;}
	/// Create this many initial elements 6D
	CkArrayOptions &setNumInitial(short ni1, short ni2, short ni3, short ni4, short ni5, short ni6)
		{numInitial=CkArrayIndex6D(ni1, ni2, ni3, ni4, ni5, ni6); return *this;}
        */

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
	CkArrayOptions &setReductionClient(CkCallback cb)
	{ reductionClient = cb; return *this; }

  //Used by the array manager:
	const CkArrayIndexMax &getNumInitial(void) const {return numInitial;}
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

	static CkArrayID ckCreateEmptyArray(void);
	static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts);

	void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx);
	void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const;
	CkArrayID ckGetArrayID(void) const { return _aid; }
	CkArray *ckLocalBranch(void) const { return _aid.ckLocalBranch(); }
	CkLocMgr *ckLocMgr(void) const;
	inline operator CkArrayID () const {return ckGetArrayID();}
	unsigned int numLocalElements() const { return ckLocMgr()->numLocalElements(); }

	void doneInserting(void);

	CK_REDUCTION_CLIENT_DECL

	void pup(PUP::er &p);
};
PUPmarshall(CProxy_ArrayBase)

class CProxyElement_ArrayBase:public CProxy_ArrayBase {
private:
	CkArrayIndexMax _idx;//<- our element's array index
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
      static void ckSendWrapper(CkArrayID _aid, CkArrayIndexMax _idx, void *m, int ep, int opts);
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
		const CkArrayIndexMax *elems, const int nElems)
		:CProxy_ArrayBase(aid), _nsid(1) { _sid = new CkSectionID(aid, elems, nElems); }
	CProxySection_ArrayBase(const CkArrayID &aid,
		const CkArrayIndexMax *elems, const int nElems, CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(aid,CK_DELCTOR_ARGS), _nsid(1) { _sid = new CkSectionID(aid, elems, nElems); }
	CProxySection_ArrayBase(const CkSectionID &sid)
		:CProxy_ArrayBase(sid._cookie.aid), _nsid(1) { _sid = new CkSectionID(sid); }
	CProxySection_ArrayBase(const CkSectionID &sid, CK_DELCTOR_PARAM)
		:CProxy_ArrayBase(sid._cookie.aid, CK_DELCTOR_ARGS), _nsid(1) { _sid = new CkSectionID(sid); }
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
    CProxySection_ArrayBase(const int n, const CkArrayID *aid, CkArrayIndexMax const * const *elems, const int *nElems)
        :CProxy_ArrayBase(aid[0]), _nsid(n) {
      if (_nsid == 1) _sid = new CkSectionID(aid[0], elems[0], nElems[0]);
      else if (_nsid > 1) {
      _sid = new CkSectionID[n];
      for (int i=0; i<n; ++i) _sid[i] = CkSectionID(aid[i], elems[i], nElems[i]);
      } else _sid = NULL;
    }
    CProxySection_ArrayBase(const int n, const CkArrayID *aid, CkArrayIndexMax const * const *elems, const int *nElems,CK_DELCTOR_PARAM)
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
	inline CkArrayID ckGetArrayIDn(int i) const {return _sid[i]._cookie.aid;}
    inline CkArrayIndexMax *ckGetArrayElements() const {return _sid[0]._elems;}
    inline CkArrayIndexMax *ckGetArrayElements(int i) const {return _sid[i]._elems;}
    inline int ckGetNumElements() const { return _sid[0]._nElems; }
	inline int ckGetNumElements(int i) const { return _sid[i]._nElems; }
	void pup(PUP::er &p);
};
PUPmarshall(CProxySection_ArrayBase)

//Simple C-like API:
void CkSendMsgArray(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts=0);
void CkSendMsgArrayInline(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts=0);
void CkBroadcastMsgArray(int entryIndex, void *msg, CkArrayID aID, int opts=0);

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
public:
  ArrayElement(void);
  ArrayElement(CkMigrateMessage *m);
  virtual ~ArrayElement();

/// Pack/unpack routine (called before and after migration)
  virtual void pup(PUP::er &p);

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

  CK_REDUCTION_CONTRIBUTE_METHODS_DECL

  inline const CkArrayID &ckGetArrayID(void) const {return thisArrayID;}

  inline int ckGetArraySize(void) const { return numInitialElements; }
protected:
  CkArray *thisArray;//My source array
  CkArrayID thisArrayID;//My source array's ID

  //More verbose form of abort
  virtual void CkAbort(const char *str) const;

private:
//Array implementation methods:
#ifndef CK_ARRAYLISTENER_MAXLEN
# define CK_ARRAYLISTENER_MAXLEN 3
#endif
  int listenerData[CK_ARRAYLISTENER_MAXLEN];

#if CMK_MEM_CHECKPOINT
friend class CkMemCheckPT;
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
        mlogData->objID.data.array.idx.asMax()=thisIndexMax;
#endif
}
  ArrayElementT(CkMigrateMessage *msg)
	:ArrayElement(msg),
	thisIndex(*(const T *)thisIndexMax.data()) {}

  const T thisIndex;/// Object array index
};

typedef int CkIndex1D;
typedef struct {int x,y;} CkIndex2D;
inline void operator|(PUP::er &p,CkIndex2D &i) {p(i.x); p(i.y);}
typedef struct {int x,y,z;} CkIndex3D;
inline void operator|(PUP::er &p,CkIndex3D &i) {p(i.x); p(i.y); p(i.z);}
typedef struct {short int w,x,y,z;} CkIndex4D;
inline void operator|(PUP::er &p,CkIndex4D &i) {p(i.w); p(i.x); p(i.y); p(i.z);}
typedef struct {short int v,w,x,y,z;} CkIndex5D;
inline void operator|(PUP::er &p,CkIndex5D &i) {p(i.v); p(i.w); p(i.x); p(i.y); p(i.z);}
typedef struct {short int x1,y1,z1,x2,y2,z2;} CkIndex6D;
inline void operator|(PUP::er &p,CkIndex6D &i) {p(i.x1); p(i.y1); p(i.z1); p(i.x2); p(i.y2); p(i.z2);}
typedef struct {int data[CK_ARRAYINDEX_MAXLEN];} CkIndexMax;
inline void operator|(PUP::er &p,CkIndexMax &i) {
  for (int j=0;j<CK_ARRAYINDEX_MAXLEN;j++) {
    p|i.data[j];
  }
}

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

///This arrayListener is in charge of delivering broadcasts to the array.
class CkArrayBroadcaster : public CkArrayListener {
  inline int &getData(ArrayElement *el) {return *ckGetData(el);}
public:
  CkArrayBroadcaster(bool _stableLocations);
  CkArrayBroadcaster(CkMigrateMessage *m);
  virtual void pup(PUP::er &p);
  virtual ~CkArrayBroadcaster();
  PUPable_decl(CkArrayBroadcaster);

  virtual void ckElementStamp(int *eltInfo) {*eltInfo=bcastNo;}

  ///Element was just created on this processor
  /// Return false if the element migrated away or deleted itself.
  virtual CmiBool ckElementCreated(ArrayElement *elt)
    { return bringUpToDate(elt); }

  ///Element just arrived on this processor (so just called pup)
  /// Return false if the element migrated away or deleted itself.
  virtual CmiBool ckElementArriving(ArrayElement *elt)
    { return bringUpToDate(elt); }

  void incoming(CkArrayMessage *msg);

  CmiBool deliver(CkArrayMessage *bcast, ArrayElement *el, bool doFree);

  void springCleaning(void);

  void flushState();
private:
  int bcastNo;//Number of broadcasts received (also serial number)
  int oldBcastNo;//Above value last spring cleaning
  //This queue stores old broadcasts (in case a migrant arrives
  // and needs to be brought up to date)
  CkQ<CkArrayMessage *> oldBcasts;
  bool stableLocations;

  CmiBool bringUpToDate(ArrayElement *el);
};

///This arrayListener is in charge of performing reductions on the array.
class CkArrayReducer : public CkArrayListener {
  CkGroupID mgrID;
  CkReductionMgr *mgr;
  typedef  contributorInfo *I;
  inline contributorInfo *getData(ArrayElement *el)
    {return (I)ckGetData(el);}
public:
  /// Attach this array to this CkReductionMgr
  CkArrayReducer(CkGroupID mgrID_);
  CkArrayReducer(CkMigrateMessage *m);
  virtual void pup(PUP::er &p);
  virtual ~CkArrayReducer();
  PUPable_decl(CkArrayReducer);

  void ckBeginInserting(void) {mgr->creatingContributors();}
  void ckEndInserting(void) {mgr->doneCreatingContributors();}

  void ckElementStamp(int *eltInfo) {mgr->contributorStamped((I)eltInfo);}

  void ckElementCreating(ArrayElement *elt)
    {mgr->contributorCreated(getData(elt));}
  void ckElementDied(ArrayElement *elt)
    {mgr->contributorDied(getData(elt));}

  void ckElementLeaving(ArrayElement *elt)
    {mgr->contributorLeaving(getData(elt));}
  CmiBool ckElementArriving(ArrayElement *elt)
    {mgr->contributorArriving(getData(elt)); return CmiTrue; }
};

void _ckArrayInit(void);

//#include "ComlibArrayListener.h"  FILE DELETED!

class CkArray : public CkReductionMgr, public CkArrMgr {
  friend class ArrayElement;
  friend class CProxy_ArrayBase;
  friend class CProxyElement_ArrayBase;

  CkMagicNumber<ArrayElement> magic; //To detect heap corruption
  CkLocMgr *locMgr;
  CkGroupID locMgrID;
  CProxy_CkArray thisProxy;
  typedef CkMigratableListT<ArrayElement> ArrayElementList;
  ArrayElementList *elements;
  bool stableLocations;

public:
//Array Creation:
  CkArray(CkArrayOptions &c,CkMarshalledMessage &initMsg,CkNodeGroupID nodereductionProxy);
  CkArray(CkMigrateMessage *m);
  CkGroupID &getGroupID(void) {return thisgroup;}

//Access & information routines
  inline CkLocMgr *getLocMgr(void) {return locMgr;}
  inline const CkArrayIndexMax &getNumInitial(void) const {return numInitial;}
  inline int homePe(const CkArrayIndex &idx) const {return locMgr->homePe(idx);}
  inline int procNum(const CkArrayIndex &idx) const {return locMgr->procNum(idx);}

  /// Return the last known processor for this array index.
  /// Valid for any possible array index.
  inline int lastKnown(const CkArrayIndex &idx) const
	  {return locMgr->lastKnown(idx);}
  /// Deliver message to this element (directly if local)
  /// doFree if is local
  inline int deliver(CkMessage *m,CkDeliver_t type,int opts=0)
	  {return locMgr->deliver(m,type,opts);}
  /// Fetch a local element via its index (return NULL if not local)
  inline ArrayElement *lookup(const CkArrayIndex &index)
	  {return (ArrayElement *)locMgr->lookup(index,thisgroup);}

//Creation:
  /// Create-after-migrate:
  virtual CkMigratable *allocateMigrated(int elChareType,const CkArrayIndex &idx,
		  	CkElementCreation_t type);

  /// Prepare creation message:
  void prepareCtorMsg(CkMessage *m,int &onPe,const CkArrayIndex &idx);

  /// Create initial array elements:
  virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg,int local=1);
  virtual void doneInserting(void);
  void remoteDoneInserting(void);

  /// Create manually:
  virtual CmiBool insertElement(CkMessage *);

/// Demand-creation:
  CmiBool demandCreateElement(const CkArrayIndex &idx,
  	int onPe,int ctor,CkDeliver_t type);

/// Broadcast communication:
  void sendBroadcast(CkMessage *msg);
  void recvBroadcast(CkMessage *msg);
  void sendExpeditedBroadcast(CkMessage *msg);
  void recvExpeditedBroadcast(CkMessage *msg) { recvBroadcast(msg); }

  void pup(PUP::er &p);
  void ckJustMigrated(void){ doneInserting(); }

  // COMLIB HACK
  // Ok, this has gone very far... getting rid of it!
  //ComlibArrayListener * calistener;
  //ComlibArrayListener * getComlibArrayListener() {return calistener;}

  virtual CmiBool isArrMgr(void) {return CmiTrue;}

private:
  CkArrayIndexMax numInitial;/// Number of initial array elements
  CmiBool isInserting;/// Are we currently inserting elements?

/// Allocate space for a new array element
  ArrayElement *allocate(int elChareType,const CkArrayIndex &idx,
	CkMessage *msg,CmiBool fromMigration);

//Spring cleaning
  void springCleaning(void);
  static void staticSpringCleaning(void *forWhom,double curWallTime);

//ArrayListeners:
//Iterate over the CkArrayListeners in this vector, calling "inside" each time.
#define CK_ARRAYLISTENER_LOOP(listVec,inside) \
  do { \
	int lIdx,lMax=listVec.size();\
	for (lIdx=0;lIdx<lMax;lIdx++) { \
		CkArrayListener *l=listVec[lIdx];\
		inside;\
	}\
  } while(0)

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
  void flushStates() { CkReductionMgr::flushStates(); CK_ARRAYLISTENER_LOOP(listeners, l->flushState()); }
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	// the mlogft only support 1D arrays, then returning the number of elements in the first dimension
	virtual int numberReductionMessages(){CkAssert(CkMyPe() == 0);return numInitial.data()[0];}
	void broadcastHomeElements(void *data,CkLocRec *rec,CkArrayIndex *index);
	static void staticBroadcastHomeElements(CkArray *arr,void *data,CkLocRec *rec,CkArrayIndex *index);
#endif

};



/*@}*/

#endif
