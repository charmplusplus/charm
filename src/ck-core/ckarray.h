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
#include "ckreduction.h"

/***********************************************************
	Utility defines, includes, etc.
*/
extern void _registerCkArray(void);

#define ALIGN8(x)       (int)((~7)&((x)+7))

//Arguments for array creation:
class CkArrayOptions {
	int numInitial;//Number of elements to create
	CkGroupID map;//Array location map object
	CkGroupID locMgr;//Location manager to bind to
 public:
 //Used by external world:
	CkArrayOptions(void); //Default: empty array
	CkArrayOptions(int numInitial_); //With initial elements

	//These functions return a copy of this so you can string them together, e.g.:
	//  foo(CkArrayOptions().setMap(mid).bindTo(aid));

	//Create this many initial elements
	CkArrayOptions &setNumInitial(int ni)
		{numInitial=ni; return *this;}

	//Use this location map
	CkArrayOptions &setMap(const CkGroupID &m)
		{map=m; return *this;}

	//Bind our elements to this array
	CkArrayOptions &bindTo(const CkArrayID &b);
	
	//Use this location manager
	CkArrayOptions &setLocationManager(const CkGroupID &l)
		{locMgr=l; return *this;}
	
  //Used by the array manager:
	int getNumInitial(void) const {return numInitial;}
	const CkGroupID &getMap(void) const {return map;}
	const CkGroupID &getLocationManager(void) const {return locMgr;}

	void pup(PUP::er &p) {
		p|numInitial;
		p|locMgr;
		p|map;
	}
};
PUPmarshall(CkArrayOptions);


//This class is a wrapper around a CkArrayIndex and ArrayID,
// used by array element proxies.  This makes the translator's
// job simpler, and the translated code smaller. 
class CProxy_ArrayBase :public CProxyBase_Delegatable {
private:
	CkArrayID _aid;
public:
	CProxy_ArrayBase() { }
	CProxy_ArrayBase(const CkArrayID &aid,CkGroupID dTo) 
		:CProxyBase_Delegatable(dTo), _aid(aid) { }
	CProxy_ArrayBase(const CkArrayID &aid) 
		:CProxyBase_Delegatable(), _aid(aid) { }

	static CkArrayID ckCreateEmptyArray(void);
	static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,CkArrayOptions opts);

	void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx);	
	void ckBroadcast(CkArrayMessage *m, int ep) const;
	CkArrayID ckGetArrayID(void) const { return _aid; }
	CkArray *ckLocalBranch(void) const { return _aid.ckLocalBranch(); }	
	inline operator CkArrayID () const {return ckGetArrayID();}

	void doneInserting(void);
	void setReductionClient(CkReductionMgr::clientFn fn,void *param=NULL);

	void pup(PUP::er &p);
};
PUPmarshall(CProxy_ArrayBase);
#define CK_DISAMBIG_ARRAY(super) \
	CK_DISAMBIG_DELEGATABLE(super) \
	inline operator CkArrayID () const {return ckGetArrayID();}\
	inline static CkArrayID ckCreateEmptyArray(void)\
	  { return super::ckCreateEmptyArray(); }\
	inline static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts)\
	  { return super::ckCreateArray(m,ctor,opts); }\
	inline void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx) \
	  { super::ckInsertIdx(m,ctor,onPe,idx); }\
	inline void ckBroadcast(CkArrayMessage *m, int ep) const \
	  { super::ckBroadcast(m,ep); } \
	inline CkArrayID ckGetArrayID(void) const \
	  { return super::ckGetArrayID();} \
	inline CkArray *ckLocalBranch(void) const \
	  { return super::ckLocalBranch(); } \
	inline void doneInserting(void) { super::doneInserting(); }\
	inline void setReductionClient(CkReductionMgr::clientFn fn,void *param=NULL)\
	  { super::setReductionClient(fn,param); }\


class CProxyElement_ArrayBase:public CProxy_ArrayBase {
private:
	CkArrayIndexMax _idx;//<- our element's array index
public:
	CProxyElement_ArrayBase() { }
	CProxyElement_ArrayBase(const CkArrayID &aid,
		const CkArrayIndex &idx,CkGroupID dTo)
		:CProxy_ArrayBase(aid,dTo), _idx(idx) { }
	CProxyElement_ArrayBase(const CkArrayID &aid, const CkArrayIndex &idx)
		:CProxy_ArrayBase(aid), _idx(idx) { }
	
	void ckInsert(CkArrayMessage *m,int ctor,int onPe);
	void ckSend(CkArrayMessage *m, int ep) const;
	void *ckSendSync(CkArrayMessage *m, int ep) const;
	const CkArrayIndex &ckGetIndex() const {return _idx;}

	ArrayElement *ckLocal(void) const;
	void pup(PUP::er &p);
};
PUPmarshall(CProxyElement_ArrayBase);
#define CK_DISAMBIG_ARRAY_ELEMENT(super) \
	CK_DISAMBIG_ARRAY(super) \
	inline void ckInsert(CkArrayMessage *m,int ctor,int onPe) \
	  { super::ckInsert(m,ctor,onPe); }\
	inline void ckSend(CkArrayMessage *m, int ep) const \
	  { super::ckSend(m,ep); }\
	inline void *ckSendSync(CkArrayMessage *m, int ep) const \
	  { return super::ckSendSync(m,ep); }\
	inline const CkArrayIndex &ckGetIndex() const \
	  { return super::ckGetIndex(); }\


class CProxySection_ArrayBase:public CProxy_ArrayBase {
private:
	CkSectionID _sid;
public:
	CProxySection_ArrayBase() { }
	CProxySection_ArrayBase(const CkArrayID &aid,
		const CkArrayIndexMax *elems, const int nElems, CkGroupID dTo)
		:CProxy_ArrayBase(aid,dTo), _sid(aid, elems, nElems) { }
	CProxySection_ArrayBase(const CkArrayID &aid, 
		const CkArrayIndexMax *elems, const int nElems) 
		:CProxy_ArrayBase(aid), _sid(aid, elems, nElems) { }
	CProxySection_ArrayBase(const CkSectionID &sid)
		:CProxy_ArrayBase(sid._cookie.aid), _sid(sid){}
	CProxySection_ArrayBase(const CkSectionID &sid, CkGroupID dTo)
		:CProxy_ArrayBase(sid._cookie.aid, dTo), _sid(sid){}
	
	void ckInsert(CkArrayMessage *m,int ctor,int onPe);
	void ckSend(CkArrayMessage *m, int ep) ;

//	ArrayElement *ckLocal(void) const;
	inline CkSectionCookie &ckGetSectionCookie() {return _sid._cookie;}
	inline CkSectionID &ckGetSectionID() {return _sid;}
        inline const CkArrayIndexMax *ckGetArrayElements() const {return _sid._elems;}
	inline const int ckGetNumElements() const { return _sid._nElems; }
	void pup(PUP::er &p);
};
PUPmarshall(CProxySection_ArrayBase);
#define CK_DISAMBIG_ARRAY_SECTION(super) \
	CK_DISAMBIG_ARRAY(super) \
	inline void ckInsert(CkArrayMessage *m,int ctor,int onPe) \
	  { super::ckInsert(m,ctor,onPe); }\
	inline void ckSend(CkArrayMessage *m, int ep) \
	  { super::ckSend(m,ep); } \
        inline CkSectionCookie &ckGetSectionCookie() \
	  { return super::ckGetSectionCookie(); } \
        inline CkSectionID &ckGetSectionID() \
	  { return super::ckGetSectionID(); } \
        inline const CkArrayIndexMax *ckGetArrayElements() const \
	  { return super::ckGetArrayElements(); } \
        inline const int ckGetNumElements() const \
	  { return super::ckGetNumElements(); }  \


/************************ Array Element *********************/

class ArrayElement : public CkMigratable
{
  friend class CkArray;
  void initBasics(void);
public:
  ArrayElement(void);
  ArrayElement(CkMigrateMessage *m);
  virtual ~ArrayElement();
  
  int numElements; //Initial number of array elements (DEPRICATED)
  
//Contribute to the given reduction type.  Data is copied, not deleted.
  void contribute(int dataSize,void *data,CkReduction::reducerType type);

//Pack/unpack routine (called before and after migration)
  virtual void pup(PUP::er &p);

//Overridden functions:
  //Called by the system just before and after migration to another processor:  
  virtual void ckAboutToMigrate(void); 
  virtual void ckJustMigrated(void); 
  virtual void ckDestroy(void);

  //Synonym for ckMigrate
  inline void migrateMe(int toPe) {ckMigrate(toPe);}

protected:
  CkArray *thisArray;//My source array
  CkArrayID thisArrayID;//My source array's ID
  
private:
//Array implementation methods:   
  int bcastNo;//Number of broadcasts received (also serial number)
  CkReductionMgr::contributorInfo reductionInfo;//My reduction information
};

//An ArrayElementT is a utility class where you are 
// constrained to a "thisIndex" of some fixed-sized type T.
template <class T>
class ArrayElementT : public ArrayElement
{
public:
  ArrayElementT(void) {thisIndex=*(T *)thisIndexMax.data();}
  ArrayElementT(CkMigrateMessage *msg) 
	:ArrayElement(msg)
	{thisIndex=*(T *)thisIndexMax.data();}
  
  T thisIndex;//Object array index
};

typedef ArrayElementT<int> ArrayElement1D;

typedef struct {int x,y;} CkIndex2D;
void operator|(PUP::er &p,CkIndex2D &i);
typedef ArrayElementT<CkIndex2D> ArrayElement2D;

typedef struct {int x,y,z;} CkIndex3D;
void operator|(PUP::er &p,CkIndex3D &i);
typedef ArrayElementT<CkIndex3D> ArrayElement3D;


/*********************** Array Manager BOC *******************/

#include "CkArray.decl.h"

class CkArray : public CkReductionMgr, public CkArrMgr {
  friend class ArrayElement;
  friend class CProxy_ArrayBase;
  friend class CProxyElement_ArrayBase;

  CkMagicNumber<ArrayElement> magic; //To detect heap corruption
  CkLocMgr *locMgr;
  CProxy_CkArray thisproxy;
  typedef CkMigratableListT<ArrayElement> ArrayElementList;
  ArrayElementList *elements;  

public:
//Array Creation:
  CkArray(const CkArrayOptions &c,CkMarshalledMessage &initMsg);
  CkGroupID &getGroupID(void) {return thisgroup;}

//Access & information routines
  inline CkLocMgr *getLocMgr(void) {return locMgr;}
  inline int getBcastNo(void) const {return bcastNo;}
  inline int getNumInitial(void) const {return numInitial;}
  inline int homePe(const CkArrayIndex &idx) const {return locMgr->homePe(idx);}

  /* Return the last known processor for this array index.
   Valid for any possible array index. */
  inline int lastKnown(const CkArrayIndex &idx) const
	  {return locMgr->lastKnown(idx);}
  //Deliver message to this element (directly if local)
  inline void deliver(CkMessage *m) 
	  {locMgr->deliver(m);}
  inline void deliverViaQueue(CkMessage *m) 
	  {locMgr->deliverViaQueue(m);}
  //Fetch a local element via its index (return NULL if not local)
  inline ArrayElement *lookup(const CkArrayIndex &index)
	  {return (ArrayElement *)locMgr->lookup(index,thisgroup);}

//Creation:
  //Create-after-migrate:
  virtual CkMigratable *allocateMigrated(int elChareType,const CkArrayIndex &idx);
  
  //Create initial:
  virtual void insertInitial(const CkArrayIndex &idx,void *ctorMsg);
  virtual void doneInserting(void);
  void remoteDoneInserting(void);

  //Create manually:
  virtual bool insertElement(CkMessage *);

//Demand-creation:
  bool demandCreateElement(const CkArrayIndex &idx,int onPe,int ctor);

//Broadcast communication:
  void sendBroadcast(CkMessage *msg);
  void recvBroadcast(CkMessage *msg);
  
private:
  int numInitial;//Number of 1D initial array elements
  CmiBool isInserting;//Are we currently inserting elements?

//Allocate space for a new array element
  ArrayElement *allocate(int elChareType,const CkArrayIndex &idx,
	int bcast,bool fromMigration);
  
//Broadcast support
  int bcastNo;//Number of broadcasts received (also serial number)
  int oldBcastNo;//Above value last spring cleaning
  //This queue stores old broadcasts (in case a migrant arrives
  // and needs to be brought up to date)
  CkQ<CkArrayMessage *> oldBcasts;
  bool bringBroadcastUpToDate(ArrayElement *el);
  void deliverBroadcast(CkArrayMessage *bcast);
  bool deliverBroadcast(CkArrayMessage *bcast,ArrayElement *el);

//Spring cleaning
  void springCleaning(void);
  static void staticSpringCleaning(void *forWhom);
};

#endif
