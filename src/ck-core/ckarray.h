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

	static CkGroupID ckCreateArray(int numInitial,CkGroupID mapID,CkArrayID boundToArray);
	static CkGroupID ckCreateArray1D(int ctorIndex,CkArrayMessage *m,
	     int numInitial,CkGroupID mapID);
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
	inline static CkGroupID ckCreateArray(int numInitial,CkGroupID mapID,CkArrayID boundToArray)\
	  { return super::ckCreateArray(numInitial,mapID,boundToArray); }\
	inline static CkGroupID ckCreateArray1D(int ctorIndex,CkArrayMessage *m,\
	     int numInitial,CkGroupID mapID)\
	  { return super::ckCreateArray1D(ctorIndex,m,numInitial,mapID); }\
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
		const CkArrayIndex &idx,CkGroupID dTo=-1)
		:CProxy_ArrayBase(aid,dTo), _idx(idx) { }
	
	void ckInsert(CkArrayMessage *m,int ctor,int onPe);
	void ckSend(CkArrayMessage *m, int ep) const;
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
	inline const CkArrayIndex &ckGetIndex() const \
	  { return super::ckGetIndex(); }\

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
  ArrayElementT(CkMigrateMessage *msg) {thisIndex=*(T *)thisIndexMax.data();}
  
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

class CkArrayCreateInfo {
 public:
	CkGroupID locMgrID;
	int numInitial;
	CkArrayCreateInfo() {}
	CkArrayCreateInfo(CkGroupID locMgrID_,int numInitial_)
		:locMgrID(locMgrID_), numInitial(numInitial_) { }
	void pup(PUP::er &p) {
		p|locMgrID;
		p|numInitial;
	}
};
PUPmarshall(CkArrayCreateInfo);

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
  CkArray(const CkArrayCreateInfo &c);
  CkGroupID &getGroupID(void) {return thisgroup;}

//Access & information routines
  inline CkLocMgr *getLocMgr(void) {return locMgr;}
  inline int getBcastNo(void) const {return bcastNo;}
  inline int homePe(const CkArrayIndex &idx) const {return locMgr->homePe(idx);}

  /* Return the last known processor for this array index.
   Valid for any possible array index. */
  inline int lastKnown(const CkArrayIndex &idx) const
	  {return locMgr->lastKnown(idx);}
  //Deliver message to this element (directly if local)
  inline void deliver(CkArrayMessage *m) 
	  {locMgr->deliver(m);}
  inline void deliverViaQueue(CkArrayMessage *m) 
	  {locMgr->deliverViaQueue(m);}
  //Fetch a local element via its index (return NULL if not local)
  inline ArrayElement *lookup(const CkArrayIndex &index)
	  {return (ArrayElement *)locMgr->lookup(index,thisgroup);}

//Creation:
  virtual CkMigratable *allocateMigrated(int elChareType,const CkArrayIndex &idx);
  virtual bool insertElement(CkArrayMessage *);
  virtual void doneInserting(void);

//Demand-creation:
  bool demandCreateElement(const CkArrayIndex &idx,int onPe,int ctor);

//Broadcast communication:
  void sendBroadcast(CkArrayMessage *msg);
  void recvBroadcast(CkArrayMessage *msg);
  
private:
  int numInitial;//Number of 1D initial array elements (backward-compatability)
  CmiBool isInserting;//Are we currently inserting elements?

//Allocate space for a new array element
  ArrayElement *allocate(int elChareType,const CkArrayIndex &idx,int bcast);
  
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
