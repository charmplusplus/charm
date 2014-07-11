/**
\file
\addtogroup CkArray

An Array is a collection of array elements (Chares) which
can be indexed by an arbitary run of bytes (a CkArrayIndex).
Elements can be inserted or removed from the array,
or migrated between processors.  Arrays are integrated with
the run-time load balancer.
Elements can also receive broadcasts and participate in
reductions.

Here's a list, valid in 2003/12, of all the different 
code paths used to create array elements:

1.) Initial inserts: all at once
CProxy_foo::ckNew(msg,n);
 CProxy_ArrayBase::ckCreateArray
  CkArray::CkArray
   CkLocMgr::populateInitial(numInitial) -> CkArrayMap::populateInitial(numInitial)
    for (idx=...)
     if (map->procNum(idx)==thisPe) 
      CkArray::insertInitial
       CkArray::prepareCtorMsg
       CkArray::insertElement
    // OR map-specific insertion logic

2.) Initial inserts: one at a time
fooProxy[idx].insert(msg,n);
 CProxy_ArrayBase::ckInsertIdx
  CkArray::prepareCtorMsg
  CkArray::insertElement

3.) Demand creation (receive side)
CkLocMgr::deliver
 CkLocMgr::deliverUnknown
  CkLocMgr::demandCreateElement
   CkArray::demandCreateElement
    CkArray::prepareCtorMsg
    CkArray::insertElement

4.) Migration (receive side)
CkLocMgr::migrateIncoming
 CkLocMgr::pupElementsFor
  CkArray::allocateMigrated



Converted from 1-D arrays 2/27/2000 by
Orion Sky Lawlor, olawlor@acm.org
*/
#include "charm++.h"
#include "register.h"
#include "ck.h"
#include "pathHistory.h"

CpvDeclare(int ,serializer);

bool _isAnytimeMigration;
bool _isStaticInsertion;
bool _isNotifyChildInRed;

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT 
#   define DEB(x) CkPrintf x  //General debug messages
#   define DEBI(x) CkPrintf x  //Index debug messages
#   define DEBC(x) CkPrintf x  //Construction debug messages
#   define DEBS(x) CkPrintf x  //Send/recv/broadcast debug messages
#   define DEBM(x) CkPrintf x  //Migration debug messages
#   define DEBL(x) CkPrintf x  //Load balancing debug messages
#   define DEBK(x) CkPrintf x  //Spring Cleaning debug messages
#   define DEBB(x) CkPrintf x  //Broadcast debug messages
#   define AA "ArrayBOC on %d: "
#   define AB ,CkMyPe()
#   define DEBUG(x) x
#else
#   define DEB(X) /*CkPrintf x*/
#   define DEBI(X) /*CkPrintf x*/
#   define DEBC(X) /*CkPrintf x*/
#   define DEBS(x) /*CkPrintf x*/
#   define DEBM(X) /*CkPrintf x*/
#   define DEBL(X) /*CkPrintf x*/
#   define DEBK(x) /*CkPrintf x*/
#   define DEBB(x) /*CkPrintf x*/
#   define str(x) /**/
#   define DEBUG(x)
#endif

///This arrayListener is in charge of delivering broadcasts to the array.
class CkArrayBroadcaster : public CkArrayListener {
  inline int &getData(ArrayElement *el) {return *ckGetData(el);}
public:
  CkArrayBroadcaster(bool _stableLocations, bool _broadcastViaScheduler);
  CkArrayBroadcaster(CkMigrateMessage *m);
  virtual void pup(PUP::er &p);
  virtual ~CkArrayBroadcaster();
  PUPable_decl(CkArrayBroadcaster);

  virtual void ckElementStamp(int *eltInfo) {*eltInfo=bcastNo;}

  ///Element was just created on this processor
  /// Return false if the element migrated away or deleted itself.
  virtual bool ckElementCreated(ArrayElement *elt)
    { return bringUpToDate(elt); }

  ///Element just arrived on this processor (so just called pup)
  /// Return false if the element migrated away or deleted itself.
  virtual bool ckElementArriving(ArrayElement *elt)
    { return bringUpToDate(elt); }

  void incoming(CkArrayMessage *msg);

  bool deliver(CkArrayMessage *bcast, ArrayElement *el, bool doFree);

  void springCleaning(void);

  void flushState();
private:
  int bcastNo;//Number of broadcasts received (also serial number)
  int oldBcastNo;//Above value last spring cleaning
  //This queue stores old broadcasts (in case a migrant arrives
  // and needs to be brought up to date)
  CkQ<CkArrayMessage *> oldBcasts;
  bool stableLocations;
  bool broadcastViaScheduler;

  bool bringUpToDate(ArrayElement *el);
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
  bool ckElementArriving(ArrayElement *elt)
    {mgr->contributorArriving(getData(elt)); return true; }
};

/*
void 
CProxyElement_ArrayBase::ckSendWrapper(void *me, void *m, int ep, int opts){
       ((CProxyElement_ArrayBase*)me)->ckSend((CkArrayMessage*)m,ep,opts);
}
*/
void
CProxyElement_ArrayBase::ckSendWrapper(CkArrayID _aid, CkArrayIndex _idx, void *m, int ep, int opts) {
	CProxyElement_ArrayBase me = CProxyElement_ArrayBase(_aid,_idx);
	((CProxyElement_ArrayBase)me).ckSend((CkArrayMessage*)m,ep,opts);
}

/*********************** CkVerboseListener ******************/
#define VL_PRINT ckout<<"VerboseListener on PE "<<CkMyPe()<<" > "

CkVerboseListener::CkVerboseListener(void)
  :CkArrayListener(0)
{
  VL_PRINT<<"INIT  Creating listener"<<endl;
}

void CkVerboseListener::ckRegister(CkArray *arrMgr,int dataOffset_)
{
  CkArrayListener::ckRegister(arrMgr,dataOffset_);
  VL_PRINT<<"INIT  Registering array manager at offset "<<dataOffset_<<endl;
}
void CkVerboseListener::ckBeginInserting(void)
{
  VL_PRINT<<"INIT  Begin inserting elements"<<endl;
}
void CkVerboseListener::ckEndInserting(void)
{
  VL_PRINT<<"INIT  Done inserting elements"<<endl;
}

void CkVerboseListener::ckElementStamp(int *eltInfo)
{
  VL_PRINT<<"LIFE  Stamping element"<<endl;
}
void CkVerboseListener::ckElementCreating(ArrayElement *elt)
{
  VL_PRINT<<"LIFE  About to create element "<<idx2str(elt)<<endl;
}
bool CkVerboseListener::ckElementCreated(ArrayElement *elt)
{
  VL_PRINT<<"LIFE  Created element "<<idx2str(elt)<<endl;
  return true;
}
void CkVerboseListener::ckElementDied(ArrayElement *elt)
{
  VL_PRINT<<"LIFE  Deleting element "<<idx2str(elt)<<endl;
}

void CkVerboseListener::ckElementLeaving(ArrayElement *elt)
{
  VL_PRINT<<"MIG  Leaving: element "<<idx2str(elt)<<endl;
}
bool CkVerboseListener::ckElementArriving(ArrayElement *elt)
{
  VL_PRINT<<"MIG  Arriving: element "<<idx2str(elt)<<endl;
  return true;
}

//Iterate over the CkArrayListeners in this vector, calling "inside" each time.
#define CK_ARRAYLISTENER_LOOP(listVec,inside) \
  do { \
	int lIdx,lMax=listVec.size();\
	for (lIdx=0;lIdx<lMax;lIdx++) { \
		CkArrayListener *l=listVec[lIdx];\
		inside;\
	}\
  } while(0)

/************************* ArrayElement *******************/
class ArrayElement_initInfo {
public:
  CkArray *thisArray;
  CkArrayID thisArrayID;
  CkArrayIndex numInitial;
  int listenerData[CK_ARRAYLISTENER_MAXLEN];
  bool fromMigration;
};

CkpvStaticDeclare(ArrayElement_initInfo,initInfo);

void ArrayElement::initBasics(void)
{
#if CMK_OUT_OF_CORE
  if (CkpvAccess(CkSaveRestorePrefetch)) 
    return; /* Just restoring from disk--don't try to set up anything. */
#endif
#if CMK_GRID_QUEUE_AVAILABLE
	grid_queue_interval = 0;
	grid_queue_threshold = 0;
	msg_count = 0;
	msg_count_grid = 0;
	border_flag = 0;

	grid_queue_interval = CmiGridQueueGetInterval ();
	grid_queue_threshold = CmiGridQueueGetThreshold ();
#endif
  ArrayElement_initInfo &info=CkpvAccess(initInfo);
  thisArray=info.thisArray;
  thisArrayID=info.thisArrayID;
  numInitialElements=info.numInitial.getCombinedCount();
  if (info.listenerData) {
    memcpy(listenerData,info.listenerData,sizeof(listenerData));
  }
  if (!info.fromMigration) {
    CK_ARRAYLISTENER_LOOP(thisArray->listeners,
			  l->ckElementCreating(this));
  }
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	if(mlogData == NULL)
		mlogData = new ChareMlogData();
	mlogData->objID.type = TypeArray;
	mlogData->objID.data.array.id = (CkGroupID)thisArrayID;
#endif
#ifdef _PIPELINED_ALLREDUCE_
	allredMgr = NULL;
#endif
}

ArrayElement::ArrayElement(void) 
{
	initBasics();
#if CMK_MEM_CHECKPOINT
        init_checkpt();
#endif
}

ArrayElement::ArrayElement(CkMigrateMessage *m) : CkMigratable(m)
{
	initBasics();
}

//Called by the system just before and after migration to another processor:  
void ArrayElement::ckAboutToMigrate(void) {
	CK_ARRAYLISTENER_LOOP(thisArray->listeners,
				l->ckElementLeaving(this));
	CkMigratable::ckAboutToMigrate();
}
void ArrayElement::ckJustMigrated(void) {
	CkMigratable::ckJustMigrated();
	CK_ARRAYLISTENER_LOOP(thisArray->listeners,
	      if (!l->ckElementArriving(this)) return;);
}

void ArrayElement::ckJustRestored(void) {
    CkMigratable::ckJustRestored();
    //empty for out-of-core emulation
}

#ifdef _PIPELINED_ALLREDUCE_
void ArrayElement::contribute2(int dataSize,const void *data,CkReduction::reducerType type,
					CMK_REFNUM_TYPE userFlag)
{
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);
	msg->setUserFlag(userFlag);
	msg->setMigratableContributor(true);
	thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}
void ArrayElement::contribute2(int dataSize,const void *data,CkReduction::reducerType type,
					const CkCallback &cb,CMK_REFNUM_TYPE userFlag)
{
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);
	msg->setUserFlag(userFlag);
	msg->setCallback(cb);
	msg->setMigratableContributor(true);
	thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}
void ArrayElement::contribute2(CkReductionMsg *msg) 
{
	msg->setMigratableContributor(true);
	thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}
void ArrayElement::contribute2(const CkCallback &cb,CMK_REFNUM_TYPE userFlag)
{
    CkReductionMsg *msg=CkReductionMsg::buildNew(0,NULL,CkReduction::nop);
    msg->setUserFlag(userFlag);
    msg->setCallback(cb);
    msg->setMigratableContributor(true);
    thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}
void ArrayElement::contribute2(CMK_REFNUM_TYPE userFlag)
{
    CkReductionMsg *msg=CkReductionMsg::buildNew(0,NULL,CkReduction::nop);
    msg->setUserFlag(userFlag);
    msg->setMigratableContributor(true);
    thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}

void ArrayElement::contribute2(CkArrayIndex myIndex, int dataSize,const void *data,CkReduction::reducerType type,
							  const CkCallback &cb,CMK_REFNUM_TYPE userFlag)
{
	// if it is a broadcast to myself and size is large
	if(cb.type==CkCallback::bcastArray && cb.d.array.id==thisArrayID && dataSize>FRAG_THRESHOLD) 
	{
		if (!allredMgr) {
			allredMgr = new AllreduceMgr();
		}
		// number of fragments
		int fragNo = dataSize/FRAG_SIZE;
		int size = FRAG_SIZE;
		// for each fragment
		for (int i=0; i<fragNo; i++) {
			// callback to defragmentor
			CkCallback defrag_cb(CkIndex_ArrayElement::defrag(NULL), thisArrayID);
			if ((0 != i) && ((fragNo-1) == i) && (0 != dataSize%FRAG_SIZE)) {
				size = dataSize%FRAG_SIZE;
			}
			CkReductionMsg *msg = CkReductionMsg::buildNew(size, (char*)data+i*FRAG_SIZE);
			// initialize the new msg
			msg->reducer            = type;
			msg->nFrags             = fragNo;
			msg->fragNo             = i;
			msg->callback           = defrag_cb;
			msg->userFlag           = userFlag;
			allredMgr->cb		= cb;
			allredMgr->cb.type	= CkCallback::sendArray;
			allredMgr->cb.d.array.idx = myIndex;
			contribute2(msg);
		}
		return;
	}
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);
	msg->setUserFlag(userFlag);
	msg->setCallback(cb);
	msg->setMigratableContributor(true);
	thisArray->contribute(&*(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],msg);
}


#else
CK_REDUCTION_CONTRIBUTE_METHODS_DEF(ArrayElement,thisArray,
   *(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],true)
#endif
// _PIPELINED_ALLREDUCE_
void ArrayElement::defrag(CkReductionMsg *msg)
{
//	CkPrintf("in defrag\n");
#ifdef _PIPELINED_ALLREDUCE_
	allredMgr->allreduce_recieve(msg);
#endif
}

/// Remote method: calls destructor
void ArrayElement::ckDestroy(void)
{
	if(_BgOutOfCoreFlag!=1){ //in case of taking core out of memory
	    CK_ARRAYLISTENER_LOOP(thisArray->listeners,
			   l->ckElementDied(this));
	}
	CkMigratable::ckDestroy();
}

//Destructor (virtual)
ArrayElement::~ArrayElement()
{
#if CMK_OUT_OF_CORE
  if (CkpvAccess(CkSaveRestorePrefetch)) 
    return; /* Just saving to disk--don't trash anything. */
#endif
  //To detect use-after-delete: 
  thisArray=(CkArray *)0xDEADa7a1;
}

void ArrayElement::pup(PUP::er &p)
{
  DEBM((AA "  ArrayElement::pup()\n" AB));
  CkMigratable::pup(p);
  thisArrayID.pup(p);
  if (p.isUnpacking())
  	thisArray=thisArrayID.ckLocalBranch();
  p(listenerData,CK_ARRAYLISTENER_MAXLEN);
#if CMK_MEM_CHECKPOINT
  p(budPEs, 2);
#endif
  p.syncComment(PUP::sync_last_system,"ArrayElement");
#if CMK_GRID_QUEUE_AVAILABLE
  p|grid_queue_interval;
  p|grid_queue_threshold;
  p|msg_count;
  p|msg_count_grid;
  p|border_flag;
  if (p.isUnpacking ()) {
    msg_count = 0;
    msg_count_grid = 0;
    border_flag = 0;
  }
#endif
}

char *ArrayElement::ckDebugChareName(void) {
	char buf[200];
	const char *className=_chareTable[ckGetChareType()]->name;
	const int *d=thisIndexMax.data();
	const short int *s=(const short int*)d;
	switch (thisIndexMax.dimension) {
	case 0:	sprintf(buf,"%s",className); break;
	case 1: sprintf(buf,"%s[%d]",className,d[0]); break;
	case 2: sprintf(buf,"%s(%d,%d)",className,d[0],d[1]); break;
	case 3: sprintf(buf,"%s(%d,%d,%d)",className,d[0],d[1],d[2]); break;
    case 4: sprintf(buf,"%s(%hd,%hd,%hd,%hd)",className,s[0],s[1],s[2],s[3]); break;
    case 5: sprintf(buf,"%s(%hd,%hd,%hd,%hd,%hd)",className,s[0],s[1],s[2],s[3],s[4]); break;
    case 6: sprintf(buf,"%s(%hd,%hd,%hd,%hd,%hd,%hd)",className,s[0],s[1],s[2],s[3],s[4],s[5]); break;
	default: sprintf(buf,"%s(%d,%d,%d,%d..)",className,d[0],d[1],d[2],d[3]); break;
	};
	return strdup(buf);
}

int ArrayElement::ckDebugChareID(char *str, int limit) {
  if (limit<21) return -1;
  str[0] = 2;
  *((int*)&str[1]) = ((CkGroupID)thisArrayID).idx;
  *((CkArrayIndex*)&str[5]) = thisIndexMax;
  return 21;
}

/// A more verbose form of abort
void ArrayElement::CkAbort(const char *str) const
{
	CkError("[%d] Array element at index %s aborting:\n",
		CkMyPe(), idx2str(thisIndexMax));
	CkMigratable::CkAbort(str);
}

void ArrayElement::recvBroadcast(CkMessage *m){
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	CkArrayMessage *bcast = (CkArrayMessage *)m;
    envelope *env = UsrToEnv(m);
	int epIdx= env->piggyBcastIdx;
    ckInvokeEntry(epIdx,bcast,true);
#endif
}

/*********************** Spring Cleaning *****************
Periodically (every minute or so) remove expired broadcasts
from the queue.

This does not get called for arrays with stable locations (all
insertions done at creation, migration only at discrete points).
*/

inline void CkArray::springCleaning(void)
{
  DEBK((AA "Starting spring cleaning\n" AB));
  broadcaster->springCleaning();
  setupSpringCleaning();
}

void CkArray::staticSpringCleaning(void *forArray,double curWallTime) {
	((CkArray *)forArray)->springCleaning();
}

void CkArray::setupSpringCleaning() {
 // set up broadcast cleaner
 if (!stableLocations)
      springCleaningCcd = CcdCallOnCondition(CcdPERIODIC_1minute,
                                             staticSpringCleaning, (void *)this);
}

/********************* Little CkArray Utilities ******************/

CProxy_ArrayBase::CProxy_ArrayBase(const ArrayElement *e)
	:CProxy(), _aid(e->ckGetArrayID())
	{}
CProxyElement_ArrayBase::CProxyElement_ArrayBase(const ArrayElement *e)
	:CProxy_ArrayBase(e), _idx(e->ckGetArrayIndex())
	{}

CkLocMgr *CProxy_ArrayBase::ckLocMgr(void) const
	{return ckLocalBranch()->getLocMgr(); }

CK_REDUCTION_CLIENT_DEF(CProxy_ArrayBase,ckLocalBranch())

CkArrayOptions::CkArrayOptions(void) //Default: empty array
	: numInitial(), start(), end(), step(), bounds(), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1) //With initial elements (1D)
	: start(CkArrayIndex1D(0)), end(CkArrayIndex1D(ni1)), step(CkArrayIndex1D(1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1, int ni2) //With initial elements (2D)
	: start(CkArrayIndex2D(0,0)), end(CkArrayIndex2D(ni1, ni2)), step(CkArrayIndex2D(1,1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1, int ni2, int ni3) //With initial elements (3D)
	: start(CkArrayIndex3D(0,0,0)), end(CkArrayIndex3D(ni1, ni2, ni3)), step(CkArrayIndex3D(1,1,1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(short int ni1, short int ni2, short int ni3,
                               short int ni4) //With initial elements (4D)
	: start(CkArrayIndex4D(0,0,0,0)),
	  end(CkArrayIndex4D(ni1, ni2, ni3, ni4)),
	  step(CkArrayIndex4D(1,1,1,1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(short int ni1, short int ni2, short int ni3,
                               short int ni4, short int ni5) //With initial elements (5D)
	: start(CkArrayIndex5D(0,0,0,0,0)),
	  end(CkArrayIndex5D(ni1, ni2, ni3, ni4, ni5)),
	  step(CkArrayIndex5D(1,1,1,1,1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(short int ni1, short int ni2, short int ni3,
                               short int ni4, short int ni5, short int ni6) //With initial elements (6D)
	: start(CkArrayIndex6D(0,0,0,0,0,0)),
	  end(CkArrayIndex6D(ni1, ni2, ni3, ni4, ni5, ni6)),
	  step(CkArrayIndex6D(1,1,1,1,1,1)),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(CkArrayIndex s, CkArrayIndex e, CkArrayIndex step)
	: start(s), end(e), step(step),
	  numInitial(end), bounds(end), map(_defaultArrayMapID)
{
    init();
}

void CkArrayOptions::init()
{
    locMgr.setZero();
    anytimeMigration = _isAnytimeMigration;
    staticInsertion = _isStaticInsertion;
    reductionClient.type = CkCallback::invalid;
    disableNotifyChildInRed = !_isNotifyChildInRed;
    broadcastViaScheduler = false;
}

CkArrayOptions &CkArrayOptions::setStaticInsertion(bool b)
{
    staticInsertion = b;
    if (b && map == _defaultArrayMapID)
	map = _fastArrayMapID;
    return *this;
}

/// Bind our elements to this array
CkArrayOptions &CkArrayOptions::bindTo(const CkArrayID &b)
{
	CkArray *arr=CProxy_CkArray(b).ckLocalBranch();
	//Stupid bug: need a way for arrays to stay the same size *FOREVER*,
	// not just initially.
	//setNumInitial(arr->getNumInitial());
	return setLocationManager(arr->getLocMgr()->getGroupID());
}

CkArrayOptions &CkArrayOptions::addListener(CkArrayListener *listener)
{
	arrayListeners.push_back(listener);
	return *this;
}

void CkArrayOptions::updateIndices() {
	bool shorts = numInitial.dimension > 3;
	start = step = end = numInitial;

	for (int d = 0; d < numInitial.dimension; d++) {
		if (shorts) {
			((short*)start.data())[d] = 0;
			((short*)step.data())[d] = 1;
		} else {
			start.data()[d] = 0;
			step.data()[d] = 1;
		}
	}
}

void CkArrayOptions::updateNumInitial() {
	if (end.dimension != start.dimension || end.dimension != step.dimension) {
		return;
	}

	bool shorts = end.dimension > 3;
	numInitial = end;
	for (int d = 0; d < end.dimension; d++) {
		int diff, increment, num;

		// Extract the current dimension of the indices
		if (shorts) {
			diff = ((short*)end.data())[d] - ((short*)start.data())[d];
			increment = ((short*)step.data())[d];
		} else {
			diff = end.data()[d] - start.data()[d];
			increment = step.data()[d];
		}

		// Compute the number of initial elements in this dimension
		num = diff / increment;
		if (diff < 0) {
			num = 0;
		} else if (diff % increment > 0) {
			num++;
		}

		// Set the current dimension of numInitial
		if (shorts) {
			((short*)numInitial.data())[d] = (short)num;
		} else {
			numInitial.data()[d] = num;
		}
	}
}

void CkArrayOptions::pup(PUP::er &p) {
	p|start;
	p|end;
	p|step;
	p|numInitial;
	p|bounds;
	p|map;
	p|locMgr;
	p|arrayListeners;
	p|reductionClient;
	p|anytimeMigration;
	p|disableNotifyChildInRed;
	p|staticInsertion;
	p|broadcastViaScheduler;
}

CkArrayListener::CkArrayListener(int nInts_) 
  :nInts(nInts_) 
{
  dataOffset=-1;
}
CkArrayListener::CkArrayListener(CkMigrateMessage *m) {
  nInts=-1; dataOffset=-1;
}
void CkArrayListener::pup(PUP::er &p) {
  p|nInts;
  p|dataOffset;
}

void CkArrayListener::ckRegister(CkArray *arrMgr,int dataOffset_)
{
  if (dataOffset!=-1) CkAbort("Cannot register an ArrayListener twice!\n");
  dataOffset=dataOffset_;
}

static CkArrayID CkCreateArray(CkArrayMessage *m, int ctor, CkArrayOptions opts)
{
  //CkAssert(CkMyPe() == 0); // Will become mandatory under 64-bit ID

  CkGroupID locMgr = opts.getLocationManager();
  if (locMgr.isZero())
  { //Create a new location manager
    CkEntryOptions  e_opts;
    e_opts.setGroupDepID(opts.getMap());       // group creation dependence
    locMgr = CProxy_CkLocMgr::ckNew(opts, &e_opts);
    opts.setLocationManager(locMgr);
  }
  //Create the array manager
  m->array_ep()=ctor;
  CkMarshalledMessage marsh(m);
  CkEntryOptions  e_opts;
  e_opts.setGroupDepID(locMgr);       // group creation dependence
#if !GROUP_LEVEL_REDUCTION
  CProxy_CkArrayReductionMgr nodereductionProxy = CProxy_CkArrayReductionMgr::ckNew();
  CkGroupID ag=CProxy_CkArray::ckNew(opts,marsh,nodereductionProxy,&e_opts);
  nodereductionProxy.setAttachedGroup(ag);
#else
  CkNodeGroupID dummyid;
  CkGroupID ag=CProxy_CkArray::ckNew(opts,marsh,dummyid,&e_opts);
#endif
  return (CkArrayID)ag;
}

CkArrayID CProxy_ArrayBase::ckCreateArray(CkArrayMessage *m,int ctor,
					  const CkArrayOptions &opts)
{
  return CkCreateArray(m, ctor, opts);
}

CkArrayID CProxy_ArrayBase::ckCreateEmptyArray(CkArrayOptions opts)
{
  return ckCreateArray((CkArrayMessage *)CkAllocSysMsg(),0,opts);
}

void CProxy_ArrayBase::ckCreateEmptyArrayAsync(CkCallback cb, CkArrayOptions opts)
{
  CkSendAsyncCreateArray(0, cb, opts, (CkArrayMessage *)CkAllocSysMsg());
}

extern IrrGroup *lookupGroupAndBufferIfNotThere(CkCoreState *ck,envelope *env,const CkGroupID &groupID);

struct CkInsertIdxMsg {
  char core[CmiReservedHeaderSize];
  CkArrayIndex idx;
  CkArrayMessage *m;
  int ctor;
  int onPe;
  CkArrayID _aid;
};

static int ckinsertIdxHdl;

void ckinsertIdxFunc(void *m)
{
  CkInsertIdxMsg *msg = (CkInsertIdxMsg *)m;
  CProxy_ArrayBase   ca(msg->_aid);
  ca.ckInsertIdx(msg->m, msg->ctor, msg->onPe, msg->idx);
  CmiFree(msg);
}

void CProxy_ArrayBase::ckInsertIdx(CkArrayMessage *m,int ctor,int proposedPe,
	const CkArrayIndex &idx)
{
  if (m==NULL) m=(CkArrayMessage *)CkAllocSysMsg();
  m->array_ep()=ctor;
  CkArray *ca = ckLocalBranch();
  if (ca == NULL) {
      CkInsertIdxMsg *msg = (CkInsertIdxMsg *)CmiAlloc(sizeof(CkInsertIdxMsg));
      msg->idx = idx;
      msg->m = m;
      msg->ctor = ctor;
      msg->onPe = proposedPe;
      msg->_aid = _aid;
      CmiSetHandler(msg, ckinsertIdxHdl);
      ca = (CkArray *)lookupGroupAndBufferIfNotThere(CkpvAccess(_coreState), (envelope*)msg,_aid);
      if (ca == NULL) return;
  }

  int hostPe = ca->findInitialHostPe(idx, proposedPe);

  int listenerData[CK_ARRAYLISTENER_MAXLEN];
  ca->prepareCtorMsg(m, listenerData);
  if (ckIsDelegated()) {
    ckDelegatedTo()->ArrayCreate(ckDelegatedPtr(),ctor,m,idx,hostPe,_aid);
    return;
  }
  
  DEBC((AA "Proxy inserting element %s on Pe %d\n" AB,idx2str(idx),hostPe));
  CProxy_CkArray(_aid)[hostPe].insertElement(m, idx, listenerData);
}

void CProxyElement_ArrayBase::ckInsert(CkArrayMessage *m,int ctorIndex,int onPe)
{
  ckInsertIdx(m,ctorIndex,onPe,_idx);
}

ArrayElement *CProxyElement_ArrayBase::ckLocal(void) const
{
  return ckLocalBranch()->lookup(_idx);
}

//pack-unpack method for CProxy_ArrayBase
void CProxy_ArrayBase::pup(PUP::er &p)
{
  CProxy::pup(p);
  _aid.pup(p);
}
void CProxyElement_ArrayBase::pup(PUP::er &p)
{
  CProxy_ArrayBase::pup(p);
  p|_idx.nInts;
  p|_idx.dimension;
  p(_idx.data(),_idx.nInts);
}

void CProxySection_ArrayBase::pup(PUP::er &p)
{
  CProxy_ArrayBase::pup(p);
  p | _nsid;
  if (p.isUnpacking()) {
    if (_nsid == 1) _sid = new CkSectionID;
    else if (_nsid > 1) _sid = new CkSectionID[_nsid];
    else _sid = NULL;
  }
  for (int i=0; i<_nsid; ++i) _sid[i].pup(p);
}

/*
 * Message type and code to create new chare arrays asynchronously.
 * Post-startup, whatever non-0 PE calls for the creation of an array will pack
 * up all of the arguments and send them to PE 0. PE 0 will then run the normal
 * creation process and send the array ID to the provided callback. This
 * ensures that up to the limit of available bits, array IDs can be represented
 * as part of a compound fixed-size ID for their elements.
 */
struct CkCreateArrayAsyncMsg : public CMessage_CkCreateArrayAsyncMsg {
  int ctor;
  CkCallback cb;
  CkArrayOptions opts;
  char *ctorPayload;

  CkCreateArrayAsyncMsg(int ctor_, CkCallback cb_, CkArrayOptions opts_)
    : ctor(ctor_), cb(cb_), opts(opts_)
  { }
};

static int ckArrayCreationHdl = 0;

void CkSendAsyncCreateArray(int ctor, CkCallback cb, CkArrayOptions opts, void *ctorMsg)
{
  CkAssert(ctorMsg);
  UsrToEnv(ctorMsg)->setMsgtype(ArrayEltInitMsg);
  PUP::sizer ps;
  CkPupMessage(ps, &ctorMsg);
  CkCreateArrayAsyncMsg *msg = new (ps.size()) CkCreateArrayAsyncMsg(ctor, cb, opts);
  PUP::toMem p(msg->ctorPayload);
  CkPupMessage(p, &ctorMsg);
  envelope *env = UsrToEnv(msg);
  CmiSetHandler(env, ckArrayCreationHdl);
  CkPackMessage(&env);
  CmiSyncSendAndFree(0, env->getTotalsize(), (char*)env);
}

static void CkCreateArrayAsync(void *vmsg)
{
  envelope *venv = static_cast<envelope*>(vmsg);
  CkUnpackMessage(&venv);
  CkCreateArrayAsyncMsg *msg = static_cast<CkCreateArrayAsyncMsg*>(EnvToUsr(venv));

  // Unpack arguments
  PUP::fromMem p(msg->ctorPayload);
  void *vm;
  CkPupMessage(p, &vm);
  CkArrayMessage *m = static_cast<CkArrayMessage*>(vm);

  // Does the caller care about the constructed array ID?
  if (!msg->cb.isInvalid()) {
    CkArrayCreatedMsg *response = new CkArrayCreatedMsg;
    response->aid = CkCreateArray(m, msg->ctor, msg->opts);

    msg->cb.send(response);
  } else {
    CkCreateArray(m, msg->ctor, msg->opts);
  }
}

/*********************** CkArray Creation *************************/
void _ckArrayInit(void)
{
  CkpvInitialize(ArrayElement_initInfo,initInfo);
  CkDisableTracing(CkIndex_CkArray::insertElement(0, CkArrayIndex(), 0));
  CkDisableTracing(CkIndex_CkArray::recvBroadcast(0));
    // disable because broadcast listener may deliver broadcast message
  CkDisableTracing(CkIndex_CkLocMgr::immigrate(0));
  // by default anytime migration is allowed
  ckinsertIdxHdl = CkRegisterHandler(ckinsertIdxFunc);
  ckArrayCreationHdl = CkRegisterHandler(CkCreateArrayAsync);
}

CkArray::CkArray(CkArrayOptions &opts,
		 CkMarshalledMessage &initMsg,
		 CkNodeGroupID nodereductionID)
  : CkReductionMgr(nodereductionID),
    locMgr(CProxy_CkLocMgr::ckLocalBranch(opts.getLocationManager())),
    locMgrID(opts.getLocationManager()),
    thisProxy(thisgroup),
    stableLocations(opts.staticInsertion && !opts.anytimeMigration),
    numInitial(opts.getNumInitial()), isInserting(true)
{
  // Register with our location manager
  locMgr->addManager(thisgroup,this);

  setupSpringCleaning();
  
  //set the field in one my parent class (CkReductionMgr)
  if(opts.disableNotifyChildInRed)
	  disableNotifyChildrenStart = true; 
  
  //Find, register, and initialize the arrayListeners
  listenerDataOffset=0;
  broadcaster=new CkArrayBroadcaster(stableLocations, opts.broadcastViaScheduler);
  addListener(broadcaster);
  reducer=new CkArrayReducer(thisgroup);
  addListener(reducer);

  // COMLIB HACK
  //calistener = new ComlibArrayListener();
  //addListener(calistener,dataOffset);

  int lNo,nL=opts.getListeners(); //User-added listeners
  for (lNo=0;lNo<nL;lNo++) addListener(opts.getListener(lNo));

  for (int l=0;l<listeners.size();l++) listeners[l]->ckBeginInserting();

  ///Set up initial elements (if any)
  locMgr->populateInitial(opts,initMsg.getMessage(),this);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	// creating the spanning tree to be used for broadcast
	children = (int *) CmiAlloc(sizeof(int) * _MLOG_BCAST_BFACTOR_);
	numChildren = 0;
	
	// computing the level of the tree this pe is in
	// we should use the geometric series formula, but now a quick and dirty code should suffice
	// PE 0 is at level 0, PEs 1.._MLOG_BCAST_BFACTOR_ are at level 1 and so on
	int level = 0;
	int aux = CmiMyPe();
	int max = CmiNumPes();
	int factor = _MLOG_BCAST_BFACTOR_;
	int startLevel = 0;
	int startNextLevel = 1;
	while(aux >= 0){
		level++;
		startLevel = startNextLevel;
		startNextLevel += factor;
		aux -= factor;
		factor *= _MLOG_BCAST_BFACTOR_;
	}

	// adding children to the tree
	int first = startNextLevel + (CmiMyPe() - startLevel) * _MLOG_BCAST_BFACTOR_;
	for(int i=0; i<_MLOG_BCAST_BFACTOR_; i++){
		if(first + i >= CmiNumPes())
			break;
		children[i] = first + i;
		numChildren++;
	}
 
#endif


  if (opts.reductionClient.type != CkCallback::invalid && CkMyPe() == 0)
      ckSetReductionClient(&opts.reductionClient);
}

CkArray::CkArray(CkMigrateMessage *m)
	:CkReductionMgr(m), thisProxy(thisgroup)
{
  locMgr=NULL;
  isInserting=true;
}

CkArray::~CkArray()
{
  if (!stableLocations)
    CcdCancelCallOnCondition(CcdPERIODIC_1minute, springCleaningCcd);
}

#if CMK_ERROR_CHECKING
inline void testPup(PUP::er &p,int shouldBe) {
  int a=shouldBe;
  p|a;
  if (a!=shouldBe)
    CkAbort("PUP direction mismatch!");
}
#else
inline void testPup(PUP::er &p,int shouldBe) {}
#endif

void CkArray::pup(PUP::er &p){
	CkReductionMgr::pup(p);
	p|numInitial;
	p|locMgrID;
	p|listeners;
	p|listenerDataOffset;
        p|stableLocations;
	testPup(p,1234);
	if(p.isUnpacking()){
		thisProxy=thisgroup;
		locMgr = CProxy_CkLocMgr::ckLocalBranch(locMgrID);
		locMgr->addManager(thisgroup,this);
		/// Restore our default listeners:
		broadcaster=(CkArrayBroadcaster *)(CkArrayListener *)(listeners[0]);
		reducer=(CkArrayReducer *)(CkArrayListener *)(listeners[1]);
                setupSpringCleaning();
	}
}

#define CK_ARRAYLISTENER_STAMP_LOOP(listenerData) do {\
  int dataOffset=0; \
  for (int lNo=0;lNo<listeners.size();lNo++) { \
    CkArrayListener *l=listeners[lNo]; \
    l->ckElementStamp(&listenerData[dataOffset]); \
    dataOffset+=l->ckGetLen(); \
  } \
} while (0)

//Called on send side to prepare array constructor message
void CkArray::prepareCtorMsg(CkMessage *m, int *listenerData)
{
  envelope *env=UsrToEnv((void *)m);
  env->setMsgtype(ArrayEltInitMsg);
  CK_ARRAYLISTENER_STAMP_LOOP(listenerData);
}

int CkArray::findInitialHostPe(const CkArrayIndex &idx, int proposedPe)
{
  int hostPe = locMgr->whichPE(idx);

  if (hostPe == -1 && proposedPe == -1)
    return procNum(idx);
  if (hostPe == -1)
    return proposedPe;
  if (proposedPe == -1)
    return hostPe;
  if (hostPe == proposedPe)
    return hostPe;

  CkAbort("hostPe for a bound element disagrees with an explicit proposedPe");
}

CkMigratable *CkArray::allocateMigrated(int elChareType, CkElementCreation_t type)
{
	ArrayElement *ret=allocate(elChareType, NULL, true, NULL);
	if (type==CkElementCreation_resume) 
	{ // HACK: Re-stamp elements on checkpoint resume--
	  //  this restores, e.g., reduction manager's gcount
		int *listenerData=ret->listenerData;
		CK_ARRAYLISTENER_STAMP_LOOP(listenerData);
	}
	return ret;
}

ArrayElement *CkArray::allocate(int elChareType, CkMessage *msg, bool fromMigration, int *listenerData)
{
	//Stash the element's initialization information in the global "initInfo"
	ArrayElement_initInfo &init=CkpvAccess(initInfo);
	init.numInitial=numInitial;
	init.thisArray=this;
	init.thisArrayID=thisgroup;
	if (listenerData) /*Have to *copy* data because msg will be deleted*/
	  memcpy(init.listenerData, listenerData, sizeof(init.listenerData));
	init.fromMigration=fromMigration;
	
	//Build the element
	int elSize=_chareTable[elChareType]->size;
	ArrayElement *elem = (ArrayElement *)malloc(elSize);
	if (elem!=NULL) setMemoryTypeChare(elem);
	return elem;
}

void CkArray::insertElement(CkMarshalledMessage &m, const CkArrayIndex &idx, int listenerData[CK_ARRAYLISTENER_MAXLEN])
{
  insertElement((CkArrayMessage*)m.getMessage(), idx, listenerData);
}

/// This method is called by ck.C or the user to add an element.
bool CkArray::insertElement(CkArrayMessage *me, const CkArrayIndex &idx, int listenerData[CK_ARRAYLISTENER_MAXLEN])
{
  CK_MAGICNUMBER_CHECK
  int onPe;
  if (locMgr->isRemote(idx,&onPe)) 
  { /* element's sibling lives somewhere else, so insert there */
    thisProxy[onPe].insertElement(me, idx, listenerData);
    return false;
  }
  int ctorIdx = me->array_ep();
  int chareType=_entryTable[ctorIdx]->chareIdx;
  ArrayElement *elt=allocate(chareType, me, false, listenerData);
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)elt)->chareIdx = -1;
#endif
  if (!locMgr->addElement(thisgroup, idx, elt, ctorIdx, (void *)me)) return false;
  CK_ARRAYLISTENER_LOOP(listeners,
      if (!l->ckElementCreated(elt)) return false;);
  return true;
}

void CProxy_ArrayBase::doneInserting(void)
{
  DEBC((AA "Broadcasting a doneInserting request\n" AB));
  //Broadcast a DoneInserting
  CProxy_CkArray(_aid).remoteDoneInserting();
}

void CProxy_ArrayBase::beginInserting(void)
{
  DEBC((AA "Broadcasting a beginInserting request\n" AB));
  CProxy_CkArray(_aid).remoteBeginInserting();
}

void CkArray::doneInserting(void)
{
  thisProxy[CkMyPe()].remoteDoneInserting();
}

void CkArray::beginInserting(void)
{
  thisProxy[CkMyPe()].remoteBeginInserting();
}

/// This is called on every processor after the last array insertion.
void CkArray::remoteDoneInserting(void)
{
  CK_MAGICNUMBER_CHECK
  if (isInserting) {
    isInserting=false;
    DEBC((AA "Done inserting objects\n" AB));
    for (int l=0;l<listeners.size();l++) listeners[l]->ckEndInserting();
    locMgr->doneInserting();
  }
}

void CkArray::remoteBeginInserting(void)
{
  CK_MAGICNUMBER_CHECK;

  if (!isInserting) {
    isInserting = true;
    DEBC((AA "Begin inserting objects\n" AB));
    for (int l=0;l<listeners.size();l++) listeners[l]->ckBeginInserting();
    locMgr->startInserting();
  }
}

bool CkArray::demandCreateElement(const CkArrayIndex &idx, int onPe, int ctor, CkDeliver_t type)
{
	CkArrayMessage *m=(CkArrayMessage *)CkAllocSysMsg();
        envelope *env = UsrToEnv(m);
        env->setMsgtype(ArrayEltInitMsg);
        env->setArrayMgr(thisgroup);
        int listenerData[CK_ARRAYLISTENER_MAXLEN];
	prepareCtorMsg(m, listenerData);
	m->array_ep()=ctor;
	
	if ((onPe!=CkMyPe()) || (type==CkDeliver_queue)) {
		DEBC((AA "Forwarding demand-creation request for %s to %d\n" AB,idx2str(idx),onPe));
		thisProxy[onPe].insertElement(m, idx, listenerData);
	} else /* local message, non-queued */ {
		//Call local constructor directly
		DEBC((AA "Demand-creating %s\n" AB,idx2str(idx)));
		return insertElement(m, idx, listenerData);
	}
	return true;
}

void CkArray::insertInitial(const CkArrayIndex &idx,void *ctorMsg)
{
	CkArrayMessage *m=(CkArrayMessage *)ctorMsg;
        int listenerData[CK_ARRAYLISTENER_MAXLEN];
	prepareCtorMsg(m, listenerData);
#if CMK_BIGSIM_CHARM
        BgEntrySplit("split-array-new");
#endif
        insertElement(m, idx, listenerData);
}

/********************* CkArray Messaging ******************/
/// Fill out a message's array fields before sending it
inline void msg_prepareSend(CkArrayMessage *msg, int ep,CkArrayID aid)
{
	envelope *env=UsrToEnv((void *)msg);
        env->setMsgtype(ForArrayEltMsg);
	env->setArrayMgr(aid);
	env->getsetArraySrcPe()=CkMyPe();
        env->setRecipientID(ck::ObjID(0));
#if CMK_SMP_TRACE_COMMTHREAD
        env->setSrcPe(CkMyPe());
#endif
	env->setEpIdx(ep);
	env->getsetArrayHops()=0;
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
	criticalPath_send(env);
	automaticallySetMessagePriority(env);
#endif
}


/// Just a non-inlined version of msg_prepareSend()
void msg_prepareSend_noinline(CkArrayMessage *msg, int ep,CkArrayID aid)
{
	envelope *env=UsrToEnv((void *)msg);
	env->setArrayMgr(aid);
	env->getsetArraySrcPe()=CkMyPe();
#if CMK_SMP_TRACE_COMMTHREAD
        env->setSrcPe(CkMyPe());
#endif
	env->setEpIdx(ep);
	env->getsetArrayHops()=0;
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
	criticalPath_send(env);
	automaticallySetMessagePriority(env);
#endif
}

void CProxyElement_ArrayBase::ckSend(CkArrayMessage *msg, int ep, int opts) const
{
#if CMK_ERROR_CHECKING
	//Check our array index for validity
	if (_idx.nInts<0) CkAbort("Array index length is negative!\n");
	if (_idx.nInts>CK_ARRAYINDEX_MAXLEN)
		CkAbort("Array index length (nInts) is too long-- did you "
			"use bytes instead of integers?\n");
#endif
	msg_prepareSend(msg, ep, ckGetArrayID());
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySend(ckDelegatedPtr(),ep,msg,_idx,ckGetArrayID());
	else 
	{ //Usual case: a direct send
	  CkArray *localbranch = ckLocalBranch();
	  if (localbranch == NULL) { // array not created yet
	    CkAbort("Cannot send a message from an array without a local branch");
	  }
	  else {
	    if (opts & CK_MSG_INLINE)
	      localbranch->deliver(msg, _idx, CkDeliver_inline, opts & (~CK_MSG_INLINE));
	    else
	      localbranch->deliver(msg, _idx, CkDeliver_queue, opts);
	  }
	}
}

void *CProxyElement_ArrayBase::ckSendSync(CkArrayMessage *msg, int ep) const
{
	CkFutureID f=CkCreateAttachedFuture(msg);
	ckSend(msg,ep);
	return CkWaitReleaseFuture(f);
}

void CkBroadcastMsgSection(int entryIndex, void *msg, CkSectionID sID, int opts     )
{
	CProxySection_ArrayBase sp(sID);
	sp.ckSend((CkArrayMessage *)msg,entryIndex,opts);
}

void CProxySection_ArrayBase::ckSend(CkArrayMessage *msg, int ep, int opts)
{
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySectionSend(ckDelegatedPtr(), ep, msg, _nsid, _sid, opts);
	else {
	  // send through all
	  for (int k=0; k<_nsid; ++k) {
	    for (int i=0; i< _sid[k]._nElems-1; i++) {
	      CProxyElement_ArrayBase ap(_sid[k]._cookie.get_aid(), _sid[k]._elems[i]);
	      void *newMsg=CkCopyMsg((void **)&msg);
	      ap.ckSend((CkArrayMessage *)newMsg,ep,opts);
	    }
	    if (_sid[k]._nElems > 0) {
	      void *newMsg= (k<_nsid-1) ? CkCopyMsg((void **)&msg) : msg;
	      CProxyElement_ArrayBase ap(_sid[k]._cookie.get_aid(), _sid[k]._elems[_sid[k]._nElems-1]);
	      ap.ckSend((CkArrayMessage *)newMsg,ep,opts);
	    }
	  }
	}
}

void CkSetMsgArrayIfNotThere(void *msg) {
  envelope *env = UsrToEnv((void *)msg);
  env->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *m = (CkArrayMessage *)msg;
  m->array_setIfNotThere(CkArray_IfNotThere_buffer);
}

void CkSendMsgArray(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  if (a == NULL)
    CkAbort("Cannot receive a message for an array without a local branch");
  else
    a->deliver(m, idx, CkDeliver_queue, opts);
}

void CkSendMsgArrayInline(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  int oldStatus = CkDisableTracing(entryIndex);     // avoid nested tracing
  a->deliver(m, idx, CkDeliver_inline, opts);
  if (oldStatus) CkEnableTracing(entryIndex);
}


/*********************** CkArray Reduction *******************/
CkArrayReducer::CkArrayReducer(CkGroupID mgrID_)
  :CkArrayListener(sizeof(contributorInfo)/sizeof(int)),
   mgrID(mgrID_)
{
  mgr=CProxy_CkReductionMgr(mgrID).ckLocalBranch();
}
CkArrayReducer::CkArrayReducer(CkMigrateMessage *m)
  :CkArrayListener(m)
{
  mgr=NULL;
}
void CkArrayReducer::pup(PUP::er &p) {
  CkArrayListener::pup(p);
  p|mgrID;
  if (p.isUnpacking())
    mgr=CProxy_CkReductionMgr(mgrID).ckLocalBranch();
}
CkArrayReducer::~CkArrayReducer() {}

/*********************** CkArray Broadcast ******************/

CkArrayBroadcaster::CkArrayBroadcaster(bool stableLocations_, bool broadcastViaScheduler_)
    :CkArrayListener(1), //Each array element carries a broadcast number
     bcastNo(0), oldBcastNo(0), stableLocations(stableLocations_), broadcastViaScheduler(broadcastViaScheduler_)
{ }

CkArrayBroadcaster::CkArrayBroadcaster(CkMigrateMessage *m)
    :CkArrayListener(m), bcastNo(-1), oldBcastNo(-1), broadcastViaScheduler(false)
{ }

void CkArrayBroadcaster::pup(PUP::er &p) {
  CkArrayListener::pup(p);
  /* Assumption: no migrants during checkpoint, so no need to
     save old broadcasts. */
  p|bcastNo;
  p|stableLocations;
  p|broadcastViaScheduler;
  if (p.isUnpacking()) {
    oldBcastNo=bcastNo; /* because we threw away oldBcasts */
  }
}

CkArrayBroadcaster::~CkArrayBroadcaster()
{
  CkArrayMessage *msg;
  while (NULL!=(msg=oldBcasts.deq())) delete msg;
}

void CkArrayBroadcaster::incoming(CkArrayMessage *msg)
{
  bcastNo++;
  DEBB((AA "Received broadcast %d\n" AB,bcastNo));

  if (stableLocations)
    return;

  CmiMemoryMarkBlock(((char *)UsrToEnv(msg))-sizeof(CmiChunkHeader));
  oldBcasts.enq((CkArrayMessage *)msg);//Stash the message for later use
}

/// Deliver a copy of the given broadcast to the given local element
bool CkArrayBroadcaster::deliver(CkArrayMessage *bcast, ArrayElement *el,
				    bool doFree)
{
  int &elBcastNo=getData(el);
  // if this array element already received this message, skip it
  if (elBcastNo >= bcastNo) return false;
  elBcastNo++;
  DEBB((AA "Delivering broadcast %d to element %s\n" AB,elBcastNo,idx2str(el)));

  CkAssert(UsrToEnv(bcast)->getMsgtype() == ForArrayEltMsg);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
  DEBUG(printf("[%d] elBcastNo %d bcastNo %d \n",CmiMyPe(),bcastNo));
  return true;
#else
  if (!broadcastViaScheduler)
    return el->ckInvokeEntry(bcast->array_ep(), bcast, doFree);
  else {
    if (!doFree) {
      CkArrayMessage *newMsg = (CkArrayMessage *)CkCopyMsg((void **)&bcast);
      bcast = newMsg;
    }
    envelope *env = UsrToEnv(bcast);
    env->setRecipientID(el->ckGetID());
    CkArrayManagerDeliver(CkMyPe(), bcast, 0);
    return true;
  }
#endif
}

/// Deliver all needed broadcasts to the given local element
bool CkArrayBroadcaster::bringUpToDate(ArrayElement *el)
{
  if (stableLocations) return true;
  int &elBcastNo=getData(el);
  if (elBcastNo<bcastNo)
  {//This element needs some broadcasts-- it must have
   //been migrating during the broadcast.
    int i,nDeliver=bcastNo-elBcastNo;
    DEBM((AA "Migrator %s missed %d broadcasts--\n" AB,idx2str(el),nDeliver));

    //Skip the old junk at the front of the bcast queue
    for (i=oldBcasts.length()-1;i>=nDeliver;i--)
      oldBcasts.enq(oldBcasts.deq());

    //Deliver the newest messages, in old-to-new order
    for (i=nDeliver-1;i>=0;i--)
    {
      CkArrayMessage *msg=oldBcasts.deq();
		if(msg == NULL)
        	continue;
      oldBcasts.enq(msg);
      if (!deliver(msg, el, false))
	return false; //Element migrated away
    }
  }
  //Otherwise, the element survived
  return true;
}


void CkArrayBroadcaster::springCleaning(void)
{
  //Remove old broadcast messages
  int nDelete=oldBcasts.length()-(bcastNo-oldBcastNo);
  if (nDelete>0) {
    DEBK((AA "Cleaning out %d old broadcasts\n" AB,nDelete));
    for (int i=0;i<nDelete;i++)
      delete oldBcasts.deq();
  }
  oldBcastNo=bcastNo;
}

void CkArrayBroadcaster::flushState() 
{ 
  bcastNo = oldBcastNo = 0; 
  CkArrayMessage *msg;
  while (NULL!=(msg=oldBcasts.deq())) delete msg;
}

void CkBroadcastMsgArray(int entryIndex, void *msg, CkArrayID aID, int opts)
{
	CProxy_ArrayBase ap(aID);
	ap.ckBroadcast((CkArrayMessage *)msg,entryIndex,opts);
}

void CProxy_ArrayBase::ckBroadcast(CkArrayMessage *msg, int ep, int opts) const
{
	envelope *env = UsrToEnv(msg);
	env->setMsgtype(ForBocMsg);
	msg->array_ep_bcast()=ep;
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArrayBroadcast(ckDelegatedPtr(),ep,msg,_aid);
	else 
	{ //Broadcast message via serializer node
	  _TRACE_CREATION_DETAILED(UsrToEnv(msg), ep);
 	  int skipsched = opts & CK_MSG_EXPEDITED; 
	  //int serializer=0;//1623802937%CkNumPes();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
                CProxy_CkArray ap(_aid);
                ap[CpvAccess(serializer)].sendBroadcast(msg);
                CkGroupID _id = _aid;
//              printf("[%d] At ckBroadcast in CProxy_ArrayBase id %d epidx %d \n",CkMyPe(),_id.idx,ep);
#else
	  if (CkMyPe()==CpvAccess(serializer))
	  {
		DEBB((AA "Sending array broadcast\n" AB));
		if (skipsched)
			CProxy_CkArray(_aid).recvExpeditedBroadcast(msg);
		else
			CProxy_CkArray(_aid).recvBroadcast(msg);
	  } else {
		DEBB((AA "Forwarding array broadcast to serializer node %d\n" AB,CpvAccess(serializer)));
		CProxy_CkArray ap(_aid);
		if (skipsched)
			ap[CpvAccess(serializer)].sendExpeditedBroadcast(msg);
		else
			ap[CpvAccess(serializer)].sendBroadcast(msg);
	  }
#endif
	}
}

/// Reflect a broadcast off this Pe:
void CkArray::sendBroadcast(CkMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	if(CkMyPe() == CpvAccess(serializer)){
#if _MLOG_BCAST_TREE_
		// Using the spanning tree to broadcast the message
		for(int i=0; i<numChildren; i++){
			CkMessage *copyMsg = (CkMessage *) CkCopyMsg((void **)&msg);
			thisProxy[children[i]].recvBroadcastViaTree(copyMsg);
		}
	
		// delivering message locally
		recvBroadcast(msg);	
#else
		//Broadcast the message to all processors
		thisProxy.recvBroadcast(msg);
#endif
	}else{
		thisProxy[CpvAccess(serializer)].sendBroadcast(msg);
	}
}
void CkArray::sendExpeditedBroadcast(CkMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	//Broadcast the message to all processors
	thisProxy.recvExpeditedBroadcast(msg);
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
int _tempBroadcastCount=0;

// Delivers a message using the spanning tree
void CkArray::recvBroadcastViaTree(CkMessage *msg)
{
	CK_MAGICNUMBER_CHECK

	// Using the spanning tree to broadcast the message
	for(int i=0; i<numChildren; i++){
		CkMessage *copyMsg = (CkMessage *) CkCopyMsg((void **)&msg);
		thisProxy[children[i]].recvBroadcastViaTree(copyMsg);
	}

	// delivering message locally
	recvBroadcast(msg);	
}

void CkArray::broadcastHomeElements(void *data,CkLocRec *rec,CkArrayIndex *index){
    if(homePe(*index)==CmiMyPe()){
        CkArrayMessage *bcast = (CkArrayMessage *)data;
        int epIdx=bcast->array_ep();
        DEBUG(CmiPrintf("[%d] gid %d broadcastHomeElements to index %s entry name %s\n",CmiMyPe(),thisgroup.idx,idx2str(*index),_entryTable[bcast->array_ep_bcast()]->name));
        CkArrayMessage *copy = (CkArrayMessage *)   CkCopyMsg((void **)&bcast);
        envelope *env = UsrToEnv(copy);
        env->sender.data.group.onPE = CkMyPe();
#if defined(_FAULT_CAUSAL_)
        env->TN = 0;
#endif
		env->SN = 0;
        env->piggyBcastIdx = epIdx;
        env->setEpIdx(CkIndex_ArrayElement::recvBroadcast(0));
        env->setArrayMgr(thisgroup);
        env->setRecipientID(ck::ObjID(thisgroup, rec->getID());
        env->setSrcPe(CkMyPe());
        env->getsetArrayHops() = 0;
        deliver(copy,CkDeliver_queue);
        _tempBroadcastCount++;
    }else{
        if(locMgr->homeElementCount != -1){
            DEBUG(CmiPrintf("[%d] gid %d skipping broadcast to index %s \n",CmiMyPe(),thisgroup.idx,idx2str(*index)));
        }
    }
}

void CkArray::staticBroadcastHomeElements(CkArray *arr,void *data,CkLocRec *rec,CkArrayIndex *index){
    arr->broadcastHomeElements(data,rec,index);
}
#else
void CkArray::recvBroadcastViaTree(CkMessage *msg){
}
#endif


/// Increment broadcast count; deliver to all local elements
void CkArray::recvBroadcast(CkMessage *m)
{
	CK_MAGICNUMBER_CHECK
	CkArrayMessage *msg=(CkArrayMessage *)m;

        // Turn the message into a real single-element message
        unsigned short ep = msg->array_ep_bcast();
        CkAssert(UsrToEnv(msg)->getGroupNum() == thisgroup);
        UsrToEnv(msg)->setMsgtype(ForArrayEltMsg);
        UsrToEnv(msg)->setArrayMgr(thisgroup);
        UsrToEnv(msg)->getsetArrayEp() = ep;

	broadcaster->incoming(msg);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        _tempBroadcastCount=0;
        locMgr->callForAllRecords(CkArray::staticBroadcastHomeElements,this,(void *)msg);
#else
#if CMK_BIGSIM_CHARM
        void *root;
        _TRACE_BG_TLINE_END(&root);
	BgSetEntryName("start-broadcast", &root);
        CkVec<void *> logs;    // store all logs for each delivery
	extern void stopVTimer();
	extern void startVTimer();
#endif
    int len = localElemVec.size();
    for (unsigned int i = 0; i < len; ++i) {
#if CMK_BIGSIM_CHARM
                //BgEntrySplit("split-broadcast");
  		stopVTimer();
                void *curlog = BgSplitEntry("split-broadcast", &root, 1);
                logs.push_back(curlog);
  		startVTimer();
#endif
		bool doFree = false;
		if (stableLocations && i == len-1) doFree = true;
		broadcaster->deliver(msg, (ArrayElement*)localElemVec[i], doFree);
	}
#endif

#if CMK_BIGSIM_CHARM
	//BgEntrySplit("end-broadcast");
	stopVTimer();
	BgSplitEntry("end-broadcast", logs.getVec(), logs.size());
	startVTimer();
#endif

	// CkArrayBroadcaster doesn't have msg buffered, and there was
	// no last delivery to transfer ownership
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	if (stableLocations)
	  delete msg;
#else
	if (stableLocations && len == 0)
	  delete msg;
#endif
}

void CkArray::flushStates() {
  CkReductionMgr::flushStates();
  // For chare arrays, and for chare arrays alone, the global and local
  // element counters in the reduction manager need to be reset to 0.
  // This is because all array elements are recreated during recovery
  // and will reregister, pushing the counts back to the correct levels.
  // For groups, the counters are set to 1 in the Group constructor.
  // However, since groups are not recreated during recovery, setting them
  // to zero in Group::flushStates() would not be followed by an increment
  // to 1 because the constructor will not be invoked.
  // Hence, these counters are reset only for chare arrays.
  resetCountersWhenFlushingStates();
  CK_ARRAYLISTENER_LOOP(listeners, l->flushState());
}

void CkArray::ckDestroy() {
  isDestroying = true;
  // Set the duringDestruction flag in the location manager. This is used to
  // indicate that the location manager is going to be destroyed so don't need
  // to send messages to remote PEs with reclaimRemote messages.
  locMgr->setDuringDestruction(true);

  for (unsigned int i = 0; i < localElemVec.size(); ++i)
    localElemVec[i]->ckDestroy();

  locMgr->deleteManager(CkGroupID(thisProxy), this);
  delete this;
}

#include "CkArray.def.h"


