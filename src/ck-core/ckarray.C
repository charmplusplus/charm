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
   CkLocMgr::populateInitial(numInitial)
    for (idx=...)
     if (map->procNum(idx)==thisPe) 
      CkArray::insertInitial
       CkArray::prepareCtorMsg
       CkArray::insertElement

2.) Initial inserts: one at a time
fooProxy[idx].insert(msg,n);
 CProxy_ArrayBase::ckInsertIdx
  CkArray::prepareCtorMsg
  CkArrayManagerInsert
   CkArray::insertElement

3.) Demand creation (receive side)
CkLocMgr::deliver
 CkLocMgr::deliverUnknown
  CkLocMgr::demandCreateElement
   CkArray::demandCreateElement
    CkArray::prepareCtorMsg
    CkArrayManagerInsert or direct CkArray::insertElement

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

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

CpvDeclare(int ,serializer);

bool _isAnytimeMigration;

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

/*
void 
CProxyElement_ArrayBase::ckSendWrapper(void *me, void *m, int ep, int opts){
       ((CProxyElement_ArrayBase*)me)->ckSend((CkArrayMessage*)m,ep,opts);
}
*/
void
CProxyElement_ArrayBase::ckSendWrapper(CkArrayID _aid, CkArrayIndexMax _idx, void *m, int ep, int opts) {
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
CmiBool CkVerboseListener::ckElementCreated(ArrayElement *elt)
{
  VL_PRINT<<"LIFE  Created element "<<idx2str(elt)<<endl;
  return CmiTrue;
}
void CkVerboseListener::ckElementDied(ArrayElement *elt)
{
  VL_PRINT<<"LIFE  Deleting element "<<idx2str(elt)<<endl;
}

void CkVerboseListener::ckElementLeaving(ArrayElement *elt)
{
  VL_PRINT<<"MIG  Leaving: element "<<idx2str(elt)<<endl;
}
CmiBool CkVerboseListener::ckElementArriving(ArrayElement *elt)
{
  VL_PRINT<<"MIG  Arriving: element "<<idx2str(elt)<<endl;
  return CmiTrue;
}


/************************* ArrayElement *******************/
class ArrayElement_initInfo {
public:
  CkArray *thisArray;
  CkArrayID thisArrayID;
  CkArrayIndexMax numInitial;
  int listenerData[CK_ARRAYLISTENER_MAXLEN];
  CmiBool fromMigration;
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
        mlogData->objID.type = TypeArray;
        mlogData->objID.data.array.id = (CkGroupID)thisArrayID;
#endif
}

ArrayElement::ArrayElement(void) 
{
	initBasics();
#if CMK_MEM_CHECKPOINT
        init_checkpt();
#endif
}

ArrayElement::ArrayElement(CkMigrateMessage *m) 
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

CK_REDUCTION_CONTRIBUTE_METHODS_DEF(ArrayElement,thisArray,
   *(contributorInfo *)&listenerData[thisArray->reducer->ckGetOffset()],true)

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
  DEBM((AA"  ArrayElement::pup()\n"AB));
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
  *((CkArrayIndexMax*)&str[5]) = thisIndexMax;
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
    ckInvokeEntry(epIdx,bcast,CmiTrue);
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
  DEBK((AA"Starting spring cleaning\n"AB));
  broadcaster->springCleaning();
}

void CkArray::staticSpringCleaning(void *forArray,double curWallTime) {
	((CkArray *)forArray)->springCleaning();
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
	:numInitial(0),map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1) //With initial elements (1D)
	:numInitial(CkArrayIndex1D(ni1)),map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1, int ni2) //With initial elements (2D)
	:numInitial(CkArrayIndex2D(ni1, ni2)),map(_defaultArrayMapID)
{
    init();
}

CkArrayOptions::CkArrayOptions(int ni1, int ni2, int ni3) //With initial elements (3D)
	:numInitial(CkArrayIndex3D(ni1, ni2, ni3)),map(_defaultArrayMapID)
{
    init();
}

void CkArrayOptions::init()
{
    locMgr.setZero();
    anytimeMigration = _isAnytimeMigration;
    staticInsertion = false;
    reductionClient.type = CkCallback::invalid;
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

void CkArrayOptions::pup(PUP::er &p) {
	p|numInitial;
	p|map;
	p|locMgr;
	p|arrayListeners;
	p|reductionClient;
	p|anytimeMigration;
	p|staticInsertion;
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

CkArrayID CProxy_ArrayBase::ckCreateArray(CkArrayMessage *m,int ctor,
					  const CkArrayOptions &opts_)
{
  CkArrayOptions opts(opts_);
  CkGroupID locMgr = opts.getLocationManager();
  if (locMgr.isZero())
  { //Create a new location manager
#if !CMK_LBDB_ON
    CkGroupID _lbdb;
#endif
    locMgr = CProxy_CkLocMgr::ckNew(opts.getMap(),_lbdb,opts.getNumInitial());
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

CkArrayID CProxy_ArrayBase::ckCreateEmptyArray(void)
{
  return ckCreateArray((CkArrayMessage *)CkAllocSysMsg(),0,CkArrayOptions());
}

void CProxy_ArrayBase::ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,
	const CkArrayIndex &idx)
{
  if (m==NULL) m=(CkArrayMessage *)CkAllocSysMsg();
  m->array_ep()=ctor;
  ckLocalBranch()->prepareCtorMsg(m,onPe,idx);
  if (ckIsDelegated()) {
  	ckDelegatedTo()->ArrayCreate(ckDelegatedPtr(),ctor,m,idx,onPe,_aid);
  	return;
  }
  
  DEBC((AA"Proxy inserting element %s on Pe %d\n"AB,idx2str(idx),onPe));
  CkArrayManagerInsert(onPe,m,_aid);
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

/*********************** CkArray Creation *************************/
void _ckArrayInit(void)
{
  CkpvInitialize(ArrayElement_initInfo,initInfo);
  CkDisableTracing(CkIndex_CkArray::insertElement(0));
  CkDisableTracing(CkIndex_CkArray::recvBroadcast(0));
    // disable because broadcast listener may deliver broadcast message
  CkDisableTracing(CkIndex_CkLocMgr::immigrate(0));
  // by default anytime migration is allowed
}

CkArray::CkArray(CkArrayOptions &opts,
		 CkMarshalledMessage &initMsg,
		 CkNodeGroupID nodereductionID)
  : CkReductionMgr(),
    locMgr(CProxy_CkLocMgr::ckLocalBranch(opts.getLocationManager())),
    locMgrID(opts.getLocationManager()),
    thisProxy(thisgroup),
    // Register with our location manager
    elements((ArrayElementList *)locMgr->addManager(thisgroup,this)),
    stableLocations(opts.staticInsertion && !opts.anytimeMigration),
    numInitial(opts.getNumInitial()), isInserting(CmiTrue)
{
  if (!stableLocations)
      CcdCallOnConditionKeep(CcdPERIODIC_1minute,
			     staticSpringCleaning, (void *)this);

  //Find, register, and initialize the arrayListeners
  listenerDataOffset=0;
  broadcaster=new CkArrayBroadcaster(stableLocations);
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
  locMgr->populateInitial(numInitial,initMsg.getMessage(),this);

  ///adding code for Reduction using nodegroups

#if !GROUP_LEVEL_REDUCTION
  CProxy_CkArrayReductionMgr  nodetemp(nodereductionID);  
  nodeProxy = nodetemp;
  //nodeProxy = new CProxy_CkArrayReductionMgr (nodereductionID);
#endif

  if (opts.reductionClient.type != CkCallback::invalid && CkMyPe() == 0)
      ckSetReductionClient(&opts.reductionClient);
}

CkArray::CkArray(CkMigrateMessage *m)
	:CkReductionMgr(m), thisProxy(thisgroup)
{
  locMgr=NULL;
  isInserting=CmiTrue;
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
	testPup(p,1234);
	if(p.isUnpacking()){
		thisProxy=thisgroup;
		locMgr = CProxy_CkLocMgr::ckLocalBranch(locMgrID);
		elements = (ArrayElementList *)locMgr->addManager(thisgroup,this);
		/// Restore our default listeners:
		broadcaster=(CkArrayBroadcaster *)(CkArrayListener *)(listeners[0]);
		reducer=(CkArrayReducer *)(CkArrayListener *)(listeners[1]);
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
void CkArray::prepareCtorMsg(CkMessage *m,int &onPe,const CkArrayIndex &idx)
{
  envelope *env=UsrToEnv((void *)m);
  env->getsetArrayIndex()=idx;
  int *listenerData=env->getsetArrayListenerData();
  CK_ARRAYLISTENER_STAMP_LOOP(listenerData);
  if (onPe==-1) onPe=procNum(idx);   // onPe may still be -1
  if (onPe!=CkMyPe()&&onPe!=-1) //Let the local manager know where this el't is
  	getLocMgr()->inform(idx,onPe);
}

CkMigratable *CkArray::allocateMigrated(int elChareType,const CkArrayIndex &idx,
			CkElementCreation_t type)
{
	ArrayElement *ret=allocate(elChareType,idx,NULL,CmiTrue);
	if (type==CkElementCreation_resume) 
	{ // HACK: Re-stamp elements on checkpoint resume--
	  //  this restores, e.g., reduction manager's gcount
		int *listenerData=ret->listenerData;
		CK_ARRAYLISTENER_STAMP_LOOP(listenerData);
	}
	return ret;
}

ArrayElement *CkArray::allocate(int elChareType,const CkArrayIndex &idx,
		     CkMessage *msg,CmiBool fromMigration) 
{
	//Stash the element's initialization information in the global "initInfo"
	ArrayElement_initInfo &init=CkpvAccess(initInfo);
	init.numInitial=numInitial;
	init.thisArray=this;
	init.thisArrayID=thisgroup;
	if (msg) /*Have to *copy* data because msg will be deleted*/
	  memcpy(init.listenerData,UsrToEnv(msg)->getsetArrayListenerData(),
		 sizeof(init.listenerData));
	init.fromMigration=fromMigration;
	
	//Build the element
	int elSize=_chareTable[elChareType]->size;
	ArrayElement *elem = (ArrayElement *)malloc(elSize);
#ifndef CMK_OPTIMIZE
	if (elem!=NULL) setMemoryTypeChare(elem);
#endif
	return elem;
}

/// This method is called by ck.C or the user to add an element.
CmiBool CkArray::insertElement(CkMessage *me)
{
  CK_MAGICNUMBER_CHECK
  CkArrayMessage *m=(CkArrayMessage *)me;
  const CkArrayIndex &idx=m->array_index();
  int onPe;
  if (locMgr->isRemote(idx,&onPe)) 
  { /* element's sibling lives somewhere else, so insert there */
  	CkArrayManagerInsert(onPe,me,thisgroup);
	return CmiFalse;
  }
  int ctorIdx=m->array_ep();
  int chareType=_entryTable[ctorIdx]->chareIdx;
  ArrayElement *elt=allocate(chareType,idx,me,CmiFalse);
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)elt)->chareIdx = -1;
#endif
  if (!locMgr->addElement(thisgroup,idx,elt,ctorIdx,(void *)m)) return CmiFalse;
  CK_ARRAYLISTENER_LOOP(listeners,
      if (!l->ckElementCreated(elt)) return CmiFalse;);
  return CmiTrue;
}

void CProxy_ArrayBase::doneInserting(void)
{
  DEBC((AA"Broadcasting a doneInserting request\n"AB));
  //Broadcast a DoneInserting
  CProxy_CkArray(_aid).remoteDoneInserting();
}

void CkArray::doneInserting(void)
{
  thisProxy[CkMyPe()].remoteDoneInserting();
}

/// This is called on every processor after the last array insertion.
void CkArray::remoteDoneInserting(void)
{
  CK_MAGICNUMBER_CHECK
  if (isInserting) {
    isInserting=CmiFalse;
    DEBC((AA"Done inserting objects\n"AB));
    for (int l=0;l<listeners.size();l++) listeners[l]->ckEndInserting();
    locMgr->doneInserting();
  }
}

CmiBool CkArray::demandCreateElement(const CkArrayIndex &idx,
	int onPe,int ctor,CkDeliver_t type)
{
	CkArrayMessage *m=(CkArrayMessage *)CkAllocSysMsg();
	prepareCtorMsg(m,onPe,idx);
	m->array_ep()=ctor;
	
	if ((onPe!=CkMyPe()) || (type==CkDeliver_queue)) {
		DEBC((AA"Forwarding demand-creation request for %s to %d\n"AB,idx2str(idx),onPe));
		CkArrayManagerInsert(onPe,m,thisgroup);
	} else /* local message, non-queued */ {
		//Call local constructor directly
		DEBC((AA"Demand-creating %s\n"AB,idx2str(idx)));
		return insertElement(m);
	}
	return CmiTrue;
}

void CkArray::insertInitial(const CkArrayIndex &idx,void *ctorMsg, int local)
{
	CkArrayMessage *m=(CkArrayMessage *)ctorMsg;
	if (local) {
	  int onPe=CkMyPe();
	  prepareCtorMsg(m,onPe,idx);
#if CMK_BLUEGENE_CHARM
          BgEntrySplit("split-array-new");
#endif
	  insertElement(m);
  	}
	else {
	  int onPe=-1;
	  prepareCtorMsg(m,onPe,idx);
	  CkArrayManagerInsert(onPe,m,getGroupID());
	}
}

/********************* CkArray Messaging ******************/
/// Fill out a message's array fields before sending it
inline void msg_prepareSend(CkArrayMessage *msg, int ep,CkArrayID aid)
{
	envelope *env=UsrToEnv((void *)msg);
	env->getsetArrayMgr()=aid;
	env->getsetArraySrcPe()=CkMyPe();
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
	env->getsetArrayMgr()=aid;
	env->getsetArraySrcPe()=CkMyPe();
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
	msg_prepareSend(msg,ep,ckGetArrayID());
	msg->array_index()=_idx;//Insert array index
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySend(ckDelegatedPtr(),ep,msg,_idx,ckGetArrayID());
	else 
	{ //Usual case: a direct send
	  CkArray *localbranch = ckLocalBranch();
	  if (localbranch == NULL) {             // array not created yet
	    CkArrayManagerDeliver(CkMyPe(), msg, 0);
          }
	  else {
	    if (opts & CK_MSG_INLINE)
	      localbranch->deliver(msg, CkDeliver_inline, opts & (~CK_MSG_INLINE));
	    else
	      localbranch->deliver(msg, CkDeliver_queue, opts);
	  }
	}
}

void *CProxyElement_ArrayBase::ckSendSync(CkArrayMessage *msg, int ep) const
{
	CkFutureID f=CkCreateAttachedFuture(msg);
	ckSend(msg,ep);
	return CkWaitReleaseFuture(f);
}

void CProxySection_ArrayBase::ckSend(CkArrayMessage *msg, int ep, int opts)
{
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySectionSend(ckDelegatedPtr(), ep, msg, _nsid, _sid, opts);
	else {
	  // send through all
	  for (int k=0; k<_nsid; ++k) {
	    for (int i=0; i< _sid[k]._nElems-1; i++) {
	      CProxyElement_ArrayBase ap(_sid[k]._cookie.aid, _sid[k]._elems[i]);
	      void *newMsg=CkCopyMsg((void **)&msg);
	      ap.ckSend((CkArrayMessage *)newMsg,ep,opts);
	    }
	    if (_sid[k]._nElems > 0) {
	      void *newMsg= (k<_nsid-1) ? CkCopyMsg((void **)&msg) : msg;
	      CProxyElement_ArrayBase ap(_sid[k]._cookie.aid, _sid[k]._elems[_sid[k]._nElems-1]);
	      ap.ckSend((CkArrayMessage *)newMsg,ep,opts);
	    }
	  }
	}
}

void CkSendMsgArray(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  m->array_index()=idx;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  if (a == NULL)
    CkArrayManagerDeliver(CkMyPe(), msg, 0);
  else
    a->deliver(m,CkDeliver_queue,opts);
}

void CkSendMsgArrayInline(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx, int opts)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  m->array_index()=idx;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  int oldStatus = CkDisableTracing(entryIndex);     // avoid nested tracing
  a->deliver(m,CkDeliver_inline,opts);
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

CkArrayBroadcaster::CkArrayBroadcaster(bool stableLocations_)
    :CkArrayListener(1), //Each array element carries a broadcast number
     bcastNo(0), oldBcastNo(0), stableLocations(stableLocations_)
{ }
CkArrayBroadcaster::CkArrayBroadcaster(CkMigrateMessage *m)
    :CkArrayListener(m), bcastNo(-1), oldBcastNo(-1)
{ }

void CkArrayBroadcaster::pup(PUP::er &p) {
  CkArrayListener::pup(p);
  /* Assumption: no migrants during checkpoint, so no need to
     save old broadcasts. */
  p|bcastNo;
  p|stableLocations;
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
  DEBB((AA"Received broadcast %d\n"AB,bcastNo));

  if (stableLocations)
    return;

  CmiMemoryMarkBlock(((char *)UsrToEnv(msg))-sizeof(CmiChunkHeader));
  oldBcasts.enq((CkArrayMessage *)msg);//Stash the message for later use
}

/// Deliver a copy of the given broadcast to the given local element
CmiBool CkArrayBroadcaster::deliver(CkArrayMessage *bcast, ArrayElement *el,
				    CmiBool doFree)
{
  int &elBcastNo=getData(el);
  // if this array element already received this message, skip it
  if (elBcastNo >= bcastNo) return CmiFalse;
  elBcastNo++;
  DEBB((AA"Delivering broadcast %d to element %s\n"AB,elBcastNo,idx2str(el)));
  int epIdx=bcast->array_ep_bcast();

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
  DEBUG(printf("[%d] elBcastNo %d bcastNo %d \n",CmiMyPe(),bcastNo));
  return CmiTrue;
#else
  return el->ckInvokeEntry(epIdx, bcast, doFree);
#endif
}

/// Deliver all needed broadcasts to the given local element
CmiBool CkArrayBroadcaster::bringUpToDate(ArrayElement *el)
{
  if (stableLocations) return CmiTrue;
  int &elBcastNo=getData(el);
  if (elBcastNo<bcastNo)
  {//This element needs some broadcasts-- it must have
   //been migrating during the broadcast.
    int i,nDeliver=bcastNo-elBcastNo;
    DEBM((AA"Migrator %s missed %d broadcasts--\n"AB,idx2str(el),nDeliver));

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
      if (!deliver(msg, el, CmiFalse))
	return CmiFalse; //Element migrated away
    }
  }
  //Otherwise, the element survived
  return CmiTrue;
}


void CkArrayBroadcaster::springCleaning(void)
{
  //Remove old broadcast messages
  int nDelete=oldBcasts.length()-(bcastNo-oldBcastNo);
  if (nDelete>0) {
    DEBK((AA"Cleaning out %d old broadcasts\n"AB,nDelete));
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
		DEBB((AA"Sending array broadcast\n"AB));
		if (skipsched)
			CProxy_CkArray(_aid).recvExpeditedBroadcast(msg);
		else
			CProxy_CkArray(_aid).recvBroadcast(msg);
	  } else {
		DEBB((AA"Forwarding array broadcast to serializer node %d\n"AB,CpvAccess(serializer)));
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
		//Broadcast the message to all processors
		thisProxy.recvBroadcast(msg);
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

void CkArray::broadcastHomeElements(void *data,CkLocRec *rec,CkArrayIndex *index){
    if(homePe(*index)==CmiMyPe()){
        CkArrayMessage *bcast = (CkArrayMessage *)data;
    int epIdx=bcast->array_ep_bcast();
        DEBUG(CmiPrintf("[%d] gid %d broadcastHomeElements to index %s entry name %s\n",CmiMyPe(),thisgroup.idx,idx2str(*index),_entryTable[bcast->array_ep_bcast()]->name));
        CkArrayMessage *copy = (CkArrayMessage *)   CkCopyMsg((void **)&bcast);
        envelope *env = UsrToEnv(copy);
        env->sender.data.group.onPE = CkMyPe();
        env->TN  = env->SN=0;
        env->piggyBcastIdx = epIdx;
        env->setEpIdx(CkIndex_ArrayElement::recvBroadcast(0));
        env->getsetArrayMgr() = thisgroup;
        env->getsetArrayIndex() = *index;
    env->getsetArrayEp() = CkIndex_ArrayElement::recvBroadcast(0);
        env->setSrcPe(CkMyPe());
        rec->deliver(copy,CkDeliver_queue);
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
#endif


/// Increment broadcast count; deliver to all local elements
void CkArray::recvBroadcast(CkMessage *m)
{
	CK_MAGICNUMBER_CHECK
	CkArrayMessage *msg=(CkArrayMessage *)m;
	broadcaster->incoming(msg);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        _tempBroadcastCount=0;
        locMgr->callForAllRecords(CkArray::staticBroadcastHomeElements,this,(void *)msg);
#else
	//Run through the list of local elements
	int idx=0, len = elements->length();
	ArrayElement *el;
#if CMK_BLUEGENE_CHARM
        void *root;
        _TRACE_BG_TLINE_END(&root);
	BgSetEntryName("start-broadcast", &root);
        CkVec<void *> logs;    // store all logs for each delivery
	extern void stopVTimer();
	extern void startVTimer();
#endif
	while (NULL!=(el=elements->next(idx))) {
#if CMK_BLUEGENE_CHARM
                //BgEntrySplit("split-broadcast");
  		stopVTimer();
                void *curlog = BgSplitEntry("split-broadcast", &root, 1);
                logs.push_back(curlog);
  		startVTimer();
#endif
		CmiBool doFree = CmiFalse;
		if (stableLocations && idx == len) doFree = CmiTrue;
		broadcaster->deliver(msg, el, doFree);
	}
#endif

#if CMK_BLUEGENE_CHARM
	//BgEntrySplit("end-broadcast");
	stopVTimer();
	BgSplitEntry("end-broadcast", logs.getVec(), logs.size());
	startVTimer();
#endif

	// CkArrayBroadcaster doesn't have msg buffered, and there was
	// no last delivery to transfer ownership
	if (stableLocations && len == 0)
	  delete msg;
}

#include "CkArray.def.h"


