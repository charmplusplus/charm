/* Generalized Chare Arrays

An Array is a collection of array elements (Chares) which
can be indexed by an arbitary run of bytes (a CkArrayIndex).
Elements can be inserted or removed from the array,
or migrated between processors.  Arrays are integrated with
the run-time load balancer.

Elements can also receive broadcasts and participate in
reductions.

Converted from 1-D arrays 2/27/2000 by
Orion Sky Lawlor, olawlor@acm.org

*/
#include "charm++.h"
#include "register.h"
#include "ck.h"

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

/************************** Debugging Utilities **************/

//For debugging: convert given index to a string (NOT threadsafe)
static const char *idx2str(const CkArrayIndex &ind)
{
  static char retBuf[80];
  retBuf[0]=0;
  for (int i=0;i<ind.nInts;i++)
  {
  	if (i>0) strcat(retBuf,";");
  	sprintf(&retBuf[strlen(retBuf)],"%d",ind.data()[i]);
  }
  return retBuf;
}
static const char *idx2str(const ArrayElement *el)
  {return idx2str(el->thisIndexMax);}

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
#endif

inline CkArrayIndexMax &CkArrayMessage::array_index(void)
{
	return UsrToEnv((void *)this)->array_index();
}

/*********************** CkVerboseListener ******************/
#define VL_PRINT ckout<<"VerboseListener on PE "<<CkMyPe()<<" > "
CkVerboseListenerCreator::CkVerboseListenerCreator(void)
{
  createCount=0;
}
//Called by the array manager to actually create the listener.
//  We ignore the index, but you could theoretically use it for something.
CkComponent *CkVerboseListenerCreator::ckLookupComponent(int useIndex)
{
  VL_PRINT<<"Creating listener for index "<<useIndex<<endl;
  return new CkVerboseListener();
}
CkComponentID CkVerboseListenerCreator::createListener(void)
{
  createCount++;
  return CkComponentID(thisgroup,createCount);
}

CkVerboseListener::CkVerboseListener(int nInts_) 
  :CkArrayListener(nInts_) {}

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
  int numInitial;
  int listenerData[CK_ARRAYLISTENER_MAXLEN];
  CmiBool fromMigration;
};

CkpvStaticDeclare(ArrayElement_initInfo,initInfo);

void ArrayElement::initBasics(void)
{
  ArrayElement_initInfo &info=CkpvAccess(initInfo);
  thisArray=info.thisArray;
  thisArrayID=info.thisArrayID;
  numElements=info.numInitial;
  if (info.listenerData) {
    memcpy(listenerData,info.listenerData,sizeof(listenerData));
  }
  if (!info.fromMigration) {
    CK_ARRAYLISTENER_LOOP(thisArray->listeners,
			  l->ckElementCreating(this));
  }
}

ArrayElement::ArrayElement(void) 
{
	initBasics();
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

CK_REDUCTION_CONTRIBUTE_METHODS_DEF(ArrayElement,thisArray,
   *(CkReductionMgr::contributorInfo *)&listenerData[thisArray->reducer.ckGetOffset()]);

/// Remote method: calls destructor
void ArrayElement::ckDestroy(void)
{
	CK_ARRAYLISTENER_LOOP(thisArray->listeners,
			   l->ckElementDied(this));
	CkMigratable::ckDestroy();
}

//Destructor (virtual)
ArrayElement::~ArrayElement()
{
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
}

/// A more verbose form of abort
void ArrayElement::CkAbort(const char *str) const
{
	CkError("Array element at index %s aborting:\n",
		idx2str(thisIndexMax));
	CkMigratable::CkAbort(str);
}

//// Checkpoint: pup to disk file
void ArrayElement::ckCheckpoint(char* fname)
{
  FILE *chkptfile=fopen(fname,"wb");
  if(chkptfile == NULL){
    CkAbort("ArrayElement::ckCheckpoint open file failed!");
  }
  PUP::toDisk p(chkptfile);
  this->pup(p);
  fclose(chkptfile);
}

void ArrayElement::ckRestart(char* fname)
{
  FILE *chkptfile=fopen(fname,"rb");
  if(chkptfile == NULL){
    CkAbort("ArrayElement::ckRestart open file failed!");
  }
  PUP::fromDisk p(chkptfile);
  this->pup(p);
  fclose(chkptfile);
}

/*********************** Spring Cleaning *****************
Periodically (every minute or so) remove expired broadcasts
from the queue.
*/

inline void CkArray::springCleaning(void)
{
  DEBK((AA"Starting spring cleaning #%d\n"AB,nSprings,nSprings))
  broadcaster.springCleaning();
}

void CkArray::staticSpringCleaning(void *forArray) {
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

CK_REDUCTION_CLIENT_DEF(CProxy_ArrayBase,ckLocalBranch());

CkArrayOptions::CkArrayOptions(void) //Default: empty array
	:numInitial(0),map(_RRMapID)
{
	locMgr.setZero();
}

CkArrayOptions::CkArrayOptions(int ni) //With initial elements
	:numInitial(ni),map(_RRMapID)
{
	locMgr.setZero();
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
CkArrayOptions &CkArrayOptions::addListener(const CkComponentID &id)
{
	arrayListeners.push_back(id);
	return *this;
}

void CkArrayOptions::pup(PUP::er &p) {
	p|numInitial;
	p|locMgr;
	p|map;
	p|arrayListeners;
}

CkArrayListener::CkArrayListener(int nInts_) 
  :nInts(nInts_) 
{
  dataOffset=-1;
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
  if (opts.getLocationManager().isZero()) 
  { //Create a new location manager
#if !CMK_LBDB_ON
    CkGroupID lbdb;
#endif
    opts.setLocationManager(CProxy_CkLocMgr::ckNew(
      opts.getMap(),lbdb,opts.getNumInitial()
      ));
  }
  //Create the array manager
  m->array_ep()=ctor;
  CkMarshalledMessage marsh(m);
  CkGroupID ag=CProxy_CkArray::ckNew(opts,marsh);
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
  	ckDelegatedTo()->ArrayCreate(ctor,m,idx,onPe,_aid);
  	return;
  }
  
  DEBC((AA"Proxy inserting element %s on Pe %d\n"AB,idx2str(idx),onPe));
  CProxy_CkArray ap(_aid);
  ap[onPe].insertElement(m);
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
  p(_idx.data(),_idx.nInts);
}

void CProxySection_ArrayBase::pup(PUP::er &p)
{
  CProxy_ArrayBase::pup(p);
  _sid.pup(p);
}

/*********************** CkArray Creation *************************/
void _ckArrayInit(void)
{
  CkpvInitialize(ArrayElement_initInfo,initInfo);
}

CkArray::CkArray(const CkArrayOptions &c,CkMarshalledMessage &initMsg)
  : CkReductionMgr(), 
  locMgr(CProxy_CkLocMgr::ckLocalBranch(c.getLocationManager())),
  thisProxy(thisgroup), reducer(this)
{
  //Registration
  elements=(ArrayElementList *)locMgr->addManager(thisgroup,this);
//  moved to _ckArrayInit()
//  CkpvInitialize(ArrayElement_initInfo,initInfo);
  CcdCallOnConditionKeep(CcdPERIODIC_1minute,staticSpringCleaning,(void *)this);

  //Set class variables
  ckEnableTracing=CmiFalse; //Prevent us from being recorded
  numInitial=c.getNumInitial();
  isInserting=CmiTrue;

  //Find, register, and initialize the arrayListeners
  int dataOffset=0;
  addListener(&broadcaster,dataOffset);
  addListener(&reducer,dataOffset);
  int lNo,nL=c.getListeners(); //User-added listeners
  for (lNo=0;lNo<nL;lNo++) addListener(c.getListener(lNo),dataOffset);
  if (dataOffset>CK_ARRAYLISTENER_MAXLEN)
    CkAbort("Too much array listener data!\n"
"You'll have to either use fewer array listeners, or increase the compile-time\n"
"constant CK_ARRAYLISTENER_MAXLEN!\n");

  for (int l=0;l<listeners.size();l++) listeners[l]->ckBeginInserting();
  
  //Set up initial elements (if any)
  locMgr->populateInitial(numInitial,initMsg.getMessage(),this);
}

//Called on send side to prepare array constructor message
void CkArray::prepareCtorMsg(CkMessage *m,int &onPe,const CkArrayIndex &idx)
{
  envelope *env=UsrToEnv((void *)m);
  env->array_index()=idx;
  int *listenerData=env->array_listenerData();
  int dataOffset=0;
  for (int lNo=0;lNo<listeners.size();lNo++) {
    CkArrayListener *l=listeners[lNo];
    l->ckElementStamp(&listenerData[dataOffset]);
    dataOffset+=l->ckGetLen();
  }
  if (onPe==-1) onPe=homePe(idx);
  if (onPe!=CkMyPe()) //Let the local manager know where this el't is
  	getLocMgr()->inform(idx,onPe);
}

CkMigratable *CkArray::allocateMigrated(int elChareType,const CkArrayIndex &idx) 
{
	return allocate(elChareType,idx,NULL,CmiTrue);
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
	  memcpy(init.listenerData,UsrToEnv(msg)->array_listenerData(),
		 sizeof(init.listenerData));
	init.fromMigration=fromMigration;
	
	//Build the element
	int elSize=_chareTable[elChareType]->size;
	return (ArrayElement *)malloc(elSize);
}

/// This method is called by the user to add an element.
CmiBool CkArray::insertElement(CkMessage *me)
{
  CK_MAGICNUMBER_CHECK
  CkArrayMessage *m=(CkArrayMessage *)me;
  const CkArrayIndex &idx=m->array_index();
  int ctorIdx=m->array_ep();
  int chareType=_entryTable[ctorIdx]->chareIdx;
  ArrayElement *elt=allocate(chareType,idx,me,CmiFalse);
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

CmiBool CkArray::demandCreateElement(const CkArrayIndex &idx,int onPe,int ctor)
{
	CkArrayMessage *m=(CkArrayMessage *)CkAllocSysMsg();
	prepareCtorMsg(m,onPe,idx);
	m->array_ep()=ctor;
	
	if (onPe==CkMyPe()) //Call local constructor directly
		return insertElement(m);
	else
		thisProxy[onPe].insertElement(m);
	return CmiTrue;
}

void CkArray::insertInitial(const CkArrayIndex &idx,void *ctorMsg)
{
	CkArrayMessage *m=(CkArrayMessage *)ctorMsg;
	int onPe=CkMyPe();
	prepareCtorMsg(m,onPe,idx);
	insertElement(m);
}

/********************* CkArray Messaging ******************/
/// Fill out a message's array fields before sending it
inline void msg_prepareSend(CkArrayMessage *msg, int ep,CkArrayID aid)
{
	envelope *env=UsrToEnv((void *)msg);
	env->array_mgr()=aid;
	env->array_srcPe()=CkMyPe();
	env->array_ep()=ep;
	env->array_hops()=0;
}
inline void msg_prepareSendImmediate(CkArrayMessage *msg, int ep,CkArrayID aid)
{
        msg_prepareSend(msg, ep, aid);
	envelope *env=UsrToEnv((void *)msg);
	CmiSetHandler(env, CpvAccessOther(CmiImmediateMsgHandlerIdx,0));
  	CmiSetXHandler(env, _charmHandlerIdx);
}
void CProxyElement_ArrayBase::ckSend(CkArrayMessage *msg, int ep) const
{
#ifndef CMK_OPTIMIZE
	//Check our array index for validity
	if (_idx.nInts<0) CkAbort("Array index length is negative!\n");
	if (_idx.nInts>CK_ARRAYINDEX_MAXLEN)
		CkAbort("Array index length (nInts) is too long-- did you "
			"use bytes instead of integers?\n");
#endif
        CmiBool immediate = msg->array_isImmediate();
#if CMK_IMMEDIATE_MSG
	if (immediate) {
	  msg_prepareSendImmediate(msg,ep,ckGetArrayID());
	  msg->array_setImmediate(CmiFalse);
        }
        else
#endif
	msg_prepareSend(msg,ep,ckGetArrayID());
	msg->array_index()=_idx;//Insert array index
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySend(ep,msg,_idx,ckGetArrayID());
	else 
	{ //Usual case: a direct send
	  ckLocalBranch()->deliverViaQueue(msg, immediate);
	}
}

void *CProxyElement_ArrayBase::ckSendSync(CkArrayMessage *msg, int ep) const
{
	CkFutureID f=CkCreateAttachedFuture(msg);
	ckSend(msg,ep);
	return CkWaitReleaseFuture(f);
}

void CProxySection_ArrayBase::ckSend(CkArrayMessage *msg, int ep)
{
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArraySectionSend(ep,msg,ckGetArrayID(),ckGetSectionCookie());
	else {
	  // send through all
	  for (int i=0; i< _sid._nElems-1; i++) {
	    CProxyElement_ArrayBase ap(ckGetArrayID(), _sid._elems[i]);
	    void *newMsg=CkCopyMsg((void **)&msg);
	    ap.ckSend((CkArrayMessage *)newMsg,ep);
	  }
	  if (_sid._nElems > 0) {
	    CProxyElement_ArrayBase ap(ckGetArrayID(), _sid._elems[_sid._nElems-1]);
	    ap.ckSend((CkArrayMessage *)msg,ep);
	  }
        }
}

void CkSendMsgArray(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  m->array_index()=idx;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  a->deliverViaQueue(m);
}
void CkSendMsgArrayInline(int entryIndex, void *msg, CkArrayID aID, const CkArrayIndex &idx)
{
  CkArrayMessage *m=(CkArrayMessage *)msg;
  m->array_index()=idx;
  msg_prepareSend(m,entryIndex,aID);
  CkArray *a=(CkArray *)_localBranch(aID);
  a->deliver(m);
}


/*********************** CkArray Reduction *******************/
CkArrayReducer::CkArrayReducer(CkReductionMgr *mgr_) 
  :CkArrayListener(sizeof(CkReductionMgr::contributorInfo)/sizeof(int)), 
   mgr(mgr_) 
{}
CkArrayReducer::~CkArrayReducer() {}

/*********************** CkArray Broadcast ******************/

CkArrayBroadcaster::CkArrayBroadcaster()
  :CkArrayListener(1) //Each array element carries a broadcast number
{
  bcastNo=oldBcastNo=0;
}

CkArrayBroadcaster::~CkArrayBroadcaster() 
{
  CkArrayMessage *msg;
  while (NULL!=(msg=oldBcasts.deq())) delete msg;
}

void CkArrayBroadcaster::incoming(CkArrayMessage *msg) 
{
  DEBB((AA"Received broadcast %d\n"AB,bcastNo));
  bcastNo++;
  oldBcasts.enq((CkArrayMessage *)msg);//Stash the message for later use
}

/// Deliver a copy of the given broadcast to the given local element
CmiBool CkArrayBroadcaster::deliver(CkArrayMessage *bcast,ArrayElement *el)
{
  int &elBcastNo=getData(el);
  elBcastNo++;
  void *newMsg=CkCopyMsg((void **)&bcast);
  DEBB((AA"Delivering broadcast %d to element %s\n"AB,elBcastNo,idx2str(el)));
  int epIdx=((CkArrayMessage *)newMsg)->array_ep();
  return el->ckInvokeEntry(epIdx,newMsg);
}

/// Deliver all needed broadcasts to the given local element
CmiBool CkArrayBroadcaster::bringUpToDate(ArrayElement *el)
{
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
      oldBcasts.enq(msg);
      if (!deliver(msg,el)) 
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
      CkFreeMsg((void *)oldBcasts.deq());
  }
  oldBcastNo=bcastNo;
}

void CkBroadcastMsgArray(int entryIndex, void *msg, CkArrayID aID)
{
	CProxy_ArrayBase ap(aID);
	ap.ckBroadcast((CkArrayMessage *)msg,entryIndex);
}

void CProxy_ArrayBase::ckBroadcast(CkArrayMessage *msg, int ep) const
{
	msg_prepareSend(msg,ep,ckGetArrayID());
	if (ckIsDelegated()) //Just call our delegateMgr
	  ckDelegatedTo()->ArrayBroadcast(ep,msg,_aid);
	else 
	{ //Broadcast message via serializer node
	  int serializer=0;//1623802937%CkNumPes();
	  if (CkMyPe()==serializer)
	  {
		DEBB((AA"Sending array broadcast\n"AB));
		CProxy_CkArray(_aid).recvBroadcast(msg);
	  } else {
		DEBB((AA"Forwarding array broadcast to serializer node %d\n"AB,serializer));
		CProxy_CkArray ap(_aid);
		ap[serializer].sendBroadcast(msg);
	  }
	}
}


/// Reflect a broadcast off this Pe:
void CkArray::sendBroadcast(CkMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	//Broadcast the message to all processors
	thisProxy.recvBroadcast(msg);
}

/// Increment broadcast count; deliver to all local elements
void CkArray::recvBroadcast(CkMessage *m)
{
	CK_MAGICNUMBER_CHECK
	CkArrayMessage *msg=(CkArrayMessage *)m;
	broadcaster.incoming(msg);
	//Run through the list of local elements
	int idx=0;
	ArrayElement *el;
	while (NULL!=(el=elements->next(idx)))
		broadcaster.deliver(msg,el);
}

#include "CkArray.def.h"





