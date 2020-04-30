/** \file cklocation.C
 *  \addtogroup CkArrayImpl
 *
 *  The location manager keeps track of an indexed set of migratable objects.
 *  It is used by the array manager to locate array elements, interact with the
 *  load balancer, and perform migrations.
 *
 *  Orion Sky Lawlor, olawlor@acm.org 9/29/2001
 */

#include "hilbert.h"
#include "partitioning_strategies.h"
#include "charm++.h"
#include "register.h"
#include "ck.h"
#include "trace.h"
#include "TopoManager.h"
#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>
#include "pup_stl.h"
#include <stdarg.h>

#if CMK_LBDB_ON
#include "LBManager.h"
#include "MetaBalancer.h"
#if CMK_GLOBAL_LOCATION_UPDATE
#include "BaseLB.h"
#include "init.h"
#endif
CkpvExtern(int, _lb_obj_index);                // for lbdb user data for obj index
#endif // CMK_LBDB_ON

CpvExtern(std::vector<NcpyOperationInfo *>, newZCPupGets); // used for ZC Pup
#ifndef CMK_CHARE_USE_PTR
CkpvExtern(int, currentChareIdx);
#endif

#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void *, CkGridObject);
#endif

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT 
#   define DEB(x) CkPrintf x  //General debug messages
#   define DEBI(x) CkPrintf x  //Index debug messages
#   define DEBC(x) CkPrintf x  //Construction debug messages
#   define DEBS(x) CkPrintf x  //Send/recv/broadcast debug messages
#   define DEBM(x) CkPrintf x  //Migration debug messages
#   define DEBL(x) CkPrintf x  //Load balancing debug messages
#   define DEBN(x) CkPrintf x  //Location debug messages
#   define DEBB(x) CkPrintf x  //Broadcast debug messages
#   define AA "LocMgr on %d: "
#   define AB ,CkMyPe()
#   define DEBUG(x) CkPrintf x
#   define DEBAD(x) CkPrintf x
#else
#   define DEB(X) /*CkPrintf x*/
#   define DEBI(X) /*CkPrintf x*/
#   define DEBC(X) /*CkPrintf x*/
#   define DEBS(x) /*CkPrintf x*/
#   define DEBM(X) /*CkPrintf x*/
#   define DEBL(X) /*CkPrintf x*/
#   define DEBN(x) /*CkPrintf x*/
#   define DEBB(x) /*CkPrintf x*/
#   define str(x) /**/
#   define DEBUG(x)   /**/
#   define DEBAD(x) /*CkPrintf x*/
#endif

/// Message size above which the runtime will buffer messages directed at
/// unlocated array elements
int _messageBufferingThreshold;

#if CMK_LBDB_ON

#if CMK_GLOBAL_LOCATION_UPDATE
void UpdateLocation(MigrateInfo& migData) {

  CkGroupID locMgrGid = ck::ObjID(migData.obj.id).getCollectionID();
  if (locMgrGid.idx == 0) {
    return;
  }

  CkLocMgr *localLocMgr = (CkLocMgr *) CkLocalBranch(locMgrGid);
  localLocMgr->updateLocation(migData.obj.id, migData.to_pe);
}
#endif

#endif

/*********************** Array Messages ************************/
CmiUInt8 CkArrayMessage::array_element_id(void)
{
  return ck::ObjID(UsrToEnv((void *)this)->getRecipientID()).getElementID();
}
unsigned short &CkArrayMessage::array_ep(void)
{
	return UsrToEnv((void *)this)->getsetArrayEp();
}
unsigned short &CkArrayMessage::array_ep_bcast(void)
{
	return UsrToEnv((void *)this)->getsetArrayBcastEp();
}
unsigned char &CkArrayMessage::array_hops(void)
{
	return UsrToEnv((void *)this)->getsetArrayHops();
}
unsigned int CkArrayMessage::array_getSrcPe(void)
{
	return UsrToEnv((void *)this)->getsetArraySrcPe();
}
unsigned int CkArrayMessage::array_ifNotThere(void)
{
	return UsrToEnv((void *)this)->getArrayIfNotThere();
}
void CkArrayMessage::array_setIfNotThere(unsigned int i)
{
	UsrToEnv((void *)this)->setArrayIfNotThere(i);
}

// given an envelope of a Charm msg, find the recipient object pointer
CkMigratable * CkArrayMessageObjectPtr(envelope *env) {
  if (env->getMsgtype() != ForArrayEltMsg)
      return NULL;   // not an array msg

  ///@todo: Delegate this to the array manager which can then deal with ForArrayEltMsg
  CkArray *mgr = CProxy_CkArray(env->getArrayMgr()).ckLocalBranch();
  return mgr ? mgr->lookup(ck::ObjID(env->getRecipientID()).getElementID()) : NULL;
}

/****************************** Out-of-Core support ********************/

#if CMK_OUT_OF_CORE
CooPrefetchManager CkArrayElementPrefetcher;
CkpvDeclare(int,CkSaveRestorePrefetch);

/**
 * Return the out-of-core objid (from CooRegisterObject)
 * that this Converse message will access.  If the message
 * will not access an object, return -1.
 */
int CkArrayPrefetch_msg2ObjId(void *msg) {
  envelope *env=(envelope *)msg;
  CkMigratable *elt = CkArrayMessageObjectPtr(env);
  return elt?elt->prefetchObjID:-1;
}

/**
 * Write this object (registered with RegisterObject)
 * to this writable file.
 */
void CkArrayPrefetch_writeToSwap(FILE *swapfile,void *objptr) {
  CkMigratable *elt=(CkMigratable *)objptr;

  //Save the element's data to disk:
  PUP::toDisk p(swapfile);
  elt->virtual_pup(p);

  //Call the element's destructor in-place (so pointer doesn't change)
  CkpvAccess(CkSaveRestorePrefetch)=1;
  elt->~CkMigratable(); //< because destructor is virtual, destroys user class too.
  CkpvAccess(CkSaveRestorePrefetch)=0;
}
	
/**
 * Read this object (registered with RegisterObject)
 * from this readable file.
 */
void CkArrayPrefetch_readFromSwap(FILE *swapfile,void *objptr) {
  CkMigratable *elt=(CkMigratable *)objptr;
  //Call the element's migration constructor in-place
  CkpvAccess(CkSaveRestorePrefetch)=1;
  int ctorIdx=_chareTable[elt->thisChareType]->migCtor;
  elt->myRec->invokeEntry(elt,(CkMigrateMessage *)0,ctorIdx,true);
  CkpvAccess(CkSaveRestorePrefetch)=0;
  
  //Restore the element's data from disk:
  PUP::fromDisk p(swapfile);
  elt->virtual_pup(p);
}

static void _CkMigratable_prefetchInit(void) 
{
  CkpvExtern(int,CkSaveRestorePrefetch);
  CkpvAccess(CkSaveRestorePrefetch)=0;
  CkArrayElementPrefetcher.msg2ObjId=CkArrayPrefetch_msg2ObjId;
  CkArrayElementPrefetcher.writeToSwap=CkArrayPrefetch_writeToSwap;
  CkArrayElementPrefetcher.readFromSwap=CkArrayPrefetch_readFromSwap;
  CooRegisterManager(&CkArrayElementPrefetcher, _charmHandlerIdx);
}
#endif

/****************************** CkMigratable ***************************/
/**
 * This tiny class is used to convey information to the 
 * newly created CkMigratable object when its constructor is called.
 */
class CkMigratable_initInfo {
public:
	CkLocRec *locRec;
	int chareType;
	bool forPrefetch; /* If true, this creation is only a prefetch restore-from-disk.*/
};

CkpvStaticDeclare(CkMigratable_initInfo,mig_initInfo);


void _CkMigratable_initInfoInit(void) {
  CkpvInitialize(CkMigratable_initInfo,mig_initInfo);
#if CMK_OUT_OF_CORE
  _CkMigratable_prefetchInit();
#endif
}

void CkMigratable::commonInit(void) {
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
#if CMK_OUT_OF_CORE
	isInCore=true;
	if (CkpvAccess(CkSaveRestorePrefetch))
		return; /* Just restoring from disk--don't touch object */
	prefetchObjID=-1; //Unregistered
#endif
	myRec=i.locRec;
	thisIndexMax=myRec->getIndex();
	thisChareType=i.chareType;
	usesAtSync=false;
	usesAutoMeasure=true;
	barrierRegistered=false;

  local_state = OFF;
  prev_load = 0.0;
  can_reset = false;

#if CMK_LBDB_ON
  if (_lb_args.metaLbOn()) {
    atsync_iteration = myRec->getMetaBalancer()->get_iteration();
    myRec->getMetaBalancer()->AdjustCountForNewContributor(atsync_iteration);
  }
#endif

#if CMK_FAULT_EVAC
	AsyncEvacuate(true);
#endif
}

CkMigratable::CkMigratable(void) {
	DEBC((AA "In CkMigratable constructor\n" AB));
	commonInit();
}
CkMigratable::CkMigratable(CkMigrateMessage *m): Chare(m) {
	commonInit();
}

int CkMigratable::ckGetChareType(void) const {return thisChareType;}

void CkMigratable::pup(PUP::er &p) {
	DEBM((AA "In CkMigratable::pup %s\n" AB,idx2str(thisIndexMax)));
	Chare::pup(p);
	p|thisIndexMax;
	p(usesAtSync);
  p(can_reset);
	p(usesAutoMeasure);
#if CMK_LBDB_ON 
	int readyMigrate = 0;
	if (p.isPacking()) readyMigrate = myRec->isReadyMigrate();
	p|readyMigrate;
	if (p.isUnpacking()) myRec->ReadyMigrate(readyMigrate);
#endif
	if(p.isUnpacking()) barrierRegistered=false;

#if CMK_FAULT_EVAC
	p | asyncEvacuate;
	if(p.isUnpacking()){myRec->AsyncEvacuate(asyncEvacuate);}
#endif
	
	ckFinishConstruction();
}

void CkMigratable::ckDestroy(void) {}
void CkMigratable::ckAboutToMigrate(void) { }
void CkMigratable::ckJustMigrated(void) { }
void CkMigratable::ckJustRestored(void) { }

CkMigratable::~CkMigratable() {
	DEBC((AA "In CkMigratable::~CkMigratable %s\n" AB,idx2str(thisIndexMax)));
#if CMK_OUT_OF_CORE
	isInCore=false;
	if (CkpvAccess(CkSaveRestorePrefetch)) 
		return; /* Just saving to disk--don't deregister anything. */
	/* We're really leaving or dying-- unregister from the ooc system*/
	if (prefetchObjID!=-1) {
		CooDeregisterObject(prefetchObjID);
		prefetchObjID=-1;
	}
#endif
#if CMK_LBDB_ON 
	if (barrierRegistered) {
	  DEBL((AA "Removing barrier for element %s\n" AB,idx2str(thisIndexMax)));
	  if (usesAtSync)
		myRec->getLBMgr()->RemoveLocalBarrierClient(ldBarrierHandle);
	}

  if (_lb_args.metaLbOn()) {
    myRec->getMetaBalancer()->AdjustCountForDeadContributor(atsync_iteration);
  }
#endif
	myRec->destroy(); /* Attempt to delete myRec if it's no longer in use */
	//To detect use-after-delete
	thisIndexMax.nInts=0;
	thisIndexMax.dimension=0;
}

void CkMigratable::CkAbort(const char *format, ...) const {
	char newmsg[256];
	va_list args;
	va_start(args, format);
	vsnprintf(newmsg, sizeof(newmsg), format, args);
	va_end(args);

	::CkAbort("CkMigratable '%s' aborting: %s", _chareTable[thisChareType]->name, newmsg);
}

void CkMigratable::ResumeFromSync(void)
{
}

void CkMigratable::UserSetLBLoad() {
	CkAbort("::UserSetLBLoad() not defined for this array element!\n");
}

#if CMK_LBDB_ON  //For load balancing:
// user can call this helper function to set obj load (for model-based lb)
void CkMigratable::setObjTime(double cputime) {
	myRec->setObjTime(cputime);
}
double CkMigratable::getObjTime() {
	return myRec->getObjTime();
}

#if CMK_LB_USER_DATA
/**
* Use this method to set user specified data to the lbdatabase.
*
* Eg usage: 
* In the application code,
*   void *data = getObjUserData(CkpvAccess(_lb_obj_index));
*   *(int *) data = val;
*
* In the loadbalancer or wherever this data is used do
*   for (int i = 0; i < stats->n_objs; i++ ) {
*     LDObjData &odata = stats->objData[i];
*     int* udata = (int *) odata.getUserData(CkpvAccess(_lb_obj_index));
*   }
*
* For a complete example look at tests/charm++/load_balancing/lb_userdata_test/
*/
void *CkMigratable::getObjUserData(int idx) {
	return myRec->getObjUserData(idx);
}
#endif

void CkMigratable::clearMetaLBData() {
//  if (can_reset) {
    local_state = OFF;
    atsync_iteration = -1;
    prev_load = 0.0;
    can_reset = false;
//  }
}

void CkMigratable::recvLBPeriod(void *data) {
  if (atsync_iteration < 0) {
    return;
  }
  int lb_period = *((int *) data);
 DEBAD(("\t[obj %s] Received the LB Period %d current iter %d state %d on PE %d\n",
     idx2str(thisIndexMax), lb_period, atsync_iteration, local_state, CkMyPe()));

  bool is_tentative;
  if (local_state == LOAD_BALANCE) {
    CkAssert(lb_period == myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative));
    return;
  }

  if (local_state == PAUSE) {
    if (atsync_iteration < lb_period) {
      local_state = DECIDED;
      ResumeFromSync();
      return;
    }
    local_state = LOAD_BALANCE;

    can_reset = true;
    //myRec->AtLocalBarrier(ldBarrierHandle);
    return;
  }
  local_state = DECIDED;
}

void CkMigratable::metaLBCallLB() {
  if(usesAtSync)
    myRec->getLBMgr()->AtLocalBarrier(ldBarrierHandle);
}

void CkMigratable::ckFinishConstruction(void)
{
//	if ((!usesAtSync) || barrierRegistered) return;
	if (usesAtSync && _lb_args.lbperiod() != -1.0)
          CkAbort("You must use AtSync or Periodic LB separately!\n");

	myRec->setMeasure(usesAutoMeasure);
	if (barrierRegistered) return;
	DEBL((AA "Registering barrier client for %s\n" AB,idx2str(thisIndexMax)));
	if (usesAtSync) {
	  ldBarrierHandle = myRec->getLBMgr()->AddLocalBarrierClient(this, &CkMigratable::ResumeFromSyncHelper);
	}
	barrierRegistered=true;
}

void CkMigratable::AtSync(int waitForMigration)
{
	if (!usesAtSync)
		CkAbort("You must set usesAtSync=true in your array element constructor to use AtSync!\n");
	if(CkpvAccess(numLoadBalancers) == 0) {
		ResumeFromSync();
		return;
	}
	myRec->AsyncMigrate(!waitForMigration);
	if (waitForMigration) ReadyMigrate(true);
	ckFinishConstruction();
  DEBL((AA "Element %s going to sync\n" AB,idx2str(thisIndexMax)));
  // model-based load balancing, ask user to provide cpu load
  if (usesAutoMeasure == false) UserSetLBLoad();

  if(_lb_psizer_on || _lb_args.metaLbOn()){
    PUP::sizer ps;
    this->virtual_pup(ps);
    if(_lb_psizer_on)
      setPupSize(ps.size());
    if(_lb_args.metaLbOn())
      myRec->getMetaBalancer()->SetCharePupSize(ps.size());
  }

  if (!_lb_args.metaLbOn()) {
    myRec->getLBMgr()->AtLocalBarrier(ldBarrierHandle);
    return;
  }

  // When MetaBalancer is turned on

  if (atsync_iteration == -1) {
    can_reset = false;
    local_state = OFF;
    prev_load = 0.0;
  }

  atsync_iteration++;
  //CkPrintf("[pe %s] atsync_iter %d && predicted period %d state: %d\n",
  //    idx2str(thisIndexMax), atsync_iteration,
  //    myRec->getMetaBalancer()->getPredictedLBPeriod(), local_state);
  double tmp = prev_load;
  prev_load = myRec->getObjTime();
  double current_load = prev_load - tmp;

  // If the load for the chares are based on certain model, then set the
  // current_load to be whatever is the obj load.
  if (!usesAutoMeasure) {
    current_load = myRec->getObjTime();
  }

  if (atsync_iteration <= myRec->getMetaBalancer()->get_finished_iteration()) {
    CkPrintf("[%d:%s] Error!! Contributing to iter %d < current iter %d\n",
      CkMyPe(), idx2str(thisIndexMax), atsync_iteration,
      myRec->getMetaBalancer()->get_finished_iteration());
    CkAbort("Not contributing to the right iteration\n");
  }

  if (atsync_iteration != 0) {
    myRec->getMetaBalancer()->AddLoad(atsync_iteration, current_load);
  }

  bool is_tentative;
  if (atsync_iteration < myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative)) {
    ResumeFromSync();
  } else if (is_tentative) {
    local_state = PAUSE;
  } else if (local_state == DECIDED) {
    DEBAD(("[%d:%s] Went to load balance iter %d\n", CkMyPe(), idx2str(thisIndexMax), atsync_iteration));
    local_state = LOAD_BALANCE;
    can_reset = true;
    //myRec->AtLocalBarrier(ldBarrierHandle);
  } else {
    DEBAD(("[%d:%s] Went to pause state iter %d\n", CkMyPe(), idx2str(thisIndexMax), atsync_iteration));
    local_state = PAUSE;
  }
}

void CkMigratable::ReadyMigrate(bool ready)
{
	myRec->ReadyMigrate(ready);
}


void CkMigratable::ResumeFromSyncHelper()
{
  DEBL((AA "Element %s resuming from sync\n" AB,idx2str(thisIndexMax)));

  if (_lb_args.metaLbOn()) {
    clearMetaLBData();
  }

  CkLocMgr *localLocMgr = myRec->getLocMgr();
  auto iter = localLocMgr->bufferedActiveRgetMsgs.find(ckGetID());
  if(iter != localLocMgr->bufferedActiveRgetMsgs.end()) {
    localLocMgr->toBeResumeFromSynced.emplace(ckGetID(), this);
  } else {
    ResumeFromSync();
  }
}

void CkMigratable::setMigratable(int migratable) 
{
	myRec->setMigratable(migratable);
}

void CkMigratable::setPupSize(size_t obj_pup_size)
{
	myRec->setPupSize(obj_pup_size);
}

struct CkArrayThreadListener {
        struct CthThreadListener base;
        CkMigratable *mig;
};

static void CkArrayThreadListener_suspend(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        a->mig->ckStopTiming();
}

static void CkArrayThreadListener_resume(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        a->mig->ckStartTiming();
}

static void CkArrayThreadListener_free(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        delete a;
}

void CkMigratable::CkAddThreadListeners(CthThread tid, void *msg)
{
        Chare::CkAddThreadListeners(tid, msg);   // for trace
        CthSetThreadID(tid, thisIndexMax.data()[0], thisIndexMax.data()[1], 
		       thisIndexMax.data()[2]);
	CkArrayThreadListener *a=new CkArrayThreadListener;
	a->base.suspend=CkArrayThreadListener_suspend;
	a->base.resume=CkArrayThreadListener_resume;
	a->base.free=CkArrayThreadListener_free;
	a->mig=this;
	CthAddListener(tid,(struct CthThreadListener *)a);
}
#else
void CkMigratable::setObjTime(double cputime) {}
double CkMigratable::getObjTime() {return 0.0;}

#if CMK_LB_USER_DATA
void *CkMigratable::getObjUserData(int idx) { return NULL; }
#endif

/* no load balancer: need dummy implementations to prevent link error */
void CkMigratable::CkAddThreadListeners(CthThread tid, void *msg)
{
}
#endif


/************************** Location Records: *********************************/


/*----------------- Local:
Matches up the array index with the local index, an
interfaces with the load balancer on behalf of the
represented array elements.
*/
CkLocRec::CkLocRec(CkLocMgr *mgr,bool fromMigration,
                   bool ignoreArrival, const CkArrayIndex &idx_, CmiUInt8 id_)
  :myLocMgr(mgr),idx(idx_), id(id_),
	 deletedMarker(NULL),running(false)
{
#if CMK_LBDB_ON
	DEBL((AA "Registering element %s with load balancer\n" AB,idx2str(idx)));
	nextPe = -1;
	asyncMigrate = false;
	readyMigrate = true;
        enable_measure = true;
#if CMK_FAULT_EVAC
	bounced  = false;
#endif
	lbmgr=mgr->getLBMgr();
	if(_lb_args.metaLbOn())
	  the_metalb=mgr->getMetaBalancer();
#if CMK_GLOBAL_LOCATION_UPDATE
	CmiUInt8 locMgrGid = mgr->getGroupID().idx;
	id_ = ck::ObjID(id_).getElementID();
	id_ |= locMgrGid << ck::ObjID().ELEMENT_BITS;
#endif        
	ldHandle=lbmgr->RegisterObj(mgr->getOMHandle(),
		id_, (void *)this,1);
	if (fromMigration) {
		DEBL((AA "Element %s migrated in\n" AB,idx2str(idx)));
		if (!ignoreArrival)  {
			lbmgr->Migrated(ldHandle, true);
		  // load balancer should ignore this objects movement
		//  AsyncMigrate(true);
		}
	}
#endif

#if CMK_FAULT_EVAC
	asyncEvacuate = true;
#endif
}
CkLocRec::~CkLocRec()
{
	if (deletedMarker!=NULL) *deletedMarker=true;
#if CMK_LBDB_ON
	stopTiming();
	DEBL((AA "Unregistering element %s from load balancer\n" AB,idx2str(idx)));
	lbmgr->UnregisterObj(ldHandle);
#endif
}
void CkLocRec::migrateMe(int toPe) //Leaving this processor
{
	//This will pack us up, send us off, and delete us
//	printf("[%d] migrating migrateMe to %d \n",CkMyPe(),toPe);
	myLocMgr->emigrate(this,toPe);
}

#if CMK_LBDB_ON
void CkLocRec::startTiming(int ignore_running) {
  	if (!ignore_running) running=true;
	DEBL((AA "Start timing for %s at %.3fs {\n" AB,idx2str(idx),CkWallTimer()));
  if (enable_measure) lbmgr->ObjectStart(ldHandle);
}
void CkLocRec::stopTiming(int ignore_running) {
	DEBL((AA "} Stop timing for %s at %.3fs\n" AB,idx2str(idx),CkWallTimer()));
	if ((ignore_running || running) && enable_measure) lbmgr->ObjectStop(ldHandle);
  	if (!ignore_running) running=false;
}
void CkLocRec::setObjTime(double cputime) {
	lbmgr->EstObjLoad(ldHandle, cputime);
}
double CkLocRec::getObjTime() {
        LBRealType walltime, cputime;
        lbmgr->GetObjLoad(ldHandle, walltime, cputime);
        return walltime;
}
#if CMK_LB_USER_DATA
void* CkLocRec::getObjUserData(int idx) {
        return lbmgr->GetDBObjUserData(ldHandle, idx);
}
#endif
#endif

// Attempt to destroy this record. If the location manager is done with the
// record (because all array elements were destroyed) then it will be deleted.
void CkLocRec::destroy(void) {
	myLocMgr->reclaim(this);
}

/**********Added for cosmology (inline function handling without parameter marshalling)***********/

LDObjHandle CkMigratable::timingBeforeCall(int* objstopped){
	LDObjHandle objHandle;
#if CMK_LBDB_ON
	if (getLBMgr()->RunningObject(&objHandle)) {
		*objstopped = 1;
		getLBMgr()->ObjectStop(objHandle);
	}
	myRec->startTiming(1);
#endif

  return objHandle;
}

void CkMigratable::timingAfterCall(LDObjHandle objHandle,int *objstopped){
	myRec->stopTiming(1);
#if CMK_LBDB_ON
	if (*objstopped) {
		 getLBMgr()->ObjectStart(objHandle);
	}
#endif

 return;
}
/****************************************************************************/


bool CkLocRec::invokeEntry(CkMigratable *obj,void *msg,
	int epIdx,bool doFree) 
{

	DEBS((AA "   Invoking entry %d on element %s\n" AB,epIdx,idx2str(idx)));
	bool isDeleted=false; //Enables us to detect deletion during processing
	deletedMarker=&isDeleted;
	startTiming();


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		envelope *env=UsrToEnv(msg);
	//	CkPrintf("ckLocation.C beginExecuteDetailed %d %d \n",env->getEvent(),env->getsetArraySrcPe());
		if (_entryTable[epIdx]->traceEnabled)
        {
            _TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(), ForChareMsg,epIdx,env->getSrcPe(), env->getTotalsize(), idx.getProjectionID(), obj);
            if(_entryTable[epIdx]->appWork)
                _TRACE_BEGIN_APPWORK();
        }
	}
#endif

	if (doFree) 
	   CkDeliverMessageFree(epIdx,msg,obj);
	else /* !doFree */
	   CkDeliverMessageReadonly(epIdx,msg,obj);


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		if (_entryTable[epIdx]->traceEnabled)
        {
            if(_entryTable[epIdx]->appWork)
                _TRACE_END_APPWORK();
			_TRACE_END_EXECUTE();
        }
	}
#endif
#if CMK_LBDB_ON
        if (!isDeleted) checkBufferedMigration();   // check if should migrate
#endif
	if (isDeleted) return false;//We were deleted
	deletedMarker=NULL;
	stopTiming();
	return true;
}

#if CMK_LBDB_ON

void CkLocRec::staticMetaLBResumeWaitingChares(LDObjHandle h, int lb_ideal_period) {
	CkLocRec *el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
	DEBL((AA "MetaBalancer wants to resume waiting chare %s\n" AB,idx2str(el->idx)));
	el->myLocMgr->informLBPeriod(el, lb_ideal_period);
}

void CkLocRec::staticMetaLBCallLBOnChares(LDObjHandle h) {
	CkLocRec *el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
	DEBL((AA "MetaBalancer wants to call LoadBalance on chare %s\n" AB,idx2str(el->idx)));
	el->myLocMgr->metaLBCallLB(el);
}

void CkLocRec::staticMigrate(LDObjHandle h, int dest)
{
	CkLocRec *el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
	DEBL((AA "Load balancer wants to migrate %s to %d\n" AB,idx2str(el->idx),dest));
	el->recvMigrate(dest);
}

void CkLocRec::recvMigrate(int toPe)
{
	// we are in the mode of delaying actual migration
 	// till readyMigrate()
	if (readyMigrate) { migrateMe(toPe); }
	else nextPe = toPe;
}

void CkLocRec::AsyncMigrate(bool use)  
{
        asyncMigrate = use; 
	lbmgr->UseAsyncMigrate(ldHandle, use);
}

bool CkLocRec::checkBufferedMigration()
{
	// we don't migrate in user's code when calling ReadyMigrate(true)
	// we postphone the action to here until we exit from the user code.
	if (readyMigrate && nextPe != -1) {
	    int toPe = nextPe;
	    nextPe = -1;
	    // don't migrate inside the object call
	    migrateMe(toPe);
	    // don't do anything
	    return true;
	}
	return false;
}

int CkLocRec::MigrateToPe()
{
	int pe = nextPe;
	nextPe = -1;
	return pe;
}

void CkLocRec::setMigratable(int migratable)
{
	if (migratable)
	  lbmgr->Migratable(ldHandle);
	else
	  lbmgr->NonMigratable(ldHandle);
}

void CkLocRec::setPupSize(size_t obj_pup_size) {
  lbmgr->setPupSize(ldHandle, obj_pup_size);
}

#endif


// Call ckDestroy for each record, which deletes the record, and ~CkLocRec()
// removes it from the hash table, which would invalidate an iterator.
void CkLocMgr::flushLocalRecs(void)
{
  while (hash.size()) {
    CkLocRec* rec = hash.begin()->second;
    callMethod(rec, &CkMigratable::ckDestroy);
  }
}

// All records are local records after the 64bit ID update
void CkLocMgr::flushAllRecs(void)
{
  flushLocalRecs();
}



/*************************** LocMgr: CREATION *****************************/
// TODO: No longer need to save options AND bounds
CkLocMgr::CkLocMgr(CkArrayOptions opts)
    : idCounter(1), thisProxy(thisgroup), thislocalproxy(thisgroup,CkMyPe()),
      bounds(opts.getBounds()), options(opts) {
	DEBC((AA "Creating new location manager %d\n" AB,thisgroup));
// moved to _CkMigratable_initInfoInit()
//	CkpvInitialize(CkMigratable_initInfo,mig_initInfo);

	duringMigration = false;

//Register with the map object
	mapID = opts.getMap();
	map=((CkArrayMap *)CkLocalBranch(mapID))->getMapObj();
	if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");

        // Figure out the mapping from indices to object IDs if one is possible
        compressor = ck::FixedArrayIndexCompressor::make(bounds);

//Find and register with the load balancer
#if CMK_LBDB_ON
        lbmgrID = _lbmgr;
        metalbID = _metalb;
#endif
        initLB(lbmgrID, metalbID);
}

CkLocMgr::CkLocMgr(CkMigrateMessage* m)
	:IrrGroup(m),thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe())
{
	duringMigration = false;
}

CkLocMgr::~CkLocMgr() {
#if CMK_LBDB_ON
  lbmgr->RemoveLocalBarrierClient(lbBarrierClient);
  lbmgr->DecreaseLocalBarrier(1);
  lbmgr->RemoveLocalBarrierReceiver(lbBarrierReceiver);
  lbmgr->UnregisterOM(myLBHandle);
#endif
}

void CkLocMgr::pup(PUP::er &p){
  if (p.isPacking() && pendingImmigrate.size() > 0)
    CkAbort("Attempting to pup location manager with buffered migration messages."
            " Likely cause is checkpointing before array creation has fully completed\n");
	IrrGroup::pup(p);
	p|mapID;
	p|mapHandle;
	p|lbmgrID;
        p|metalbID;
        p|bounds;
	if(p.isUnpacking()) {
		thisProxy=thisgroup;
		CProxyElement_CkLocMgr newlocalproxy(thisgroup,CkMyPe());
		thislocalproxy=newlocalproxy;
		//Register with the map object
		map=((CkArrayMap *)CkLocalBranch(mapID))->getMapObj();
		if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
                CkArrayIndex emptyIndex;
		// _lbmgr is the fixed global groupID
		initLB(lbmgrID, metalbID);
                compressor = ck::FixedArrayIndexCompressor::make(bounds);
#if __FAULT__
        int count = 0;
        p | count;
        DEBUG(CmiPrintf("[%d] Unpacking Locmgr %d has %d home elements\n",CmiMyPe(),thisgroup.idx,count));
        for(int i=0;i<count;i++){
            CkArrayIndex idx;
            int pe = 0;
            p | idx;
            p | pe;
  //          CmiPrintf("[%d] idx %s is a home element exisiting on pe %d\n",CmiMyPe(),idx2str(idx),pe);
            inform(idx, lookupID(idx), pe);
            CmiUInt8 id = lookupID(idx);
            CkLocRec *rec = elementNrec(id);
            CmiAssert(rec!=NULL);
            CmiAssert(lastKnown(idx) == pe);
        }
#endif
		// delay doneInserting when it is unpacking during restart.
		// to prevent load balancing kicking in
		if (!CkInRestarting()) 
			doneInserting();
	}else{
 /**
 * pack the indexes of elements which have their homes on this processor
 * but dont exist on it.. needed for broadcast after a restart
 * indexes of local elements dont need to be packed
 * since they will be recreated later anyway
 */
#if __FAULT__
        int count=0;
        std::vector<int> pe_list;
        std::vector<CmiUInt8> idx_list;
        for (auto itr = id2pe.begin(); itr != id2pe.end(); ++itr)
            if (homePe(itr->first) == CmiMyPe() && itr->second != CmiMyPe())
            {
                idx_list.push_back(itr->first);
                pe_list.push_back(itr->second);
                count++;
            }

        p | count;
        // syncft code depends on this exact arrangement:
        for (int i=0; i<count; i++)
        {
          p | idx_list[i];
          p | pe_list[i];
        }
#endif

	}
}

/// Add a new local array manager to our list.
void CkLocMgr::addManager(CkArrayID id,CkArray *mgr)
{
  CK_MAGICNUMBER_CHECK
  DEBC((AA "Adding new array manager\n" AB));
  managers[id] = mgr;
  auto i = pendingImmigrate.begin();
  while (i != pendingImmigrate.end()) {
    auto msg = *i;
    if (msg->nManagers <= managers.size()) {
      i = pendingImmigrate.erase(i);
      immigrate(msg);
    } else {
      i++;
    }
  }
}

void CkLocMgr::deleteManager(CkArrayID id, CkArray *mgr) {
  CkAssert(managers[id] == mgr);
  managers.erase(id);

  if (managers.size() == 0)
    delete this;
}

//Tell this element's home processor it now lives "there"
void CkLocMgr::informHome(const CkArrayIndex &idx,int nowOnPe)
{
	int home=homePe(idx);
	if (home!=CkMyPe() && home!=nowOnPe) {
		//Let this element's home Pe know it lives here now
		DEBC((AA "  Telling %s's home %d that it lives on %d.\n" AB,idx2str(idx),home,nowOnPe));
		thisProxy[home].updateLocation(idx, lookupID(idx), nowOnPe);
	}
}

CkLocRec *CkLocMgr::createLocal(const CkArrayIndex &idx, 
		bool forMigration, bool ignoreArrival,
		bool notifyHome)
{
	DEBC((AA "Adding new record for element %s\n" AB,idx2str(idx)));
	CmiUInt8 id = lookupID(idx);

	CkLocRec *rec=new CkLocRec(this, forMigration, ignoreArrival, idx, id);
	insertRec(rec, id);
        inform(idx, id, CkMyPe());

	if (notifyHome) { informHome(idx,CkMyPe()); }
	return rec;
}

// Used to handle messages that were buffered because of active rgets in progress
void CkLocMgr::processAfterActiveRgetsCompleted(CmiUInt8 id) {

    // Call ckJustMigrated
    CkLocRec *myLocRec = elementNrec(id);
    callMethod(myLocRec, &CkMigratable::ckJustMigrated);

    // Call ResumeFromSync on elements that were waiting for rgets
    auto iter2 = toBeResumeFromSynced.find(id);
    if(iter2 != toBeResumeFromSynced.end()) {
      iter2->second->ResumeFromSync();
      toBeResumeFromSynced.erase(iter2);
    }

    // Deliver buffered messages to the elements that were waiting on rgets
    auto iter = bufferedActiveRgetMsgs.find(id);
    if(iter != bufferedActiveRgetMsgs.end()) {
      std::vector<CkArrayMessage *> bufferedMsgs = iter->second;
      bufferedActiveRgetMsgs.erase(iter);
      for(auto msg : bufferedMsgs) {
        CmiHandleMessage(UsrToEnv(msg));
      }
    }
}


void CkLocMgr::deliverAnyBufferedMsgs(CmiUInt8 id, MsgBuffer &buffer)
{
    auto itr = buffer.find(id);
    // If there are no buffered msgs, don't do anything
    if (itr == buffer.end()) return;

    std::vector<CkArrayMessage*> messagesToFlush;
    messagesToFlush.swap(itr->second);

    // deliver all buffered messages
    for (int i = 0; i < messagesToFlush.size(); ++i)
    {
        CkArrayMessage *m = messagesToFlush[i];
        deliverMsg(m, UsrToEnv(m)->getArrayMgr(), id, NULL, CkDeliver_queue);
    }

    CkAssert(itr->second.empty()); // Nothing should have been added, since we
                                   // ostensibly know where the object lives

    // and then, delete the entry in this table of buffered msgs
    buffer.erase(itr);
}

CmiUInt8 CkLocMgr::getNewObjectID(const CkArrayIndex &idx)
{
  CmiUInt8 id;
  if (!lookupID(idx, id)) {
    id = idCounter++ + ((CmiUInt8)CkMyPe() << 24);
    insertID(idx,id);
  }
  return id;
}

//Add a new local array element, calling element's constructor
bool CkLocMgr::addElement(CkArrayID mgr,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{
	CK_MAGICNUMBER_CHECK

        CmiUInt8 id = getNewObjectID(idx);

	CkLocRec *rec = elementNrec(id);
	if (rec == NULL)
	{ //This is the first we've heard of that element-- add new local record
		rec=createLocal(idx,false,false,true);
#if CMK_GLOBAL_LOCATION_UPDATE
                if (homePe(idx) != CkMyPe()) {
                  DEBC((AA "Global location broadcast for new element idx %s "
                        "assigned to %d \n" AB, idx2str(idx), CkMyPe()));
                  thisProxy.updateLocation(id, CkMyPe());
                }
#endif
                
	}
	//rec is *already* local-- must not be the first insertion	
        else
		deliverAnyBufferedMsgs(id, bufferedShadowElemMsgs);
	if (!addElementToRec(rec, managers[mgr], elt, ctorIdx, ctorMsg)) return false;
	elt->ckFinishConstruction();
	return true;
}

//As above, but shared with the migration code
bool CkLocMgr::addElementToRec(CkLocRec *rec,CkArray *mgr,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{//Insert the new element into its manager's local list
  CmiUInt8 id = lookupID(rec->getIndex());
  if (mgr->getEltFromArrMgr(id))
    CkAbort("Cannot insert array element twice!");
  mgr->putEltInArrMgr(id, elt); //Local element table

//Call the element's constructor
	DEBC((AA "Constructing element %s of array\n" AB,idx2str(rec->getIndex())));
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
	i.locRec=rec;
	i.chareType=_entryTable[ctorIdx]->chareIdx;

#ifndef CMK_CHARE_USE_PTR
  int callingChareIdx = CkpvAccess(currentChareIdx);
  CkpvAccess(currentChareIdx) = -1;
#endif

	if (!rec->invokeEntry(elt,ctorMsg,ctorIdx,true)) return false;

#ifndef CMK_CHARE_USE_PTR
  CkpvAccess(currentChareIdx) = callingChareIdx;
#endif

#if CMK_OUT_OF_CORE
	/* Register new element with out-of-core */
	PUP::sizer p_getSize;
	elt->virtual_pup(p_getSize);
	elt->prefetchObjID=CooRegisterObject(&CkArrayElementPrefetcher,p_getSize.size(),elt);
#endif
	
	return true;
}

// TODO: suppressIfHere doesn't seem to be useful anymore because we return
// early when peToTell == CkMyPe()
void CkLocMgr::requestLocation(const CkArrayIndex &idx, const int peToTell,
                               bool suppressIfHere, int ifNonExistent, int chareType, CkArrayID mgr) {
  int onPe = -1;
  DEBN(("%d requestLocation for %s peToTell %d\n", CkMyPe(), idx2str(idx), peToTell));

  if (peToTell == CkMyPe())
    return;

  CmiUInt8 id;
  if (lookupID(idx,id)) {
    // We found the ID so update the location for peToTell
    onPe = lastKnown(idx);
    thisProxy[peToTell].updateLocation(idx, id, onPe);
  } else {
    // We don't know the ID so buffer the location request
    DEBN(("%d Buffering ID/location req for %s\n", CkMyPe(), idx2str(idx)));
    bufferedLocationRequests[idx].emplace_back(peToTell, suppressIfHere);

    switch (ifNonExistent) {
    case CkArray_IfNotThere_createhome:
      demandCreateElement(idx, chareType, CkMyPe(), mgr);
      break;
    case CkArray_IfNotThere_createhere:
      demandCreateElement(idx, chareType, peToTell, mgr);
      break;
    default:
      break;
    }
  }
}

void CkLocMgr::requestLocation(CmiUInt8 id, const int peToTell,
                               bool suppressIfHere) {
  int onPe = -1;
  DEBN(("%d requestLocation for %u peToTell %d\n", CkMyPe(), id, peToTell));

  if (peToTell == CkMyPe())
    return;

  onPe = lastKnown(id);

  if (suppressIfHere && peToTell == CkMyPe())
    return;

  thisProxy[peToTell].updateLocation(id, onPe);
}

void CkLocMgr::updateLocation(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe) {
  DEBN(("%d updateLocation for %s on %d\n", CkMyPe(), idx2str(idx), nowOnPe));
  inform(idx, id, nowOnPe);
  deliverAnyBufferedMsgs(id, bufferedRemoteMsgs);
}

void CkLocMgr::updateLocation(CmiUInt8 id, int nowOnPe) {
  DEBN(("%d updateLocation for %s on %d\n", CkMyPe(), idx2str(idx), nowOnPe));
  inform(id, nowOnPe);
  deliverAnyBufferedMsgs(id, bufferedRemoteMsgs);
}

void CkLocMgr::inform(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe) {
  // On restart, conservatively determine the next 'safe' ID to
  // generate for new elements by the max over all of the elements with
  // IDs corresponding to each PE
  if (CkInRestarting()) {
    CmiUInt8 maskedID = id & ((1u << 24) - 1);
    CmiUInt8 origPe = id >> 24;
    if (origPe == CkMyPe()) {
      if (maskedID >= idCounter)
        idCounter = maskedID + 1;
    } else {
      if (origPe < CkNumPes())
        thisProxy[origPe].updateLocation(idx, id, nowOnPe);
    }
  }

  insertID(idx,id);
  id2pe[id] = nowOnPe;

  auto itr = bufferedLocationRequests.find(idx);
  if (itr != bufferedLocationRequests.end()) {
    for (std::vector<std::pair<int, bool> >::iterator i = itr->second.begin();
         i != itr->second.end(); ++i) {
      int peToTell = i->first;
      DEBN(("%d Replying to buffered ID/location req to pe %d\n", CkMyPe(), peToTell));
      if (peToTell != CkMyPe())
        thisProxy[peToTell].updateLocation(idx, id, nowOnPe);
    }
    bufferedLocationRequests.erase(itr);
  }

  deliverAnyBufferedMsgs(id, bufferedMsgs);

  auto idx_itr = bufferedIndexMsgs.find(idx);
  if (idx_itr != bufferedIndexMsgs.end()) {
    vector<CkArrayMessage*> &msgs = idx_itr->second;
    for (int i = 0; i < msgs.size(); ++i) {
      envelope *env = UsrToEnv(msgs[i]);
      CkGroupID mgr = ck::ObjID(env->getRecipientID()).getCollectionID();
      env->setRecipientID(ck::ObjID(mgr, id));
      deliverMsg(msgs[i], mgr, id, &idx, CkDeliver_queue);
    }
    bufferedIndexMsgs.erase(idx_itr);
  }
}

void CkLocMgr::inform(CmiUInt8 id, int nowOnPe) {
  id2pe[id] = nowOnPe;
  deliverAnyBufferedMsgs(id, bufferedMsgs);
}



/*************************** LocMgr: DELETION *****************************/
// This index may no longer be used -- check if any of our managers are still
// using it, and if not delete it and clean up all traces of it on other PEs.
void CkLocMgr::reclaim(CkLocRec* rec) {
	CK_MAGICNUMBER_CHECK
	// Return early if the record is still in use by any of our arrays
	for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		if (itr->second->lookup(rec->getID())) return;
	}
	removeFromTable(rec->getID());
	
	DEBC((AA "Destroying record for element %s\n" AB,idx2str(rec->getIndex())));
	if (!duringMigration) 
	{ //This is a local element dying a natural death
		int home=homePe(rec->getIndex());
		if (home!=CkMyPe())
#if CMK_MEM_CHECKPOINT
	        if (!CkInRestarting()) // all array elements are removed anyway
#endif
	        if (!duringDestruction)
	            thisProxy[home].reclaimRemote(rec->getIndex(),CkMyPe());
	}
	delete rec;
}

// The location record associated with idx has been deleted on a remote PE, so
// we should free all of our caching associated with that index.
void CkLocMgr::reclaimRemote(const CkArrayIndex &idx,int deletedOnPe) {
	DEBC((AA "Our element %s died on PE %d\n" AB,idx2str(idx),deletedOnPe));

	CmiUInt8 id;
	if (!lookupID(idx, id)) CkAbort("Cannot find ID for the given index\n");

	// Delete the id and index from our location caching
	id2pe.erase(id);
	idx2id.erase(idx);

	// Assert that there were no undelivered messages for the dying element
	CkAssert(bufferedMsgs.count(id) == 0);
	CkAssert(bufferedRemoteMsgs.count(id) == 0);
	CkAssert(bufferedShadowElemMsgs.count(id) == 0);
	CkAssert(bufferedLocationRequests.count(idx) == 0);
	CkAssert(bufferedIndexMsgs.count(idx) == 0);
}

void CkLocMgr::removeFromTable(const CmiUInt8 id) {
#if CMK_ERROR_CHECKING
	//Make sure it's actually in the table before we delete it
	if (NULL==elementNrec(id))
		CkAbort("CkLocMgr::removeFromTable called on invalid index!");
#endif
		hash.erase(id);
#if CMK_ERROR_CHECKING
	//Make sure it's really gone
	if (NULL!=elementNrec(id))
		CkAbort("CkLocMgr::removeFromTable called, but element still there!");
#endif
}

/************************** LocMgr: MESSAGING *************************/
/// Deliver message to this element, going via the scheduler if local
/// @return 0 if object local, 1 if not
int CkLocMgr::deliverMsg(CkArrayMessage *msg, CkArrayID mgr, CmiUInt8 id, const CkArrayIndex* idx, CkDeliver_t type, int opts) {
  CkLocRec *rec = elementNrec(id);

#if CMK_LBDB_ON
  if ((idx || compressor) && type==CkDeliver_queue && !(opts & CK_MSG_LB_NOTRACE) && lbmgr->CollectingCommStats())
  {
#if CMK_GLOBAL_LOCATION_UPDATE
    CmiUInt8 locMgrGid = thisgroup.idx;
    id = ck::ObjID(id).getElementID();
    id |= locMgrGid << ck::ObjID().ELEMENT_BITS;
#endif
    lbmgr->Send(myLBHandle
                   , id
                   , UsrToEnv(msg)->getTotalsize()
                   , lastKnown(id)
                   , 1);
  }
#endif

  // Known, remote location or unknown location
  if (rec == NULL)
  {
    // known location
    int destPE = whichPE(id);
    if (destPE != -1)
    {
#if CMK_FAULT_EVAC
      if((!CmiNodeAlive(destPE) && destPE != allowMessagesOnly)){
        CkAbort("Cannot send to a chare on a dead node");
      }
#endif
      msg->array_hops()++;
      CkArrayManagerDeliver(destPE,msg,opts);
      return true;
    }
    // unknown location
    deliverUnknown(msg,idx,type,opts);
    return true;
  }

  // Send via the msg q
  if (type==CkDeliver_queue)
  {
    CkArrayManagerDeliver(CkMyPe(),msg,opts);
    return true;
  }

  CkAssert(mgr == UsrToEnv(msg)->getArrayMgr());
  CkArray *arr = managers[mgr];
  if (!arr) {
    bufferedShadowElemMsgs[id].push_back(msg);
    return true;
  }
  CkMigratable *obj = arr->lookup(id);
  if (obj==NULL) {//That sibling of this object isn't created yet!
    if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer)
      return demandCreateElement(msg, rec->getIndex(), CkMyPe(),type);
    else { // BUFFERING message for nonexistent element
      bufferedShadowElemMsgs[id].push_back(msg);
      return true;
    }
  }
        
  if (msg->array_hops()>1)
    multiHop(msg);
#if CMK_LBDB_ON
  // if there is a running obj being measured, stop it temporarily
  LDObjHandle objHandle;
  bool wasAnObjRunning = false;
  if ((wasAnObjRunning = lbmgr->RunningObject(&objHandle)))
    lbmgr->ObjectStop(objHandle);
#endif
  // Finally, call the entry method
  bool result = ((CkLocRec*)rec)->invokeEntry(obj,(void *)msg,msg->array_ep(), true);
#if CMK_LBDB_ON
  if (wasAnObjRunning) lbmgr->ObjectStart(objHandle);
#endif
  return result;
}

void CkLocMgr::sendMsg(CkArrayMessage *msg, CkArrayID mgr, const CkArrayIndex &idx, CkDeliver_t type, int opts) {
  CK_MAGICNUMBER_CHECK
  DEBS((AA "send %s\n" AB,idx2str(idx)));
  envelope *env = UsrToEnv(msg);
  env->setMsgtype(ForArrayEltMsg);

  checkInBounds(idx);

  if (type==CkDeliver_queue)
    _TRACE_CREATION_DETAILED(env, msg->array_ep());

  CmiUInt8 id;
  if (lookupID(idx, id)) {
    env->setRecipientID(ck::ObjID(mgr, id));
    deliverMsg(msg, mgr, id, &idx, type, opts);
    return;
  }

  env->setRecipientID(ck::ObjID(mgr, 0));

  int home = homePe(idx);
  if (home != CkMyPe()) {
    if (bufferedIndexMsgs.find(idx) == bufferedIndexMsgs.end())
      thisProxy[home].requestLocation(idx, CkMyPe(), false, msg->array_ifNotThere(), _entryTable[env->getEpIdx()]->chareIdx, mgr);
    bufferedIndexMsgs[idx].push_back(msg);

    return;
  }

  // We are the home, and there's no ID for this index yet - i.e. its
  // construction hasn't reached us yet.
  if (managers.find(mgr) == managers.end()) {
    // Even the manager for this array hasn't been constructed here yet
    if (CkInRestarting()) {
      // during restarting, this message should be ignored
      delete msg;
    } else {
      // Eventually, the manager will be created, and the element inserted, and
      // it will get pulled back out
      // 
      // XXX: Is demand creation ever possible in this case? I don't see why not
      bufferedIndexMsgs[idx].push_back(msg);
    }
    return;
  }

  // Buffer the msg
  bufferedIndexMsgs[idx].push_back(msg);

  // If requested, demand-create the element:
  if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
    demandCreateElement(msg, idx, -1, type);
  }
}

/// This index is not hashed-- somehow figure out what to do.
void CkLocMgr::deliverUnknown(CkArrayMessage *msg, const CkArrayIndex* idx, CkDeliver_t type, int opts)
{
  CK_MAGICNUMBER_CHECK
  CmiUInt8 id = msg->array_element_id();
  int home;
  if (idx) home = homePe(*idx);
  else home = homePe(id);

  if (home != CkMyPe()) {// Forward the message to its home processor
    id2pe[id] = home;
    if (UsrToEnv(msg)->getTotalsize() < _messageBufferingThreshold) {
      DEBM((AA "Forwarding message for unknown %u to home %d \n" AB, id, home));
      msg->array_hops()++;
      CkArrayManagerDeliver(home, msg, opts);
    } else {
      DEBM((AA "Buffering message for unknown %u, home %d \n" AB, id, home));
      if (bufferedRemoteMsgs.find(id) == bufferedRemoteMsgs.end())
        thisProxy[home].requestLocation(id, CkMyPe(), false);
      bufferedRemoteMsgs[id].push_back(msg);
    }
  } else { // We *are* the home processor:
    //Check if the element's array manager has been registered yet:
    //No manager yet-- postpone the message (stupidly)
    if (managers.find(UsrToEnv((void*)msg)->getArrayMgr()) == managers.end()) {
      if (CkInRestarting()) {
        // during restarting, this message should be ignored
        delete msg;
      } else {
        CkArrayManagerDeliver(CkMyPe(),msg);
      }
    } else { // Has a manager-- must buffer the message
      // Buffer the msg
      bufferedMsgs[id].push_back(msg);
      // If requested, demand-create the element:
      if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
        CkAbort("Demand creation of elements is currently unimplemented");
      }
    }
  }
}

void CkLocMgr::demandCreateElement(const CkArrayIndex &idx, int chareType, int onPe, CkArrayID mgr)
{
  int ctor=_chareTable[chareType]->getDefaultCtor();
  if (ctor==-1) CkAbort("Can't create array element to handle message--\n"
                        "The element has no default constructor in the .ci file!\n");

  //Find the manager and build the element
  DEBC((AA "Demand-creating element %s on pe %d\n" AB,idx2str(idx),onPe));
  inform(idx, getNewObjectID(idx), onPe);
  CProxy_CkArray(mgr)[onPe].demandCreateElement(idx, ctor, CkDeliver_inline);
}

bool CkLocMgr::demandCreateElement(CkArrayMessage *msg, const CkArrayIndex &idx, int onPe,CkDeliver_t type)
{
	CK_MAGICNUMBER_CHECK
	int chareType=_entryTable[msg->array_ep()]->chareIdx;
	int ctor=_chareTable[chareType]->getDefaultCtor();
	if (ctor==-1) CkAbort("Can't create array element to handle message--\n"
			      "The element has no default constructor in the .ci file!\n");
	if (onPe==-1) 
	{ //Decide where element needs to live
		if (msg->array_ifNotThere()==CkArray_IfNotThere_createhere) 
			onPe=UsrToEnv(msg)->getsetArraySrcPe();
		else //Createhome
			onPe=homePe(idx);
	}
	
	//Find the manager and build the element
	DEBC((AA "Demand-creating element %s on pe %d\n" AB,idx2str(idx),onPe));
	CProxy_CkArray(UsrToEnv((void *)msg)->getArrayMgr())[onPe].demandCreateElement(idx, ctor, type);
        return onPe == CkMyPe();
}

//This message took several hops to reach us-- fix it
void CkLocMgr::multiHop(CkArrayMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	int srcPe=msg->array_getSrcPe();
	if (srcPe==CkMyPe())
          DEB((AA "Odd routing: local element %u is %d hops away!\n" AB, msg->array_element_id(),msg->array_hops()));
	else
	{//Send a routing message letting original sender know new element location
          DEBS((AA "Sending update back to %d for element %u\n" AB, srcPe, msg->array_element_id()));
          thisProxy[srcPe].updateLocation(msg->array_element_id(), CkMyPe());
	}
}

void CkLocMgr::checkInBounds(const CkArrayIndex &idx)
{
#if CMK_ERROR_CHECKING
  if (bounds.nInts > 0) {
    CkAssert(idx.dimension == bounds.dimension);
    bool shorts = idx.dimension > 3;

    for (int i = 0; i < idx.dimension; ++i) {
      unsigned int thisDim = shorts ? idx.indexShorts[i] : idx.index[i];
      unsigned int thatDim = shorts ? bounds.indexShorts[i] : bounds.index[i];
      CkAssert(thisDim < thatDim);
    }
  }
#endif
}

/************************** LocMgr: ITERATOR *************************/
CkLocation::CkLocation(CkLocMgr *mgr_, CkLocRec *rec_)
	:mgr(mgr_), rec(rec_) {}
	
const CkArrayIndex &CkLocation::getIndex(void) const {
	return rec->getIndex();
}

CmiUInt8 CkLocation::getID() const {
	return rec->getID();
}

void CkLocation::destroyAll() {
	mgr->callMethod(rec, &CkMigratable::ckDestroy);
}

void CkLocation::pup(PUP::er &p) {
	mgr->pupElementsFor(p,rec,CkElementCreation_migrate);
}

CkLocIterator::~CkLocIterator() {}

/// Iterate over our local elements:
void CkLocMgr::iterate(CkLocIterator &dest) {
  //Poke through the hash table for local ArrayRecs.
  for (LocRecHash::iterator it = hash.begin(); it != hash.end(); it++) {
    CkLocation loc(this,it->second);
    dest.addLocation(loc);
  }
}




/************************** LocMgr: MIGRATION *************************/
void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec *rec,
		CkElementCreation_t type,bool rebuild)
{
	p.comment("-------- Array Location --------");

	//First pup the element types
	// (A separate loop so ckLocal works even in element pup routines)
    for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		int elCType;
                CkArray *arr = itr->second;
		if (!p.isUnpacking())
		{ //Need to find the element's existing type
			CkMigratable *elt = arr->getEltFromArrMgr(rec->getID());
			if (elt) elCType=elt->ckGetChareType();
			else elCType=-1; //Element hasn't been created
		}
		p(elCType);
		if (p.isUnpacking() && elCType!=-1) {
			//Create the element
			CkMigratable *elt = arr->allocateMigrated(elCType, type);
			int migCtorIdx=_chareTable[elCType]->getMigCtor();
			//Insert into our tables and call migration constructor
			if (!addElementToRec(rec,arr,elt,migCtorIdx,NULL)) return;
                        if (type==CkElementCreation_resume)
                        { // HACK: Re-stamp elements on checkpoint resume--
                          //  this restores, e.g., reduction manager's gcount
                          arr->stampListenerData(elt);
                        }
		}
	}
	//Next pup the element data
    for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getID());
		if (elt!=NULL)
                {
                        elt->virtual_pup(p);
#if CMK_ERROR_CHECKING
                        if (p.isUnpacking()) elt->sanitycheck();
#endif
                }
	}
#if CMK_MEM_CHECKPOINT
	if(rebuild){
	  ArrayElement *elt;
	  std::vector<CkMigratable *> list;
	  migratableList(rec, list);
	  CmiAssert(!list.empty());
	  for (int l=0; l<list.size(); l++) {
		//    reset, may not needed now
		// for now.
		for (int i=0; i<CK_ARRAYLISTENER_MAXLEN; i++) {
			ArrayElement * elt = (ArrayElement *)list[l];
		  contributorInfo *c=(contributorInfo *)&elt->listenerData[i];
		  if (c) c->redNo = 0;
		}
	  }
		
	}
#endif
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec *rec,CkMigratable_voidfn_t fn)
{
    for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getID());
		if (el) (el->* fn)();
	}
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec *rec,CkMigratable_voidfn_arg_t fn,     void * data)
{
    for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getID());
		if (el) (el->* fn)(data);
	}
}

/// return a list of migratables in this local record
void CkLocMgr::migratableList(CkLocRec *rec, std::vector<CkMigratable *> &list)
{
        for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
                CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getID());
                if (elt) list.push_back(elt);
        }
}

/// Migrate this local element away to another processor.
void CkLocMgr::emigrate(CkLocRec *rec,int toPe)
{
	CK_MAGICNUMBER_CHECK
	if (toPe==CkMyPe()) return; //You're already there!

#if CMK_FAULT_EVAC
	/*
		if the toProcessor is already marked as invalid, dont emigrate
		Shouldn't happen but might
	*/
	if(!CmiNodeAlive(toPe)){
		return;
	}
#endif

	CkArrayIndex idx=rec->getIndex();
        CmiUInt8 id = rec->getID();

#if CMK_OUT_OF_CORE
	/* Load in any elements that are out-of-core */
    for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getIndex());
		if (el) if (!el->isInCore) CooBringIn(el->prefetchObjID);
	}
#endif

	//Let all the elements know we're leaving
	callMethod(rec,&CkMigratable::ckAboutToMigrate);
	/*EVAC*/

//First pass: find size of migration message
	size_t bufSize;
	{
		PUP::sizer p;
		pupElementsFor(p,rec,CkElementCreation_migrate);
		bufSize=p.size(); 
	}
#if CMK_ERROR_CHECKING
	if (bufSize > std::numeric_limits<int>::max()) {
		CkAbort("Cannot migrate an object with size greater than %d bytes!\n", std::numeric_limits<int>::max());
	}
#endif

//Allocate and pack into message
	CkArrayElementMigrateMessage *msg = new (bufSize, 0) CkArrayElementMigrateMessage(idx, id,
#if CMK_LBDB_ON
		rec->isAsyncMigrate(),
#else
		false,
#endif
		bufSize, managers.size(),
#if CMK_FAULT_EVAC
    rec->isBounced()
#else
    false
#endif
    );

	{
		PUP::toMem p(msg->packData); 
		p.becomeDeleting(); 
		pupElementsFor(p,rec,CkElementCreation_migrate);
		if (p.size()!=bufSize) {
			CkError("ERROR! Array element claimed it was %zu bytes to a "
				"sizing PUP::er, but copied %zu bytes into the packing PUP::er!\n",
				bufSize,p.size());
			CkAbort("Array element's pup routine has a direction mismatch.\n");
		}
	}

	DEBM((AA "Migrated index size %s to %d \n" AB,idx2str(idx),toPe));	

	thisProxy[toPe].immigrate(msg);

	duringMigration=true;
	for (auto itr = managers.begin(); itr != managers.end(); ++itr) {
		itr->second->deleteElt(id);
	}
	duringMigration=false;

	//The element now lives on another processor-- tell ourselves and its home
	inform(idx, id, toPe);
	informHome(idx,toPe);

#if !CMK_LBDB_ON && CMK_GLOBAL_LOCATION_UPDATE
        DEBM((AA "Global location update. idx %s " 
              "assigned to %d \n" AB,idx2str(idx),toPe));
        thisProxy.updateLocation(id, toPe);
#endif

	CK_MAGICNUMBER_CHECK
}

#if CMK_LBDB_ON
void CkLocMgr::informLBPeriod(CkLocRec *rec, int lb_ideal_period) {
	callMethod(rec,&CkMigratable::recvLBPeriod, (void *)&lb_ideal_period);
}

void CkLocMgr::metaLBCallLB(CkLocRec *rec) {
	callMethod(rec, &CkMigratable::metaLBCallLB);
}
#endif

/**
  Migrating array element is arriving on this processor.
*/
void CkLocMgr::immigrate(CkArrayElementMigrateMessage *msg)
{
	const CkArrayIndex &idx=msg->idx;
		
	PUP::fromMem p(msg->packData); 
	
	if (msg->nManagers < managers.size())
		CkAbort("Array element arrived from location with fewer managers!\n");
	if (msg->nManagers > managers.size()) {
		//Some array managers haven't registered yet -- buffer the message
		DEBM((AA "Buffering %s immigrate msg waiting for array registration\n" AB,idx2str(idx)));
		pendingImmigrate.push_back(msg);
		return;
	}

	insertID(idx,msg->id);

	//Create a record for this element
	CkLocRec *rec=createLocal(idx,true,msg->ignoreArrival,false /* home told on departure */ );
	
	envelope *env = UsrToEnv(msg);
	CmiAssert(CpvAccess(newZCPupGets).empty()); // Ensure that vector is empty
	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_migrate);
	bool zcRgetsActive = !CpvAccess(newZCPupGets).empty();
	if(zcRgetsActive) {
		// newZCPupGets is not empty, rgets need to be launched
		// newZCPupGets is populated with NcpyOperationInfo during pupElementsFor by pup_buffer calls that require Rgets
		// Issue Rgets using the populated newZCPupGets vector
		zcPupIssueRgets(msg->id, this);
	}
	CpvAccess(newZCPupGets).clear(); // Clear this to reuse the vector
	if (p.size()!=msg->length) {
		CkError("ERROR! Array element claimed it was %d bytes to a"
			"packing PUP::er, but %zu bytes in the unpacking PUP::er!\n",
			msg->length,p.size());
		CkError("(I have %zu managers; it claims %d managers)\n",
			managers.size(), msg->nManagers);
		
		CkAbort("Array element's pup routine has a direction mismatch.\n");
	}

#if CMK_FAULT_EVAC
	/*
			if this element came in as a result of being bounced off some other process,
			then it needs to be resumed. It is assumed that it was bounced because load 
			balancing caused it to move into a processor which later crashed
	*/
	if(msg->bounced){
		callMethod(rec,&CkMigratable::ResumeFromSync);
	}
#endif

	if(!zcRgetsActive) {
		//Let all the elements know we've arrived
		callMethod(rec,&CkMigratable::ckJustMigrated);
	}

#if CMK_FAULT_EVAC
	/*
		If this processor has started evacuating array elements on it 
		dont let new immigrants in. If they arrive send them to what
		would be their correct homePE.
		Leave a record here mentioning the processor where it got sent
	*/
	if(CkpvAccess(startedEvac)){
		int newhomePE = getNextPE(idx);
		DEBM((AA "Migrated into failed processor index size %s resent to %d \n" AB,idx2str(idx),newhomePE));	
		int targetPE=getNextPE(idx);
		//set this flag so that load balancer is not informed when
		//this element migrates
		rec->AsyncMigrate(true);
		rec->Bounced(true);
		emigrate(rec,targetPE);
	}
#endif

	delete msg;
}

void CkLocMgr::restore(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p)
{
	insertID(idx,id);

	CkLocRec *rec=createLocal(idx,false,false,false);
	
	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_restore);

	callMethod(rec,&CkMigratable::ckJustRestored);
}


/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
void CkLocMgr::resume(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p, bool notify,bool rebuild)
{
	insertID(idx,id);

	CkLocRec *rec=createLocal(idx,false,false,notify /* home doesn't know yet */ );

	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_resume,rebuild);

	callMethod(rec,&CkMigratable::ckJustMigrated);
}

/********************* LocMgr: UTILITY ****************/
void CkMagicNumber_impl::badMagicNumber(
	int expected,const char *file,int line,void *obj) const
{
	CkError("FAILURE on pe %d, %s:%d> Expected %p's magic number "
		"to be 0x%08x; but found 0x%08x!\n", CkMyPe(),file,line,obj,
		expected, magic);
	CkAbort("Bad magic number detected!  This implies either\n"
		"the heap or a message was corrupted!\n");
}
CkMagicNumber_impl::CkMagicNumber_impl(int m) :magic(m) { }

int CkLocMgr::whichPE(const CkArrayIndex &idx) const
{
  CmiUInt8 id;
  if (!lookupID(idx, id))
    return -1;

  IdPeMap::const_iterator itr = id2pe.find(id);
  return (itr != id2pe.end() ? itr->second : -1);
}

int CkLocMgr::whichPE(const CmiUInt8 id) const
{
  IdPeMap::const_iterator itr = id2pe.find(id);
  return (itr != id2pe.end() ? itr->second : -1);
}

//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(const CkArrayIndex &idx) {
	CkLocMgr *vthis=(CkLocMgr *)this;//Cast away "const"
	int pe = whichPE(idx);
	if (pe==-1) return homePe(idx);
	else{
#if CMK_FAULT_EVAC
		if(!CmiNodeAlive(pe)){
			CkAbort("Last known PE is no longer alive");
		}
#endif
		return pe;
	}	
}

//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(CmiUInt8 id) {
  int pe = whichPE(id);
  if (pe==-1) return homePe(id);
  else{
#if CMK_FAULT_EVAC
    if(!CmiNodeAlive(pe)){
      CkAbort("Last known PE is no longer alive");
    }
#endif
    return pe;
  }	
}

/// Return true if this array element lives on another processor
bool CkLocMgr::isRemote(const CkArrayIndex &idx,int *onPe) const
{
    int pe = whichPE(idx);
    /* not definitely a remote element */
    if (pe == -1 || pe == CkMyPe())
        return false;
    // element is indeed remote
    *onPe = pe;
    return true;
}

static const char *rec2str[]={
    "base (INVALID)",//Base class (invalid type)
    "local",//Array element that lives on this Pe
};


// If we are deleting our last array manager set duringDestruction to true to
// avoid sending out unneeded reclaimRemote messages.
void CkLocMgr::setDuringDestruction(bool _duringDestruction) {
  duringDestruction = (_duringDestruction && managers.size() == 1);
}

//Add given element array record at idx, replacing the existing record
void CkLocMgr::insertRec(CkLocRec *rec, const CmiUInt8 &id) {
    CkLocRec *old_rec = elementNrec(id);
    hash[id] = rec;
    delete old_rec;
}

//Call this on an unrecognized array index
static void abort_out_of_bounds(const CkArrayIndex &idx)
{
  CkPrintf("ERROR! Unknown array index: %s\n",idx2str(idx));
  CkAbort("Array index out of bounds\n");
}

//Look up array element in hash table.  Index out-of-bounds if not found.
// TODO: Could this take an ID instead?
CkLocRec *CkLocMgr::elementRec(const CkArrayIndex &idx) {
#if ! CMK_ERROR_CHECKING
//Assume the element will be found
  return hash[lookupID(idx)];
#else
//Include an out-of-bounds check if the element isn't found
  CmiUInt8 id;
  CkLocRec *rec = NULL;
  if (lookupID(idx, id) && (rec = elementNrec(id))) {
	return rec;
  } else {
	if (rec==NULL) abort_out_of_bounds(idx);
	return NULL;
  }
#endif
}

//Look up array element in hash table.  Return NULL if not there.
CkLocRec *CkLocMgr::elementNrec(const CmiUInt8 id) {
  LocRecHash::iterator it = hash.find(id);
  return it == hash.end() ? NULL : it->second;
}

struct LocalElementCounter :  public CkLocIterator
{
    unsigned int count;
    LocalElementCounter() : count(0) {}
    void addLocation(CkLocation &loc)
	{ ++count; }
};

unsigned int CkLocMgr::numLocalElements()
{
    LocalElementCounter c;
    iterate(c);
    return c.count;
}


/********************* LocMgr: LOAD BALANCE ****************/

#if !CMK_LBDB_ON
//Empty versions of all load balancer calls
void CkLocMgr::initLB(CkGroupID lbmgrID_, CkGroupID metalbID_) {}
void CkLocMgr::startInserting(void) {}
void CkLocMgr::doneInserting(void) {}
void CkLocMgr::dummyAtSync(void) {}
void CkLocMgr::AtSyncBarrierReached(void) {}
#endif


#if CMK_LBDB_ON
void CkLocMgr::initLB(CkGroupID lbmgrID_, CkGroupID metalbID_)
{ //Find and register with the load balancer
	lbmgr = (LBManager *)CkLocalBranch(lbmgrID_);
	if (lbmgr == nullptr)
		CkAbort("LBManager not yet created?\n");
	DEBL((AA "Connected to load balancer %p\n" AB,lbmgr));
	if(_lb_args.metaLbOn()){
	  the_metalb = (MetaBalancer *)CkLocalBranch(metalbID_);
	  if (the_metalb == 0)
		  CkAbort("MetaBalancer not yet created?\n");
	}
	// Register myself as an object manager
	LDOMid myId;
	myId.id = thisgroup;
	LDCallbacks myCallbacks;
	myCallbacks.migrate = (LDMigrateFn)CkLocRec::staticMigrate;
	myCallbacks.setStats = NULL;
	myCallbacks.queryEstLoad = NULL;
  myCallbacks.metaLBResumeWaitingChares =
      (LDMetaLBResumeWaitingCharesFn)CkLocRec::staticMetaLBResumeWaitingChares;
  myCallbacks.metaLBCallLBOnChares =
      (LDMetaLBCallLBOnCharesFn)CkLocRec::staticMetaLBCallLBOnChares;
	myLBHandle = lbmgr->RegisterOM(myId,this,myCallbacks);

	// Tell the lbdb that I'm registering objects
	lbmgr->RegisteringObjects(myLBHandle);

	/*Set up the dummy barrier-- the load balancer needs
	  us to call Registering/DoneRegistering during each AtSync,
	  and this is the only way to do so.
	*/
	lbBarrierReceiver = lbmgr->AddLocalBarrierReceiver(this, &CkLocMgr::AtSyncBarrierReached);
	lbBarrierClient = lbmgr->AddLocalBarrierClient(this, &CkLocMgr::dummyResumeFromSync);
	dummyAtSync();
}

void CkLocMgr::dummyAtSync(void)
{
	DEBL((AA "dummyAtSync called\n" AB));
	lbmgr->AtLocalBarrier(lbBarrierClient);
}

void CkLocMgr::dummyResumeFromSync()
{
	DEBL((AA "DummyResumeFromSync called\n" AB));
	lbmgr->DoneRegisteringObjects(myLBHandle);
	dummyAtSync();
}
void CkLocMgr::AtSyncBarrierReached()
{
	DEBL((AA "AtSyncBarrierReached called\n" AB));
	lbmgr->RegisteringObjects(getOMHandle());
}

void CkLocMgr::startInserting(void)
{
	lbmgr->RegisteringObjects(myLBHandle);
}
void CkLocMgr::doneInserting(void)
{
	lbmgr->DoneRegisteringObjects(myLBHandle);
}
#endif

#include "CkLocation.def.h"


