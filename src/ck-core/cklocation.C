/*
Location manager: keeps track of an indexed set of migratable
objects.  Used by the array manager to locate array elements,
interact with the load balancer, and perform migrations.

Orion Sky Lawlor, olawlor@acm.org 9/29/2001
*/
#include "charm++.h"
#include "register.h"
#include "ck.h"
#include "trace.h"

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

/************************** Debugging Utilities **************/
//For debugging: convert given index to a string
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

static const char *idx2str(const CkArrayMessage *m)
{
	return idx2str(((CkArrayMessage *)m)->array_index());
}

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
#   define AA "LocMgr on %d: "
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


#if CMK_LBDB_ON
/*LBDB object handles are fixed-sized, and not necc.
the same size as ArrayIndices.
*/
static LDObjid idx2LDObjid(const CkArrayIndex &idx)
{
  LDObjid r;
  int i;
  const int *data=idx.data();
  if (OBJ_ID_SZ>=idx.nInts) {
    for (i=0;i<idx.nInts;i++)
      r.id[i]=data[i];
    for (i=idx.nInts;i<OBJ_ID_SZ;i++)
      r.id[i]=0;
  } else {
    //Must hash array index into LBObjid
    int j;
    for (j=0;j<OBJ_ID_SZ;j++)
    	r.id[j]=data[j];
    for (i=0;i<idx.nInts;i++)
      for (j=0;j<OBJ_ID_SZ;j++)
        r.id[j]+=circleShift(data[i],22+11*i*(j+1))+
          circleShift(data[i],21-9*i*(j+1));
  }
  return r;
}
#endif

/************************* Array Index *********************
Array Index class.  An array index is just a 
a run of bytes used to look up an object in a hash table.
*/
typedef unsigned char uc;

inline CkHashCode CkArrayIndex::hash(void) const
{
        register int i;
	register const int *d=data();
	register CkHashCode ret=d[0];
	for (i=1;i<nInts;i++)
		ret +=circleShift(d[i],10+11*i)+circleShift(d[i],9+7*i);
	return ret;
}
CkHashCode CkArrayIndex::staticHash(const void *v,size_t)
	{return ((const CkArrayIndex *)v)->hash();}

inline int CkArrayIndex::compare(const CkArrayIndex &i2) const
{
	const CkArrayIndex &i1=*this;
#if CMK_1D_ONLY
	return i1.data()[0]==i2.data()[0];
#else
	const int *d1=i1.data();
	const int *d2=i2.data();
	int l=i1.nInts;
	if (l!=i2.nInts) return 0;
	for (int i=0;i<l;i++)
		if (d1[i]!=d2[i])
			return 0;
	//If we got here, the two keys must have exactly the same data
	return 1;
#endif
}
int CkArrayIndex::staticCompare(const void *k1,const void *k2,size_t /*len*/)
{
	return ((const CkArrayIndex *)k1)->
		compare(*(const CkArrayIndex *)k2);
}

void CkArrayIndex::pup(PUP::er &p) 
{
	p(nInts);
	p(data(),nInts);
}

/*********************** Array Messages ************************/
inline CkArrayIndexMax &CkArrayMessage::array_index(void)
{
	return UsrToEnv((void *)this)->array_index();
}
unsigned short &CkArrayMessage::array_ep(void)
{
	return UsrToEnv((void *)this)->array_ep();
}
unsigned char &CkArrayMessage::array_hops(void)
{
	return UsrToEnv((void *)this)->array_hops();
}
unsigned int CkArrayMessage::array_getSrcPe(void)
{
	return UsrToEnv((void *)this)->array_srcPe();
}
unsigned int CkArrayMessage::array_ifNotThere(void)
{
	return UsrToEnv((void *)this)->getIfNotThere();
}
void CkArrayMessage::array_setIfNotThere(unsigned int i)
{
	UsrToEnv((void *)this)->setIfNotThere(i);
}

/*********************** Array Map ******************
Given an array element index, an array map tells us 
the index's "home" Pe.  This is the Pe the element will
be created on, and also where messages to this element will
be forwarded by default.
*/

CkArrayMap::CkArrayMap(void) { }
CkArrayMap::~CkArrayMap() { }
int CkArrayMap::registerArray(int numElements,CkArrayID aid)
{return 0;}

void CkArrayMap::populateInitial(int arrayHdl,int numElements,void *ctorMsg,CkArrMgr *mgr)
{
	if (numElements==0) return;
	int thisPe=CkMyPe();
	for (int i=0;i<numElements;i++) {
		//Make 1D indices
		CkArrayIndex1D idx(i);
		if (procNum(arrayHdl,idx)==thisPe)
			mgr->insertInitial(idx,CkCopyMsg(&ctorMsg));
	}
	mgr->doneInserting();
	CkFreeMsg(ctorMsg);
}

CkGroupID _RRMapID;

/**
 *The default map object-- round-robin homes.  This is 
 * almost always what you want.
 */
class RRMap : public CkArrayMap
{
public:
  RRMap(void)
  {
	  DEBC((AA"Creating RRMap\n"AB));
  // CkPrintf("Pe %d creating RRMap\n",CkMyPe());
  }
  RRMap(CkMigrateMessage *m) {}
  int procNum(int /*arrayHdl*/, const CkArrayIndex &i)
  {
#if 1
    if (i.nInts==1) {
      //Map 1D integer indices in simple round-robin fashion
      return (i.data()[0])%CkNumPes();
    }
    else 
#endif
      {
	//Map other indices based on their hash code, mod a big prime.
	unsigned int hash=(i.hash()+739)%1280107;
	return (hash % CkNumPes());
      }
  }
};

CkpvStaticDeclare(double*, rem);

class arrInfo {
 private:
   int _nelems;
   int *_map;
   void distrib(int *speeds);
 public:
   arrInfo(int n, int *speeds)
   {
     _nelems = n;
     _map = new int[_nelems];
     distrib(speeds);
   }
   ~arrInfo() { delete[] _map; }
   int getMap(const CkArrayIndex &i);
};

static int cmp(const void *first, const void *second)
{
  int fi = *((const int *)first);
  int si = *((const int *)second);
  return ((CkpvAccess(rem)[fi]==CkpvAccess(rem)[si]) ? 
          0 : 
          ((CkpvAccess(rem)[fi]<CkpvAccess(rem)[si]) ? 
          1 : (-1)));
}

void
arrInfo::distrib(int *speeds)
{
  double total = 0.0;
  int npes = CkNumPes();
  int i,j,k;
  for(i=0;i<npes;i++)
    total += (double) speeds[i];
  double *nspeeds = new double[npes];
  for(i=0;i<npes;i++)
    nspeeds[i] = (double) speeds[i] / total;
  int *cp = new int[npes];
  for(i=0;i<npes;i++)
    cp[i] = (int) (nspeeds[i]*_nelems);
  int nr = 0;
  for(i=0;i<npes;i++)
    nr += cp[i];
  nr = _nelems - nr;
  if(nr != 0)
  {
    CkpvAccess(rem) = new double[npes];
    for(i=0;i<npes;i++)
      CkpvAccess(rem)[i] = (double)_nelems*nspeeds[i] - cp[i];
    int *pes = new int[npes];
    for(i=0;i<npes;i++)
      pes[i] = i;
    qsort(pes, npes, sizeof(int), cmp);
    for(i=0;i<nr;i++)
      cp[pes[i]]++;
    delete[] CkpvAccess(rem);
    delete[] pes;
  }
  k = 0;
  for(i=0;i<npes;i++)
  {
    for(j=0;j<cp[i];j++)
      _map[k++] = i;
  }
  delete[] nspeeds;
  delete[] cp;
}

int
arrInfo::getMap(const CkArrayIndex &i)
{
  if(i.nInts==1)
    return _map[i.data()[0]];
  else
    return _map[((i.hash()+739)%1280107)%_nelems];
}

CkpvStaticDeclare(int*, speeds);

#if CMK_USE_PROP_MAP
typedef struct _speedmsg
{
  char hdr[CmiMsgHeaderSizeBytes];
  int pe;
  int speed;
} speedMsg;

static void _speedHdlr(void *m)
{
  speedMsg *msg = (speedMsg *) m;
  CkpvAccess(speeds)[msg->pe] = msg->speed;
  CmiFree(m);
}

void _propMapInit(void)
{
  CkpvInitialize(int*, speeds);
  CkpvAccess(speeds) = new int[CkNumPes()];
  int hdlr = CkRegisterHandler((CmiHandler)_speedHdlr);
  CmiPrintf("[%d]Measuring processor speed for prop. mapping...\n", CkMyPe());
  int s = LDProcessorSpeed();
  speedMsg msg;
  CmiSetHandler(&msg, hdlr);
  msg.pe = CkMyPe();
  msg.speed = s;
  CmiSyncBroadcast(sizeof(msg), &msg);
  CkpvAccess(speeds)[CkMyPe()] = s;
  int i;
  for(i=1;i<CkNumPes();i++)
    CmiDeliverSpecificMsg(hdlr);
}
#else
void _propMapInit(void)
{
  CkpvInitialize(int*, speeds);
  CkpvAccess(speeds) = new int[CkNumPes()];
  int i;
  for(i=0;i<CkNumPes();i++)
    CkpvAccess(speeds)[i] = 1;
}
#endif
/**
 * A proportional map object-- tries to map more objects to
 * faster processors and fewer to slower processors.  Also
 * attempts to ensure good locality by mapping nearby elements
 * together.
 */
class PropMap : public CkArrayMap
{
private:
  CkVec<arrInfo *> arrs;
public:
  PropMap(void)
  {
    CkpvInitialize(double*, rem);
    DEBC((AA"Creating PropMap\n"AB));
  }
  PropMap(CkMigrateMessage *m) {}
  int registerArray(int numElements,CkArrayID aid)
  {
    int idx = arrs.length();
    arrs.insertAtEnd(new arrInfo(numElements, CkpvAccess(speeds)));
    return idx;
  }
  int procNum(int arrayHdl, const CkArrayIndex &i)
  {
    return arrs[arrayHdl]->getMap(i);
  }
};

class CkMapsInit : public Chare 
{
public:
  CkMapsInit(CkArgMsg *msg) {
    _RRMapID = CProxy_RRMap::ckNew();
    delete msg;
  }
  CkMapsInit(CkMigrateMessage *m) {}
};


/****************************** CkMigratable ***************************/
/**
 * This tiny class is used to convey information to the 
 * newly created CkMigratable object when its constructor is called.
 */
class CkMigratable_initInfo {
public:
	CkLocRec_local *locRec;  
	int chareType;
};

CkpvStaticDeclare(CkMigratable_initInfo,mig_initInfo);

void _CkMigratable_initInfoInit(void) {
  CkpvInitialize(CkMigratable_initInfo,mig_initInfo);
}

void CkMigratable::commonInit(void) {
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
	myRec=i.locRec;
	thisIndexMax=myRec->getIndex();
	thisChareType=i.chareType;
	usesAtSync=CmiFalse;
	barrierRegistered=CmiFalse;
}

CkMigratable::CkMigratable(void) {
	DEBC((AA"In CkMigratable constructor\n"AB));
	commonInit();
}
CkMigratable::CkMigratable(CkMigrateMessage *m) {
	commonInit();
}

void CkMigratable::pup(PUP::er &p) {
	DEBM((AA"In CkMigratable::pup %s\n"AB,idx2str(thisIndexMax)));
	Chare::pup(p);
	p|thisIndexMax;
	p(usesAtSync);
	ckFinishConstruction();
}

void CkMigratable::ckDestroy(void) {
	DEBC((AA"In CkMigratable::ckDestroy %s\n"AB,idx2str(thisIndexMax)));
	myRec->destroy();
}

void CkMigratable::ckAboutToMigrate(void) { }
void CkMigratable::ckJustMigrated(void) { }

CkMigratable::~CkMigratable() {
	DEBC((AA"In CkMigratable::~CkMigratable %s\n"AB,idx2str(thisIndexMax)));
	/*Might want to tell myRec about our doom here--
	it's difficult to avoid some kind of circular-delete, though.
	*/
#if CMK_LBDB_ON 
	if (barrierRegistered) {
		DEBL((AA"Removing barrier for element %s\n"AB,idx2str(thisIndexMax)));
		myRec->getLBDB()->RemoveLocalBarrierClient(ldBarrierHandle);
	}
#endif
	//To detect use-after-delete
	thisIndexMax.nInts=-123456;
}

void CkMigratable::CkAbort(const char *why) const {
	CkError("CkMigratable '%s' aborting:\n",_chareTable[thisChareType]->name);
	::CkAbort(why);
}

void CkMigratable::ResumeFromSync(void)
{
	CkAbort("::ResumeFromSync() not defined for this array element!\n");
}
#if CMK_LBDB_ON  //For load balancing:
void CkMigratable::ckFinishConstruction(void) 
{
	if ((!usesAtSync) || barrierRegistered) return;
	DEBL((AA"Registering barrier client for %s\n"AB,idx2str(thisIndexMax)));
	ldBarrierHandle = myRec->getLBDB()->AddLocalBarrierClient(
		(LDBarrierFn)staticResumeFromSync,(void*)(this));
	barrierRegistered=CmiTrue;
}
void CkMigratable::AtSync(void)
{
	if (!usesAtSync) 
		CkAbort("You must set usesAtSync=CmiTrue in your array element constructor to use AtSync!\n");
	ckFinishConstruction();
	DEBL((AA"Element %s going to sync\n"AB,idx2str(thisIndexMax)));
	myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
}
void CkMigratable::staticResumeFromSync(void* data)
{
	CkMigratable *el=(CkMigratable *)data;
	DEBL((AA"Element %s resuming from sync\n"AB,idx2str(el->thisIndexMax)));
	el->ResumeFromSync();
}
void CkMigratable::setMigratable(int migratable) 
{
	myRec->setMigratable(migratable);
}
#endif


/*CkMigratableList*/
CkMigratableList::CkMigratableList() {}
CkMigratableList::~CkMigratableList() {}

void CkMigratableList::setSize(int s) {
	el.setSize(s);
	el.length()=s;
}

void CkMigratableList::put(CkMigratable *v,int atIdx) {
#ifndef CMK_OPTIMIZE
	if (atIdx>=length())
		CkAbort("Internal array manager error (CkMigrableList::put index out of bounds)");
#endif
	el[atIdx]=v;
}


/************************** Location Records: *********************************/

//---------------- Base type:
void CkLocRec::weAreObsolete(const CkArrayIndex &idx) {}
CkLocRec::~CkLocRec() { }
void CkLocRec::beenReplaced(void)
    {/*Default: ignore replacement*/}  

//Return the represented array element; or NULL if there is none
CkMigratable *CkLocRec::lookupElement(CkArrayID aid) {return NULL;}

//Return the last known processor; or -1 if none
int CkLocRec::lookupProcessor(void) {return -1;}


/*----------------- Local: 
Matches up the array index with the local index, and 
interfaces with the load balancer on behalf of the 
represented array elements.
*/
CkLocRec_local::CkLocRec_local(CkLocMgr *mgr,CmiBool fromMigration,
  const CkArrayIndex &idx_,int localIdx_) 
	:CkLocRec(mgr),idx(idx_),localIdx(localIdx_),
	 running(CmiFalse),deletedMarker(NULL)
{
#if CMK_LBDB_ON
	DEBL((AA"Registering element %s with load balancer\n"AB,idx2str(idx)));
	the_lbdb=mgr->getLBDB();
	ldHandle=the_lbdb->RegisterObj(mgr->getOMHandle(),
		idx2LDObjid(idx),(void *)this,1);
	if (fromMigration) {
		DEBL((AA"Element %s migrated in\n"AB,idx2str(idx)));
		the_lbdb->Migrated(ldHandle);
	}
#endif
}
CkLocRec_local::~CkLocRec_local()
{
	if (deletedMarker!=NULL) *deletedMarker=CmiTrue;
	myLocMgr->reclaim(idx,localIdx);
#if CMK_LBDB_ON
	stopTiming();
	DEBL((AA"Unregistering element %s from load balancer\n"AB,idx2str(idx)));
	the_lbdb->UnregisterObj(ldHandle);
#endif
}
void CkLocRec_local::migrateMe(int toPe) //Leaving this processor
{
	//This will pack us up, send us off, and delete us
	myLocMgr->migrate(this,toPe);
}

#if CMK_LBDB_ON
void CkLocRec_local::startTiming(void) {
  	running=CmiTrue; 
	DEBL((AA"Start timing for %s at %.3fs {\n"AB,idx2str(idx),CkWallTimer()));
  	the_lbdb->ObjectStart(ldHandle);
}
void CkLocRec_local::stopTiming(void) {
	DEBL((AA"} Stop timing for %s at %.3fs\n"AB,idx2str(idx),CkWallTimer()));
  	if (running) the_lbdb->ObjectStop(ldHandle);
  	running=CmiFalse;
}
#endif

void CkLocRec_local::destroy(void) //User called destructor
{
	//Our destructor does all the needed work
	delete this; 
}
//Return the represented array element; or NULL if there is none
CkMigratable *CkLocRec_local::lookupElement(CkArrayID aid) {
	return myLocMgr->lookupLocal(localIdx,aid);
}

//Return the last known processor; or -1 if none
int CkLocRec_local::lookupProcessor(void) {
	return CkMyPe();
}

CkLocRec::RecType CkLocRec_local::type(void)
{
	return local;
}

void CkLocRec_local::addedElement(void) 
{
	//Push everything in the half-created queue into the system--
	// anything not ready yet will be put back in.
	while (!halfCreated.isEmpty())
		myLocMgr->getLocalProxy().deliver(halfCreated.deq());
}

CmiBool CkLocRec_local::isObsolete(int nSprings,const CkArrayIndex &idx_)
{ 
	int len=halfCreated.length();
	if (len!=0) {
		/* This is suspicious-- the halfCreated queue should be extremely
		 transient.  It's possible we just looked at the wrong time, though;
		 so this is only a warning. 
		*/
		CkPrintf("CkLoc WARNING> %d messages still around for uncreated element %s!\n",
			 len,idx2str(idx));
	}
	//A local element never expires
	return CmiFalse;
}

CmiBool CkLocRec_local::invokeEntry(CkMigratable *obj,void *msg,int epIdx) {
	DEBS((AA"   Invoking entry %d on element %s\n"AB,epIdx,idx2str(idx)));
	CmiBool isDeleted=CmiFalse; //Enables us to detect deletion during processing
	deletedMarker=&isDeleted;
	startTiming();
	if (msg) {
		envelope *env=UsrToEnv(msg);
		_TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(),
		     ForChareMsg,epIdx,env->array_srcPe(), env->getTotalsize());
	}
	_entryTable[epIdx]->call(msg, obj);
	if (msg) _TRACE_END_EXECUTE();
	if (isDeleted) return CmiFalse;//We were deleted
	deletedMarker=NULL;
	stopTiming();
	return CmiTrue;
}

CmiBool CkLocRec_local::deliver(CkArrayMessage *msg,CmiBool viaScheduler)
{
	if (viaScheduler) {
		myLocMgr->getLocalProxy().deliver(msg);
		return CmiTrue;
	}
	else
	{
		CkMigratable *obj=myLocMgr->lookupLocal(localIdx,
			UsrToEnv(msg)->array_mgr());
		if (obj==NULL) {//That sibling of this object isn't created yet!
			if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
				return myLocMgr->demandCreateElement(msg,CkMyPe());
			}
			else {
				DEBS((AA"   BUFFERING message for nonexistent element %s!\n"AB,idx2str(this->idx)));
				halfCreated.enq(msg);
				return CmiTrue;
			}
		}
			
		if (msg->array_hops()>1)
			myLocMgr->multiHop(msg);
		return invokeEntry(obj,(void *)msg,msg->array_ep());
	}
}

#if CMK_LBDB_ON
void CkLocRec_local::staticMigrate(LDObjHandle h, int dest)
{
	CkLocRec_local *el=(CkLocRec_local *)h.user_ptr;
	DEBL((AA"Load balancer wants to migrate %s to %d\n"AB,idx2str(el->idx),dest));
	el->migrateMe(dest);
}
#endif

void CkLocRec_local::setMigratable(int migratable)
{
	if (migratable)
  	  the_lbdb->Migratable(ldHandle);
	else
  	  the_lbdb->NonMigratable(ldHandle);
}

/**
 * Represents a deleted array element (and prevents re-use).
 * These are a debugging aid, usable only by uncommenting a line in
 * the element destruction code.
 */
class CkLocRec_dead:public CkLocRec {
public:
	CkLocRec_dead(CkLocMgr *Narr):CkLocRec(Narr) {}
  
	virtual RecType type(void) {return dead;}
  
	virtual CmiBool deliver(CkArrayMessage *msg,CmiBool viaScheduler) {
		CkPrintf("Dead array element is %s.\n",idx2str(msg->array_index()));
		CkAbort("Send to dead array element!\n");
		return CmiFalse;
	}
	virtual void beenReplaced(void) 
		{CkAbort("Can't re-use dead array element!\n");}
  
	//Return if this element is now obsolete (it isn't)
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {return CmiFalse;}	
};

/**
 * This is the abstract superclass of arrayRecs that keep track of their age,
 * and eventually expire. Its kids are remote and buffering.
 */
class CkLocRec_aging:public CkLocRec {
private:
	int lastAccess;//Age when last accessed
protected:
	//Update our access time
	inline void access(void) {
		lastAccess=myLocMgr->getSpringCount();
	}
	//Return if we are "stale"-- we were last accessed a while ago
	CmiBool isStale(void) {
		if (myLocMgr->getSpringCount()-lastAccess>3) return CmiTrue;
		else return CmiFalse;
	}
public:
	CkLocRec_aging(CkLocMgr *Narr):CkLocRec(Narr) {
		lastAccess=myLocMgr->getSpringCount();
	}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
	//virtual void pup(PUP::er &p) { CkLocRec::pup(p); p(lastAccess); }
};


/**
 * Represents a remote array element.  This is just a PE number.
 */
class CkLocRec_remote:public CkLocRec_aging {
private:
	int onPe;//The last known Pe for this element
public:
	CkLocRec_remote(CkLocMgr *Narr,int NonPe)
		:CkLocRec_aging(Narr) 
		{
			onPe=NonPe;
#ifndef CMK_OPTIMIZE
			if (onPe==CkMyPe())
				CkAbort("ERROR!  'remote' array element on this Pe!\n");
#endif
		}
	//Return the last known processor for this element
	int lookupProcessor(void) {
		return onPe;
	}  
	virtual RecType type(void) {return remote;}
  
	//Send a message for this element.
	virtual CmiBool deliver(CkArrayMessage *msg,CmiBool viaScheduler) {
		access();//Update our modification date
		msg->array_hops()++;
		DEBS((AA"   Forwarding message for element %s to %d (REMOTE)\n"AB,
		      idx2str(msg->array_index()),onPe));
		myLocMgr->getProxy()[onPe].deliver(msg);
		return CmiTrue;
	}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
		if (myLocMgr->isHome(idx)) 
			//Home elements never become obsolete
			// if they did, we couldn't deliver messages to that element.
			return CmiFalse;
		else if (isStale())
			return CmiTrue;//We haven't been used in a long time
		else
			return CmiFalse;//We're fairly recent
	}
	//virtual void pup(PUP::er &p) { CkLocRec_aging::pup(p); p(onPe); }
};


/**
 * Buffers messages until record is replaced in the hash table, 
 * then delivers all messages to the replacing record.  This is 
 * used when a message arrives for a local element that has not
 * yet been created, buffering messages until the new element finally
 * checks in.
 *
 * It's silly to send a message to an element you won't ever create,
 * so this kind of record causes an abort "Stale array manager message!"
 * if it's left undelivered too long.
 */
class CkLocRec_buffering:public CkLocRec_aging {
private:
	CkQ<CkArrayMessage *> buffer;//Buffered messages.
public:
	CkLocRec_buffering(CkLocMgr *Narr):CkLocRec_aging(Narr) {}
	virtual ~CkLocRec_buffering() {
		if (0!=buffer.length())
			CkAbort("Messages abandoned in array manager buffer!\n");
	}
  
	virtual RecType type(void) {return buffering;}
  
	//Send (or buffer) a message for this element.
	//  If idx==NULL, the index is packed in the message.
	//  If idx!=NULL, the index must be packed in the message.
	virtual CmiBool deliver(CkArrayMessage *msg,CmiBool viaScheduler) {
		DEBS((AA" Queued message for %s\n"AB,idx2str(msg->array_index())));
		buffer.enq(msg);
		return CmiTrue;
	}
  
	//This is called when this ArrayRec is about to be replaced.
	// We dump all our buffered messages off on the next guy,
	// who should know what to do with them.
	virtual void beenReplaced(void) {
		DEBS((AA" Delivering queued messages:\n"AB));
		CkArrayMessage *m;
		while (NULL!=(m=buffer.deq())) {
			DEBS((AA"Sending buffered message to %s\n"AB,idx2str(m->array_index())));
			myLocMgr->deliverViaQueue(m);
		}
	}
  
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
		if (isStale()) {
			/*This indicates something is seriously wrong--
			  buffers should be short-lived.*/
			CkPrintf("%d stale array message(s) found!\n",buffer.length());
			CkArrayMessage *msg=buffer.deq();
			CkPrintf("Addressed to: ");
			CkPrintEntryMethod(msg->array_ep());
			CkPrintf(" index %s\n",idx2str(idx));
			if (myLocMgr->isHome(idx)) 
				CkPrintf("is this an out-of-bounds array index, or was it never created?\n");
			else //Idx is a remote-home index
				CkPrintf("why weren't they forwarded?\n");
			
			CkAbort("Stale array manager message(s)!\n");
		}
		return CmiFalse;
	}
  
/*  virtual void pup(PUP::er &p) {
    CkLocRec_aging::pup(p);
    CkArray::pupArrayMsgQ(buffer, p);
    }*/
};

/*********************** Spring Cleaning *****************/
/**
 * Used to periodically flush out unused remote element pointers.
 *
 * Cleaning often will free up memory quickly, but slow things
 * down because the cleaning takes time and some not-recently-referenced
 * remote element pointers might be valid and used some time in 
 * the future.
 *
 * Also used to determine if buffered messages have become stale.
 */
inline void CkLocMgr::springCleaning(void)
{
  nSprings++;

  //Poke through the hash table for old ArrayRecs.
  void *objp;
  void *keyp;
  CkHashtableIterator *it=hash.iterator();
  while (NULL!=(objp=it->next(&keyp))) {
    CkLocRec *rec=*(CkLocRec **)objp;
    CkArrayIndex &idx=*(CkArrayIndex *)keyp;
    if (rec->isObsolete(nSprings,idx)) {
      //This record is obsolete-- remove it from the table
      DEBK((AA"Cleaning out old record %s\n"AB,idx2str(idx)));
      hash.remove(*(CkArrayIndexMax *)&idx);
      delete rec;
      it->seek(-1);//retry this hash slot
    }
  }
  delete it;
}
void CkLocMgr::staticSpringCleaning(void *forWhom) {
	DEBK((AA"Starting spring cleaning at %.2f\n"AB,CkWallTimer()));
	((CkLocMgr *)forWhom)->springCleaning();
}

/*************************** LocMgr: CREATION *****************************/
CkLocMgr::CkLocMgr(CkGroupID mapID_,CkGroupID lbdbID_,int numInitial) 
	:thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe()),
	 hash(17,0.3)
{
	DEBC((AA"Creating new location manager %d\n"AB,thisgroup));
// moved to _CkMigratable_initInfoInit()
//	CkpvInitialize(CkMigratable_initInfo,mig_initInfo);
	
	ckEnableTracing=CmiFalse; //Prevent us from being recorded
	managers.init();
	nManagers=0;
  	firstManager=NULL;
	firstFree=localLen=0;
	duringMigration=CmiFalse;
	nSprings=0;
	CcdCallOnConditionKeepOnPE(CcdPERIODIC_1minute,staticSpringCleaning,(void *)this, CkMyPe());

//Register with the map object
	mapID=mapID_;
	map=(CkArrayMap *)CkLocalBranch(mapID);
	if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
	mapHandle=map->registerArray(numInitial,thisgroup);
	
//Find and register with the load balancer
	initLB(lbdbID_);
}


/// Add a new local array manager to our list.
/// Returns a new CkMigratableList for the manager to store his 
/// elements in.
CkMigratableList *CkLocMgr::addManager(CkArrayID id,CkArrMgr *mgr)
{
	magic.check();
	DEBC((AA"Adding new array manager\n"AB));
	//Link new manager into list
	ManagerRec *n=&managers.find(id);
	n->next=firstManager;
	n->mgr=mgr;
	n->elts.setSize(localLen);
	nManagers++;
	firstManager=n;
	return &n->elts;
}

/// Return the next unused local element index.
int CkLocMgr::nextFree(void) {
	if (firstFree>=localLen) 
	{//Need more space in the local index arrays-- enlarge them
		int oldLen=localLen;
		localLen=localLen*2+8;
		DEBC((AA"Growing the local list from %d to %d...\n"AB,oldLen,localLen));
		for (ManagerRec *m=firstManager;m!=NULL;m=m->next)
			m->elts.setSize(localLen);
		//Update the free list
		freeList.setSize(localLen);
		for (int i=oldLen;i<localLen;i++)
			freeList[i]=i+1;
	}
	int localIdx=firstFree;
	if (localIdx==-1) CkAbort("CkLocMgr free list corrupted!");
	firstFree=freeList[localIdx];
	freeList[localIdx]=-1; //Mark as used
	return localIdx;
}

CkLocRec_remote *CkLocMgr::insertRemote(const CkArrayIndex &idx,int nowOnPe)
{
	DEBS((AA"Remote element %s lives on %d\n"AB,idx2str(idx),nowOnPe));
	CkLocRec_remote *rem=new CkLocRec_remote(this,nowOnPe);
	insertRec(rem,idx);
	return rem;
}

//This element now lives on the given Pe
void CkLocMgr::inform(const CkArrayIndex &idx,int nowOnPe)
{
	if (nowOnPe==CkMyPe()) 
		return; //Never insert a "remote" record pointing here
	CkLocRec *rec=elementNrec(idx);
	if (rec!=NULL && rec->type()==CkLocRec::local)
		return; //Never replace a local element's record!
	insertRemote(idx,nowOnPe);
}

//Tell this element's home processor it now lives "there"
void CkLocMgr::informHome(const CkArrayIndex &idx,int nowOnPe)
{
	int home=homePe(idx);
	if (home!=CkMyPe() && home!=nowOnPe) {
		//Let this element's home Pe know it lives here now
		DEBC((AA"  Telling %s's home %d that it lives on %d.\n"AB,idx2str(idx),home,nowOnPe));
		thisProxy[home].updateLocation(idx,nowOnPe);
	}
}

//Add a new local array element, calling element's constructor
CmiBool CkLocMgr::addElement(CkArrayID id,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{
	magic.check();
	CkLocRec *oldRec=elementNrec(idx);
	CkLocRec_local *rec;
	if (oldRec==NULL||oldRec->type()!=CkLocRec::local) 
	{ //This is the first we've heard of that element-- add new local record
		int localIdx=nextFree();
		DEBC((AA"Adding new record for element %s at local index %d\n"AB,idx2str(idx),localIdx));
		rec=new CkLocRec_local(this,CmiFalse,idx,localIdx);
		insertRec(rec,idx); //Add to global hashtable
		informHome(idx,CkMyPe());
	} else 
	{ //rec is *already* local-- must not be the first insertion	
		rec=((CkLocRec_local *)oldRec);
		rec->addedElement();
	}
	if (!addElementToRec(rec,&managers.find(id),elt,ctorIdx,ctorMsg)) return CmiFalse;
	elt->ckFinishConstruction();
	return CmiTrue;
}

//As above, but shared with the migration code
CmiBool CkLocMgr::addElementToRec(CkLocRec_local *rec,ManagerRec *m,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{//Insert the new element into its manager's local list
	int localIdx=rec->getLocalIndex();
	const CkArrayIndex &idx=rec->getIndex();
	if (m->elts.get(localIdx)!=NULL) CkAbort("Cannot insert array element twice!");
	m->elts.put(elt,localIdx); //Local element table
	
//Call the element's constructor
	DEBC((AA"Constructing element %s of array\n"AB,idx2str(idx)));
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
	i.locRec=rec;
	i.chareType=_entryTable[ctorIdx]->chareIdx;
	if (!rec->invokeEntry(elt,ctorMsg,ctorIdx)) return CmiFalse;
	
	return CmiTrue;
}
void CkLocMgr::updateLocation(const CkArrayIndexMax &idx,int nowOnPe) {
	inform(idx,nowOnPe);
}

/*************************** LocMgr: DELETION *****************************/
/// This index will no longer be used-- delete the associated elements
void CkLocMgr::reclaim(const CkArrayIndex &idx,int localIdx) {
	magic.check();
	DEBC((AA"Destroying element %s (local %d)\n"AB,idx2str(idx),localIdx));
	//Delete, and mark as empty, each array element
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		delete m->elts.get(localIdx);
		m->elts.empty(localIdx);
	}
	
	removeFromTable(idx);
	
	//Link local index into free list
	freeList[localIdx]=firstFree;
	firstFree=localIdx;
	if (!duringMigration) 
	{ //This is a local element dying a natural death
		int home=homePe(idx);
		if (home!=CkMyPe())
			thisProxy[home].reclaimRemote(idx,CkMyPe());
	/*	//Install a zombie to keep the living from re-using this index.
		insertRecN(new CkLocRec_dead(this),idx); */
	}
}

void CkLocMgr::reclaimRemote(const CkArrayIndexMax &idx,int deletedOnPe) {
	DEBC((AA"Our element %s died on PE %d\n"AB,idx2str(idx),deletedOnPe));
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) return; //We never knew him
	if (rec->type()==CkLocRec::local) return; //He's already been reborn
	removeFromTable(idx);
	delete rec;
}
void CkLocMgr::removeFromTable(const CkArrayIndex &idx) {
#ifndef CMK_OPTIMIZE
	//Make sure it's actually in the table before we delete it
	if (NULL==elementNrec(idx))
		CkAbort("CkLocMgr::removeFromTable called on invalid index!");
#endif
	hash.remove(*(CkArrayIndexMax *)&idx);
#ifndef CMK_OPTIMIZE
	//Make sure it's really gone
	if (NULL!=elementNrec(idx))
		CkAbort("CkLocMgr::removeFromTable called, but element still there!");
#endif
}

/************************** LocMgr: MESSAGING *************************/
/// Deliver message to this element, going via the scheduler if local
void CkLocMgr::deliverViaQueue(CkMessage *m) {
	magic.check();
	CkArrayMessage *msg=(CkArrayMessage *)m;
	const CkArrayIndex &idx=msg->array_index();
	DEBS((AA"deliverViaQueue %s\n"AB,idx2str(idx)));
#if CMK_LBDB_ON
	the_lbdb->Send(myLBHandle,idx2LDObjid(idx),UsrToEnv(msg)->getTotalsize());
#endif
	CkLocRec *rec=elementNrec(idx);
	if (rec!=NULL)
		rec->deliver(msg,CmiTrue);
	else deliverUnknown(msg);
}
/// Deliver message directly to this element
CmiBool CkLocMgr::deliver(CkMessage *m) {
	magic.check();
	CkArrayMessage *msg=(CkArrayMessage *)m;
	const CkArrayIndex &idx=msg->array_index();
	DEBS((AA"deliver %s\n"AB,idx2str(idx)));
	CkLocRec *rec=elementNrec(idx);
	if (rec!=NULL)
		return rec->deliver(msg,CmiFalse);
	else 
		return deliverUnknown(msg);
}

/// This index is not hashed-- somehow figure out what to do.
CmiBool CkLocMgr::deliverUnknown(CkArrayMessage *msg)
{
	magic.check();
	const CkArrayIndex &idx=msg->array_index();
	int onPe=homePe(idx);
	if (onPe!=CkMyPe()) 
	{// Forward the message to its home processor
		DEBM((AA"Forwarding message for unknown %s\n"AB,idx2str(idx)));
		msg->array_hops()++;
		thisProxy[onPe].deliver(msg);
		return CmiTrue;
	}
	else
	{// We *are* the home processor-- decide what to do
	  int nt=msg->array_ifNotThere();
	  if (nt==CkArray_IfNotThere_buffer)
	  {//Just buffer the message
		DEBC((AA"Adding buffer for unknown element %s\n"AB,idx2str(idx)));
		CkLocRec *rec=new CkLocRec_buffering(this);
		insertRecN(rec,idx);
		return rec->deliver(msg,CmiTrue);	       
	  }
	  else 
		return demandCreateElement(msg,-1);
	}
}

CmiBool CkLocMgr::demandCreateElement(CkArrayMessage *msg,int onPe)
{
	magic.check();
	const CkArrayIndex &idx=msg->array_index();
	int chareType=_entryTable[msg->array_ep()]->chareIdx;
	int ctor=_chareTable[chareType]->getDefaultCtor();
	if (ctor==-1) CkAbort("Can't create array element to handle message--\n"
			      "The element has no default constructor in the .ci file!\n");
	if (onPe==-1) 
	{ //Decide where element needs to live
		if (msg->array_ifNotThere()==CkArray_IfNotThere_createhere) 
			onPe=UsrToEnv(msg)->array_srcPe();
		else //Createhome
			onPe=homePe(idx);
	}
	
	//Find the manager and build the element
	DEBC((AA"Demand-creating element %s on pe %d\n"AB,idx2str(idx),onPe));
	CkArrMgr *mgr=managers.find(UsrToEnv((void *)msg)->array_mgr()).mgr;
	CmiBool created=mgr->demandCreateElement(idx,onPe,ctor);

	//Try the delivery again-- it should succeed this time
	deliver(msg);
	
	return created;
}

//This message took several hops to reach us-- fix it
void CkLocMgr::multiHop(CkArrayMessage *msg)
{
	magic.check();
	int hopCount=msg->array_hops();
	int srcPe=msg->array_getSrcPe();
	if (srcPe==CkMyPe())
		DEB((AA"Odd routing: local element %s is %d hops away!\n"AB,idx2str(msg),hopCount));
	else
	{//Send a routing message letting original sender know new element location
		DEBS((AA"Sending update back to %d for element\n"AB,srcPe,idx2str(msg)));
		thisProxy[srcPe].updateLocation(msg->array_index(),CkMyPe());
	}
}

/************************** LocMgr: MIGRATION *************************/

CkMigratable *CkArrMgr::allocateMigrated(int elChareType,const CkArrayIndex &idx)
{
	return (CkMigratable *)malloc(_chareTable[elChareType]->size);
}

void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec_local *rec)
{
	register ManagerRec *m;
	int localIdx=rec->getLocalIndex();
	
	//First pup the element types
	// (A separate loop so ckLocal works even in element pup routines)
	for (m=firstManager;m!=NULL;m=m->next) {
		int elCType;
		if (!p.isUnpacking()) 
		{ //Need to find the element's existing type
			CkMigratable *elt=m->element(localIdx);
			if (elt) elCType=elt->ckGetChareType();
			else elCType=-1; //Element hasn't been created
		}
		p(elCType);
		if (p.isUnpacking() && elCType!=-1) {
			//Create the element
			CkMigratable *elt=m->mgr->allocateMigrated(elCType,rec->getIndex());
			int migCtorIdx=_chareTable[elCType]->getMigCtor();
			//Insert into our tables and call migration constructor
			if (!addElementToRec(rec,m,elt,migCtorIdx,NULL)) return;
		}
	}
	//Next pup the element data
	for (m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *elt=m->element(localIdx);
		if (elt!=NULL) elt->pup(p);
	}
}

/// Migrate this element to another processor.
void CkLocMgr::migrate(CkLocRec_local *rec,int toPe)
{
	magic.check();
	if (toPe==CkMyPe()) return; //You're already there!

	int localIdx=rec->getLocalIndex();
	CkArrayIndexMax idx=rec->getIndex();

	//Let all the elements know we're leaving
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *el=m->element(localIdx);
		if (el) el->ckAboutToMigrate();
	}

//First pass: find size of migration message
	int bufSize;
	{ 
		PUP::sizer p; 
		p(nManagers);
		pupElementsFor(p,rec);
		bufSize=p.size(); 
	}
	
//Allocate and pack into message
	int doubleSize=bufSize/sizeof(double)+1;
	CkArrayElementMigrateMessage *msg = 
		new (doubleSize, 0) CkArrayElementMigrateMessage;
	msg->length=bufSize;
	CkArrayMessage *amsg=(CkArrayMessage *)msg;
	{
		PUP::toMem p(msg->packData); 
		p.becomeDeleting(); 
		p(nManagers);
		pupElementsFor(p,rec);
		if (p.size()!=bufSize) {
			CkError("ERROR! Array element claimed it was %d bytes to a"
				"sizing PUP::er, but copied %d bytes into the packing PUP::er!\n",
				bufSize,p.size());
			CkAbort("Array element's pup routine has a direction mismatch.\n");
		}
	}
	amsg->array_index()=idx;
	DEBM((AA"Migrated index size %s\n"AB,idx2str(amsg->array_index())));	

//Send off message and delete old copy
	thisProxy[toPe].migrateIncoming(msg);
	duringMigration=CmiTrue;
	delete rec; //Removes elements, hashtable entries, local index
	duringMigration=CmiFalse;
	//The element now lives on another processor-- tell ourselves and its home
	inform(idx,toPe);
	informHome(idx,toPe);
}

void CkLocMgr::migrateIncoming(CkArrayElementMigrateMessage *msg)
{
	CkArrayMessage *amsg=(CkArrayMessage *)msg;
	const CkArrayIndex &idx=amsg->array_index();
	PUP::fromMem p(msg->packData); 
	
	int nMsgMan;
	p(nMsgMan);
	if (nMsgMan<nManagers)
		CkAbort("Array element arrived from location with fewer managers!\n");
	if (nMsgMan>nManagers) {
		//Some array managers haven't registered yet-- throw it back
		DEBM((AA"Busy-waiting for array registration on migrating %s\n"AB,idx2str(idx)));
		thisProxy[CkMyPe()].migrateIncoming(msg);
		return;
	}

	//Create a record for this element
	int localIdx=nextFree();
	CkLocRec_local *rec=new CkLocRec_local(this,CmiTrue,idx,localIdx);
	insertRec(rec,idx); //Add to global hashtable
	
	//Create the new elements as we unpack the message
	pupElementsFor(p,rec);
	if (p.size()!=msg->length) {
		CkError("ERROR! Array element claimed it was %d bytes to a"
			"packing PUP::er, but %d bytes in the unpacking PUP::er!\n",
			msg->length,p.size());
		CkError("(I have %d managers; he claims %d managers)\n",
			nManagers,nMsgMan);
		
		CkAbort("Array element's pup routine has a direction mismatch.\n");
	}
	
	//Let all the elements know we've arrived
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *el=m->element(localIdx);
		if (el) el->ckJustMigrated();
	}
	delete msg;
}

/********************* LocMgr: UTILITY ****************/
void CkMagicNumber_impl::badMagicNumber(int expected) const
{
	CkError("Expected magic number 0x%08x; found 0x%08x!\n",expected,magic);
	CkAbort("Bad magic number detected!  This implies either\n"
		"the heap or a message was corrupted!\n");
}
CkMagicNumber_impl::CkMagicNumber_impl(int m) :magic(m) { }

//Look up the object with this array index, or return NULL
CkMigratable *CkLocMgr::lookup(const CkArrayIndex &idx,CkArrayID aid) {
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) return NULL;
	else return rec->lookupElement(aid);
}
//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(const CkArrayIndex &idx) const {
	CkLocMgr *vthis=(CkLocMgr *)this;//Cast away "const"
	CkLocRec *rec=vthis->elementNrec(idx);
	int pe=-1;
	if (rec!=NULL) pe=rec->lookupProcessor();
	if (pe==-1) return homePe(idx);
	else return pe;
}

static const char *rec2str[]={
    "base (INVALID)",//Base class (invalid type)
    "local",//Array element that lives on this Pe
    "remote",//Array element that lives on some other Pe
    "buffering",//Array element that was just created
    "dead"//Deleted element (for debugging)
};

//Add given element array record at idx, replacing the existing record
void CkLocMgr::insertRec(CkLocRec *rec,const CkArrayIndex &idx) {
	CkLocRec *old=elementNrec(idx);
	insertRecN(rec,idx);
	if (old!=NULL) {
		DEBC((AA"  replaces old rec(%s) for %s\n"AB,rec2str[old->type()],idx2str(idx)));
		//There was an old element at this location
		if (old->type()==CkLocRec::local && rec->type()==CkLocRec::local) {
			CkPrintf("ERROR! Duplicate array index: %s\n",idx2str(idx));
			CkAbort("Duplicate array index used");
		}
		old->beenReplaced();
		delete old;
	}
}

//Add given record, when there is guarenteed to be no prior record
void CkLocMgr::insertRecN(CkLocRec *rec,const CkArrayIndex &idx) {
	DEBC((AA"  adding new rec(%s) for %s\n"AB,rec2str[rec->type()],idx2str(idx)));
	hash.put(*(CkArrayIndexMax *)&idx)=rec;
}

//Call this on an unrecognized array index
static void abort_out_of_bounds(const CkArrayIndex &idx)
{
  CkPrintf("ERROR! Unknown array index: %s\n",idx2str(idx));
  CkAbort("Array index out of bounds\n");
}

//Look up array element in hash table.  Index out-of-bounds if not found.
CkLocRec *CkLocMgr::elementRec(const CkArrayIndex &idx) {
#ifdef CMK_OPTIMIZE
//Assume the element will be found
	return hash.getRef(*(CkArrayIndexMax *)&idx);
#else
//Include an out-of-bounds check if the element isn't found
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) abort_out_of_bounds(idx);
	return rec;
#endif
}

//Look up array element in hash table.  Return NULL if not there.
CkLocRec *CkLocMgr::elementNrec(const CkArrayIndex &idx) {
	return hash.get(*(CkArrayIndexMax *)&idx);
}

/********************* LocMgr: LOAD BALANCE ****************/

#if !CMK_LBDB_ON
//Empty versions of all load balancer calls
void CkLocMgr::initLB(CkGroupID lbdbID_) {}
void CkLocMgr::doneInserting(void) {}
void CkLocMgr::dummyAtSync(void) {}
#endif


#if CMK_LBDB_ON
void CkLocMgr::initLB(CkGroupID lbdbID_)
{ //Find and register with the load balancer
	the_lbdb = (LBDatabase *)CkLocalBranch(lbdbID_);
	if (the_lbdb == 0)
		CkAbort("LBDatabase not yet created?\n");
	DEBL((AA"Connected to load balancer %p\n"AB,the_lbdb));

	// Register myself as an object manager
	LDOMid myId;
	myId.id = thisgroup;
	LDCallbacks myCallbacks;
	myCallbacks.migrate = (LDMigrateFn)CkLocRec_local::staticMigrate;
	myCallbacks.setStats = NULL;
	myCallbacks.queryEstLoad = NULL;
	myLBHandle = the_lbdb->RegisterOM(myId,this,myCallbacks);  
	
	// Tell the lbdb that I'm registering objects
	the_lbdb->RegisteringObjects(myLBHandle);  
	
	/*Set up the dummy barrier-- the load balancer needs 
	  us to call Registering/DoneRegistering during each AtSync,
	  and this is the only way to do so.
	*/
	the_lbdb->AddLocalBarrierReceiver(
		(LDBarrierFn)staticRecvAtSync,(void*)(this));    	
	dummyBarrierHandle = the_lbdb->AddLocalBarrierClient(
		(LDResumeFn)staticDummyResumeFromSync,(void*)(this));
	dummyAtSync();
}
void CkLocMgr::dummyAtSync(void)
{
	DEBL((AA"dummyAtSync called\n"AB));
	the_lbdb->AtLocalBarrier(dummyBarrierHandle);
}

void CkLocMgr::staticDummyResumeFromSync(void* data)
{      ((CkLocMgr*)data)->dummyResumeFromSync(); }
void CkLocMgr::dummyResumeFromSync()
{
	DEBL((AA"DummyResumeFromSync called\n"AB));
	the_lbdb->DoneRegisteringObjects(myLBHandle);
	dummyAtSync();
}
void CkLocMgr::staticRecvAtSync(void* data)
{      ((CkLocMgr*)data)->recvAtSync(); }
void CkLocMgr::recvAtSync()
{
	DEBL((AA"recvAtSync called\n"AB));
	the_lbdb->RegisteringObjects(myLBHandle);
}

void CkLocMgr::doneInserting(void) 
{
	the_lbdb->DoneRegisteringObjects(myLBHandle);
}
#endif

#include "CkLocation.def.h"


