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

#define thisproxy CProxy_CkArray(thisgroup)

/************************** Debugging Utilities **************/

//For debugging: convert given index to a string
static const char *idx2str(const CkArrayIndex &ind)
{
	static char retBuf[500];
	const char *hexTable="0123456789abcdef";
	char *r=retBuf;//Write into retBuf
	int i,len;
	const unsigned char *d=ind.getKey(len);
	*r='{';*++r='0';*++r='x';
	for (i=0;i<len;i++)
	{
		*++r=hexTable[(*d)>>4];
		*++r=hexTable[(*d)&0x0f];
		*++r=' ';
		if ((i%4)==3) 
			*r='/';//Overwrite last space with slash
		d++;
	}
	*r='}';//Overwrite last space with a closing brace
	*++r=0;//Append trailing NULL
	return retBuf;
}
static const char *idx2str(const ArrayElement *el)
  {return idx2str(*el->thisindex);}

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT 
#   define DEB(x) CkPrintf x  //General debug messages
#   define DEBI(x) //CkPrintf x  //Index debug messages
#   define DEBC(x) //CkPrintf x  //Construction debug messages
#   define DEBS(x) //CkPrintf x  //Send/recv/broadcast debug messages
#   define DEBM(x) CkPrintf x  //Migration debug messages
#   define DEBL(x) CkPrintf x  //Load balancing debug messages
#   define DEBK(x) //CkPrintf x  //Spring Cleaning debug messages
#   define DEBB(x) //CkPrintf x  //Broadcast debug messages
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

/************************* Array Index *********************
Array Index class.  An array index is just a HashKey-- 
a run of bytes used to look up an object in a hash table.
An Array Index cannot be modified once it is created.
*/
typedef unsigned char uc;

//Simple array indices:
const uc *CkArrayIndex1D::getKey(/*out*/ int &len) const
{len=1*sizeof(int);return (const uc *)&index;}

const uc *CkArrayIndex2D::getKey(/*out*/ int &len) const
{len=2*sizeof(int);return (const uc *)index;}

const uc *CkArrayIndex3D::getKey(/*out*/ int &len) const
{len=3*sizeof(int);return (const uc *)index;}

const uc *CkArrayIndex4D::getKey(/*out*/ int &len) const
{len=4*sizeof(int);return (const uc *)index;}

///// ArrayIndexConst ////
//Finally, here the key is a run of bytes whose length can vary at run time.  
// Is generic because it can contain the data of *any* kind of ArrayIndex.

CkArrayIndexConst::CkArrayIndexConst(int len,const void *srcData)//Copy given data
{
	nBytes=len; 
	constData=(const unsigned char *)srcData;
}

CkArrayIndexConst::CkArrayIndexConst(const CkArrayIndex &that)//Copy given index's data
{
	constData=that.getKey(nBytes);
}
const unsigned char *CkArrayIndexConst::getKey(/*out*/ int &len) const
{
	len=nBytes;
	return constData;
}

#if CMK_LBDB_ON
/*One problem with the LBDB is that the object handles 
(the nordic-sounding LBObjid) are fixed-sized-- they are 
always 4 ints.  Of course, this means it's possible that 
two distinct array indices will hash to the same handle, 
thus counting the messages and CPU time for those array 
elements together instead of separately.  Since there are 
(for 32-bit ints) 2^128 LBObjid's, this should be rare enough 
not to seriously impact the load balance.
*/
static LDObjid idx2LDObjid(const CkArrayIndex &idx)
{
	int i,len, lenInInts;
	LDObjid r;
	for (i=0;i<OBJ_ID_SZ;i++)
		r.id[i]=0;
	const unsigned char *data=idx.getKey(len);
	const int *id=(const int *)data;
	lenInInts = len/sizeof(int);
	for (i=0;i<lenInInts;i++)
		r.id[i%OBJ_ID_SZ]^=id[i]+(id[i]<<(24+i/4));
	return r;
}
#endif

/*********************** Array Map ******************
Given an array element index, an array map tells us 
the index's "home" PE.  This is the PE the element will
be created on, and also where messages to this element will
be forwarded by default.
*/

CkArrayMap::CkArrayMap(void) {}
int CkArrayMap::registerArray(CkArrayMapRegisterMessage *) 
{return 0;}
int CkArrayMap::procNum(int arrayHdl,const CkArrayIndex &element) 
{return 0;}

CkGroupID _RRMapID;

class RRMap : public CkArrayMap
{
public:
  RRMap(void)
  {
  // CkPrintf("PE %d creating RRMap\n",CkMyPe());
  }
  int registerArray(CkArrayMapRegisterMessage *msg)
  {
    delete msg;
    return 0;
  }
  int procNum(int /*arrayHdl*/, const CkArrayIndex &i)
  {
#if 1
	if (i.len()==sizeof(int)) {
		//Map 1D integer indices in simple round-robin fashion
		return (*(int *)i.data())%CkNumPes();
	}
	else 
#endif
	{
		//Map other indices based on their hash code, mod a big prime.
		unsigned int hash=(i.getHashCode()+739)%1280107;
		return (hash % CkNumPes());
	}
  }
};

class CkArrayInit : public Chare 
{
public:
  CkArrayInit(CkArgMsg *msg) {
    _RRMapID = CProxy_RRMap::ckNew();
    delete msg;
  }
};

/************************** ArrayElement *******************
An array element is a chare that lives inside the array.
Unlike regular chares, array elements can migrate from one
PE to another.  Each element has a unique index.
*/

//Remote method: calls destructor
void ArrayElement::destroy(void)
{
	thisArray->contributorDied(&reductionInfo);
	delete this;
}

//Destructor (virtual)
ArrayElement::~ArrayElement()
{
	thisArray->ElementDying((new CkArrayRemoveMsg)->insert(*thisindex));
	lbUnregister();
	delete thisindex;//Delete heap-allocated index object
//To detect use-after-delete: 
	thisArray=(CkArray *)0xDEADa7a1;
	thisindex=(CkArrayIndex *)0xDEAD1de9;
}


//Contribute to the given reduction type.  Data is copied, not deleted.
void ArrayElement::contribute(int dataSize,void *data,CkReduction::reducerType type)
{
	thisArray->contribute(&reductionInfo,
		CkReductionMsg::buildNew(dataSize,data,type));
}

//For migration:
void ArrayElement::migrateMe(int where)
{
	if (where!=CkMyPe())
	{
		thisArray->migrateMe(this,where);
	}
}

void ArrayElement::pup(PUP::er &p)
{
	DEBM((AA"  ArrayElement::pup()\n"AB));
	p(bcastNo);
	reductionInfo.pup(p);
#if CMK_LBDB_ON
	p(usesAtSync);
#endif
}


#if CMK_LBDB_ON //Load balancer utilities:
void ArrayElement::AtSync(void) 
{
	if (!usesAtSync)
		CkAbort("You must set usesAtSync=CmiTrue in your array element constructor to use AtSync!\n");
	DEBL((AA"Element %s at sync\n"AB,idx2str(this))); 
	thisArray->the_lbdb->AtLocalBarrier(ldBarrierHandle);
}
void ArrayElement::ResumeFromSync(void)
{
	CkAbort("No ResumeFromSync() defined for this array element!\n");
}
void ArrayElement::staticResumeFromSync(void* data)
{
  ArrayElement* el = (ArrayElement*)(data);
  el->ResumeFromSync();
}
void ArrayElement::staticMigrate(LDObjHandle h, int dest)
{
  ArrayElement *el=(ArrayElement *)h.user_ptr;
  DEBL((AA"Load balancer wants to migrate %s to %d\n"AB,idx2str(el),dest));
  el->migrateMe(dest);
}
void ArrayElement::lbRegister(void)//Connect to load balancer
{
	DEBL((AA"Registering element %s with load balancer (%s AtSync)\n"AB,idx2str(this),usesAtSync?"and":"without"));	
	ldHandle = thisArray->the_lbdb->
		RegisterObj(thisArray->myLBHandle,
		idx2LDObjid(*thisindex),(void *)this,1);
	
	if (usesAtSync)
		ldBarrierHandle = thisArray->the_lbdb->AddLocalBarrierClient(
			(LDBarrierFn)staticResumeFromSync,
			(void*)(this));
}

void ArrayElement::lbUnregister(void)//Disconnect from load balancer
{
	DEBL((AA"Unregistering element %s from load balancer (%s AtSync)\n"AB,idx2str(this),usesAtSync?"and":"without"));	
	thisArray->the_lbdb->UnregisterObj(ldHandle);
	if (usesAtSync)
		thisArray->the_lbdb->RemoveLocalBarrierClient(ldBarrierHandle);
}
#else //not CMK_LDBD_ON
void ArrayElement::lbRegister(void) {}//Connect to load balancer
void ArrayElement::lbUnregister(void) {}//Disconnect from load balancer
#endif


///// 1-D array element utility routines:
ArrayElement1D::ArrayElement1D(void)
{
	numElements=thisArray->numInitial;
	thisIndex=*(int *)thisindex->data();
}
ArrayElement1D::ArrayElement1D(ArrayElementMigrateMessage *msg)
{
	numElements=thisArray->numInitial;
	thisIndex=*(int *)thisindex->data();
}

void ArrayElement1D::pup(PUP::er &p)
{
	ArrayElement::pup(p);//Pack superclass
	p(numElements);
}


void operator|(PUP::er &p,CkArray_index2D &i) {p(i.x);p(i.y);}
void operator|(PUP::er &p,CkArray_index3D &i) {p(i.x);p(i.y);p(i.z);}

/*********************** CkArrayRec ***********************
These objects represent array elements in the main array hash table.
There are arrayRecs for just-created objects, objects that live here,
objects that live somewhere else, objects that are moving somewhere else,
etc. 
*/
//This is the abstract superclass
class CkArrayRec {
protected:
	CkArray *arr;
	int lastAccess;//Age when last accessed
	//Called when we discover we are obsolete before we suicide
	virtual void weAreObsolete(const CkArrayIndex &idx) {}
public:
	CkArrayRec(CkArray *Narr) {arr=Narr;lastAccess=arr->nSprings;}
	virtual ~CkArrayRec() {}

	typedef enum {
		base=0,//Base class (invalid type)
		local,//Array element that lives on this PE
		remote,//Array element that lives on some other PE
		buffering,//Array element that was just created
		buffering_migrated,//Array element that just left
		dead//Deleted element (for debugging)
	} RecType;
	//Return the type of this ArrayRec:
	virtual RecType type(void) {return base;}
	
	//Send (or buffer) a message for this element.
	// if viaSchedulerQ, deliver it via Scheduler's message Queue.
	virtual void send(CkArrayMessage *msg) = 0;
	
	//This is called when this ArrayRec is about to be replaced.
	// It is only used to deliver buffered element messages.
	virtual void beenReplaced(void) 
	{/*Default: ignore replacement*/}	
	
	//Return if this rec is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
	
	//Return this rec's array element; or NULL if none
	virtual ArrayElement *element(void) {return NULL;}
};


//Represents a local array element
class CkArrayRec_local:public CkArrayRec {
private:
	ArrayElement *el;//Our local element
	void killElement(void)
	{
		if (el!=NULL)
			delete el;
	}
public:
	CkArrayRec_local(CkArray *Narr,ArrayElement *Nel):CkArrayRec(Narr)
	  {el=Nel;arr->num.local++;}
	virtual ~CkArrayRec_local() 
	{
		killElement();
		arr->num.local--;
	}

	virtual RecType type(void) {return local;}

	//Deliver a message to this local element.
	virtual void send(CkArrayMessage *msg)
	{
		arr->deliverLocal(msg,el);
	}
	//Return if this element is now obsolete (it isn't)
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {return CmiFalse;}
	
	//Return this rec's array element
	virtual ArrayElement *element(void) {return el;}
	
	//Give up ownership of this element
	ArrayElement *releaseElement(void) 
	{ArrayElement *ret=el; el=NULL; return ret;}
};

//Represents a deleted array element (prevents re-use)
class CkArrayRec_dead:public CkArrayRec {
public:
	CkArrayRec_dead(CkArray *Narr):CkArrayRec(Narr) {}

	virtual RecType type(void) {return dead;}

	virtual void send(CkArrayMessage *msg)
	{
		CkPrintf("Dead array element is %s.\n",idx2str(msg->index()));
		CkAbort("Send to dead array element!\n");
	}
	virtual void yourSuccessorIs(CkArrayRec *nextGuy) 
	{CkAbort("Can't re-use dead array element!\n");}
	
	//Return if this element is now obsolete (it isn't)
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {return CmiFalse;}
	
};

//This is the abstract superclass of arrayRecs that keep track of their age.
// Its kids are remote and buffering.
class CkArrayRec_aging:public CkArrayRec {
private:
	int lastAccess;//Age when last accessed
protected:
	void access(void)//Update our access time
	{
		lastAccess=arr->nSprings;
	}
	//Return if we are "stale"-- we were last accessed a while ago
	CmiBool isStale(void) 
	{
		if (arr->nSprings-lastAccess>3) return CmiTrue;
		else return CmiFalse;
	}
public:
	CkArrayRec_aging(CkArray *Narr):CkArrayRec(Narr) {lastAccess=arr->nSprings;}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
};


//Represents a remote array element
class CkArrayRec_remote:public CkArrayRec_aging {
	friend class CkArray;
private:
	int onPE;//The last known PE for this element
public:
	CkArrayRec_remote(CkArray *Narr,int NonPE):CkArrayRec_aging(Narr)
	{
		onPE=NonPE;
		if (onPE==CkMyPe())
			CkAbort("ERROR!  'remote' array element on this PE!\n");
	}

	virtual RecType type(void) {return remote;}

	//Send a message for this element.
	virtual void send(CkArrayMessage *msg)
	{
		access();//Update our modification date
		arr->deliverRemote(msg, onPE);
	}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)
	{
		if (arr->isHome(idx)) 
		//Home elements never become obsolete
		// if they did, we couldn't deliver messages to that element.
			return CmiFalse;
		else if (isStale())
			return CmiTrue;//We haven't been used in a long time
		else
			return CmiFalse;//We're fairly recent
	}
};


/*Buffers messages until record is replaced in the hash table, 
then delivers all messages to the replacing record.  This is 
used when an array element migrates, buffering messages until he 
arrives safely [replacing us by a CkArrayRec_remote]; 
or when a local element is created, buffering messages until 
the new element checks in [replacing us by a CkArrayRec_local].
*/
class CkArrayRec_buffering:public CkArrayRec_aging {
private:
	CkQ<CkArrayMessage *> buffer;//Buffered messages.
public:
	CkArrayRec_buffering(CkArray *Narr):CkArrayRec_aging(Narr) {}
	virtual ~CkArrayRec_buffering() {
		if (0!=buffer.length())
			CkAbort("Messages abandoned in array manager buffer!\n");
	}

	virtual RecType type(void) {return buffering;}

	//Send (or buffer) a message for this element.
	//  If idx==NULL, the index is packed in the message.
	//  If idx!=NULL, the index must be packed in the message.
	virtual void send(CkArrayMessage *msg)
	{
		buffer.enq(msg);
	}
	
	//This is called when this ArrayRec is about to be replaced.
	// We dump all our buffered messages off on the next guy,
	// who should know what to do with them.
	virtual void beenReplaced(void) 
	{
		DEBS((AA" Delivering queued messages\n"AB));
		CkArrayMessage *m;
		while (NULL!=(m=buffer.deq()))
		{
			DEBS((AA"Sending buffered message to %s\n"AB,idx2str(m->index())));
			arr->Send(m);
		}
	}
	
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)
	{
		if (isStale())
		{/*This indicates something is seriously wrong--
		buffers should be short-lived.*/
			CkPrintf("%d stale array message(s) found!\n",buffer.length());
			CkPrintf("Addressed to %s--",idx2str(idx));
			if (arr->isHome(idx)) 
				CkPrintf("is this an out-of-bounds array index?\n");
			else //Idx is a remote-home index
				CkPrintf("why weren't they forwarded?\n");
			
			CkAbort("Stale array manager message(s)!\n");
		}
		return CmiFalse;
	}
};

//Represents a local array element that just migrated away.
// Buffers messages until a remote record indicates the
// element has arrived safely.
class CkArrayRec_buffering_migrated:public CkArrayRec_buffering {
public:
	CkArrayRec_buffering_migrated(CkArray *Narr)
		:CkArrayRec_buffering(Narr)
	  {arr->num.migrating++;}
	virtual ~CkArrayRec_buffering_migrated()
	  {arr->num.migrating--;}

	virtual RecType type(void) {return buffering_migrated;}
};


//Add given element array record (which then owns it) at idx.
void CkArray::insertRec(CkArrayRec *rec,const CkArrayIndex &idx)
{
	CkArrayRec *old=(CkArrayRec *)hash.put(idx,rec);
	if (old!=NULL)
	{//There was an old element at this location
		if (old->type()==CkArrayRec::local && rec->type()==CkArrayRec::local) {
			CkPrintf("ERROR! Duplicate array index: %s\n",idx2str(idx));
			CkAbort("Duplicate array index used");
		}
		old->beenReplaced();
		delete old;
	}
}

//Call this on an unrecognized array index
static void abort_out_of_bounds(const CkArrayIndex &idx)
{
	CkPrintf("ERROR! Unknown array index: %s\n",idx2str(idx));
	CkAbort("Array index out of bounds\n");
}

//Look up array element in hash table.  
// Aborts with index out-of-bounds error if not found.
CkArrayRec *CkArray::elementRec(const CkArrayIndex &idx)
{
	CkArrayRec *ret=(CkArrayRec *)hash.get(idx);
	if (ret==NULL) abort_out_of_bounds(idx);
	return ret;
}

/*********************** Spring Cleaning *****************
Node zero periodically (every minute or so) broadcasts a
"spring cleaning" message to all nodes.  The nodes then
clean out old broadcasts (assuming nobody is going to migrate
in and need a broadcast from last spring) and old arrayRecs
(e.g. stale cached pointers to remote elements which may
have moved on or been deleted by now).

Cleaning often will free up memory quickly, but slow things
down because the cleaning takes time and some not-recently-referenced
remote element pointers might be valid and used some time in 
the future.
*/

#define SECONDS_Btw_SPRINGS 60
#define springCheck() \
	if ((CkMyPe()==0) && (CmiWallTimer()-lastCleaning>SECONDS_Btw_SPRINGS))\
	{\
		thisproxy.SpringCleaning();\
		lastCleaning=CmiWallTimer();\
	}

//Housecleaning: called periodically from node zero--
// Remove old broadcasts.
// Look for, and remove, old ArrayRec's.
void CkArray::SpringCleaning(void)
{
	DEBK((AA"Starting spring cleaning #%d (%.0f s since last)\n"AB,nSprings,CmiWallTimer()-lastCleaning));
	nSprings++;
	lastCleaning=CmiWallTimer();
	//Remove old broadcast messages
	int nDelete=oldBcasts.length()-(bcastNo-oldBcastNo);
	if (nDelete>0)
	{
		DEBK((AA"Cleaning out %d old broadcasts\n"AB,nDelete));
		for (int i=0;i<nDelete;i++)
			CkFreeMsg((void *)oldBcasts.deq());
	}
	oldBcastNo=bcastNo;
	
	//Poke through the hash table for old ArrayRecs.
	void *obj;
	HashKey *key;
	HashtableIterator *it=hash.objects();
	while (NULL!=(obj=it->next(&key)))
	{
		CkArrayRec *rec=(CkArrayRec *)obj;
		CkArrayIndex &idx=*(CkArrayIndex *)key;
		if (rec->isObsolete(nSprings,idx))
		{//This record is obsolete-- remove it from the table
			DEBK((AA"Cleaning out old record for %s\n"AB,idx2str(idx)));
			hash.remove(idx);
			delete rec;
			it->seek(-1);//retry this hash slot
		}
	}
	delete it;
}

/********************* Little CkArray Utilities ******************/

void CProxy_CkArrayBase::setReductionClient(CkReductionMgr::clientFn fn,void *param)
{ ((CkArray *)CkLocalBranch(_aid))->setClient(fn,param); }

//Fetch a local element via its index (return NULL if not local)
ArrayElement *CkArray::getElement(const CkArrayIndex &idx)
{
	CkArrayRec *rec=elementNrec(idx);
	if (rec!=NULL)
		return rec->element();
	else
		return NULL;
}  

/*********************** CkArray Creation ************************
CkArray creation is a several-step process:
	-The static CreateArray method creates the Array BOC and
sends its constructor a CkArrayCreateMessage.
	-The CkArray constructor finds the map object and returns.
	-The user adds elements to the array with insert
	-The user calls doneInserting and starts sending messages.
*/
class CkArrayCreateMsg : public CMessage_CkArrayCreateMsg
{
public:
  int numInitial;
  CkGroupID mapID;
  CkGroupID loadbalancer;
  CkArrayElementType type;
};

CkGroupID CkArray::CreateArray(int numInitialElements,
                               CkGroupID mapID,
                               ChareIndexType elementChare,
                               EntryIndexType elementMigrator)
{
  CkGroupID group;

  CkArrayCreateMsg *msg = new CkArrayCreateMsg;

  msg->numInitial=numInitialElements;
  msg->mapID = mapID;
  msg->type.chareType = elementChare;
  msg->type.migrateType = elementMigrator;
#if CMK_LBDB_ON
  msg->loadbalancer = lbdb;
#endif
  group = CProxy_CkArray::ckNew(msg);
  return group;
}

CkArray::CkArray(CkArrayCreateMsg *msg) : CkReductionMgr()
{
  //Set class variables
  type = msg->type;
  numInitial=msg->numInitial;
  num.local=num.migrating=0;
  bcastNo=oldBcastNo=0;
  nSprings=0;
  isInserting=CmiTrue;
  lastCleaning=CmiWallTimer();   
  
#if CMK_LBDB_ON
  DEBL((AA"Connecting to load balancer...\n"AB));
  //Find and register with the load balancer
  the_lbdb = CProxy_LBDatabase(msg->loadbalancer).ckLocalBranch();
  if (the_lbdb == 0)
    CkPrintf("[%d] LBDatabase not created?\n",CkMyPe());
  DEBL((AA"Connected to load balancer %p\n"AB,the_lbdb));

  // Register myself as an object manager
  LDOMid myId;
  myId.id = (int)thisgroup;
  LDCallbacks myCallbacks;
  myCallbacks.migrate = (LDMigrateFn)ArrayElement::staticMigrate;
  myCallbacks.setStats = (LDStatsFn)staticSetStats;
  myCallbacks.queryEstLoad = (LDQueryEstLoadFn)staticQueryLoad;
  myLBHandle = the_lbdb->RegisterOM(myId,this,myCallbacks);  

  // Tell the lbdb that I'm registering objects
  the_lbdb->RegisteringObjects(myLBHandle);  
  
  //Add a barrier reciever so we can fake Registering/DoneRegister calls
  the_lbdb->AddLocalBarrierReceiver(
    	(LDBarrierFn)staticRecvAtSync,(void*)(this));
    	
  /*Set up the dummy barrier-- the load balancer needs somebody
  to call AtSync on each PE, so if there are no array elements 
  the array BOC has to do it.
  */
  dummyBarrierHandle = the_lbdb->AddLocalBarrierClient(
    (LDResumeFn)staticDummyResumeFromSync,(void*)(this));

  // Activate the AtSync for this one immediately.  Note, that since
  // we have not yet called DoneRegisteringObjects(), nothing
  // will happen yet.
  thisproxy.DummyAtSync(CkMyPe());
  
#endif

	mapID=msg->mapID;map=NULL;
	delete msg;
	
	if (NULL!=CkLocalBranch(mapID))
		initAfterMap();
	else //Wait for the map to be created 
		CProxy_CkArrayMap(mapID).callMeBack(new CkGroupInitCallbackMsg(
			static_initAfterMap,(void *)this));
}
  
void CkArray::static_initAfterMap(void *dis)
 {((CkArray *)dis)->initAfterMap();}
void CkArray::initAfterMap(void)
{
	//The map is alive-- register with it
	map=(CkArrayMap *)CkLocalBranch(mapID);
	if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
	CkArrayMapRegisterMessage *mapMsg = new CkArrayMapRegisterMessage;
	mapMsg->numElements = numInitial;
	mapMsg->array = this;
	mapHandle=map->registerArray(mapMsg);
	
	//Don't start reduction until all elements have been inserted.
	CkReductionMgr::creatingContributors();
}

//Create 1D initial array elements
void CProxy_CkArrayBase::create1Dinitial(int ctorIndex,int numInitial,
	CkArrayMessage *inM)
{
	DEBC(("In createInitial-- will build %d elements\n",numInitial));
	if (numInitial>0)
	{//Build some 1D elements: (mostly for backward compatability)
		CProxy_CkArray aProxy(_aid);
		CkArray *aMgr=(CkArray *)CkLocalBranch(_aid);
		for (int i=0;i<numInitial;i++)
		{
			CkArrayMessage *m=NULL;
			if (inM!=NULL)
			{ 
				if (i!=numInitial-1)
					m=(CkArrayMessage *)CkCopyMsg((void **)&inM);
				else
					m=inM;//Last time around, send off the original message
			}
			insertAtIdx(ctorIndex,-1,CkArrayIndex1D(i),m);
		}
		DEBC(("Done building elements\n"));
		doneInserting();
	}
}

void CProxy_CkArrayBase::doInsert(int ctorIndex,int onPE,CkArrayMessage *m)
{
	if (_idx==NULL) CkAbort("Must use arrayProxy[index].insert()!\n"); 
	insertAtIdx(ctorIndex,onPE,*_idx,m);
	delete _idx;//<- this lets us just blow away old element proxies.
	_idx=NULL;
}
void CProxy_CkArrayBase::insertAtIdx(int ctorIndex,int onPE,
	const CkArrayIndex &idx,CkArrayMessage *m)
{
	if (m==NULL) m=new CkArrayElementCreateMsg;
	m->type.create.ctorIndex=ctorIndex;
	CkArray *aMgr=(CkArray *)CkLocalBranch(_aid);
	if (onPE==-1) onPE=aMgr->homePE(idx);
	
	DEBC((AA"Proxy inserting element %s on PE %d\n"AB,idx2str(idx),onPE));
	CProxy_CkArray(_aid).InsertElement(m->insert(idx),onPE);
}

//Allocate a new, uninitialized array element of the given (chare) type
// and owning the given index.
ArrayElement *CkArray::newElement(int chareType,CkArrayIndex *ind)
{
	ArrayElement *el=(ArrayElement *)malloc(_chareTable[chareType]->size);
	el->thisindex=ind;
	el->thisArray=this;
	el->thisArrayID=thisgroup;
	return el;
}

//Call the user's given constructor, passing the given message.
void CkArray::ctorElement(ArrayElement *el,int ctor,CkArrayMessage *m)
{
	curElementIsDead=CmiFalse;
	
	//Call the user's constructor
	_entryTable[ctor]->call(m, (void *)el);
}


//This method is called by the user to add an element.
void CkArray::InsertElement(CkArrayMessage *m)
{
	const CkArrayIndexConst &idx=m->index();
	int homePe=homePE(idx);
 //Create the element on this PE
	DEBC((AA"  adding local element %s\n"AB,idx2str(idx)));
	if (homePe!=CkMyPe())
	{//Let this element's home PE know it lives here now
		DEBC((AA"  Telling %s's home %d that it lives here.\n"AB,idx2str(idx),homePe));
		thisproxy.UpdateLocation((new CkArrayUpdateMsg)->insert(idx),
			homePe);
	}
	
	//Build the element
	CkArrayIndex *elIdx=m->copyIndex();
	ArrayElement *el=newElement(type.chareType,elIdx);
#if CMK_LBDB_ON //Load balancer utilities:
	el->usesAtSync=CmiFalse;
#endif
	el->bcastNo=bcastNo;
	contributorCreated(&el->reductionInfo);
	
	//Call the element's constructor (keeps message m)
	ctorElement(el,m->type.create.ctorIndex,m);
		
	if (!curElementIsDead) //<- element may have immediately migrated away or died.
	{
		el->lbRegister();// Register the object with the load balancer
		insertRec(new CkArrayRec_local(this,el),*elIdx);
	}	
	/*Was:	CkCreateChare(type.chareType,m->type.create.ctorIndex,m, 0, CkMyPe());*/
}

void CProxy_CkArrayBase::doneInserting(void)
{
	DEBC((AA"Broadcasting a doneInserting request\n"AB));
	//Broadcast a DoneInserting
	CProxy_CkArray(_aid).DoneInserting();
}

//This is called after the last array insertion.
void CkArray::DoneInserting(void)
{
	if (isInserting)
	{
		isInserting=CmiFalse;
		DEBC((AA"Done inserting objects\n"AB));
#if CMK_LBDB_ON
		the_lbdb->DoneRegisteringObjects(myLBHandle);
#endif
		CkReductionMgr::doneCreatingContributors();
		if (bcastNo>0)
		{//These broadcasts were delayed until all elements were inserted--
			DEBC((AA"  Delivering insert-delayed broadcasts\n"AB));
			for (int i=0;i<bcastNo;i++)
				deliverBroadcast(oldBcasts[i]);
		}
	}
}

/*Called from element destructor-- removes
element from hash table, notifies family, etc.
*/
void CkArray::ElementDying(CkArrayRemoveMsg *m)
{
	//Remove the element from our hashtable
	const CkArrayIndex &idx=m->index();
	CkArrayRec *rec=elementNrec(idx);
	if (rec==NULL) {delete m;return; }//<- ignore this one.  We never knew him.
	CkArrayRec::RecType rtype=rec->type();
	curElementIsDead=CmiTrue;
	
	if (rtype==CkArrayRec::buffering_migrated)
	{//This is just the old copy of a migrator-- ignore him
		DEBC((AA"Ignoring death of migrating element %s\n"AB,idx2str(idx)));
		delete m;
		return;
	}
	hash.remove(idx);
	if (rtype==CkArrayRec::local)
	{//This is a local element dying a natural death
		//Detach him from his arrayRec (prevents double-delete)
		ArrayElement *el=((CkArrayRec_local *)rec)->releaseElement();
#if CMK_LBDB_ON
		the_lbdb->ObjectStop(el->ldHandle);
#endif
		//Forward the death notice to the home
		if (homePE(idx)!=CkMyPe())
			thisproxy.ElementDying(m,homePE(idx));
		else delete m;
	} else delete m;
	delete rec;
/*	//Install a zombie to keep the living from re-using this index.
	insertRec(new CkArrayRec_dead(this),idx); */
	springCheck();//Check if it's spring cleaning time
}


/************************ Migration *********************/

void CkArray::migrateMe(ArrayElement *el, int where)
{
	const CkArrayIndex &idx=*el->thisindex;
	DEBM((AA"Migrating element %s to %d\n"AB,idx2str(idx),where));
	
	//Pack the element and send it off
	int bufSize;
	{ PUP::sizer p; el->pup(p); bufSize=p.size(); }
	ArrayElementMigrateMessage *msg = new (&bufSize, 0) ArrayElementMigrateMessage;
	msg->type.create.fromPE=CkMyPe();
	msg->type.create.ctorIndex=type.migrateType;
	{ PUP::toMem p(msg->packData); el->pup(p);}
	msg =msg->insert(idx);
	DEBM((AA"Migrated index size %s\n"AB,idx2str(msg->index())));
	contributorLeaving(&el->reductionInfo);
	
	thisproxy.RecvMigratedElement(msg,where);
	
	//Switch this element's CkArrayRec to buffering--
	// This will store his messages until he arrives safely.
	insertRec(new CkArrayRec_buffering_migrated(this),idx);
}

void CkArray::RecvMigratedElement(ArrayElementMigrateMessage *msg)
{
	{
		const CkArrayIndexConst &idx=msg->index();
		int srcPE=msg->type.create.fromPE;
		DEBM((AA"Recv'd migrating element %s from %d\n"AB,idx2str(idx),srcPE));
	
		//Send update to home & sender to let him know the migration came out OK
		int home=homePE(idx);
		if (home!=srcPE && home!=CkMyPe())
			thisproxy.UpdateLocation((new CkArrayUpdateMsg)->insert(idx),home);
		thisproxy.UpdateLocation((new CkArrayUpdateMsg)->insert(idx),srcPE);
	}

/*Was: CkCreateChare(type.chareType, type.migrateType, msg, 0, CkMyPe());*/
	//Create the new element on this PE (passing on the msg)
	CkArrayIndex *elIdx=msg->copyIndex();
	ArrayElement *el=newElement(type.chareType,elIdx);
	ctorElement(el,msg->type.create.ctorIndex,msg);
	
	if (curElementIsDead) //<- element may have immediately migrated away or died.
		{CkFreeMsg((void *)msg);return;}

	//Unpack the element's fields (must be *after* ctor since this is a virtual function)
	el->bcastNo=-1;//<- to make sure they call their superclass's pup
	{ PUP::fromMem p(msg->packData); el->pup(p); }
	if (el->bcastNo==-1) CkAbort("You forgot to call ArrayElement1D::pup from your array element's pup routine!\n");
	CkFreeMsg((void *)msg);//<- delete the old message
	contributorArriving(&el->reductionInfo);
	
	el->lbRegister();// Register the object with the load balancer
	
#if CMK_LBDB_ON
	the_lbdb->Migrated(el->ldHandle);
#endif
	//Catch up on any missed broadcasts
	bringBroadcastUpToDate(el);		
	
	if (!curElementIsDead)
		//Put the new guy in the hash table
		insertRec(new CkArrayRec_local(this,el),*elIdx);
}

/*This is called when a message is received with a hopcount
greater than one-- it tells us to direct messages for the given
array element straight to the given PE.
*/
void CkArray::UpdateLocation(CkArrayUpdateMsg *msg)
{
	const CkArrayIndexConst &idx=msg->index();
	int onPE=msg->type.msg.fromPE;
	DEBM((AA"Recv'd location update for %s from %d\n"AB,idx2str(idx),onPE));
	if (onPE!=CkMyPe())
	{
		CkArrayRec *rec=elementNrec(idx);
		if (rec!=NULL && rec->type()==CkArrayRec::remote)
			//There's already a remote record-- just update it
			((CkArrayRec_remote *)rec)->onPE=onPE;
		else if (rec!=NULL && rec->type()==CkArrayRec::local)
			DEBM((AA" Ignoring location update for local element %s\n"AB,idx2str(idx)));
		else
			insertRec(new CkArrayRec_remote(this,onPE),idx);
	}
	delete msg;
}


/********************* CkArray Messaging ******************/

void CkArray::deliverLocal(CkArrayMessage *msg,ArrayElement *el)
{
	DEBS((AA"Delivering local message for element %s\n"AB,idx2str(el)));
	int hopCount=msg->type.msg.hopCount;
	if (hopCount>1)
	{//This message took several hops to reach us.
		int srcPE=msg->type.msg.fromPE;
		if (srcPE==CkMyPe())
			DEB((AA"Odd routing: local element %s is %d hops away!\n"AB,idx2str(el),hopCount));
		else
		{//Send a routing message letting original sender know new element location
			DEBS((AA"Sending update back to %d for element %s\n"AB,srcPE,idx2str(el)));
			thisproxy.UpdateLocation((new CkArrayUpdateMsg)->insert(*el->thisindex),srcPE);
		}
	}
	curElementIsDead=CmiFalse;
	
	int entry=msg->type.msg.entryIndex;
#if CMK_LBDB_ON
	the_lbdb->ObjectStart(el->ldHandle);
	_entryTable[entry]->call(msg, el);
	if (!curElementIsDead)
		the_lbdb->ObjectStop(el->ldHandle);
#else
	_entryTable[entry]->call(msg, el);
#endif
}

void CkArray::deliverRemote(CkArrayMessage *msg,int onPE)
{
	DEBS((AA"Forwarding message for %s to %d\n"AB,idx2str(msg->index()),onPE));
	msg->type.msg.hopCount++;
	thisproxy.RecvForElement(msg, onPE);
}

void CProxy_CkArrayBase::send(CkArrayMessage *msg, int entryIndex)
{
	msg->type.msg.fromPE = CkMyPe();
	msg->type.msg.entryIndex = entryIndex;
	msg->type.msg.hopCount = 0;
	msg=msg->insert(*_idx);//Insert array index into message
	CProxy_CkArray(_aid).Send(msg, CkMyPe());
	delete _idx;
}
//Put given message 
void CkArray::Send(CkArrayMessage *msg)
{
	const CkArrayIndex &idx=msg->index();
	
#if CMK_LBDB_ON
	the_lbdb->Send(myLBHandle,idx2LDObjid(idx),UsrToEnv(msg)->getTotalsize());
#endif
	CkArrayRec *rec=elementNrec(idx);
	if (rec!=NULL)
	{//This index *is* in the hash table-- just call send
		DEBS((AA"Sending to hashed element %s\n"AB,idx2str(idx)));
		rec->send(msg);
	}
	else
	{//This index is not hashed-- send to its "home" processor
		int onPE=homePE(idx);
		deliverRemote(msg,onPE);
	}
	springCheck();//Check if it's spring cleaning time
}

//This receives a message from the net destined for a 
// (probably) local element.
void CkArray::RecvForElement(CkArrayMessage *msg)
{
	const CkArrayIndexConst &idx=msg->index();
	DEBS((AA"RecvForElement %s\n"AB,idx2str(idx)));
	CkArrayRec *rec=elementNrec(idx);
	if (rec==NULL)
	{ //Element not found in hash table-- add an entry for it
		DEBC((AA"Adding buffer for unknown element %s\n"AB,idx2str(idx)));
		rec=new CkArrayRec_buffering(this);
		insertRec(rec,idx);
	}
	rec->send(msg);
}

/*********************** CkArray Broadcast ******************/

void CProxy_CkArrayBase::broadcast(CkArrayMessage *msg, int entryIndex)
{
	msg->type.msg.fromPE = CkMyPe();
	msg->type.msg.entryIndex=entryIndex;
	msg->type.msg.hopCount=0;
	int serializer=0;//1623802937%CkNumPes();
	if (CkMyPe()==serializer)
	{
		DEBB((AA"Sending array broadcast\n"AB));
		CProxy_CkArray(_aid).RecvBroadcast(msg);
	} else {
		DEBB((AA"Forwarding array broadcast to serializer node %d\n"AB,serializer));
		CProxy_CkArray(_aid).SendBroadcast(msg,serializer);
	}
}
//Reflect a broadcast off this PE:
void CkArray::SendBroadcast(CkArrayMessage *msg)
{
	thisproxy.RecvBroadcast(msg);
}

//Increment broadcast count; deliver to all local elements
void CkArray::RecvBroadcast(CkArrayMessage *msg)
{
	bcastNo++;
	DEBB((AA"Received broadcast %d\n"AB,bcastNo));
	if (!isInserting)
		deliverBroadcast(msg);
	
	oldBcasts.enq(msg);//Stash the message for later
	springCheck();//Check if it's time for spring cleaning
}

//Deliver a copy of the given broadcast to all local elements
void CkArray::deliverBroadcast(CkArrayMessage *bcast)
{
	//Poke through the hash table for local elements.
	void *obj;HashKey *key;
	HashtableIterator *it=hash.objects();
	while (NULL!=(obj=it->next(&key)))
	{
		CkArrayRec *rec=(CkArrayRec *)obj;
		ArrayElement *el=rec->element();
		if (el!=NULL && el->bcastNo<bcastNo)
		//el hasn't heard this broadcast yet--
			deliverBroadcast(bcast,el);
	}
	delete it;
}

//Deliver a copy of the given broadcast to the given local element
void CkArray::deliverBroadcast(CkArrayMessage *bcast,ArrayElement *el)
{
	el->bcastNo++;
	void *newMsg=CkCopyMsg((void **)&bcast);
	DEBB((AA"Delivering broadcast %d to element %s\n"AB,el->bcastNo,idx2str(el)));
	deliverLocal((CkArrayMessage *)newMsg,el);
}
//Deliver a copy of the given broadcast to the given local element
void CkArray::bringBroadcastUpToDate(ArrayElement *el)
{
	if (el->bcastNo<bcastNo)
	{//This element needs some broadcasts-- it must have
	//been migrating during the broadcast.
		int i,nDeliver=bcastNo-el->bcastNo;
		DEBM((AA"Migrator %s missed %d broadcasts--\n"AB,idx2str(el),nDeliver));
	//Skip the old junk at the front of the bcast queue
		for (i=oldBcasts.length()-1;i>=nDeliver;i--)
			oldBcasts.enq(oldBcasts.deq());
	//Deliver the newest messages, in old-to-new order 
		for (i=nDeliver-1;i>=0;i--)
		{
			CkArrayMessage *msg=oldBcasts.deq();
			deliverBroadcast(msg,el);
			oldBcasts.enq(msg);
		}
	}
}

/************************** Load Balancing **********************/

#if !CMK_LBDB_ON
void CkArray::DummyAtSync(void) {}
#else //CMK_LBDB_ON

/*
The DummyAtSync is needed because the LBDB needs to get
an AtLocalBarrier on each PE *even* if there are no 
array elements.  Hence the Array BOC itself does a 
"dummy" AtLocalBarrier which is immediately triggered
(and then, on the next resumeFromSync, retriggered).
OSL, 3/11/2000
*/

void CkArray::DummyAtSync(void)
{
  DEBL((AA"DummyAtSync called\n"AB));
  the_lbdb->AtLocalBarrier(dummyBarrierHandle);
}

void CkArray::staticDummyResumeFromSync(void* data)
{
  CkArray* me = (CkArray*)(data);
  me->dummyResumeFromSync();
}
void CkArray::dummyResumeFromSync()
{
  DEBL((AA"DummyResumeFromSync called\n"AB));
  the_lbdb->DoneRegisteringObjects(myLBHandle);// <-- NOT true! OSL
  thisproxy.DummyAtSync(CkMyPe());
}
/*
Somehow, adding the fake DoneRegisteringObjects above
and RegisteringObjects below prevents a DummyAtSync/
dummyResumeFromSync loop in programs that don't have any
usesAtSync array elements.  Smells like a hack to me.
OSL, 3/12/2000
*/
void CkArray::staticRecvAtSync(void* data)
{
  ((CkArray*)data)->recvAtSync();
}
void CkArray::recvAtSync()
{
  DEBL((AA"recvAtSync called\n"AB));
  the_lbdb->RegisteringObjects(myLBHandle);// <-- NOT true! OSL
}


//These functions never seem to get called-- OSL, 3/7/2000
void CkArray::staticSetStats(LDOMHandle _h, int _state)
{
  ((CkArray*)(_h.user_ptr))->setStats(_h,_state);   
}
void CkArray::setStats(LDOMHandle _h, int _state)
{
  CkPrintf("%s(%d)[%d]: SetStats request received\n",
	   __FILE__,__LINE__,CkMyPe());
}

void CkArray::staticQueryLoad(LDOMHandle _h)
{
  ((CkArray*)_h.user_ptr)->queryLoad(_h);
}
void CkArray::queryLoad(LDOMHandle _h)
{
  CkPrintf("%s(%d)[%d]: QueryLoad request received\n",
	   __FILE__,__LINE__,CkMyPe());
}

#endif // CMK_LBDB_ON


/************************ Messages **********************/

CkArrayIndexMsg::CkArrayIndexMsg() //Index message constructor-- length is zero
{
/*//This code seems to trash *some* received messages
// (e.g. migration messages), which is pretty odd.
	DEBI((AA"In message %p constructor\n"AB,this));
	indexLength=-123654;

//It goes along with this snippet, which should go in insert, below
	else if (idxLen<=indexLength)
	{//We've already allocated a spot for this (big) index at the end of this message
		DEBI((AA"Stuffing large index %s in pre-allocated end of message %p\n"AB,idx2str(idx),this));
		dest=this;
		store=(int *)((char *)dest+dest->indexStore[0]);
	}
*/
}

//Extract a heap-allocated copy of this message's array index
CkArrayIndex *CkArrayIndexMsg::copyIndex(void)
{
	int len=indexLength;const void *data;
	if (indexLength<=(int)CKARRAYINDEX_STORELEN)
		data=(const void *)indexStore;//Index data located in indexStore array
	else //Data is stored at end of this message
		data=(const void *)((char *)this+indexStore[0]);
	CkArrayIndex *ret=CkArrayIndex::newIndex(len,data);
	DEBI((AA"Extracting heap copy of index %s from message %p\n"AB,idx2str(*ret),this));
	return ret;
}
//Extract a read-only pointer to this message's array index
const CkArrayIndexConst CkArrayIndexMsg::index(void) const
{
	int len=indexLength;const void *data;
	if (indexLength<=(int)CKARRAYINDEX_STORELEN)
		data=(const void *)indexStore;//Index data in indexStore array
	else //Data is stored at end of this message
		data=(const void *)((char *)this+indexStore[0]);
	DEBI((AA"R/O copy of index %s pulled from message %p\n"AB,idx2str(CkArrayIndexConst(len,(const unsigned char *)data)),this));
	return CkArrayIndexConst(len,(const unsigned char *)data);
}
/*Store the given index into this message (somewhere).
We'd like messages to be fixed-sized, but we'd also
like to allow an arbitrarily-long array index.  The compromise
is to always reserve CKARRRAYINDEX_STORELEN bytes for the index--
if the actual index can fit in this space, it's copied in.
If the index is too long to fit in this array, we have to
reallocate the message and copy the index in at the end.
(in this case, indexStore[0] gives the offset to the index data)
Hence we want to pick CKARRRAYINDEX_STORELEN big enough that
this slow reallocation is rare, but small enough not to waste space.

This procedure is normally wrapped by the templated version
CkArrayIndexMsgT<MESG>::insert(), which just calls this and then
does a type cast to a MESG pointer.
*/
CkArrayIndexMsg *CkArrayIndexMsg::insertArrayIndex(const CkArrayIndex &idx)
{
	DEBI((AA"Inserting index %s into message %p\n"AB,idx2str(idx),this));
	int idxLen;
	const void *idxData=(const void *)idx.getKey(idxLen);
	CkArrayIndexMsg *dest;//Message to return
	int *store;//Location to store the index data
	if (idxLen<=(int)CKARRAYINDEX_STORELEN)
		{dest=this;store=dest->indexStore;}
	else
	{//Index too big to fit in start of this message-- allocate a new, bigger
	// message and copy the index in at the end.
		DEBI((AA"Resizing message %p to make room for index %s\n"AB,this,idx2str(idx)));
		register void *srcMsg = (void *)this;
		register envelope *srcEnv = UsrToEnv(srcMsg);
		register unsigned char msgidx = srcEnv->getMsgIdx();
		if(!srcEnv->isPacked() && _msgTable[msgidx]->pack)
		{//Turn source into a flat array of bytes
			srcMsg = _msgTable[msgidx]->pack(srcMsg);srcEnv=UsrToEnv(srcMsg);
			srcEnv->setPacked(1);
		}
		//Make a message big enough to hold the old message and new index
		register int newSize = srcEnv->getTotalsize()+idxLen;
		DEBI((AA"New message will be %d bytes long\n"AB,newSize));
		register envelope *newEnv = (envelope *) CmiAlloc(newSize);
		memcpy(newEnv, srcEnv, srcEnv->getTotalsize());
		newEnv->setTotalsize(newSize);//<- this is the only field that differs from srcEnv
		
		if(srcEnv->isPacked() && _msgTable[msgidx]->unpack)
		{//Turn source back into a message
			srcMsg = _msgTable[msgidx]->unpack(srcMsg);srcEnv=UsrToEnv(srcMsg);
			srcEnv->setPacked(0);
		}
		register void *newMsg = EnvToUsr(newEnv);
		if(newEnv->isPacked() && _msgTable[msgidx]->unpack) 
		{//Turn duplicate back into a message
			newMsg = _msgTable[msgidx]->unpack(newMsg);newEnv=UsrToEnv(newMsg);
			newEnv->setPacked(0);
		}
		dest=(CkArrayIndexMsg *)newMsg;
		dest->indexStore[0]=newSize-idxLen-sizeof(envelope);
		store=(int *)((char *)dest+dest->indexStore[0]);
		CkFreeMsg((void *)this);//We're not needed any more
	}
	//Write the index into the destination message
	dest->indexLength=idxLen;
	memcpy(store,idxData,idxLen);
	return dest;
}

CkArrayUpdateMsg::CkArrayUpdateMsg(void)
{
	type.msg.fromPE=CkMyPe();
}

void *
ArrayElementMigrateMessage::alloc(int msgnum,int size,int *array,int priobits)
{
  int totalsize;
  totalsize = size + array[0] + 8;
  ArrayElementMigrateMessage *newMsg = (ArrayElementMigrateMessage *)
    CkAllocMsg(msgnum,totalsize,priobits);
  DEBM((AA"  Allocated varsize message %d, %d bytes at %p\n"AB,msgnum,totalsize,newMsg));
  newMsg->packData = (char *)newMsg + ALIGN8(size);
  return (void *) newMsg;
}

void *
ArrayElementMigrateMessage::pack(ArrayElementMigrateMessage* in)
{
  in->packData = (void*)((char*)in->packData-(char *)&(in->packData));
  return (void*) in;
}

ArrayElementMigrateMessage* 
ArrayElementMigrateMessage::unpack(void *in)
{
  ArrayElementMigrateMessage *me = new (in) ArrayElementMigrateMessage;
  me->packData = (char *)&(me->packData) + (size_t)me->packData;
  return me;
}

#include "CkArray.def.h"




