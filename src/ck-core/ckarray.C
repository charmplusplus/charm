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
  {return idx2str(el->thisindex);}

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

//General (length-independent) hash/compare functions
CkHashCode CkArrayIndex_hashN(const void *keyData,size_t /*len*/)
{
	return ((CkArrayIndexMax *)keyData)->hash();
}
inline CkHashCode CkArrayIndex_hashN_fast(const CkArrayIndexMax &key)
{
	return key.hash();
}

inline int CkArrayIndex_compareN_fast(const CkArrayIndexMax &i1,const CkArrayIndexMax &i2)
{
#if ONEDONLY
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
int CkArrayIndex_compareN(const void *k1,const void *k2,size_t /*len*/)
{
	return CkArrayIndex_compareN_fast(
		*(const CkArrayIndexMax *)k1,
		*(const CkArrayIndexMax *)k2);
}

void CkArrayIndex::pup(PUP::er &p) 
{
	p(nInts);
	p(data(),nInts);
}

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
/*********************** Array Messages ************************/
inline CkArrayIndexMax &CkArrayMessage::array_index(void)
{
	return UsrToEnv((void *)this)->array_index();
}
inline unsigned short &CkArrayMessage::array_ep(void)
{
	return UsrToEnv((void *)this)->array_ep();
}
inline unsigned char &CkArrayMessage::array_hops(void)
{
	return UsrToEnv((void *)this)->array_hops();
}
inline unsigned int CkArrayMessage::array_getSrcPe(void)
{
	return UsrToEnv((void *)this)->array_srcPe();
}
inline void CkArrayMessage::array_setSrcPe(void)
{
	UsrToEnv((void *)this)->array_srcPe()=CkMyPe();
}



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
  RRMap(CkMigrateMessage *m) {}
  int registerArray(CkArrayMapRegisterMessage *msg)
  {
    delete msg;
    return 0;
  }
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

class CkArrayInit : public Chare 
{
public:
  CkArrayInit(CkArgMsg *msg) {
    _RRMapID = CProxy_RRMap::ckNew();
    delete msg;
  }
  CkArrayInit(CkMigrateMessage *m) {}
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
  thisArray->localElementDying(this);
  lbUnregister();
  //To detect use-after-delete: 
  thisArray=(CkArray *)0xDEADa7a1;
  thisindex.nInts=-123456;
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
  if (where!=CkMyPe()) {
    thisArray->migrateMe(this,where);
  }
}

void ArrayElement::pup(PUP::er &p)
{
  DEBM((AA"  ArrayElement::pup()\n"AB));
  Chare::pup(p);
  thisArrayID.pup(p);
  if (p.isUnpacking())
  	thisArray=thisArrayID.ckLocalBranch();
  thisindex.pup(p);
  p(thisChareType);
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
		idx2LDObjid(thisindex),(void *)this,1);
  
  if (usesAtSync)
    ldBarrierHandle = thisArray->the_lbdb->AddLocalBarrierClient(
								 (LDBarrierFn)staticResumeFromSync,
								 (void*)(this));
}

void ArrayElement::lbUnregister(void)//Disconnect from load balancer
{
  DEBL((AA"Unregistering element %s from load balancer (%s AtSync)\n"AB,idx2str(this),usesAtSync?"and":"without"));	
  thisArray->the_lbdb->ObjectStop(ldHandle);
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
  thisIndex=thisindex.data()[0];
}
ArrayElement1D::ArrayElement1D(CkMigrateMessage *msg)
{
  numElements=thisArray->numInitial;
  thisIndex=thisindex.data()[0];
}

void ArrayElement1D::pup(PUP::er &p)
{
  ArrayElement::pup(p);//Pack superclass
  p(numElements);
  p(thisIndex);
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
  friend class CkArray;
protected:
  CkArray *arr;
  int lastAccess;//Age when last accessed
  //Called when we discover we are obsolete before we suicide
  virtual void weAreObsolete(const CkArrayIndex &idx) {}
 public:
  CkArrayRec() {}
  CkArrayRec(CkArray *Narr) {arr=Narr;lastAccess=arr->nSprings;}
  virtual ~CkArrayRec() {}
  typedef enum {
    base=0,//Base class (invalid type)
    local,//Array element that lives on this PE
    remote,//Array element that lives on some other PE
    buffering,//Array element that was just created
    dead//Deleted element (for debugging)
  } RecType;
  //Return the type of this ArrayRec:
  virtual RecType type(void) {return base;}
  
  //Send (or buffer) a message to this element.
  virtual void send(CkArrayMessage *msg) = 0;

  //Receive a message for this element (default: forward to him)
  virtual void recv(CkArrayMessage *msg) {send(msg);}
  
  //This is called when this ArrayRec is about to be replaced.
  // It is only used to deliver buffered element messages.
  virtual void beenReplaced(void) 
    {/*Default: ignore replacement*/}	
  
  //Return if this rec is now obsolete
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
  
  //Return this rec's array element; or NULL if none
  virtual ArrayElement *element(void) {return NULL;}
  virtual void pup(PUP::er &p) { p(lastAccess); }
};


//Represents a local array element
class CkArrayRec_local:public CkArrayRec {
  friend class CkArray;
private:
  ArrayElement *el;//Our local element
  void killElement(void) {
    if (el!=NULL)
      delete el;
  }
public:
  CkArrayRec_local() : CkArrayRec() {}
  CkArrayRec_local(CkArray *Narr,ArrayElement *Nel):CkArrayRec(Narr)
    {el=Nel;arr->num.local++;}

  virtual ~CkArrayRec_local() {
    killElement();
    arr->num.local--;
  }

  virtual RecType type(void) {return local;}
  
  //Deliver a message to this local element, going via the
  // message queue.
  virtual void send(CkArrayMessage *msg) {
    CProxy_CkArray(arr->getGroupID()).RecvForElement(msg,CkMyPe());
  }

  //Deliver a message to this local element.
  virtual void recv(CkArrayMessage *msg);

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
  CkArrayRec_dead() : CkArrayRec() {}
  CkArrayRec_dead(CkArray *Narr):CkArrayRec(Narr) {}
  
  virtual RecType type(void) {return dead;}
  
  virtual void send(CkArrayMessage *msg) {
    CkPrintf("Dead array element is %s.\n",idx2str(msg->array_index()));
    CkAbort("Send to dead array element!\n");
  }
  virtual void beenReplaced(void) 
    {CkAbort("Can't re-use dead array element!\n");}
  
  //Return if this element is now obsolete (it isn't)
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {return CmiFalse;}	
};

//This is the abstract superclass of arrayRecs that keep track of their age.
// Its kids are remote and buffering.
class CkArrayRec_aging:public CkArrayRec {
  friend class CkArray;
private:
  int lastAccess;//Age when last accessed
 protected:
  //Update our access time
  inline void access(void) {
    lastAccess=arr->nSprings;
  }
  //Return if we are "stale"-- we were last accessed a while ago
  CmiBool isStale(void) {
    if (arr->nSprings-lastAccess>3) return CmiTrue;
    else return CmiFalse;
  }
 public:
  CkArrayRec_aging() : CkArrayRec() {}
  CkArrayRec_aging(CkArray *Narr):CkArrayRec(Narr) {lastAccess=arr->nSprings;}
  //Return if this element is now obsolete
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
  virtual void pup(PUP::er &p) { CkArrayRec::pup(p); p(lastAccess); }
};


//Represents a remote array element
class CkArrayRec_remote:public CkArrayRec_aging {
  friend class CkArray;
 private:
  int onPE;//The last known PE for this element
 public:
  CkArrayRec_remote() : CkArrayRec_aging() {} 
  CkArrayRec_remote(CkArray *Narr,int NonPE):
    CkArrayRec_aging(Narr) {
    onPE=NonPE;
#ifndef CMK_OPTIMIZE
    if (onPE==CkMyPe())
      CkAbort("ERROR!  'remote' array element on this PE!\n");
#endif
  }
  
  virtual RecType type(void) {return remote;}
  
  //Send a message for this element.
  virtual void send(CkArrayMessage *msg) {
    access();//Update our modification date
    arr->deliverRemote(msg, onPE);
  }
  //Return if this element is now obsolete
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
    if (arr->isHome(idx)) 
      //Home elements never become obsolete
      // if they did, we couldn't deliver messages to that element.
      return CmiFalse;
    else if (isStale())
      return CmiTrue;//We haven't been used in a long time
    else
      return CmiFalse;//We're fairly recent
  }
  virtual void pup(PUP::er &p) { CkArrayRec_aging::pup(p); p(onPE); }
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
  CkArrayRec_buffering() : CkArrayRec_aging() {}
  CkArrayRec_buffering(CkArray *Narr):CkArrayRec_aging(Narr) {}
  virtual ~CkArrayRec_buffering() {
    if (0!=buffer.length())
      CkAbort("Messages abandoned in array manager buffer!\n");
  }
  
  virtual RecType type(void) {return buffering;}
  
  //Send (or buffer) a message for this element.
  //  If idx==NULL, the index is packed in the message.
  //  If idx!=NULL, the index must be packed in the message.
  virtual void send(CkArrayMessage *msg) {
    buffer.enq(msg);
  }
  
  //This is called when this ArrayRec is about to be replaced.
  // We dump all our buffered messages off on the next guy,
  // who should know what to do with them.
  virtual void beenReplaced(void) {
    DEBS((AA" Delivering queued messages\n"AB));
    CkArrayMessage *m;
    while (NULL!=(m=buffer.deq())) {
      DEBS((AA"Sending buffered message to %s\n"AB,idx2str(m->array_index())));
      arr->RecvForElement(m);
    }
  }
  
  //Return if this element is now obsolete
  virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
    if (isStale()) {
      /*This indicates something is seriously wrong--
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
  
  virtual void pup(PUP::er &p) {
    CkArrayRec_aging::pup(p);
    CkArray::pupArrayMsgQ(buffer, p);
  }
};

//Add given element array record (which then owns it) at idx.
void CkArray::insertRec(CkArrayRec *rec,const CkArrayIndex &idx)
{
  CkArrayRec *old=NULL;
  old=elementNrec(idx);
  hash.put(*(CkArrayIndexMax *)&idx)=rec;
  if (old!=NULL) {
    //There was an old element at this location
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
	CkArrayRec *rec=elementNrec(idx);
	if (rec==NULL) abort_out_of_bounds(idx);
	return rec;
}


CkArrayRec *CkArray::elementNrec(const CkArrayIndex &idx)
{
#if CMK_TEMPLATE_MEMBERS_BROKEN
	return hash.get(*(CkArrayIndexMax *)&idx); //Slower version
#else
		return hash.template get_fast<
			CkArrayIndex_hashN_fast,CkArrayIndex_compareN_fast 
		  >(*(CkArrayIndexMax *)&idx);
#endif
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

//Housecleaning: called periodically from node zero--
// Remove old broadcasts.
// Look for, and remove, old ArrayRec's.
inline void CkArray::SpringCleaning(void)
{
  static double lastCleaning=0;
  double thisCleaning=0;
  DEBK((AA"Starting spring cleaning #%d (%.0f s since last)\n"AB,nSprings,(thisCleaning=CmiWallTimer())-lastCleaning));
  lastCleaning=thisCleaning;
  nSprings++;
  //Remove old broadcast messages
  int nDelete=oldBcasts.length()-(bcastNo-oldBcastNo);
  if (nDelete>0) {
    DEBK((AA"Cleaning out %d old broadcasts\n"AB,nDelete));
    for (int i=0;i<nDelete;i++)
      CkFreeMsg((void *)oldBcasts.deq());
  }
  oldBcastNo=bcastNo;
  
  //Poke through the hash table for old ArrayRecs.
  void *objp;
  void *keyp;
  CkHashtableIterator *it=hash.iterator();
  while (NULL!=(objp=it->next(&keyp))) {
    CkArrayRec *rec=*(CkArrayRec **)objp;
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

#define SECONDS_Btw_SPRINGS 60

static void springCheck(void *forArray) {
	((CkArray *)forArray)->SpringCleaning();
	CcdCallFnAfter((CcdVoidFn)springCheck,forArray,SECONDS_Btw_SPRINGS*1000);
}

/********************* Little CkArray Utilities ******************/

void CProxy_CkArrayBase::setReductionClient(CkReductionMgr::clientFn fn,void *param)
{ ckLocalBranch()->setClient(fn,param); }

//Fetch a local element via its index (return NULL if not local)
ArrayElement *CkArray::getElement(const CkArrayIndex &idx)
{
  CkArrayRec *rec=elementNrec(idx);
  if (rec!=NULL)
    return rec->element();
  else
    return NULL;
}  

//Create 1D initial array elements
void CProxy_CkArrayBase::base_insert1D(int ctorIndex,int numInitial,
	CkArrayMessage *inM)
{
  DEBC(("In createInitial-- will build %d elements\n",numInitial));
  if (numInitial>0) {
    //Build some 1D elements: (mostly for backward compatability)
    for (int i=0;i<numInitial;i++) {
      CkArrayMessage *m=NULL;
      if (inM!=NULL) { 
	if (i!=numInitial-1)
	  m=(CkArrayMessage *)CkCopyMsg((void **)&inM);
	else
	  m=inM;//Last time around, send off the original message
      }
      base_insert(ctorIndex,-1,CkArrayIndex1D(i),m);
    }
    DEBC(("Done building elements\n"));
    doneInserting();
  }
}

void CProxy_CkArrayBase::base_insert(int ctorIndex,int onPE,CkArrayMessage *m)
{
  if (_idx.nInts==-1) CkAbort("Must use arrayProxy[index].insert()!\n"); 
  base_insert(ctorIndex,onPE,_idx,m);
}

//pack-unpack method for CProxy_CkArrayBase
void CProxy_CkArrayBase::pup(PUP::er &p)
{
  CkArrayID::pup(p);
  
  p(_aid);
  p(_idx.nInts);
  if (_idx.nInts!=-1)
  	p(_idx.data(),_idx.nInts);
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
};

//static method
CkGroupID CkArray::CreateArray(CkGroupID mapID,int numInitial)
{
  CkGroupID group;

  CkArrayCreateMsg *msg = new CkArrayCreateMsg;
  msg->numInitial=numInitial;
  msg->mapID = mapID;
#if CMK_LBDB_ON
  msg->loadbalancer = lbdb;
#endif
  group = CProxy_CkArray::ckNew(msg);
  return group;
}


CkArray::CkArray(CkMigrateMessage *) :CkReductionMgr(),
	hash(17,0.75,CkArrayIndex_hashN,CkArrayIndex_compareN)
{
  if (CkMyPe()==0) CcdCallFnAfter((CcdVoidFn)springCheck,
  	(void *)&thisgroup,SECONDS_Btw_SPRINGS*1000);
}

CkArray::CkArray(CkArrayCreateMsg *msg) :CkReductionMgr(),
	hash(17,0.75,CkArrayIndex_hashN,CkArrayIndex_compareN)
{
  //Set class variables
  numInitial=msg->numInitial;
  num.local=num.migrating=0;
  bcastNo=oldBcastNo=0;
  nSprings=0;
  isInserting=CmiTrue;
  CcdCallFnAfter((CcdVoidFn)springCheck,(void *)this,SECONDS_Btw_SPRINGS*1000);

#if CMK_LBDB_ON
  initLB(CProxy_LBDatabase(msg->loadbalancer).ckLocalBranch());
#endif
 
  mapID=msg->mapID;map=NULL;
  delete msg;
  
  if (NULL!=_localBranch(mapID))
    initAfterMap();
  else //Wait for the map to be created 
    CProxy_CkArrayMap(mapID).callMeBack(
					new CkGroupInitCallbackMsg
					(
					 static_initAfterMap,
					 (void *)this
					 )
					);
}
  
void CkArray::static_initAfterMap(void *dis)
 {((CkArray *)dis)->initAfterMap();}

void CkArray::initAfterMap(void)
{
  //The map is alive-- register with it
  map=(CkArrayMap *)_localBranch(mapID);
  if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
  CkArrayMapRegisterMessage *mapMsg = new CkArrayMapRegisterMessage;
  mapMsg->numElements = numInitial;
  mapMsg->array = this;
  mapHandle=map->registerArray(mapMsg);
  
  //Don't start reduction until all elements have been inserted.
  CkReductionMgr::creatingContributors();
}

//Allocate a new, uninitialized array element of the given (chare) type
// and owning the given index.
ArrayElement *CkArray::newElement(int chareType,const CkArrayIndex &ind)
{
  ArrayElement *el=(ArrayElement *)malloc(_chareTable[chareType]->size);
  el->thisindex=ind;
  el->thisArray=this;
  el->thisArrayID=thisgroup;
  el->thisChareType=chareType;
  return el;
}

//Call the user's given constructor, passing the given message.
void CkArray::ctorElement(ArrayElement *el,int ctor,void *msg)
{
  curElementIsDead=CmiFalse;
  
  //Call the user's constructor
  void *tmpobj = CpvAccess(_currentChare);
  CpvAccess(_currentChare) = (void*) el;
  _entryTable[ctor]->call(msg, (void *)el);
  CpvAccess(_currentChare) = tmpobj;
}


//This method is called by the user to add an element.
void CkArray::InsertElement(CkArrayMessage *m)
{
  const CkArrayIndex &idx=m->array_index();
  int homePe=homePE(idx);
  //Create the element on this PE
  DEBC((AA"  adding local element %s\n"AB,idx2str(idx)));
  if (homePe!=CkMyPe()) {
    //Let this element's home PE know it lives here now
    DEBC((AA"  Telling %s's home %d that it lives here.\n"AB,idx2str(idx),homePe));
    thisproxy.UpdateLocation(new CkArrayUpdateMsg(idx),homePe);
  }
  //Build the element
  ArrayElement *el=newElement(_entryTable[m->array_ep()]->chareIdx,idx);
#if CMK_LBDB_ON //Load balancer utilities:
  el->usesAtSync=CmiFalse;
#endif
  el->bcastNo=bcastNo;
  contributorCreated(&el->reductionInfo);
  
  //Call the element's constructor (keeps message m)
  ctorElement(el,m->array_ep(),(void *)m);
  
  if (!curElementIsDead) { 
    //<- element may have immediately migrated away or died.
    el->lbRegister();// Register the object with the load balancer
    insertRec(new CkArrayRec_local(this,el),el->thisindex);
  }
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
  if (isInserting) {
    isInserting=CmiFalse;
    DEBC((AA"Done inserting objects\n"AB));
#if CMK_LBDB_ON
    the_lbdb->DoneRegisteringObjects(myLBHandle);
#endif
    CkReductionMgr::doneCreatingContributors();
    if (bcastNo>0) {
      //These broadcasts were delayed until all elements were inserted--
      DEBC((AA"  Delivering insert-delayed broadcasts\n"AB));
      for (int i=0;i<bcastNo;i++)
	deliverBroadcast(oldBcasts[i]);
    }
  }
}

/*
  Called from element destructor-- removes
  element from hash table, notifies family, etc.
*/

void CkArray::localElementDying(ArrayElement *el)
{
  const CkArrayIndex &idx=el->thisindex;
  curElementIsDead=CmiTrue;
  CkArrayRec *rec=elementNrec(idx);
  if (rec!=NULL) {
    CkArrayRec::RecType rtype=rec->type();
    if (rtype==CkArrayRec::remote) {
      //This is just the old copy of a migrator-- ignore him
      DEBC((AA"Ignoring death of migrating element %s\n"AB,idx2str(idx)));
      return;
    }
    else if (rtype==CkArrayRec::local) {
      //This is a local element dying a natural death
      //Detach him from his arrayRec (prevents double-delete)
      ((CkArrayRec_local *)rec)->releaseElement();
      //Forward a death notice to his home
      int home=homePE(idx);
      if (home!=CkMyPe())
        thisproxy.ElementDying(new CkArrayRemoveMsg(idx),home);
    }
    hash.remove(*(CkArrayIndexMax *)&idx);
    delete rec;
  /*	//Install a zombie to keep the living from re-using this index.
	insertRec(new CkArrayRec_dead(this),idx); */
  }
}

void CkArray::ElementDying(CkArrayRemoveMsg *m)
{
  //Remove this (non-local) element from our hashtable
  const CkArrayIndex &idx=m->array_index();
  delete elementNrec(idx);
  delete m;
}


/************************ Migration *********************/

void CkArray::migrateMe(ArrayElement *el, int where)
{
  const CkArrayIndex &idx=el->thisindex;
  DEBM((AA"Migrating element %s to %d\n"AB,idx2str(idx),where));
  
  //Pack the element and send it off
  int bufSize;
  { PUP::sizer p; el->pup(p); bufSize=p.size(); }
  CkArrayElementMigrateMessage *msg = 
    new (&bufSize, 0) CkArrayElementMigrateMessage;
  msg->array_ep()=_chareTable[el->thisChareType]->getMigCtor();
  { PUP::toMem p(msg->packData); p.becomeDeleting(); el->pup(p);}
  msg->array_index()=idx;
  DEBM((AA"Migrated index size %s\n"AB,idx2str(msg->array_index())));
  contributorLeaving(&el->reductionInfo);
  
  thisproxy.RecvMigratedElement(msg,where);
  
  //Switch this element's CkArrayRec to remote--
  insertRec(new CkArrayRec_remote(this,where),idx);
  curElementIsDead=true;
}

void CkArray::RecvMigratedElement(CkArrayElementMigrateMessage *msg)
{
  const CkArrayIndex &idx=msg->array_index();
  {
    int srcPE=msg->array_getSrcPe();
    DEBM((AA"Recv'd migrating element %s from %d\n"AB,idx2str(idx),srcPE));
    
    //Send update to home to let him know about the migration
    int home=homePE(idx);
    if (home!=srcPE && home!=CkMyPe())
      thisproxy.UpdateLocation(new CkArrayUpdateMsg(idx),home);
  }

  //Create the new element on this PE (passing on the msg)
  ArrayElement *el=newElement(_entryTable[msg->array_ep()]->chareIdx,idx);
  ctorElement(el,msg->array_ep(),(CkMigrateMessage *)NULL);
  
  if (curElementIsDead) {CkFreeMsg((void *)msg);return;}
  
  //Unpack the element's fields (must be *after* ctor since this is a virtual function)
  el->bcastNo=-1;//<- to make sure they call their superclass's pup
  { PUP::fromMem p(msg->packData); el->pup(p); }
  if (curElementIsDead) {CkFreeMsg((void *)msg);return;}
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
    insertRec(new CkArrayRec_local(this,el),el->thisindex);
}

/*
  This is called when a message is received with a hopcount
  greater than one-- it tells us to direct messages for the given
  array element straight to the given PE.
*/
void CkArray::UpdateLocation(CkArrayUpdateMsg *msg)
{
  const CkArrayIndex &idx=msg->array_index();
  int onPE=msg->array_getSrcPe();
  DEBM((AA"Recv'd location update for %s from %d\n"AB,idx2str(idx),onPE));
  if (onPE!=CkMyPe()) {
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

inline void CkArray::deliverLocal(CkArrayMessage *msg,ArrayElement *el)
{//This is a local element-- deliver a message to him
	DEBS((AA"Delivering local message for element %s\n"AB,idx2str(el)));
	int hopCount=msg->array_hops();
	if (hopCount>1)
	{//This message took several hops to reach us.
		int srcPE=msg->array_getSrcPe();
		if (srcPE==CkMyPe())
			DEB((AA"Odd routing: local element %s is %d hops away!\n"AB,idx2str(el),hopCount));
		else
		{//Send a routing message letting original sender know new element location
			DEBS((AA"Sending update back to %d for element %s\n"AB,srcPE,idx2str(el)));
			thisproxy.UpdateLocation(new CkArrayUpdateMsg(el->thisindex),srcPE);
		}
	}
	curElementIsDead=CmiFalse;
	
	int entry=msg->array_ep();
#if CMK_LBDB_ON
	the_lbdb->ObjectStart(el->ldHandle);
	_entryTable[entry]->call(msg, el);
	if (!curElementIsDead)
		 the_lbdb->ObjectStop(el->ldHandle);
#else
	_entryTable[entry]->call(msg, el);
#endif
}
void CkArrayRec_local::recv(CkArrayMessage *msg) {
    arr->deliverLocal(msg,el);
}

inline void CkArray::deliverRemote(CkArrayMessage *msg,int onPE)
{//This element is on another PE-- forward message there
	DEBS((AA"Forwarding message for %s to %d\n"AB,idx2str(msg->array_index()),onPE));
	msg->array_hops()++;
	thisproxy.RecvForElement(msg, onPE);
}


inline void CkArray::deliverUnknown(CkArrayMessage *msg)
{//This index is not hashed-- send to its "home" processor
	const CkArrayIndex &idx=msg->array_index();
	int onPE=homePE(idx);
	if (onPE!=CkMyPe())
		deliverRemote(msg,onPE);
	else
	{// We *are* the home processor-- this element will be created soon
		DEBC((AA"Adding buffer for unknown element %s\n"AB,idx2str(idx)));
		CkArrayRec *rec=new CkArrayRec_buffering(this);
		insertRec(rec,idx);
		rec->send(msg);
	}
}


//Deliver given (pre-addressed) message 
inline void CkArray::Send(CkArrayMessage *msg)
{
	const CkArrayIndex &idx=msg->array_index();
#if CMK_LBDB_ON
	the_lbdb->Send(myLBHandle,idx2LDObjid(idx),UsrToEnv(msg)->getTotalsize());
#endif
	CkArrayRec *rec=elementNrec(idx);
	if (rec!=NULL)
		rec->send(msg);
	else deliverUnknown(msg);
}

//This receives a message from the net destined for a 
// (probably) local element.  
inline void CkArray::RecvForElement(CkArrayMessage *msg)
{
	const CkArrayIndex &idx=msg->array_index();
	DEBS((AA"RecvForElement %s\n"AB,idx2str(idx)));
	CkArrayRec *rec=elementNrec(idx);
	if (rec!=NULL)
		rec->recv(msg);
	else deliverUnknown(msg);
}

void CProxy_CkArrayBase::base_insert(int ctorIndex,int onPE,
	const CkArrayIndex &idx,CkArrayMessage *m)
{
  if (m==NULL) m=new CkArrayElementCreateMsg;
  m->array_index()=idx;
  m->array_ep()=ctorIndex;
  if (onPE==-1) onPE=ckLocalBranch()->homePE(idx);
  
  DEBC((AA"Proxy inserting element %s on PE %d\n"AB,idx2str(idx),onPE));
  CProxy_CkArray(_aid).InsertElement(m,onPE);
}

void CProxy_CkArrayBase::base_send(CkArrayMessage *msg, int entryIndex) const
{
	msg->array_setSrcPe();
	msg->array_ep() = entryIndex;
	msg->array_hops() = 0;
#ifndef CMK_OPTIMIZE
	if (_idx.nInts<0) CkAbort("Array index length is negative!\n");
	if (_idx.nInts>CK_ARRAYINDEX_MAXLEN)
		CkAbort("Array index length (nInts) is too long-- did you "
			"use bytes instead of integers?\n");
#endif
	msg->array_index()=_idx;//Insert array index into message
	ckLocalBranch()->Send(msg);
}


/*********************** CkArray Broadcast ******************/

void CProxy_CkArrayBase::base_broadcast(CkArrayMessage *msg, int entryIndex) const
{
	msg->array_setSrcPe();
	msg->array_ep()=entryIndex;
	msg->array_hops()=0;
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
}

//Deliver a copy of the given broadcast to all local elements
void CkArray::deliverBroadcast(CkArrayMessage *bcast)
{
	//Poke through the hash table for local elements.
	void **objp;
	CkHashtableIterator *it=hash.iterator();
	while (NULL!=(objp=(void **)it->next()))
	{
		CkArrayRec *rec=*(CkArrayRec **)objp;
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

void CkArray::initLB(LBDatabase *Nlbdb)
{
  DEBL((AA"Connecting to load balancer...\n"AB));
  //Find and register with the load balancer
  the_lbdb = Nlbdb;
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
}


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

CkArrayUpdateMsg::CkArrayUpdateMsg(const CkArrayIndex &idx) 
{
	array_index()=idx;
	array_setSrcPe();
}

void *
CkArrayElementMigrateMessage::alloc(int msgnum,int size,int *array,int priobits)
{
  int totalsize;
  totalsize = size + array[0] + 8;
  CkArrayElementMigrateMessage *newMsg = (CkArrayElementMigrateMessage *)
    CkAllocMsg(msgnum,totalsize,priobits);
  DEBM((AA"  Allocated varsize message %d, %d bytes at %p\n"AB,msgnum,totalsize,newMsg));
  newMsg->packData = (char *)newMsg + ALIGN8(size);
  return (void *) newMsg;
}

void *
CkArrayElementMigrateMessage::pack(CkArrayElementMigrateMessage* in)
{
  in->packData = (void*)((char*)in->packData-(char *)&(in->packData));
  return (void*) in;
}

CkArrayElementMigrateMessage* 
CkArrayElementMigrateMessage::unpack(void *in)
{
  CkArrayElementMigrateMessage *me = new (in) CkArrayElementMigrateMessage;
  me->packData = (char *)&(me->packData) + (size_t)me->packData;
  return me;
}

/********************* Checkpointing **********************/

//static
void CkArray::pupArrayMsgQ(CkQ<CkArrayMessage *> &q, PUP::er &p)
{
  CkArrayMessage *msg;
  envelope *env;
  int nMsgs, size;
  
  if (p.isPacking()) nMsgs = q.length();
  p(nMsgs);
  for(int i = 0; i < nMsgs; i++) {
    if (p.isPacking()) {
      msg = q.deq();
      env = UsrToEnv(msg);
      _packFn((void **)&env);
      size = env->getTotalsize();
    }
    p(size);
    if (p.isUnpacking()) env = (envelope *) CmiAlloc(size);
    p((void *) env, size);
    _unpackFn((void **)&env);
    q.enq((CkArrayMessage *)EnvToUsr(env));
  }
}

CkArrayRec* CkArray::pupArrayRec(PUP::er &p, CkArrayRec *rec, CkArrayIndex *idx)
{
  CkArrayRec::RecType type;
  char    ch, *rtypes;
  ArrayElement *el;
  CkArrayRec_local *lrec;
  
  rtypes = (char *) "0lrbmd";
  
  if (p.isPacking()) {
    type = rec->type();
    ch = rtypes[type];
  }
  p(ch);
  switch(ch) {
  case 'l':
    if (p.isUnpacking()) {
      int elType;
      lrec = new CkArrayRec_local();
      p(elType);
	  el = newElement(elType, *idx);
	  ctorElement(el, _chareTable[elType]->getMigCtor(), 0);
      rec = lrec;
    }
    else {
      el = rec->element();
      p(el->thisChareType);
    }
    rec->pup(p);
    el->pup(p);
    if (p.isUnpacking()) {
	  el->lbRegister();
      lrec->el = el;
	}
  case 'r':
    if (p.isUnpacking()) 
      rec = new CkArrayRec_remote();
    rec->pup(p);
  case 'b':
    if (p.isUnpacking())
      rec = new CkArrayRec_buffering();
    rec->pup(p);
  case 'd':
    rec = new CkArrayRec_dead();
    rec->pup(p);
  }
  if (p.isUnpacking()) rec->arr = this;
  
  return rec;
}

void CkArray::pupHashTable(PUP::er &p)
{
/***** FIXME
  CkHashtableIterator *it=NULL;
  CkArrayIndex *idx;
  HashKey *key;
  CkArrayRec *rec;
  int nElem, len;
  unsigned char *data;
  
  if (p.isPacking()) 
    nElem = hash.numElements();
  
  p(nElem);
  if (p.isPacking()) {
    it = hash.iterator();
    it->seekStart();
  }
  for(int i=0; i < nElem; i++) {
    if (p.isPacking()) {
      rec = (CkArrayRec *) it->next(&key);
      data = (unsigned char *) key->getKey(len);
    }
    p(len);
    if (p.isUnpacking()) 
	  data = new unsigned char[len];
    p((void *) data, len);
    if (p.isUnpacking()) {
      idx = CkArrayIndex::newIndex(len, data);
	  key = idx;
      delete [] data;
    }
	else idx = NULL;
    rec = pupArrayRec(p, rec, idx);
    if (p.isUnpacking()) 
      hash.put(*key, rec);
  }
*/
}

void CkArray::pup(PUP::er &p) 
{
  CkReductionMgr::pup(p);
  
#if CMK_LBDB_ON
  if (p.isUnpacking()) {
    initLB(CProxy_LBDatabase(lbdb).ckLocalBranch());
  }
#endif
  p(numInitial);
  p(curElementIsDead);
  p(num.local);
  p(num.migrating);
  CmiBool oldIsInserting;
  p(nSprings);
  if (p.isUnpacking())
	isInserting = CmiTrue;
  else
	oldIsInserting = isInserting;
  p(oldIsInserting);
  p(bcastNo);
  p(oldBcastNo);
  p(mapID);
  p(mapHandle);
  map = (CkArrayMap *) _localBranch(mapID);
  pupArrayMsgQ(oldBcasts, p);
  pupHashTable(p);
}

#include "CkArray.def.h"




