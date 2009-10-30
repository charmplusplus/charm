
/*
Charm++ support for fault tolerance of
In memory synchronous checkpointing and restart

written by Gengbin Zheng, gzheng@uiuc.edu
           Lixia Shi,     lixiashi@uiuc.edu

added 12/18/03:

To support fault tolerance while allowing migration, it uses double
checkpointing scheme for each array element (not a infallible scheme).
In this version, checkpointing is done based on array elements. 
Each array element individully sends its checkpoint data to two buddies.

In this implementation, assume only one failure happens at a time,
or two failures on two processors which are not buddy to each other;
also assume there is no failure during a checkpointing or restarting phase.

Restart phase contains two steps:
1. Converse level restart: the newly created process for the failed
   processor recover its system data (no array elements) from 
   its backup processor.
2. Charm++ level restart: CkMemCheckPT gets control and recover array 
   elements and reset all states of system groups to be consistent.

added 3/14/04:
1. also support for double in-disk checkpoint/restart
   set "where" to CkCheckPoint_inDISK/CkCheckPoint_inMEM in init()

added 4/16/04:
1. also support the case when there is a pool of extra processors.
   set CK_NO_PROC_POOL to 0.

TODO:
1. checkpoint scheme can be reimplemented based on per processor scheme;
 restart phase should restore/reset group table, etc on all processors, thus flushStates() can be eliminated.
2. crash at checkpointing phase currently is catastrophic, can be fixed by storing another transient checkpoints.

*/

#include "unistd.h"

#include "charm++.h"
#include "ck.h"
#include "register.h"
#include "conv-ccs.h"
#include <signal.h>

// pick buddy processor from a different physical node
#define NODE_CHECKPOINT                        1

// assume NO extra processors--1
// assume extra processors--0
#define CK_NO_PROC_POOL				1

#define DEBUGF      // CkPrintf

// static, so that it is accessible from Converse part
int CkMemCheckPT::inRestarting = 0;
double CkMemCheckPT::startTime;
char *CkMemCheckPT::stage;
CkCallback CkMemCheckPT::cpCallback;

int _memChkptOn = 1;			// checkpoint is on or off

CkGroupID ckCheckPTGroupID;		// readonly


/// @todo the following declarations should be moved into a separate file for all 
// fault tolerant strategies

#ifdef CMK_MEM_CHECKPOINT
// name of the kill file that contains processes to be killed 
char *killFile;                                               
// flag for the kill file         
int killFlag=0;
// variable for storing the killing time
double killTime=0.0;
#endif

/// checkpoint buffer for processor system data, remove static to make icpc 10.1 pass with -O
CpvDeclare(CkProcCheckPTMessage*, procChkptBuf);

// compute the backup processor
// FIXME: avoid crashed processors
inline int ChkptOnPe() { return (CkMyPe()+1)%CkNumPes(); }

inline int CkMemCheckPT::BuddyPE(int pe)
{
  int budpe;
#if NODE_CHECKPOINT
    // buddy is the processor with same rank on the next physical node
  int r1 = CmiPhysicalRank(pe);
  int budnode = CmiPhysicalNodeID(pe);
  do {
    budnode = (budnode+1)%CmiNumPhysicalNodes();
    int *pelist;
    int num;
    CmiGetPesOnPhysicalNode(budnode, &pelist, &num);
    budpe = pelist[r1 % num];
  } while (isFailed(budpe));
  if (budpe == pe) {
    CmiPrintf("[%d] Error: failed to find a buddy processor on a different node.\n", pe);
    CmiAbort("Failed to find a buddy processor");
  }
#else
  budpe = pe;
  while (budpe == pe || isFailed(budPe)) 
          budPe = (budPe+1)%CkNumPes();
#endif
  return budpe;
}

// called in array element constructor
// choose and register with 2 buddies for checkpoiting 
#if CMK_MEM_CHECKPOINT
void ArrayElement::init_checkpt() {
	if (_memChkptOn == 0) return;
	// only master init checkpoint
        if (thisArray->getLocMgr()->firstManager->mgr!=thisArray) return;

        budPEs[0] = CkMyPe();
        budPEs[1] = CProxy_CkMemCheckPT(ckCheckPTGroupID).ckLocalBranch()->BuddyPE(CkMyPe());
	CmiAssert(budPEs[0] != budPEs[1]);
        // inform checkPTMgr
        CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
	//CmiPrintf("[%d] ArrayElement::init_checkpt array %d %p pe: %d %d\n", CkMyPe(), ((CkGroupID)thisArrayID).idx, this, budPEs[0], budPEs[1]);
        checkptMgr[budPEs[0]].createEntry(thisArrayID, thisArray->getLocMgr()->getGroupID(), thisIndexMax, budPEs[1]);        
	checkptMgr[budPEs[1]].createEntry(thisArrayID, thisArray->getLocMgr()->getGroupID(), thisIndexMax, budPEs[0]);
}
#endif

// entry function invoked by checkpoint mgr asking for checkpoint data
void ArrayElement::inmem_checkpoint(CkArrayCheckPTReqMessage *m) {
#if CMK_MEM_CHECKPOINT
  //DEBUGF("[p%d] HERE checkpoint to %d %d \n", CkMyPe(), budPEs[0], budPEs[1]);
  CkLocMgr *locMgr = thisArray->getLocMgr();
  CmiAssert(myRec!=NULL);
  int size;
  {
        PUP::sizer p;
        locMgr->pupElementsFor (p, myRec, CkElementCreation_migrate);
        size = p.size();
  }
  int packSize = size/sizeof(double) +1;
  CkArrayCheckPTMessage *msg =
                 new (packSize, 0) CkArrayCheckPTMessage;
  msg->len = size;
  msg->index =thisIndexMax;
  msg->aid = thisArrayID;
  msg->locMgr = locMgr->getGroupID();
  msg->cp_flag = 1;
  {
        PUP::toMem p(msg->packData);
        locMgr->pupElementsFor (p, myRec, CkElementCreation_migrate);
  }

  CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
  checkptMgr.recvData(msg, 2, budPEs);
  delete m;
#endif
}

// checkpoint holder class - for memory checkpointing
class CkMemCheckPTInfo: public CkCheckPTInfo
{
  CkArrayCheckPTMessage *ckBuffer;
public:
  CkMemCheckPTInfo(CkArrayID a, CkGroupID loc, CkArrayIndexMax idx, int pno): 
	            CkCheckPTInfo(a, loc, idx, pno)
  {
    ckBuffer = NULL;
  }
  ~CkMemCheckPTInfo() 
  {
    if (ckBuffer) delete ckBuffer; 
  }
  inline void updateBuffer(CkArrayCheckPTMessage *data) 
  {
    CmiAssert(data!=NULL);
    if (ckBuffer) delete ckBuffer;
    ckBuffer = data;
  }    
  inline CkArrayCheckPTMessage * getCopy()
  {
    if (ckBuffer == NULL) {
      CmiPrintf("[%d] recoverArrayElements: element does not have checkpoint data.", CkMyPe());
      CmiAbort("Abort!");
    }
    return (CkArrayCheckPTMessage *)CkCopyMsg((void **)&ckBuffer);
  }     
  inline void updateBuddy(int b1, int b2) {
     CmiAssert(ckBuffer);
     ckBuffer->bud1 = b1; ckBuffer->bud2 = b2;
  }
  inline int getSize() { 
     CmiAssert(ckBuffer);
     return ckBuffer->len; 
  }
};

// checkpoint holder class - for in-disk checkpointing
class CkDiskCheckPTInfo: public CkCheckPTInfo 
{
  char *fname;
  int bud1, bud2;
  int len; 			// checkpoint size
public:
  CkDiskCheckPTInfo(CkArrayID a, CkGroupID loc, CkArrayIndexMax idx, int pno, int myidx): CkCheckPTInfo(a, loc, idx, pno)
  {
#if CMK_USE_MKSTEMP
    fname = new char[64];
    sprintf(fname, "/tmp/ckpt%d-%d-XXXXXX", CkMyPe(), myidx);
    mkstemp(fname);
#else
    fname=tmpnam(NULL);
#endif
    bud1 = bud2 = -1;
    len = 0;
  }
  ~CkDiskCheckPTInfo() 
  {
    remove(fname);
  }
  inline void updateBuffer(CkArrayCheckPTMessage *data) 
  {
    double t = CmiWallTimer();
    // unpack it
    envelope *env = UsrToEnv(data);
    CkUnpackMessage(&env);
    data = (CkArrayCheckPTMessage *)EnvToUsr(env);
    FILE *f = fopen(fname,"wb");
    PUP::toDisk p(f);
    CkPupMessage(p, (void **)&data);
    // delay sync to the end because otherwise the messages are blocked
//    fsync(fileno(f));
    fclose(f);
    bud1 = data->bud1;
    bud2 = data->bud2;
    len = data->len;
    delete data;
    //CmiPrintf("[%d] updateBuffer took %f seconds. \n", CkMyPe(), CmiWallTimer()-t);
  }
  inline CkArrayCheckPTMessage * getCopy()	// get a copy of checkpoint
  {
    CkArrayCheckPTMessage *data;
    FILE *f = fopen(fname,"rb");
    PUP::fromDisk p(f);
    CkPupMessage(p, (void **)&data);
    fclose(f);
    data->bud1 = bud1;				// update the buddies
    data->bud2 = bud2;
    return data;
  }
  inline void updateBuddy(int b1, int b2) {
     bud1 = b1; bud2 = b2;
  }
  inline int getSize() { 
     return len; 
  }
};

CkMemCheckPT::CkMemCheckPT(int w)
{
#if CK_NO_PROC_POOL
  if (CkNumPes() <= 2) {
#else
  if (CkNumPes()  == 1) {
#endif
    if (CkMyPe() == 0)  CkPrintf("Warning: CkMemCheckPT disabled!\n");
    _memChkptOn = 0;
  }
  inRestarting = 0;
  recvCount = peCount = 0;
  where = w;
}

CkMemCheckPT::~CkMemCheckPT()
{
  int len = ckTable.length();
  for (int i=0; i<len; i++) {
    delete ckTable[i];
  }
}

void CkMemCheckPT::pup(PUP::er& p) 
{ 
  CBase_CkMemCheckPT::pup(p); 
  p|cpStarter;
  p|thisFailedPe;
  p|failedPes;
  p|ckCheckPTGroupID;		// recover global variable
  p|cpCallback;			// store callback
  p|where;			// where to checkpoint
  if (p.isUnpacking()) {
    recvCount = peCount = 0;
  }
}

// called by checkpoint mgr to restore an array element
void CkMemCheckPT::inmem_restore(CkArrayCheckPTMessage *m) 
{
#if CMK_MEM_CHECKPOINT
  DEBUGF("[%d] inmem_restore restore: mgr: %d \n", CmiMyPe(), m->locMgr);  
  // m->index.print();
  PUP::fromMem p(m->packData);
  CkLocMgr *mgr = CProxy_CkLocMgr(m->locMgr).ckLocalBranch();
  CmiAssert(mgr);
  mgr->resume(m->index, p);

  // find a list of array elements bound together
  ArrayElement *elt = (ArrayElement *)mgr->lookup(m->index, m->aid);
  CmiAssert(elt);
  CkLocRec_local *rec = elt->myRec;
  CkVec<CkMigratable *> list;
  mgr->migratableList(rec, list);
  CmiAssert(list.length() > 0);
  for (int l=0; l<list.length(); l++) {
    elt = (ArrayElement *)list[l];
    elt->budPEs[0] = m->bud1;
    elt->budPEs[1] = m->bud2;
    //    reset, may not needed now
    // for now.
    for (int i=0; i<CK_ARRAYLISTENER_MAXLEN; i++) {
      contributorInfo *c=(contributorInfo *)&elt->listenerData[i];
      if (c) c->redNo = 0;
    }
  }
#endif
}

// return 1 if pe is a crashed processor
int CkMemCheckPT::isFailed(int pe)
{
  for (int i=0; i<failedPes.length(); i++)
    if (failedPes[i] == pe) return 1;
  return 0;
}

// add pe into history list of all failed processors
void CkMemCheckPT::failed(int pe)
{
  if (isFailed(pe)) return;
  failedPes.push_back(pe);
}

int CkMemCheckPT::totalFailed()
{
  return failedPes.length();
}

// create an checkpoint entry for array element of aid with index.
void CkMemCheckPT::createEntry(CkArrayID aid, CkGroupID loc, CkArrayIndexMax index, int buddy)
{
  // error check, no duplicate
  int idx, len = ckTable.size();
  for (idx=0; idx<len; idx++) {
    CkCheckPTInfo *entry = ckTable[idx];
    if (index == entry->index) {
      if (loc == entry->locMgr) {
	  // bindTo array elements
          return;
      }
        // for array inheritance, the following check may fail
        // because ArrayElement constructor of all superclasses are called
      if (aid == entry->aid) {
        CkPrintf("[%d] CkMemCheckPT::createEntry a duplciated entry for arrayID %d:", CkMyPe(), ((CkGroupID)aid).idx); index.print(); CkPrintf("\n");
        CmiAbort("CkMemCheckPT::createEntry a duplciated entry");
      }
    }
  }
  CkCheckPTInfo *newEntry;
  if (where == CkCheckPoint_inMEM)
    newEntry = new CkMemCheckPTInfo(aid, loc, index, buddy);
  else
    newEntry = new CkDiskCheckPTInfo(aid, loc, index, buddy, len+1);
  ckTable.push_back(newEntry);
  //CkPrintf("[%d] CkMemCheckPT::createEntry for arrayID %d:", CkMyPe(), ((CkGroupID)aid).idx); index.print(); CkPrintf("\n");
}

// loop through my checkpoint table and ask checkpointed array elements
// to send me checkpoint data.
void CkMemCheckPT::doItNow(int starter, CkCallback &cb)
{
  cpCallback = cb;
  cpStarter = starter;
  if (CkMyPe() == cpStarter) {
    startTime = CmiWallTimer();
    CkPrintf("[%d] Start checkpointing  starter: %d... \n", CkMyPe(), cpStarter);
  }

  int len = ckTable.length();
  for (int i=0; i<len; i++) {
    CkCheckPTInfo *entry = ckTable[i];
      // always let the bigger number processor send request
    if (CkMyPe() < entry->pNo) continue;
      // call inmem_checkpoint to the array element, ask it to send
      // back checkpoint data via recvData().
    CkArrayCheckPTReqMessage *msg = new CkArrayCheckPTReqMessage;
    CkSendMsgArray(CkIndex_ArrayElement::inmem_checkpoint(NULL),(CkArrayMessage *)msg,entry->aid,entry->index);
  }
    // if my table is empty, then I am done
  if (len == 0) thisProxy[cpStarter].cpFinish();

  // pack and send proc level data
  sendProcData();
}

// don't handle array elements
static inline void _handleProcData(PUP::er &p)
{
    // save readonlys, and callback BTW
    CkPupROData(p);

    // save mainchares into MainChares.dat
    if(CkMyPe()==0) CkPupMainChareData(p, (CkArgMsg*)NULL);
	
#if CMK_FT_CHARE
    // save non-migratable chare
    CkPupChareData(p);
#endif

    // save groups into Groups.dat
    CkPupGroupData(p);

    // save nodegroups into NodeGroups.dat
    if(CkMyRank()==0) CkPupNodeGroupData(p);
}

void CkMemCheckPT::sendProcData()
{
  // find out size of buffer
  int size;
  {
    PUP::sizer p;
    _handleProcData(p);
    size = p.size();
  }
  int packSize = size;
  CkProcCheckPTMessage *msg = new (packSize, 0) CkProcCheckPTMessage;
  DEBUGF("[%d] CkMemCheckPT::sendProcData - size: %d\n", CkMyPe(), size);
  {
    PUP::toMem p(msg->packData);
    _handleProcData(p);
  }
  msg->pe = CkMyPe();
  msg->len = size;
  msg->reportPe = cpStarter;  //in case other processor isn't in checkpoint mode
  thisProxy[ChkptOnPe()].recvProcData(msg);
}

void CkMemCheckPT::recvProcData(CkProcCheckPTMessage *msg)
{
  if (CpvAccess(procChkptBuf)) delete CpvAccess(procChkptBuf);
  CpvAccess(procChkptBuf) = msg;
//CmiPrintf("[%d] CkMemCheckPT::recvProcData report to %d\n", CkMyPe(), msg->reportPe);
  thisProxy[msg->reportPe].cpFinish();
}

// ArrayElement call this function to give us the checkpointed data
void CkMemCheckPT::recvData(CkArrayCheckPTMessage *msg)
{
  int len = ckTable.length();
  int idx;
  for (idx=0; idx<len; idx++) {
    CkCheckPTInfo *entry = ckTable[idx];
    if (msg->locMgr == entry->locMgr && msg->index == entry->index) break;
  }
  CkAssert(idx < len);
  int isChkpting = msg->cp_flag;
  ckTable[idx]->updateBuffer(msg);
  if (isChkpting) {
      // all my array elements have returned their inmem data
      // inform starter processor that I am done.
    recvCount ++;
    if (recvCount == ckTable.length()) {
      if (where == CkCheckPoint_inMEM) {
        thisProxy[cpStarter].cpFinish();
      }
      else if (where == CkCheckPoint_inDISK) {
        // another barrier for finalize the writing using fsync
        CkCallback localcb(CkIndex_CkMemCheckPT::syncFiles(NULL),thisgroup);
        contribute(0,NULL,CkReduction::sum_int,localcb);
      }
      else
        CmiAbort("Unknown checkpoint scheme");
      recvCount = 0;
    } 
  }
}

// only used in disk checkpointing
void CkMemCheckPT::syncFiles(CkReductionMsg *m)
{
  delete m;
#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
  system("sync");
#endif
  thisProxy[cpStarter].cpFinish();
}

// only is called on cpStarter when checkpoint is done
void CkMemCheckPT::cpFinish()
{
  CmiAssert(CkMyPe() == cpStarter);
  peCount++;
    // now that all processors have finished, activate callback
  if (peCount == 2*(CkNumPes())) {
    CmiPrintf("[%d] Checkpoint finished in %f seconds, sending callback ... \n", CkMyPe(), CmiWallTimer()-startTime);
    cpCallback.send();
    peCount = 0;
    thisProxy.report();
  }
}

// for debugging, report checkpoint info
void CkMemCheckPT::report()
{
  int objsize = 0;
  int len = ckTable.length();
  for (int i=0; i<len; i++) {
    CkCheckPTInfo *entry = ckTable[i];
    CmiAssert(entry);
    objsize += entry->getSize();
  }
  CmiAssert(CpvAccess(procChkptBuf));
  CkPrintf("[%d] Checkpointed Object size: %d len: %d Processor data: %d\n", CkMyPe(), objsize, len, CpvAccess(procChkptBuf)->len);
}

/*****************************************************************************
			RESTART Procedure
*****************************************************************************/

// master processor of two buddies
inline int CkMemCheckPT::isMaster(int buddype)
{
  int mype = CkMyPe();
//CkPrintf("ismaster: %d %d\n", pe, mype);
  if (CkNumPes() - totalFailed() == 2) {
    return mype > buddype;
  }
  for (int i=1; i<CkNumPes(); i++) {
    int me = (buddype+i)%CkNumPes();
    if (isFailed(me)) continue;
    if (me == mype) return 1;
    else return 0;
  }
  return 0;
}

#ifdef CKLOCMGR_LOOP
#undef CKLOCMGR_LOOP
#endif

// loop over all CkLocMgr and do "code"
#define  CKLOCMGR_LOOP(code)	{	\
  int numGroups = CkpvAccess(_groupIDTable)->size(); 	\
  for(int i=0;i<numGroups;i++) {	\
    IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();	\
    if(obj->isLocMgr())  {	\
      CkLocMgr *mgr = (CkLocMgr*)obj;	\
      code	\
    }	\
  }	\
 }

#if 0
// helper class to pup all elements that belong to same ckLocMgr
class ElementDestoryer : public CkLocIterator {
private:
        CkLocMgr *locMgr;
public:
        ElementDestoryer(CkLocMgr* mgr_):locMgr(mgr_){};
        void addLocation(CkLocation &loc) {
		CkArrayIndexMax idx=loc.getIndex();
		CkPrintf("[%d] destroy: ", CkMyPe()); idx.print();
		loc.destroy();
        }
};
#endif

// restore the bitmap vector for LB
void CkMemCheckPT::resetLB(int diepe)
{
#if CMK_LBDB_ON
  int i;
  char *bitmap = new char[CkNumPes()];
  // set processor available bitmap
  get_avail_vector(bitmap);

  for (i=0; i<failedPes.length(); i++)
    bitmap[failedPes[i]] = 0; 
  bitmap[diepe] = 0;

#if CK_NO_PROC_POOL
  set_avail_vector(bitmap);
#endif

  // if I am the crashed pe, rebuild my failedPEs array
  if (CkMyPe() == diepe)
    for (i=0; i<CkNumPes(); i++) 
      if (bitmap[i]==0) failed(i);

  delete [] bitmap;
#endif
}

// in case when failedPe dies, everybody go through its checkpoint table:
// destory all array elements
// recover lost buddies
// reconstruct all array elements from check point data
// called on all processors
void CkMemCheckPT::restart(int diePe)
{
#if CMK_MEM_CHECKPOINT
  double curTime = CmiWallTimer();
  if (CkMyPe() == diePe)
    CkPrintf("[%d] Process data restored in %f seconds\n", CkMyPe(), curTime - startTime);
  stage = (char*)"resetLB";
  startTime = curTime;
  CkPrintf("[%d] CkMemCheckPT ----- restart.\n",CkMyPe());

#if CK_NO_PROC_POOL
  failed(diePe);	// add into the list of failed pes
#endif
  thisFailedPe = diePe;

  if (CkMyPe() == diePe) CmiAssert(ckTable.length() == 0);

  inRestarting = 1;
                                                                                
  // disable load balancer's barrier
  if (CkMyPe() != diePe) resetLB(diePe);

  CKLOCMGR_LOOP(mgr->startInserting(););

  thisProxy[0].quiescence(CkCallback(CkIndex_CkMemCheckPT::removeArrayElements(), thisProxy));
#endif
}

// loally remove all array elements
void CkMemCheckPT::removeArrayElements()
{
#if CMK_MEM_CHECKPOINT
  int len = ckTable.length();
  double curTime = CmiWallTimer();
  CkPrintf("[%d] CkMemCheckPT ----- %s len:%d in %f seconds.\n",CkMyPe(),stage,len,curTime-startTime);
  stage = (char*)"removeArrayElements";
  startTime = curTime;

  if (cpCallback.isInvalid()) CkAbort("Didn't set restart callback\n");;
  if (CkMyPe()==thisFailedPe) CmiAssert(len == 0);

  // get rid of all buffering and remote recs
  // including destorying all array elements
  CKLOCMGR_LOOP(mgr->flushAllRecs(););

//  CKLOCMGR_LOOP(ElementDestoryer chk(mgr); mgr->iterate(chk););

  thisProxy[0].quiescence(CkCallback(CkIndex_CkMemCheckPT::resetReductionMgr(), thisProxy));
#endif
}

// flush state in reduction manager
void CkMemCheckPT::resetReductionMgr()
{
  //CkPrintf("[%d] CkMemCheckPT ----- resetReductionMgr\n",CkMyPe());
  int numGroups = CkpvAccess(_groupIDTable)->size();
  for(int i=0;i<numGroups;i++) {
    CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
    IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
    obj->flushStates();
    obj->ckJustMigrated();
  }
  // reset again
  //CpvAccess(_qd)->flushStates();

  thisProxy[0].quiescence(CkCallback(CkIndex_CkMemCheckPT::recoverBuddies(), thisProxy));
}

// recover the lost buddies
void CkMemCheckPT::recoverBuddies()
{
  int idx;
  int len = ckTable.length();
  // ready to flush reduction manager
  // cannot be CkMemCheckPT::restart because destory will modify states
  double curTime = CmiWallTimer();
  CkPrintf("[%d] CkMemCheckPT ----- %s  in %f seconds\n",CkMyPe(), stage, curTime-startTime);
  stage = (char *)"recoverBuddies";
  startTime = curTime;

  // recover buddies
  for (idx=0; idx<len; idx++) {
    CkCheckPTInfo *entry = ckTable[idx];
    if (entry->pNo == thisFailedPe) {
#if CK_NO_PROC_POOL
      // find a new buddy
      int budPe = CkMyPe();
//      while (budPe == CkMyPe() || isFailed(budPe)) budPe = CrnRand()%CkNumPes();
      while (budPe == CkMyPe() || isFailed(budPe)) 
          budPe = (budPe+1)%CkNumPes();
      entry->pNo = budPe;
#else
      int budPe = thisFailedPe;
#endif
      thisProxy[budPe].createEntry(entry->aid, entry->locMgr, entry->index, CkMyPe());
      CkArrayCheckPTMessage *msg = entry->getCopy();
      msg->cp_flag = 0;            // not checkpointing
      thisProxy[budPe].recvData(msg);
    }
  }

  if (CkMyPe() == 0)
    CkStartQD(CkCallback(CkIndex_CkMemCheckPT::recoverArrayElements(), thisProxy));
}

// restore array elements
void CkMemCheckPT::recoverArrayElements()
{
  double curTime = CmiWallTimer();
  int len = ckTable.length();
  CkPrintf("[%d] CkMemCheckPT ----- %s len: %d in %f seconds \n",CkMyPe(), stage, len, curTime-startTime);
  stage = (char *)"recoverArrayElements";
  startTime = curTime;

  // recover all array elements
  int count = 0;
  for (int idx=0; idx<len; idx++)
  {
    CkCheckPTInfo *entry = ckTable[idx];
#if CK_NO_PROC_POOL
    // the bigger one will do 
//    if (CkMyPe() < entry->pNo) continue;
    if (!isMaster(entry->pNo)) continue;
#else
    // smaller one do it, which has the original object
    if (CkMyPe() == entry->pNo+1 || 
        CkMyPe()+CkNumPes() == entry->pNo+1) continue;
#endif
//CkPrintf("[%d] restore idx:%d aid:%d loc:%d ", CkMyPe(), idx, (CkGroupID)(entry->aid), entry->locMgr); entry->index.print();

    entry->updateBuddy(CkMyPe(), entry->pNo);
    CkArrayCheckPTMessage *msg = entry->getCopy();
    // gzheng
    //checkptMgr[CkMyPe()].inmem_restore(msg);
    inmem_restore(msg);
    count ++;
  }
//CkPrintf("[%d] recoverArrayElements restore %d objects\n", CkMyPe(), count);

  if (CkMyPe() == 0)
    CkStartQD(CkCallback(CkIndex_CkMemCheckPT::finishUp(), thisProxy));
}

// on every processor
// turn load balancer back on
void CkMemCheckPT::finishUp()
{
  //CkPrintf("[%d] CkMemCheckPT::finishUp\n", CkMyPe());
  CKLOCMGR_LOOP(mgr->doneInserting(););
  
  inRestarting = 0;

  if (CkMyPe() == 0)
  {
       CkPrintf("[%d] CkMemCheckPT ----- %s in %f seconds\n",CkMyPe(), stage, CmiWallTimer()-startTime);
       CkStartQD(cpCallback);
  } 
#if CK_NO_PROC_POOL
  if (CkNumPes()-totalFailed() <=2) {
    if (CkMyPe()==0) CkPrintf("Warning: CkMemCheckPT disabled!\n");
    _memChkptOn = 0;
  }
#endif
}

// called only on 0
void CkMemCheckPT::quiescence(CkCallback &cb)
{
  static int pe_count = 0;
  pe_count ++;
  CmiAssert(CkMyPe() == 0);
  //CkPrintf("quiescence %d %d\n", pe_count, CkNumPes());
  if (pe_count == CkNumPes()) {
    pe_count = 0;
    cb.send();
  }
}

// User callable function - to start a checkpoint
// callback cb is used to pass control back
void CkStartMemCheckpoint(CkCallback &cb)
{
#if CMK_MEM_CHECKPOINT
  if (_memChkptOn == 0) {
    CkPrintf("Warning: In-Memory checkpoint has been disabled! \n");
    cb.send();
    return;
  }
  if (CkInRestarting()) {
      // trying to checkpointing during restart
    cb.send();
    return;
  }
    // store user callback and user data
  CkMemCheckPT::cpCallback = cb;

    // broadcast to start check pointing
  CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
  checkptMgr.doItNow(CkMyPe(), cb);
#else
  // when mem checkpoint is disabled, invike cb immediately
  cb.send();
#endif
}

void CkRestartCheckPoint(int diePe)
{
CkPrintf("CkRestartCheckPoint  CkMemCheckPT GID:%d\n", ckCheckPTGroupID.idx);
  CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
  // broadcast
  checkptMgr.restart(diePe);
}

static int _diePE = -1;

// callback function used locally by ccs handler
static void CkRestartCheckPointCallback(void *ignore, void *msg)
{
CkPrintf("CkRestartCheckPointCallback activated for diePe: %d\n", _diePE);
  CkRestartCheckPoint(_diePE);
}

// Converse function handles
static int askProcDataHandlerIdx;
static int restartBcastHandlerIdx;
static int recoverProcDataHandlerIdx;
static int restartBeginHandlerIdx;

static void restartBeginHandler(char *msg)
{
#if CMK_MEM_CHECKPOINT
  static int count = 0;
  CmiFree(msg);
  CmiAssert(CkMyPe() == _diePE);
  count ++;
  if (count == CkNumPes()) {
    CkRestartCheckPointCallback(NULL, NULL);
    count = 0;
  }
#endif
}

static void restartBcastHandler(char *msg)
{
#if CMK_MEM_CHECKPOINT
  // advance phase counter
  CkMemCheckPT::inRestarting = 1;
  _diePE = *(int *)(msg+CmiMsgHeaderSizeBytes);
  // gzheng
  if (CkMyPe() != _diePE) cur_restart_phase ++;

  CkPrintf("[%d] restartBcastHandler cur_restart_phase=%d _diePE:%d.\n", CkMyPe(), cur_restart_phase, _diePE);

  // reset QD counters
  if (CkMyPe() != _diePE) CpvAccess(_qd)->flushStates();

/*  gzheng
  if (CkMyPe()==_diePE)
      CkRestartCheckPointCallback(NULL, NULL);
*/
  CmiFree(msg);

  char restartmsg[CmiMsgHeaderSizeBytes];
  CmiSetHandler(restartmsg, restartBeginHandlerIdx);
  CmiSyncSend(_diePE, CmiMsgHeaderSizeBytes, (char *)&restartmsg);
#endif
}

extern void _initDone();

// called on crashed processor
static void recoverProcDataHandler(char *msg)
{
#if CMK_MEM_CHECKPOINT
   int i;
   envelope *env = (envelope *)msg;
   CkUnpackMessage(&env);
   CkProcCheckPTMessage* procMsg = (CkProcCheckPTMessage *)(EnvToUsr(env));
   cur_restart_phase = procMsg->cur_restart_phase;
   CmiPrintf("[%d] ----- recoverProcDataHandler  cur_restart_phase:%d\n", CkMyPe(), cur_restart_phase);
   cur_restart_phase ++;
   CpvAccess(_qd)->flushStates();

   // restore readonly, mainchare, group, nodegroup
//   int temp = cur_restart_phase;
//   cur_restart_phase = -1;
   PUP::fromMem p(procMsg->packData);
   _handleProcData(p);

   CProxy_CkMemCheckPT(ckCheckPTGroupID).ckLocalBranch()->resetLB(CkMyPe());
   // gzheng
   CKLOCMGR_LOOP(mgr->startInserting(););

   char reqmsg[CmiMsgHeaderSizeBytes+sizeof(int)];
   *(int *)(&reqmsg[CmiMsgHeaderSizeBytes]) = CkMyPe();
   CmiSetHandler(reqmsg, restartBcastHandlerIdx);
   CmiSyncBroadcastAll(CmiMsgHeaderSizeBytes+sizeof(int), (char *)&reqmsg);
   CmiFree(msg);

   _initDone();
#endif
}

// called on its backup processor
// get backup message buffer and sent to crashed processor
static void askProcDataHandler(char *msg)
{
#if CMK_MEM_CHECKPOINT
    int diePe = *(int *)(msg+CmiMsgHeaderSizeBytes);
    CkPrintf("[%d] restartBcastHandler called with '%d' cur_restart_phase:%d.\n",CmiMyPe(),diePe, cur_restart_phase);
    envelope *env = (envelope *)(UsrToEnv(CpvAccess(procChkptBuf)));
    CmiAssert(CpvAccess(procChkptBuf)!=NULL);
    CmiAssert(CpvAccess(procChkptBuf)->pe == diePe);

    CpvAccess(procChkptBuf)->cur_restart_phase = cur_restart_phase;

    CkPackMessage(&env);
    CmiSetHandler(env, recoverProcDataHandlerIdx);
    CmiSyncSendAndFree(CpvAccess(procChkptBuf)->pe, env->getTotalsize(), (char *)env);
    CpvAccess(procChkptBuf) = NULL;
#endif
}

void CkMemRestart(const char *dummy, CkArgMsg *args)
{
#if CMK_MEM_CHECKPOINT
   _diePE = CkMyPe();
   CmiPrintf("[%d] I am restarting  cur_restart_phase:%d \n",CmiMyPe(), cur_restart_phase);
   CkMemCheckPT::startTime = CmiWallTimer();
   CkMemCheckPT::inRestarting = 1;
   char msg[CmiMsgHeaderSizeBytes+sizeof(int)];
   *(int *)(&msg[CmiMsgHeaderSizeBytes]) = CkMyPe();
   cur_restart_phase = 9999;             // big enough to get it processed
   CmiSetHandler(msg, askProcDataHandlerIdx);
   int pe = ChkptOnPe();
   CmiSyncSend(pe, CmiMsgHeaderSizeBytes+sizeof(int), (char *)&msg);
   cur_restart_phase=0;    // allow all message to come in
#else
   CmiAbort("Fault tolerance is not support, rebuild charm++ with 'syncft' option");
#endif
}

// can be called in other files
// return true if it is in restarting
int CkInRestarting()
{
#if CMK_MEM_CHECKPOINT
  // gzheng
  if (cur_restart_phase == 9999 || cur_restart_phase == 0) return 1;
  //return CProxy_CkMemCheckPT(ckCheckPTGroupID).ckLocalBranch()->inRestarting;
  return CkMemCheckPT::inRestarting;
#else
  return 0;
#endif
}

/*****************************************************************************
                module initialization
*****************************************************************************/

static int arg_where = CkCheckPoint_inMEM;

#if CMK_MEM_CHECKPOINT
void init_memcheckpt(char **argv)
{
    if (CmiGetArgFlagDesc(argv, "+ftc_disk", "Double-disk Checkpointing")) {
      arg_where = CkCheckPoint_inDISK;
    }
}
#endif

class CkMemCheckPTInit: public Chare {
public:
  CkMemCheckPTInit(CkArgMsg *m) {
#if CMK_MEM_CHECKPOINT
    if (arg_where == CkCheckPoint_inDISK) {
      CkPrintf("Charm++> Double-disk Checkpointing. \n");
    }
    ckCheckPTGroupID = CProxy_CkMemCheckPT::ckNew(arg_where);
    CkPrintf("Charm++> CkMemCheckPTInit mainchare is created!\n");
#endif
  }
};

// initproc
void CkRegisterRestartHandler( )
{
#if CMK_MEM_CHECKPOINT
  askProcDataHandlerIdx = CkRegisterHandler((CmiHandler)askProcDataHandler);
  recoverProcDataHandlerIdx = CkRegisterHandler((CmiHandler)recoverProcDataHandler);
  restartBcastHandlerIdx = CkRegisterHandler((CmiHandler)restartBcastHandler);
  restartBeginHandlerIdx = CkRegisterHandler((CmiHandler)restartBeginHandler);

  CpvInitialize(CkProcCheckPTMessage *, procChkptBuf);
  CpvAccess(procChkptBuf) = NULL;

#if 1
  // for debugging
  CkPrintf("[%d] PID %d \n", CkMyPe(), getpid());
//  sleep(4);
#endif
#endif
}

/// @todo: the following definitions should be moved to a separate file containing
// structures and functions about fault tolerance strategies

/**
 *  * @brief: function for killing a process                                             
 *   */
#ifdef CMK_MEM_CHECKPOINT
#if CMK_HAS_GETPID
void killLocal(void *_dummy,double curWallTime){
        printf("[%d] KillLocal called at %.6lf \n",CkMyPe(),CmiWallTimer());          
        if(CmiWallTimer()<killTime-1){
                CcdCallFnAfter(killLocal,NULL,(killTime-CmiWallTimer())*1000);        
        }else{  
                kill(getpid(),SIGKILL);                                               
        }              
} 
#else
void killLocal(void *_dummy,double curWallTime){
  CmiAbort("kill() not supported!");
}
#endif
#endif

#ifdef CMK_MEM_CHECKPOINT
/**
 * @brief: reads the file with the kill information
 */
void readKillFile(){
        FILE *fp=fopen(killFile,"r");
        if(!fp){
                return;
        }
        int proc;
        double sec;
        while(fscanf(fp,"%d %lf",&proc,&sec)==2){
                if(proc == CkMyPe()){
                        killTime = CmiWallTimer()+sec;
                        printf("[%d] To be killed after %.6lf s (MEMCKPT) \n",CkMyPe(),sec);
                        CcdCallFnAfter(killLocal,NULL,sec*1000);
                }
        }
        fclose(fp);
}
#endif

#include "CkMemCheckpoint.def.h"


