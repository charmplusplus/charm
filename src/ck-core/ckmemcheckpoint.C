
/*
Charm++ support for fault tolerance of
In memory synchronous checkpointing and restart

written by Gengbin Zheng, gzheng@uiuc.edu
           Lixia Shi,     lixiashi@uiuc.edu

added 12/18/03

To ensure fault tolerance while allowing migration, it uses double
checkpointing scheme for each array element.
In this version, checkpointing is done based on array elements. 
Each array element individully sends its checkpoint data to two buddies.

In this implementation, assume at a time only one failure happens,
and there is no failure during a checkpointing or restarting phase.

Restart phase contains two steps:
1. Converse level restart where only the newly created process for the failed
   processor is working on restoring the system data (except array elements)
   from its backup processor.
2. CkMemCheckPT gets control and recover array elements and reset all
   states to be consistent.

*/

#include "charm++.h"
#include "ck.h"
#include "register.h"
#include "conv-ccs.h"

#define DEBUGF      CkPrintf

CkGroupID ckCheckPTGroupID;		// readonly

CkCallback CkMemCheckPT::cpCallback;    // static

CpvStaticDeclare(CkProcCheckPTMessage*, procChkptBuf);

// compute the backup processor
// FIXME: avoid crashed processors
inline int ChkptOnPe() { return (CkMyPe()+1)%CkNumPes(); }

// called in array element constructor
// choose and register with 2 buggies for checkpoiting 
void ArrayElement::init_checkpt() {
	// CmiPrintf("[%d] ArrayElement::init_checkpt %d\n", CkMyPe(), info.fromMigration);
        budPEs[0] = (CkMyPe()-1+CkNumPes())%CkNumPes();
        budPEs[1] = (CkMyPe()+1)%CkNumPes();
        // inform checkPTMgr
        CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
        checkptMgr[budPEs[0]].createEntry(thisArrayID, thisArray->getLocMgr()->getGroupID(), thisIndexMax, budPEs[1]);        
	checkptMgr[budPEs[1]].createEntry(thisArrayID, thisArray->getLocMgr()->getGroupID(), thisIndexMax, budPEs[0]);
}

// entry function invoked by checkpoint mgr asking for checkpoint data
void ArrayElement::inmem_checkpoint(CkArrayCheckPTReqMessage *m) {
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
}

// called by checkpoint mgr to restore an array element
void CkMemCheckPT::inmem_restore(CkArrayCheckPTMessage *m) 
{
  //DEBUGF("[%d] inmem_restore restore", CmiMyPe());  m->index.print();
  PUP::fromMem p(m->packData);
  CkLocMgr *mgr = CProxy_CkLocMgr(m->locMgr).ckLocalBranch();
  mgr->resume(m->index, p);
  // 	reset, may not needed now
  ArrayElement *elt = (ArrayElement *)mgr->lookup(m->index, m->aid);
  CmiAssert(elt);
  elt->budPEs[0] = m->bud1;
  elt->budPEs[1] = m->bud2;
  // for now.
  for (int i=0; i<CK_ARRAYLISTENER_MAXLEN; i++) {
    contributorInfo *c=(contributorInfo *)&elt->listenerData[i];
    if (c) c->redNo = 0;
  }
/*
  contributorInfo *c=(contributorInfo *)&elt->listenerData[elt->thisArray->reducer->ckGetOffset()];
  if (c) c->redNo = 0;
*/
}

CkMemCheckPT::CkMemCheckPT()
{
  if (CkNumPes() <= 3 && CkMyPe() == 0) 
    CkPrintf("CkMemCheckPT disabled!\n");
  inRestarting = 0;
  recvCount = peCount = 0;
  qdCallback = NULL;
}

CkMemCheckPT::~CkMemCheckPT()
{
  if (qdCallback) delete qdCallback;
}

void CkMemCheckPT::pup(PUP::er& p) 
{ 
  CBase_CkMemCheckPT::pup(p); 
  p|cpStarter;
  p|thisFailedPe;
  p|failedPes;
  p|ckCheckPTGroupID;		// recover global variable
  p|cpCallback;			// store callback
  if (p.isUnpacking()) {
    recvCount = peCount = 0;
    qdCallback = NULL;
  }
}

// return 1 is pe was a crashed processor
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

// create an checkpoint entry for array element of aid with index.
void CkMemCheckPT::createEntry(CkArrayID aid, CkGroupID loc, CkArrayIndexMax index, int buddy)
{
  // error check, no duplicate
  int idx, len = ckTable.size();
  for (idx=0; idx<len; idx++) {
    CkMemCheckPTInfo *entry = ckTable[idx];
    if (aid == entry->aid && index == entry->index) break;
  }
  if (idx<len) {
    CkPrintf("[%d] CkMemCheckPT::createEntry a duplciated entry. \n", CkMyPe());
    CmiAbort("CkMemCheckPT::createEntry a duplciated entry");
  }
  CkMemCheckPTInfo *newEntry = new CkMemCheckPTInfo(aid, loc, index, buddy);
  ckTable.push_back(newEntry);
}

// loop through my checkpoint table and ask checkpointed array elements
// to send me checkpoint data.
void CkMemCheckPT::doItNow(int starter, CkCallback &cb)
{
  cpCallback = cb;
  cpStarter = starter;
  CkPrintf("[%d] Start checkpointing ... \n", CkMyPe());

//  if (iFailed()) return;
  int len = ckTable.length();
  for (int i=0; i<len; i++) {
    CkMemCheckPTInfo *entry = ckTable[i];
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
static void _handleProcData(PUP::er &p)
{
    // save readonlys, and callback BTW
    CkPupROData(p);

    // save mainchares into MainChares.dat
    if(CkMyPe()==0) {
	CkPupMainChareData(p);
    }
	
    // save groups into Groups.dat
    CkPupGroupData(p);

    // save nodegroups into NodeGroups.dat
    CkPupNodeGroupData(p);
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
  CkProcCheckPTMessage *msg =
                 new (packSize, 0) CkProcCheckPTMessage;
  DEBUGF("[%d] CkMemCheckPT::sendProcData - size: %d\n", CkMyPe(), size);
  {
    PUP::toMem p(msg->packData);
    _handleProcData(p);
  }
  msg->pe = CkMyPe();
  msg->len = size;
  thisProxy[ChkptOnPe()].recvProcData(msg);
}

void CkMemCheckPT::recvProcData(CkProcCheckPTMessage *msg)
{
  if (CpvAccess(procChkptBuf)) delete CpvAccess(procChkptBuf);
  CpvAccess(procChkptBuf) = msg;
  cpStarter = 0;    // fix me
  thisProxy[cpStarter].cpFinish();
}

// ArrayElement call this function to give us the checkpointed data
void CkMemCheckPT::recvData(CkArrayCheckPTMessage *msg)
{
  int len = ckTable.length();
  int idx;
  for (idx=0; idx<len; idx++) {
    CkMemCheckPTInfo *entry = ckTable[idx];
    if (msg->aid == entry->aid && msg->index == entry->index) break;
  }
  CkAssert(idx < len);
  ckTable[idx]->updateBuffer(msg);
    // all my array elements have returned their inmem data
    // inform starter processor that I am done.
  if (msg->cp_flag) {
    recvCount ++;
    if (recvCount == ckTable.length()) {
      thisProxy[cpStarter].cpFinish();
      recvCount = 0;
    } 
  }
}

// only is called on cpStarter
void CkMemCheckPT::cpFinish()
{
  CmiAssert(CkMyPe() == 0);
  peCount++;
    // now all processors have finished, activate callback
  if (peCount == 2*(CkNumPes())) {
    CmiPrintf("[%d] Checkpoint finished, sending callback ... \n", CkMyPe());
    cpCallback.send();
    peCount = 0;
  }
}

/*****************************************************************************
			RESTART Procedure
*****************************************************************************/

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

// restore the bitmap vector for LB
void CkMemCheckPT::resetLB(int diepe)
{
  int i;
  char *bitmap = new char[CkNumPes()];
  // set processor available bitmap
  get_avail_vector(bitmap);

  for (i=0; i<failedPes.length(); i++)
    bitmap[failedPes[i]] = 0; 
  bitmap[diepe] = 0;
  set_avail_vector(bitmap);

  // if I am the crashed pe, rebuild my failedPEs array
  if (CkMyPe() == diepe)
  for (i=0; i<CkNumPes(); i++) 
    if (bitmap[i]==0) failed(bitmap[i]);

  delete [] bitmap;
}

// in case when failedPe dies, everybody go through its check point table:
// destory all array elements
// recover lost buddies
// reconstruct all array elements from check point data
// called on all processors
void CkMemCheckPT::restart(int diePe)
{
  failed(diePe);	// add into the list of failed pes
  thisFailedPe = diePe;

  CkPrintf("[%d] CkMemCheckPT ----- restart.\n",CkMyPe());
  // clean array chkpt table
  if (CkMyPe() == diePe) ckTable.length() = 0;

  inRestarting = 1;
                                                                                
  // disable load balancer's barrier
  if (CkMyPe() != diePe) resetLB(diePe);

  CKLOCMGR_LOOP(mgr->startInserting(););

  thisProxy[0].quiescence(CkCallback(CkIndex_CkMemCheckPT::removeArrayElements(), thisProxy));
  // afterwards, the QD detection should work again
}

void CkMemCheckPT::removeArrayElements()
{
  int len = ckTable.length();
  CkPrintf("[%d] CkMemCheckPT ----- removeArrayElements len:%d.\n",CkMyPe(),len);

  if (cpCallback.isInvalid()) CkAbort("Don't set restart callback\n");;
  if (CkMyPe()==thisFailedPe) CmiAssert(len == 0);

  // get rid of all buffering and remote recs
  CKLOCMGR_LOOP(mgr->flushStates(););

  // first phase: destroy all existing array elements
  for (int idx=0; idx<len; idx++) {
    CkMemCheckPTInfo *entry = ckTable[idx];
    // the bigger number PE do the destory
    if (CkMyPe() < entry->pNo && entry->pNo != thisFailedPe) continue;
    CkArrayMessage *msg = (CkArrayMessage *)CkAllocSysMsg();
    msg->array_setIfNotThere(CkArray_IfNotThere_buffer);
    CkSendMsgArray(CkIndex_ArrayElement::ckDestroy(),msg,entry->aid,entry->index);
    //CkCallback cb(CkIndex_ArrayElement::ckDestroy(), entry->index, entry->aid);
    //cb.send(msg);
//CkPrintf("[%d] Destory: ", CkMyPe()); entry->index.print();
  }

  if (CkMyPe() == 0)
  CkStartQD(CkCallback(CkIndex_CkMemCheckPT::resetReductionMgr(), thisProxy));
}

// flush state in reduction manager
void CkMemCheckPT::resetReductionMgr()
{
  CkPrintf("[%d] CkMemCheckPT ----- resetReductionMgr\n",CkMyPe());
  int numGroups = CkpvAccess(_groupIDTable)->size();
  for(int i=0;i<numGroups;i++) {
    CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
    IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
    obj->flushStates();
    obj->ckJustMigrated();
  }
  // reset again
  //CkResetQD();
  if (CkMyPe() == 0)
    CkStartQD(CkCallback(CkIndex_CkMemCheckPT::recoverBuddies(), thisProxy));
}

// recover the lost buddies
void CkMemCheckPT::recoverBuddies()
{
  int idx;
  int len = ckTable.length();
  // ready to flush reduction manager
  // cannot be CkMemCheckPT::restart because destory will modify states
  CkPrintf("[%d] CkMemCheckPT ----- recoverBuddies  len: %d\n",CkMyPe(),len);

  //if (iFailed()) return;   ??????

  // recover buddies
  for (idx=0; idx<len; idx++) {
    CkMemCheckPTInfo *entry = ckTable[idx];
    if (entry->pNo == thisFailedPe) {
      int budPe = CkMyPe();
      while (budPe == CkMyPe() || isFailed(budPe)) budPe = CrnRand()%CkNumPes();
      entry->pNo = budPe;
      thisProxy[budPe].createEntry(entry->aid, entry->locMgr, entry->index, CkMyPe());
      CmiAssert(entry->ckBuffer);
      CkArrayCheckPTMessage *msg = (CkArrayCheckPTMessage *)CkCopyMsg((void **)&entry->ckBuffer);
      msg->cp_flag = 0;            // not checkpointing
      thisProxy[budPe].recvData(msg);
    }
  }

  if (CkMyPe() == 0)
    CkStartQD(CkCallback(CkIndex_CkMemCheckPT::recoverArrayElements(), thisProxy));
}

// restore 
void CkMemCheckPT::recoverArrayElements()
{
  CkPrintf("[%d] CkMemCheckPT ----- recoverArrayElements\n",CkMyPe());
  //if (iFailed()) return;

  // recover all array elements
  int len = ckTable.length();
  for (int idx=0; idx<len; idx++)
  {
    CkMemCheckPTInfo *entry = ckTable[idx];
    // the bigger one will do 
    if (CkMyPe() < entry->pNo) continue;
//CkPrintf("[%d] restore idx:%d  ", CkMyPe(), idx); entry->index.print();
    if (entry->ckBuffer == NULL) CmiAbort("recoverArrayElements: element does not have checkpoint data.");
    entry->ckBuffer->bud1 = CkMyPe(); entry->ckBuffer->bud2 = entry->pNo;
    CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
    CkArrayCheckPTMessage *msg = (CkArrayCheckPTMessage *)CkCopyMsg((void **)&entry->ckBuffer);
    checkptMgr[CkMyPe()].inmem_restore(msg);
  }

  if (CkMyPe() == 0)
    CkStartQD(CkCallback(CkIndex_CkMemCheckPT::finishUp(), thisProxy));
}

// on every processor
// turn load balancer back on
void CkMemCheckPT::finishUp()
{
  int i;
  int numGroups = CkpvAccess(_groupIDTable)->size();
  for(i=0;i<numGroups;i++) {
    CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
    IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
    if (obj->isLocMgr()) 
      ((CkLocMgr *)obj)->doneInserting();
  }
  
  inRestarting = 0;

  if (CkMyPe() == 0)
  {
       CkPrintf("Restart finished, sending out callback ...\n");
       CkStartQD(cpCallback);
  } 
}

// called only on 0
void CkMemCheckPT::quiescence(CkCallback cb)
{
  static int pe_count = 0;
  pe_count ++;
  CmiAssert(CkMyPe() == 0);
  CkPrintf("quiescence %d\n", pe_count);
  if (pe_count == CkNumPes()) {
    pe_count = 0;
    cb.send();
  }
}

// function called by user to start a check point
// callback cb is used to pass control back
void CkStartCheckPoint(CkCallback &cb)
{
#if CMK_MEM_CHECKPOINT
    // store user callback and user data
  CkMemCheckPT::cpCallback = cb;

    // broadcast to start check pointing
  CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
  checkptMgr.doItNow(CkMyPe(), cb);
#endif
}

void CkRestartCheckPoint(int diePe)
{
CkPrintf("CkRestartCheckPoint  CkMemCheckPT GID:%d\n", ckCheckPTGroupID.idx);
  CProxy_CkMemCheckPT checkptMgr(ckCheckPTGroupID);
  // broadcast
  checkptMgr.restart(diePe);
}

static int _diePE;

// callback function used locally by ccs handler
static void CkRestartCheckPointCallback(void *ignore, void *msg)
{
CkPrintf("CkRestartCheckPointCallback activated for diePe: %d\n", _diePE);
  CkRestartCheckPoint(_diePE);
}

static int askProcDataHandlerIdx;
static int restartBcastHandlerIdx;
static int recoverProcDataHandlerIdx;

static void restartBcastHandler(char *msg)
{
  // advance phase counter
  cur_restart_phase ++;
  _diePE = *(int *)(msg+CmiMsgHeaderSizeBytes);

  CkPrintf("[%d] restartBcastHandler cur_restart_phase=%d _diePE:%d.\n", CkMyPe(), cur_restart_phase, _diePE);

  // reset QD counters
  CpvAccess(_qd)->flushStates();

  if (CkMyPe()==_diePE)
      CkRestartCheckPointCallback(NULL, NULL);
  CmiFree(msg);
}

extern void _initDone();

// called on crashed processor
static void recoverProcDataHandler(char *msg)
{
   int i;
   CmiPrintf("[%d] ----- recoverProcDataHandler  cur_restart_phase:%d\n", CkMyPe(), cur_restart_phase);
   envelope *env = (envelope *)msg;
   CkUnpackMessage(&env);
   CkProcCheckPTMessage* procMsg = (CkProcCheckPTMessage *)(EnvToUsr(env));
   cur_restart_phase = procMsg->cur_restart_phase;
   // restore readonly, mainchare, group, nodegroup
   PUP::fromMem p(procMsg->packData);
   _handleProcData(p);

   CProxy_CkMemCheckPT(ckCheckPTGroupID).ckLocalBranch()->resetLB(CkMyPe());

   char reqmsg[CmiMsgHeaderSizeBytes+sizeof(int)];
   *(int *)(&reqmsg[CmiMsgHeaderSizeBytes]) = CkMyPe();
   CmiSetHandler(reqmsg, restartBcastHandlerIdx);
   CmiSyncBroadcastAll(CmiMsgHeaderSizeBytes+sizeof(int), (char *)&reqmsg);
   CmiFree(msg);

   _initDone();
}

// called on its backup processor
// get backup message buffer and sent to crashed processor
static void askProcDataHandler(char *msg)
{
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
}

void CkMemRestart(){
   CmiPrintf("[%d] I am restarting  cur_restart_phase:%d \n",CmiMyPe(), cur_restart_phase);
   char msg[CmiMsgHeaderSizeBytes+sizeof(int)];
   *(int *)(&msg[CmiMsgHeaderSizeBytes]) = CkMyPe();
   cur_restart_phase = 999999;             // big enough to get it processed
   CmiSetHandler(msg, askProcDataHandlerIdx);
   int pe = ChkptOnPe();
   CmiSyncSend(pe, CmiMsgHeaderSizeBytes+sizeof(int), (char *)&msg);
   cur_restart_phase=-1;
}

/*****************************************************************************
                module initialization
*****************************************************************************/

class CkMemCheckPTInit: public Chare {
public:
  CkMemCheckPTInit(CkArgMsg *m) {
#if CMK_MEM_CHECKPOINT
    ckCheckPTGroupID = CProxy_CkMemCheckPT::ckNew();
    CkPrintf("CkMemCheckPTInit main chare created!\n");
#endif
  }
};

// return true if it is in restarting
int CkInRestart()
{
  return CProxy_CkMemCheckPT(ckCheckPTGroupID).ckLocalBranch()->inRestarting;
}

// initproc
void CkRegisterRestartHandler( )
{
#if CMK_MEM_CHECKPOINT
  askProcDataHandlerIdx = CkRegisterHandler((CmiHandler)askProcDataHandler);
  recoverProcDataHandlerIdx = CkRegisterHandler((CmiHandler)recoverProcDataHandler);
  restartBcastHandlerIdx = CkRegisterHandler((CmiHandler)restartBcastHandler);

  CpvInitialize(CkProcCheckPTMessage *, procChkptBuf);
  CpvAccess(procChkptBuf) = NULL;

#if 1
  // for debugging
  CkPrintf("[%d] PID %d\n", CkMyPe(), getpid());
  sleep(6);
#endif
#endif
}

#include "CkMemCheckpoint.def.h"


