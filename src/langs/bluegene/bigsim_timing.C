
#include "stdio.h"

#include "blue.h"
#include "blue_impl.h"
#include "blue_timing.h"

#define INTEGRITY_CHECK		1

typedef minHeap<bgTimeLog *>  BgTimelogHeap;

/*** CPVs ***/

CpvStaticDeclare(int,bgCorrectionHandler);
CpvStaticDeclare(int, msgCounter);
CpvDeclare(int, heartbeatHandler);
CpvDeclare(int, heartbeatBcastHandler);

int bgSkipEndFlag=0;
int traceBluegeneLinked=0;
extern int programExit;
static int deadlock = 0;

bgTimeLog *bgCurLog;

#if USE_MULTISEND
CkVec<char *>   *corrMsgBucket;
#endif
int processCount = 0;
int corrMsgCount = 0;

double gvt = 0.0;
double minCorrectTimestamp = INVALIDTIME;

/**  Externs **/

CpvExtern(int,bgStatCollectHandler);
extern void statsCollectionHandlerFunc(void *msg);

void bgCorrectionFunc(char *msg);
void initHeartbeat();

/**
  NOTE: known problems for timing correction in Charm++
1. in CkStartQD(), messages can be sent from QD handler function and we cannot
   catch the dependence when system reachs quiesence;
2. in ckreduction, because current reduction is not implemented in SDAG,
   the correction for reduction can be wrong.
*/

/**
  init Cpvs of timing module
*/
void BgInitTiming()
{
  CpvInitialize(int, msgCounter);
  CpvAccess(msgCounter) = 0;
#if USE_MULTISEND
  corrMsgBucket = new CkVec<char *>[CmiNumPes()];
#endif
#if BLUEGENE_TIMING
  CpvInitialize(int,bgCorrectionHandler);
  cva(bgCorrectionHandler) = CmiRegisterHandler((CmiHandler) bgCorrectionFunc);
  CpvInitialize(int,bgStatCollectHandler);
  cva(bgStatCollectHandler) = CmiRegisterHandler((CmiHandler) statsCollectionHandlerFunc);
#endif

  initHeartbeat();
}

// close current log
void BgSkipEndExecuteEvent() {
  bgSkipEndFlag=1; 
  if (genTimeLog && traceBluegeneLinked) {
    BgTimeLine &tline = tTIMELINE;
    CmiAssert(tline.length()>0); 
    bgTimeLog *log = tline[tline.length()-1];
    log->closeLog();
  }
}

void *BgCreateEvent(int eidx)
{
  bgTimeLog *entry = new bgTimeLog();
  entry->ep = eidx;
  entry->srcnode = BgMyNode();
  entry->startTime = BgGetCurTime();
  return (void *)entry;
}

void BgInsertLog(void* log){

  BgTimeLineRec &tlinerec = tTIMELINEREC;
  if(tlinerec[tlinerec.length()-1]->endTime == 0.0)
    CmiPrintf("\nERROR tried to insert %s after %s\n",((bgTimeLog*)log)->name,tlinerec[tlinerec.length()-1]->name);

  tlinerec.enq((bgTimeLog*)log, 1);
}

// end current one and start a new one same ep
// This is used for broadcast where several entry functions coexist
// in one starting bglog
// the parent log is bgCurLog
void BgEntrySplit()
{
  if (!genTimeLog) return;
  CmiAssert(tTHREADTYPE == WORK_THREAD);
  BgTimeLineRec &tlinerec = tTIMELINEREC;
  BgTimeLine &tline = tlinerec.timeline;
  if (bgSkipEndFlag == 0) {
    bgTimeLog *log = tline[tline.length()-1];
    log->closeLog();
  }
  else
    bgSkipEndFlag = 0;

  // make up a new bglog to start, setting up dependencies.
  bgTimeLog *newLog = new bgTimeLog(-1, "broadcast", BgGetTime());
  newLog->addBackwardDep(bgCurLog);
  tlinerec.enq(newLog, 1);
}

// must be called inside a timelog
double BgGetRecvTime()
{
  if (genTimeLog)  {
    int len = tTIMELINE.length();
    if (len) return tTIMELINE[len-1]->recvTime;
  }
  return 0.0;
}

// HACK for barrier to escape corrections
void BgResetRecvTime()
{
  if (genTimeLog) {
    int len = tTIMELINE.length();
    if (len) {
      bgTimeLog *log = tTIMELINE[len-1];
      log->recvTime = log->effRecvTime = log->startTime;
      log->doCorrect = 0;
    }
  }
}

void BgMsgSetTiming(char *msg)
{
  CmiBgMsgID(msg) = CpvAccess(msgCounter)++;
  CmiBgMsgSrcPe(msg) = BgMyNode();	// global serial number
}

// dstNode is the dest bg node, can be -1
bgMsgEntry::bgMsgEntry(char *msg, int dstNode, int tid, int local)
{
  msgID = CmiBgMsgID(msg);
//  sendtime = BgGetCurTime();
  recvTime = CmiBgMsgRecvTime(msg);
  dstPe = dstNode;
  tID = tid;                   // CmiBgMsgThreadID(msg);
#if DELAY_SEND
  sendMsg = NULL;
  if (!local && correctTimeLog) sendMsg = msg;
#endif
}

#if DELAY_SEND
void bgMsgEntry::send() {
  if (!sendMsg) return;
  CmiBgMsgRecvTime(sendMsg) = recvTime;
  if (dstPe >= 0) {
    CmiSyncSendAndFree(nodeInfo::Global2PE(dstPe),CmiBgMsgLength(sendMsg),sendMsg);
  }
  else {
    CmiSyncBroadcastAllAndFree(CmiBgMsgLength(sendMsg),sendMsg);
  }
  sendMsg = NULL;
}
#endif

bgTimeLog::bgTimeLog(bgTimeLog *log)
{
  strncpy(name,log->name,20);
  ep = log->ep;
  startTime = log->startTime;
  recvTime = log->recvTime;
  endTime = 0.0;
  execTime = 0.0;
  srcnode = log->srcnode;
  msgID = log->msgID;

  seqno = 0;
  effRecvTime = recvTime;
  doCorrect = 1;
}

bgTimeLog::bgTimeLog(int epc, char* namestr,double sTime)
{ 
  if(namestr == NULL)
    namestr = "dummyname";
  strncpy(name,namestr,20);
  ep = epc;
  startTime = sTime;
  recvTime = -1.0;//stime;
  endTime = execTime = 0.0;
  srcnode = msgID = -1;

  oldStartTime= startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
}

// for SDAG, somewhere else will set the effective recv time.
bgTimeLog::bgTimeLog(int epc, char* namestr, double sTime, double eTime)
{
  if(namestr == NULL)
    namestr = "dummyname";
  strncpy(name,namestr, 20);
  ep = epc;
  startTime = sTime;
  recvTime = -1.0; //sTime;
  endTime = eTime;
  setExecTime();
  srcnode = -1;
  msgID = -1;

  oldStartTime = startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
}


bgTimeLog::bgTimeLog(int epc, char *msg)
{
  strcpy(name,"msgep");
  ep = epc;
  startTime = BgGetCurTime();
  recvTime = msg?CmiBgMsgRecvTime(msg):0;//startTime;
  endTime = 0.0;
  execTime = 0.0;
  srcnode = msg?CmiBgMsgSrcPe(msg):-1;
  msgID = msg?CmiBgMsgID(msg):-1;

  oldStartTime=startTime;
  effRecvTime = recvTime;
  seqno = 0;
//  doCorrect = msg?CkMsgDoCorrect(msg):1;
  doCorrect = 1;

  if (genTimeLog && !doCorrect) {
//CmiPrintf("TIME: %f\n", startTime);
      recvTime = effRecvTime = startTime;
  }
}


bgTimeLog::~bgTimeLog()
{
  for (int i=0; i<msgs.length(); i++)
    delete msgs[i];
}


void bgTimeLog::setExecTime(){
  execTime = endTime - startTime;
  if(execTime < EPSILON && execTime > -EPSILON)
    execTime = 0.0;
  CmiAssert(execTime >= 0.0);
}

void bgTimeLog::closeLog() 
{ 
    endTime = BgGetCurTime(); 
    setExecTime();
    
//    if (correctTimeLog) BgAdjustTimeLineInsert(tTIMELINEREC);
}


void bgTimeLog::print(int node, int th)
{
  CmiPrintf("<<== [%d th:%d] ep:%d name:%s startTime:%f endTime:%f srcnode:%d msgID:%d\n", node, th, ep, name,startTime, endTime, srcnode, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("==>>\n");
}


void bgTimeLog::write(FILE *fp)
{ 
  int i;
  fprintf(fp,"<<==  %p ep:%d name:%s startTime:%f endTime:%f recvime:%f effRecvTime:%e srcnode:%d msgID:%d seqno:%d\n", this, ep, name,startTime, endTime, recvTime, effRecvTime, srcnode, msgID, seqno);
  for (i=0; i<msgs.length(); i++)
   msgs[i]->write(fp);
  // fprintf(fp,"\nbackwardDeps [%d]:\n",backwardDeps.length());
  fprintf(fp, "backward: ");
  for (i=0; i<backwardDeps.length(); i++)
    fprintf(fp,"[%p] ",backwardDeps[i]);
  fprintf(fp, "\n");
  fprintf(fp, "forward: ");
  for (i=0; i<forwardDeps.length(); i++)
    fprintf(fp,"[%p] ",forwardDeps[i]);
  fprintf(fp, "\n");
  fprintf(fp, "==>>\n");
}


void bgTimeLog::addBackwardDep(bgTimeLog* log){
  
 CmiAssert(recvTime < 0.);
  if(log != NULL){
    backwardDeps.insertAtEnd(log);
    log->forwardDeps.insertAtEnd(this);
    effRecvTime = max(effRecvTime, log->effRecvTime);
  }
}

void bgTimeLog::addBackwardDeps(CkVec<bgTimeLog*> logs){

  /*put backward and forward dependents*/
  for(int i=0;i<logs.length();i++)
    addBackwardDep(logs[i]);
}

void bgTimeLog::addBackwardDeps(CkVec<void*> logs){

  /*put backward and forward dependents*/
  for(int i=0;i<logs.length();i++)
    addBackwardDep((bgTimeLog*)(logs[i]));
}

// return the last eff recv time
double bgTimeLog::getEndOfBackwardDeps(){
  
  double maxEndTime =0.0;
  for(int i=0;i<backwardDeps.length();i++)
//    maxEndTime = max(maxEndTime,backwardDeps[i]->endTime);
    maxEndTime = max(maxEndTime,backwardDeps[i]->effRecvTime);
      
  return maxEndTime;
}

inline int bgTimeLog::adjustTimeLog(double tAdjust, BgTimeLine& tline, int mynode, int sendImmediately)
{
	//arg tAdjust is a relative time
	if(isZero(tAdjust)) return 0;
	oldStartTime = startTime;

	for(int i=0; i<msgs.length(); i++) 
	{
            bgMsgEntry *msgEntry = msgs[i];
	    // update new msg send and recv time
//	    msgEntry->sendtime += tAdjust;
	    msgEntry->recvTime += tAdjust;

//            if (gvt > 0.0004 && msgEntry->recvTime < 0.00005 ) CmiAbort("GOD");
	    if(minCorrectTimestamp > msgEntry->recvTime)
	      minCorrectTimestamp = msgEntry->recvTime;

#if DELAY_SEND
	    if (!sendImmediately) continue;
#endif

	    // send correction messages
	    bgCorrectionMsg *msg = (bgCorrectionMsg *)CmiAlloc(sizeof(bgCorrectionMsg));
	    msg->msgID = msgEntry->msgID;
	    msg->tID = msgEntry->tID;
	    //msg->tAdjust is absolute recvTime at destination node
	    msg->tAdjust = msgEntry->recvTime;  // new recvTime
	    msg->destNode = msgEntry->dstPe;
	    msg->srcNode = nodeInfo::Local2Global(mynode);
		
	    CmiSetHandler(msg, CpvAccess(bgCorrectionHandler));
#if ! USE_MULTISEND
	    if (msgs[i]->dstPe < 0) {
	      CmiSyncBroadcastAllAndFree(sizeof(bgCorrectionMsg),(char*)msg);
	    }
	    else{
	      CmiSyncSendAndFree(BgNodeToPE(msgs[i]->dstPe),sizeof(bgCorrectionMsg),(char*)msg);
	    }  
#else
	    if (msgs[i]->dstPe < 0) {
	      corrMsgBucket[0].push_back((char *)msg);
	      for (int i=1; i<CmiNumPes(); i++)  {
	        char *newmsg = CmiCopyMsg((char *)msg, sizeof(bgCorrectionMsg));
	        corrMsgBucket[i].push_back(newmsg);
	      }
	    }
	    else {
	      corrMsgBucket[BgNodeToPE(msgs[i]->dstPe)].push_back((char *)msg);
	    }
#endif
	}
	return 1;
}


int BgGetIndexFromTime(double effT, int seqno, BgTimeLineRec &tlinerec)
{
  int idx;
  int commit = tlinerec.commit;
#if 0
  int low = 0, high = commit;
#endif
  int startIdx = tlinerec.startIdx;
  int low = startIdx, high = commit;
  while(low<high) {
    idx = (low + high)/2;
    bgTimeLog *curLog = tlinerec[idx];  
    if(isLess(effT, curLog->effRecvTime) ||
         (isEqual(effT, curLog->effRecvTime) && curLog->seqno > seqno)) {
      high = idx;
    }
    else {
      low = idx+1;
      if(low == commit) {
	 idx = low;
	 break;
      }
    }
  }
  CmiAssert(idx<=commit && idx>=startIdx);
  return idx;
}

// tLog can be (1) a nromal entry function, or (2) a sdag
void BgGetMsgStartTime(bgTimeLog* tLog, BgTimeLineRec &tline, int* index)
{
	/* ASSUMPTION: BgGetMsgStartTime is called only if necessary */

	// binary search: index of FIRST entry in tline whose recvTime is 
	// greater than arg 'recvTime'
	int low = tline.startIdx, high = tline.commit;
	int idx = 0;
	double endOfDeps = tLog->getEndOfBackwardDeps();
	double effRecvTime  = max(tLog->recvTime, endOfDeps);
  	int commit = tline.commit;
	
        if (commit>0 && tline[commit-1]->effRecvTime<effRecvTime) {
	  *index = commit;
	  return;
	}
	// now make sure effRecvTime of tLog is in the range of 0 ... commit-1
#if 1
	while(low<high) {
	  idx = (low + high)/2;
	  bgTimeLog *curLog = tline[idx];  
	  if( (isEqual(effRecvTime, curLog->effRecvTime) && curLog->seqno > tLog->seqno)|| isLess(effRecvTime, curLog->effRecvTime)) {
	    high = idx;
	  }
	  else {
	    low = idx+1;
	    if(low == tline.commit) {
	      idx = low;
	      break;
	    }
	  }
	}
#else
/*
	for(idx=0;idx<tline.length();idx++)
	  if(tline[idx]->effRecvTime > tLog->recvTime)
	    break;
*/
	// only need to search to "commit", since only this part is sorted.
	for(idx=startIdx;idx<commit;idx++) {
	  bgTimeLog *curLog = tline[idx];
	  CmiAssert(curLog->effRecvTime>=0.0);
	  if (curLog->effRecvTime > effRecvTime) break;
          if (curLog->effRecvTime==effRecvTime && tLog->seqno<curLog->seqno) break;
	}
#endif

//CmiPrintf("SEE searching: %e found: %d %e len:%d commit:%d\n", effRecvTime, idx, tline[idx]->effRecvTime, tline.length(), tline.commit);
	*index = idx;
}



void updateEffRecvTime(minHeap<bgTimeLog*>* inpQPtr, bgTimeLog *log){
  int i,j;
  if (log) {
    int nChanged=0;
    for (i=0; i<log->forwardDeps.length();i++) {
      bgTimeLog *l = log->forwardDeps[i];
#if INTEGRITY_CHECK
      for(j=0;j<inpQPtr->length();j++) 
        if ((*inpQPtr)[j] == l) break;
      CmiAssert(j<inpQPtr->length());
#endif
      double oldEffRecvTime = l->effRecvTime;
      l->effRecvTime = max(l->getEndOfBackwardDeps(), l->recvTime);
      if (oldEffRecvTime != l->effRecvTime) nChanged++;
    }
    if (nChanged) 
      inpQPtr->buildHeap();
  }
  else {
    for(i=0;i<(inpQPtr->length());i++)  {
      (*inpQPtr)[i]->effRecvTime 
         = max((*inpQPtr)[i]->getEndOfBackwardDeps(), (*inpQPtr)[i]->recvTime);
//      inpQPtr->update(i);
    }
    inpQPtr->buildHeap();
 }
// inpQPtr->integrityCheck(0);
 return;
}

// integrity check
// all bgLog should be already assigned a value for effRecvTime
// all effRecvTime should in ascend order
// all same effRecvTime bgLog is in the ascend order of seqno - determinism
static void BgIntegrityCheck(BgTimeLine &tline)
{
  if (tline.length()) {
    if (tline[0]->effRecvTime < 0) {
       tline[0]->write(stdout);
       CmiAbort("Failed! - effRecvTime < 0\n");
    }
  }
  for (int i=1; i<tline.length(); i++) {
    if (tline[i]->effRecvTime < 0) {
       tline[i]->write(stdout);
       CmiAbort("Failed! - effRecvTime < 0\n");
    }
    else if (isEqual(tline[i]->effRecvTime, tline[i-1]->effRecvTime)) {
      if (tline[i-1]->seqno > tline[i]->seqno) {
        CmiPrintf("%e %e at %d\n", tline[i]->effRecvTime, tline[i-1]->effRecvTime, i);
        tline[i-1]->write(stdout);
        tline[i]->write(stdout);
        CmiAssert(tline[i]->seqno > 0);
        CmiAbort("Failed in seqno!\n");
      }
    }
    else if (isLess(tline[i]->effRecvTime, tline[i-1]->effRecvTime)) {
      CmiPrintf("%e %e at %d\n", tline[i]->effRecvTime, tline[i-1]->effRecvTime, i);
      tline[i-1]->write(stdout);
      tline[i]->write(stdout);
      CmiAbort("Failed!\n");
    }
  }
#if 0
  // make sure no correction message need to be sent
  for (int i=0; i<tline.length(); i++) {
    if (!isZero(tline[i]->startTime - tline[i]->oldStartTime)) {
      CmiPrintf("startT: %e oldStartT:%e\n", tline[i]->startTime, tline[i]->oldStartTime);
      tline[i]->write(stdout);
      if (i-1>=0) tline[i-1]->write(stdout);
      if (i+1<tline.length()) tline[i+1]->write(stdout);
    }
    CmiAssert(isZero(tline[i]->startTime - tline[i]->oldStartTime));
  }
#endif
}
  
int BgGetTimeLineIndexByRecvTime(BgTimeLineRec &tlinerec, bgTimeLog *tlog, int mynode, int tID)
{
  int index;
  BgGetMsgStartTime(tlog, tlinerec, &index);
  return index;
}


/* used by batch processing 2 */
int BgAdjustTimeLineFromIndex(int index, BgTimeLineRec &tlinerec, int mynode)
{
  int i;
  int len = tlinerec.length();
  if (index >= len) return 0;

  // CkQ <bgTimeLog*> insertList;
  minHeap<bgTimeLog*> insertList;
  BgTimeLine &tline = tlinerec.timeline;

  processCount ++;
  if (processCount%1000 == 0) CmiPrintf("[%d:%d] process:%d corrMsg:%d time:%fms curTime:%fms %d:%d\n", CmiMyPe(), mynode, processCount, corrMsgCount, tline[index]->endTime*1000.0, len>0?tlinerec[len-1]->endTime*1000.0:0.0, index, len);

  CmiAssert(index >= tlinerec.startIdx);

  // search for the min recv time
  int commit = tlinerec.commit;
  for (i=commit; i<tline.length(); i++) {
    bgTimeLog *clog = tline[i];
    int idx;
    BgGetMsgStartTime(clog, tlinerec, &idx);
    if (idx < commit) commit = idx;
  }
  index = min(index, commit);

  //Two pass initialization
  //First Pass: Infinite ERT for all but the first
#if 0
  for(i=index;i<len;i++){
    tline[index]->effRecvTime = INVALIDTIME;
    insertList.enq(tline.remove(index));
  }
#else
 
  for(i=index;i<len;i++){
    tline[i]->effRecvTime = INVALIDTIME;
    insertList.add(tline[i]);
  }
  insertList.buildHeap();
  tline.removeFrom(index);	
#endif

  //Second Pass
  updateEffRecvTime(&insertList, NULL);

  //Move entries from insertList to tline
  bgTimeLog* temp = NULL;
  while(!insertList.isEmpty()){

    //   temp = getLeastERTLog(tlinerec, &insertList, temp); 
    temp = insertList.deq();
    if(temp->effRecvTime == INVALIDTIME)
      CmiAbort("\nFATAL ERROR:Tried to insert infinite ERT value into tline!!!!\n\n");
    
    temp->oldStartTime = temp->startTime;

    temp->startTime = temp->effRecvTime;
    if (tline.length()>0) {
      if (tline[tline.length()-1]->endTime > temp->startTime)
	temp->startTime = tline[tline.length()-1]->endTime;
    }
    temp->endTime = temp->execTime + temp->startTime;
    CmiAssert(temp->endTime >=  temp->startTime );

    if (temp->endTime < .0) CmiAbort("Should not happen.\n");
    
    tline.enq(temp);
    updateEffRecvTime(&insertList, temp);

  }

  // integrity check
#if INTEGRITY_CHECK
  BgIntegrityCheck(tline);
#endif

  
#if LIMITED_SEND
  BgSendPendingCorrections(tlinerec,mynode);
#endif

  // whole timeline is sorted
  tlinerec.commit = tline.length();
  return 1;
}



//Real adjusting of the timeline
int BgAdjustTimeLineByIndex(int oldIdx, double tAdjustAbs, BgTimeLineRec &tlinerec, int mynode){

  int idx,i,newIdx,commit;;
  double startTime=0;
  //CkQ <bgTimeLog*> insertList;
  bgTimeLog *tlog, *clog, *temp = NULL;
  BgTimeLine &tline = tlinerec.timeline;
  int len = tline.length();

  if(len==1) {
    tlinerec.commit = 1;
    return -1;
  }

#if 0
double t = CmiWallTimer();
CmiPrintf("BgAdjustTimeLineByIndex BEGIN\n");
#endif

  CmiAssert(tlinerec.commit>=1);
  commit = tlinerec.commit; 

  // search for the min recv time
  for (i=commit; i<len; i++) {
    clog = tline[i];
    BgGetMsgStartTime(clog, tlinerec, &idx);
    if (idx < commit) commit = idx;
  }
  idx = min(commit, oldIdx);
  tlog = tlinerec[oldIdx];
  tlog->recvTime = tAdjustAbs;
  BgGetMsgStartTime(tlog, tlinerec, &newIdx);
  idx = min(idx,newIdx);

  if (idx == len) return -1;

//  processCount ++;
//  if (processCount%1000 == 0) CmiPrintf("[%d:%d] BgAdjustTimeLineByIndex: procCount:%d %f %d:%d\n", CmiMyPe(), mynode, processCount, tline[oldIdx]->endTime*1000.0, idx, len);

  minHeap<bgTimeLog*> insertList(len-idx);

  //Two pass initialization
  //First Pass: Infinite ERT for all but the first
#if 0
  for(int i=idx;i<len;i++){
    if(i!=idx)
      tline[idx]->effRecvTime = INVALIDTIME;
    insertList.enq(tline.remove(idx));
  }
#else
  for(i=idx;i<len;i++){
    if(i!=idx)
      tline[i]->effRecvTime = INVALIDTIME;
    insertList.add(tline[i]);
  }
  tline.removeFrom(idx);	
  insertList.buildHeap();
#endif

#if 0
CmiPrintf("BgAdjustTimeLineByIndex FIRST PASS: %f\n", CmiWallTimer()-t);
t = CmiWallTimer();
#endif

  //Second Pass
  updateEffRecvTime(&insertList, NULL);

  //Move entries from insertList to tline
  while(!insertList.isEmpty()){

    //    temp = getLeastERTLog(tlinerec, &insertList, temp);
    temp = insertList.deq();
    
    if(temp->effRecvTime == INVALIDTIME)
      CmiAbort("\nFATAL ERROR:Tried to insert infinite ERT value into tline!!!!\n\n");
    
    temp->oldStartTime = temp->startTime;

    temp->startTime = temp->effRecvTime;
    if (tline.length()>0) {
      if (tline[tline.length()-1]->endTime > temp->startTime)
	temp->startTime = tline[tline.length()-1]->endTime;
    }
    temp->endTime = temp->execTime + temp->startTime;
    CmiAssert(temp->endTime >=  temp->startTime );

    if (temp->endTime < .0) CmiAbort("Should not happen.\n");
    
    tlinerec.enq(temp, 0);
    updateEffRecvTime(&insertList, temp);
  }

  // whole timeline is sorted
  tlinerec.commit = tline.length();
#if 0
CmiPrintf("BgAdjustTimeLineByIndex END %d %d %f\n", idx, len, CmiWallTimer()-t);
#endif
  return idx;
}



#if 0

void BgBeginCorrection(BgTimeLineRec &tlinerec, int mynode, int tid)
{
#if LIMITED_SEND
  return;
#endif

  BgTimeLine &tline = tlinerec.timeline;
  for(int i=0;i<tline.length();i++) {
    bgTimeLog *log = tline[i];
    log->oldStartTime = log->startTime;
    //  log->oldEndTime = log->endTime;
  }
}

#endif

//Send correction msgs
void BgFinishCorrection(BgTimeLineRec &tlinerec, int mynode, int tid, int idx, int sendImmediately)
{
  BgTimeLine &tline = tlinerec.timeline;
#if !LIMITED_SEND
  int c=0;
  for(int i=idx;i<tline.length();i++) {
    bgTimeLog *log = tline[i];
    int send=sendImmediately;
#if DELAY_SEND
     if(!send) {
       send = 1;
       for(int j=0;j<tlinerec.sendingLogs.length();j++)
         if(log == tlinerec.sendingLogs[j])  { send = 0; }
     }
#endif
    c += log->adjustTimeLog(log->startTime-log->oldStartTime,tline,mynode,send);
  }
//if (c) CmiPrintf("[%d] BgFinishCorrection send %d corr msgs\n", CmiMyPe(), c);
#else
  BgSendPendingCorrections(tlinerec,mynode);
#endif

}

void BgSendPendingCorrections(BgTimeLineRec &tlinerec, int mynode)
{
  int count = 0;
  int sendIdx =  tlinerec.correctSendIdx;
  int& newSendIdx =  tlinerec.correctSendIdx;
  BgTimeLine &tline = tlinerec.timeline;

#if THROTTLE_WORK
  while (1) 
#else
  while(count < CORRECTSENDLEN && newSendIdx < tlinerec.length())
#endif
  {
    double diff = tline[newSendIdx]->startTime - tline[newSendIdx]->oldStartTime;
#if THROTTLE_WORK
    if(tline[newSendIdx]->startTime>gvt+LEASH) break;
#endif
    if(!isZero(diff)){
      tline[newSendIdx]->adjustTimeLog(diff,tline,mynode,1);
      count ++;
    }
    newSendIdx++;
  }

  if(sendIdx != newSendIdx){
    if(newSendIdx == tlinerec.length())
      CQdProcess(CpvAccess(cQdState),1);
    //CmiPrintf("Created:%d Processed:%d\n",CQdGetCreated(CpvAccess(cQdState)), CQdGetProcessed(CpvAccess(cQdState)));
  }

}

#if USE_MULTISEND
void BgSendBufferedCorrMsgs()
{
  int i, j;
  for (i=0; i<CmiNumPes(); i++) {
    int len = corrMsgBucket[i].size();
    if (len == 0) continue;
    if (len < 100) {
      for (j=0; j<len; j++)
        CmiSyncSendAndFree(i,sizeof(bgCorrectionMsg),corrMsgBucket[i][j]);
    }
    else {
      int *sizes = new int[len];
      for (j=0; j<len; j++) sizes[j] = sizeof(bgCorrectionMsg);
      CmiMultipleSend(i, len, sizes, corrMsgBucket[i].getVec());
      for (j=0; j<len; j++)  CmiFree(corrMsgBucket[i][j]);
      delete [] sizes;
    }
    corrMsgBucket[i].freeBlock();
  }
}
#endif

// move the last entry in BgTimeLine to it's proper position
int BgAdjustTimeLineInsert(BgTimeLineRec &tlinerec)
{
  int mynode = tMYNODEID;
  int len = tlinerec.length();

  if (len <= tlinerec.startIdx) return 0;

  BGSTATE(2,"BgAdjustTimeLineInsert {");
//  BgBeginCorrection(tlinerec, mynode, -1);
  int minIdx = BgAdjustTimeLineByIndex(len-1,tlinerec[len-1]->recvTime,tlinerec, mynode);

#if !LIMITED_SEND
  int send;
  if (minIdx != -1) {
#if DELAY_SEND
    send = 0;
#else
    send = 1;
#endif
    BgFinishCorrection(tlinerec, mynode, -1, minIdx, send);
  }
#endif
#if USE_MULTISEND
  BgSendBufferedCorrMsgs();
#endif
  BGSTATE(2,"} BgAdjustTimeLineInsert");
  return minIdx;
}

static bgTimeLog *BgGetTimeLogOnThread(BgTimeLineRec &tline, int srcnode, int msgID, int *index)
{
  int idxOld = tline.length()-1;
  while (idxOld >= 0)  {
    if (tline[idxOld]->msgID == msgID && tline[idxOld]->srcnode == srcnode) break;
    idxOld--;
  }

  *index = idxOld;
  if (idxOld == -1) return NULL;
  return tline[idxOld];
}

bgTimeLog *BgGetTimeLog(BgTimeLineRec *tline, CmiInt2 tID, int srcnode, int msgID, int *index)
{
  if (tID == ANYTHREAD) {
    for (int i=0; i<cva(numWth); i++) {
      bgTimeLog *tlog = BgGetTimeLogOnThread(tline[i], srcnode, msgID, index);
      if (tlog) return tlog;
    }
    return NULL;
  }
  else {
    return BgGetTimeLogOnThread(tline[tID], srcnode, msgID, index);
  }
}

int BgAdjustTimeLineForward(int srcnode, int msgID, double tAdjustAbs, BgTimeLineRec &tline, int mynode, int tid)
{
  /* ASSUMPTION: BgAdjustTimeLineForward is called only if necessary */
  /* ASSUMPTION: no error testing needed */
  BGSTATE4(2,"BgAdjustTimeLineForward on bgnode:%d tid:%d srcnode:%d tAdjustAbs:%f{", mynode,tid,srcnode,tAdjustAbs);

  int idxOld = tline.length()-1;
  while (idxOld >= 0)  {
    if (tline[idxOld]->msgID == msgID && tline[idxOld]->srcnode == srcnode) break;
    idxOld--;
  }

//CmiPrintf("BgAdjustTimeLineForward finding pe:%d mid:%d idx:%d\n", srcPe, msgID, idxOld);
  if (idxOld == -1) return 0;
  int status = BgAdjustTimeLineByIndex(idxOld, tAdjustAbs, tline, mynode);

  BGSTATE(2,"} BgAdjustTimeLineForward");
  return status;
}

void BgPrintThreadTimeLine(int pe, int th, BgTimeLine &tline)
{
  for (int i=0; i<tline.length(); i++)
    tline[i]->print(pe, th);
}

void BgWriteThreadTimeLine(char **argv, int x, int y, int z, int th, BgTimeLine &tline)
{
  char *fname = (char *)malloc(strlen(argv[0])+100);
  sprintf(fname, "%s-%d-%d-%d.%d.log", argv[0], x,y,z,th);
  FILE *fp = fopen(fname, "w+");
  CmiAssert(fp!=NULL);
  for (int i=0; i<tline.length(); i++)
    tline[i]->write(fp);
  fclose(fp);
  free(fname);
}

/*****************************************************************************
		TimeLog correction with trace projection
*****************************************************************************/

void bgAddProjEvent(void *data, double t, bgEventCallBackFn fn, void *uPtr, int eType)
{
  if (!genTimeLog) return;
  CmiAssert(tTHREADTYPE == WORK_THREAD);

  BgTimeLineRec &tlinerec = tTIMELINEREC;
  BgTimeLine &tline = tlinerec.timeline;
  CmiAssert(tline.length() > 0);
  bgTimeLog *tlog = tline[tline.length()-1];
  // make sure this time log entry is not closed
  //if ((tlog->endTime == 0.0) ||(t <= tlog->endTime)) 
  // if the last log is closed, this is a standalone event
  if (tlog->endTime != 0.0) {
    // ignore standalone event
    return;
    double endT = tlog->endTime;
    tlog = new bgTimeLog(-1, "standalone", endT, endT);	
    tlog->recvTime = tlog->effRecvTime = endT;
    tlinerec.enq(tlog, 0);
  }
  tlog->addEvent(data, t, fn, uPtr, eType);
}


// trace projections callback update projections with new timestamp after
// timing correction
void bgUpdateProj(int eType)
{
  BgTimeLine &tline = tTIMELINE;
  for (int i=0; i< tline.length(); i++) {
      tline[i]->updateEvents(eType);
  }
}

/******************************************************************************
               Timing Correction on Timeline
******************************************************************************/

extern int updateRealMsgs(bgCorrectionMsg *cm, int nodeidx);

// must be called on each processor
extern "C"
void BgStartCorrection()
{

    BgTimeLineRec *tlinerec = cva(nodeinfo)[tMYNODEID].timelines;
    if(tlinerec->startCorrFlag==0){
      tlinerec->setStartIdx();
      tlinerec->startCorrFlag = 1;
    }
}

// handle one correction mesg
static inline int handleCorrectionMsg(int mynode, BgTimeLineRec *logs, bgCorrectionMsg *m, int delmsg)
{
	CmiInt2 tID = m->tID;
	if (tID == ANYTHREAD) {
	  int found = 0;
	  for (tID=0; tID<cva(numWth); tID++) {
	    // search for the msg
            BgTimeLine &tline = logs[tID].timeline;	
	    for (int j=0; j<tline.length(); j++)
	      if (tline[j]->msgID==m->msgID && tline[j]->srcnode==m->srcNode) {
		  found = 1; break; 
	      }
            if (found) break;
	  }
	  if (!found) {
//	    CmiPrintf("Correction message arrived early. \n");
		return 0;
	  }
	}
	//CmiPrintf("tAdjust: %f\n", m->tAdjust);
	BgTimeLineRec &tlinerec = logs[tID];
	if (BgAdjustTimeLineForward(m->srcNode, m->msgID, m->tAdjust, tlinerec, mynode, tID) == 0) {
	    // if delayCheckFlag == 0, it is ok to get rid of this msg
	    if (!programExit) return 0;
	}
#if 0
        // update thread timer
        cva(nodeinfo)[mynode].threadinfo[tID]->currTime = tlinerec[tlinerec.length()-1]->endTime;
#endif
	// msg has been processed, delete it if ask to
	if (delmsg) CmiFree(m);
	return 1;
}

/* no need to consider the case when tID is ANYTHREAD */
static inline int batchHandleCorrectionMsg(int mynode, BgTimeLineRec *tlinerecs, CmiInt2 tID, bgCorrectionQ &cmsg, int *mIdx)
{
  int i;
  int worked = 0;
  CmiAssert(tID != ANYTHREAD);
  BgTimeLineRec &tlinerec = tlinerecs[tID];
  bgCorrectionQ tmpQ;
  int len = cmsg.length();
  int minIdx = -1;
  double minTime = INVALIDTIME;
  bgTimeLog *minLog = NULL;
  tlinerec.minCorrection = INVALIDTIME;
  BGSTATE2(2,"batchHandleCorrectionMsg (bgnode:%d len:%d) {", mynode, len);
  for (i=0; i<len; i++) {
    bgCorrectionMsg *cm = cmsg.deq();
    if (cm->tAdjust >= 0.) {
      int oldIdx;
      bgTimeLog *tlog = BgGetTimeLogOnThread(tlinerec, cm->srcNode, cm->msgID, &oldIdx);
      if (tlog==NULL) {
//	if (!programExit) {             // HACK ?
	if (1) {
	  tmpQ.enq(cm);
          if (cm->tAdjust< tlinerec.minCorrection) tlinerec.minCorrection=cm->tAdjust;
          continue;
        }
      }
      else {
//        if (cm->tAdjust< tlinerec.minCorrection) tlinerec.minCorrection=cm->tAdjust;
        if (tlog->doCorrect == 1) {
	double oldRecvTime = tlog->effRecvTime;
        tlog->recvTime = cm->tAdjust;
        double endOfDeps = tlog->getEndOfBackwardDeps();
//        double effRecvTime  = max(tlog->recvTime, endOfDeps);
        double effRecvTime  = tlog->recvTime;
	if (endOfDeps != INVALIDTIME) effRecvTime =  max(effRecvTime,endOfDeps);
	effRecvTime = min(oldRecvTime, effRecvTime);
	if (effRecvTime != INVALIDTIME && (isLess(effRecvTime,minTime) ||
            isEqual(effRecvTime, minTime) && tlog->seqno < minLog->seqno)) {
	  minTime = effRecvTime; 
	  minLog = tlog;
        }
        worked = 1;
        }
      }
      // counter for processed correction message
      stateCounters.corrMsgProcCnt++;
    }
    CQdProcess(CpvAccess(cQdState), 1);
    CmiFree(cm);
  }  /* for */
  if (minLog != NULL) {
    minIdx = BgGetIndexFromTime(minTime, minLog->seqno, tlinerec);
    minIdx = min(minIdx, tlinerec.commit);
  }

  int qlen = tmpQ.length();
  for (i=0; i<qlen; i++) cmsg.enq(tmpQ.deq());
/*
if (cmsg.length()>0)
CmiPrintf("QUEUE LEN %d: %d\n", mynode, cmsg.length());
*/

  if (minIdx != -1) {
#if LIMITED_SEND
    int& sendIdx = tlinerec.correctSendIdx ;
    int newSendIdx;
    newSendIdx = min(sendIdx,minIdx);
    if(sendIdx != newSendIdx){
      if(sendIdx == tlinerec.length())
	CQdCreate(CpvAccess(cQdState),1);
      sendIdx = newSendIdx;
    }
#endif
    if (BgAdjustTimeLineFromIndex(minIdx, tlinerec, mynode)) {
#if 0
      threadInfo *threadinfo = cva(nodeinfo)[mynode].threadinfo[0];
      threadinfo->currTime = tlinerec[tlinerec.length()-1]->endTime;
#endif
    }
  }
  BGSTATE2(2,"batchHandleCorrectionMsg (bgnode:%d len:%d) {", mynode, len);
//  tlinerec.minCorrection = INVALIDTIME;
  // return the index of least affected log
  *mIdx = minIdx;
  return 1;
}


// process all the correction messages for one node
// nodeidx is local to the current real processor
void processCorrectionMsg(int nodeidx)
{
    int tID=0;
    BgTimeLineRec *tlinerec = cva(nodeinfo)[nodeidx].timelines;
    bgCorrectionQ &cmsg = cva(nodeinfo)[nodeidx].cmsg;
    int len = cmsg.length();
    int worked = 0;
    int minIdx = -1;
    if (len ==0) goto end;

    BGSTATE1(2,"processCorrectionMsg (len:%d) {", len);
//CmiPrintf("[%d:%d] processCorrectionMsg len:%d\n", CmiMyPe(), nodeidx, len);

    for (tID=0; tID<cva(numWth); tID++) {
      worked = batchHandleCorrectionMsg(nodeidx, tlinerec, tID, cmsg, &minIdx);
      if (worked && minIdx !=-1)
        BgFinishCorrection(tlinerec[tID], nodeidx, tID, minIdx);
    }

    BGSTATE(2,"} processCorrectionMsg ");
end:
#if LIMITED_SEND
    if(worked == 0)
      for (tID=0; tID<cva(numWth); tID++)
        BgSendPendingCorrections(tlinerec[tID], nodeidx);
#endif

    return;
}

void processBufferCorrectionMsgs(void *ignored)
{
//CmiPrintf("[%d] processBufferCorrectionMsgs called\n", CmiMyPe());
  for (int i=0; i<BgNodeSize(); i++)
    processCorrectionMsg(i);
#if USE_MULTISEND
  BgSendBufferedCorrMsgs();
#endif
#if !THROTTLE_WORK
  CcdCallFnAfter(processBufferCorrectionMsgs,NULL,CHECK_INTERVAL);
#endif
}

// enqueue one correction message, invalidate old ones too.
// nodeidx is local to the current real processor
static void enqueueCorrectionMsg(int nodeidx, bgCorrectionMsg* m)
{
    CmiAssert(nodeidx >= 0);
    bgCorrectionQ &cmsg = cva(nodeinfo)[nodeidx].cmsg;
    BgTimeLineRec &tlinerec = cva(nodeinfo)[nodeidx].timelines[0];

    // ignore the correction message if it is before a specific time
    if(tlinerec.startCorrFlag == 0){
      CmiFree(m);
      return;
    }

    // if there is already a same destination correction message in the queue
    // remove it, since it is now obsolete
    int removed = 0;
    int msgLen = cmsg.length();
    for (int i=0; i<msgLen; i++) {
      bgCorrectionMsg* cm = cmsg[i];
      if (cm->msgID == m->msgID && cm->srcNode == m->srcNode && cm->tID == m->tID) {
        cm->tAdjust = m->tAdjust;
	removed = 1;
        stateCounters.corrMsgCCCnt++;
	//Guna
//	cmsg.update(i);
	break;
      }
    }
    // correction cancel real
    if (updateRealMsgs(m, nodeidx)) removed = 1;
    if (removed)  {
      CmiFree(m);
    }
    else {
#if 1
      CQdCreate(CpvAccess(cQdState), 1);
#endif
      cmsg.enq(m);
      stateCounters.corrMsgEnqCnt++;
    }
    corrMsgCount++;
    if (m->tAdjust<tlinerec.minCorrection) tlinerec.minCorrection=m->tAdjust;
}

// Converse handler for correction msgs
void bgCorrectionFunc(char *msg)
{
    int i, j;
    static double lastCheck = .0;

    bgCorrectionMsg* m = (bgCorrectionMsg*)msg;
    int nodeidx = m->destNode;
    CmiInt2 tID = m->tID;

    if (nodeidx < 0) {
      // copy correction msg to each thread of each node
      CmiAssert(nodeidx == -1);
      CmiAssert(tID == ANYTHREAD);
      for (i=0; i<BgNodeSize(); i++) {
	for (tID=0; tID<cva(numWth); tID++) {
	  bgCorrectionMsg *newMsg = (bgCorrectionMsg*)CmiCopyMsg(msg, sizeof(bgCorrectionMsg));
	  newMsg->destNode = nodeInfo::Local2Global(i);   // global node seqno
	  newMsg->tID = tID;
          enqueueCorrectionMsg(i, newMsg);
	}
      }
      CmiFree(m);
    }
    else {
      CmiAssert(nodeidx>=0);
      nodeidx = nodeInfo::Global2Local(nodeidx);	
      enqueueCorrectionMsg(nodeidx, m);
    }

#if DELAY_CHECK
    // when delay checking, just return after enqueue the corr msg.
    return;
#endif
#if !THROTTLE_WORK
    //only correct message every 0.02s, otherwise msgs are queued
    if (CmiWallTimer() - lastCheck < 0.02 && delayCheckFlag ) {
      return; 
    }
    lastCheck = CmiWallTimer();
#endif

    // start processing
    if (nodeidx < 0) {
      for (i=0; i<BgNodeSize(); i++)
        processCorrectionMsg(i);
    }
    else {
      processCorrectionMsg(nodeidx);
    }
}

/******************************************************************************
               implement heart beat
TODO:
1. broadcast with ANYTHREAD, not just enq to every thread
2. delay check for correction message should be very short.
******************************************************************************/
class HeartBeatMsg {
  char core[CmiBlueGeneMsgHeaderSizeBytes];
public:
//  int rcounter, ccounter, ecounter, ncounter;
  StateCounters counters;
  int newInterval;
  double  gvt;
};

#define TREEWIDTH     4

static int nChildren = 0;
static int children[TREEWIDTH];
static int parent = -1;

#define HEARTBEAT_INTERVAL   30		/* 30us walltime */
#define HEARTBEAT_MIN        5
#define HEARTBEAT_MAX        100
#define GVT_INC       	     0.00005         /* 50us bg time */

static int hearbeatInterval = HEARTBEAT_INTERVAL;

void recvGVT(char *msg);

static double findLeastTime()
{
  // find the least of all buffered message timer and all correction timer
  double minT = INVALIDTIME;
  int i;

  for (int nodeidx=0; nodeidx<cva(numNodes); nodeidx++) {
    BgTimeLineRec *tlinerecs = cva(nodeinfo)[nodeidx].timelines;
    threadInfo **threadinfos = cva(nodeinfo)[nodeidx].threadinfo;
    ckMsgQueue* affinityQs = cva(nodeinfo)[nodeidx].affinityQ;
    ckMsgQueue& nodeQ = cva(nodeinfo)[nodeidx].nodeQ;
    bgCorrectionQ &cmsgQ = cva(nodeinfo)[nodeidx].cmsg;
    CthThread *threadTable = cva(nodeinfo)[nodeidx].threadTable;

    //min in correction msg Q  
    for (i=0; i<cmsgQ.length(); i++){
      bgCorrectionMsg* cmsg = cmsgQ[i]; 
      if (cmsg->tAdjust>.0){
	int index;
	bgTimeLog* tlog;
	//Comapre startTime
	tlog = BgGetTimeLog(tlinerecs, cmsg->tID, cmsg->srcNode, cmsg->msgID, &index);
	if(tlog != NULL){
	  if(tlog->startTime < minT)
	    minT = tlog->startTime;
	}
	//Compare tAdjust
	if(cmsg->tAdjust < minT)
	  minT = cmsg->tAdjust;
      }
    }
    //min in affinityQ
    for(i=0;i<cva(numWth);i++){
	ckMsgQueue &aQ = affinityQs[i];
#if 0
	if (aQ.length() && deadlock)  {
  	  double nextT = CmiBgMsgRecvTime(aQ[0]);
  	  unsigned int prio = (unsigned int)(nextT*1e7)+1;
  	  CthAwakenPrio(threadTable[i], CQS_QUEUEING_IFIFO, sizeof(int), &prio);
	}
#endif
	for(int j=0;j<aQ.length();j++){
          double t = CmiBgMsgRecvTime(aQ[j]);
	  if(t < minT) { minT = t; }
	}
    }
    //min in nodeQ
    for(i=0;i<nodeQ.length();i++){
        if(CmiBgMsgRecvTime(nodeQ[i])< minT)
	    minT = CmiBgMsgRecvTime(nodeQ[i]);
    }
  }
  if(minCorrectTimestamp < minT)
    minT = minCorrectTimestamp;
  minCorrectTimestamp = INVALIDTIME;

  return minT;
}

static void sendHeartbeat(double t, StateCounters &counters)
{
  HeartBeatMsg *msg = (HeartBeatMsg *)CmiAlloc(sizeof(HeartBeatMsg));
  msg->gvt = t;
  msg->counters = counters;
  CmiSetHandler(msg, cva(heartbeatHandler));
  CQdCreate(CpvAccess(cQdState), -1);
  if (parent == -1) {
//    CmiPrintf("HEART BEAT %f Count:%d %d %d %d ival:%d %d at %f\n", gvt, msg->counters.realMsgProcCnt,msg->counters.corrMsgProcCnt,msg->counters.corrMsgEnqCnt, msg->counters.corrMsgCRCnt, hearbeatInterval, programExit,CmiWallTimer());
    CmiSetHandler(msg, cva(heartbeatHandler));
    CmiSyncSendAndFree(0, sizeof(HeartBeatMsg), msg);
  }
  else
    CmiSyncSendAndFree(parent, sizeof(HeartBeatMsg), msg);
}

//Only called for the leaf
static void sendHeartbeatFunc()
{
  double local_gvt = findLeastTime();
  sendHeartbeat(local_gvt, stateCounters);
}

//Only called for inner-node
void heartbeatHandlerFunc(char *msg)
{
  CQdProcess(CpvAccess(cQdState), -1);
  static int reported = 0;
  static double localGvt = INVALIDTIME;
  static double lastGvt = INVALIDTIME;
  static StateCounters  oldCount, newCount;
  reported ++;
  HeartBeatMsg *m = (HeartBeatMsg*)msg;
  localGvt = min(localGvt, m->gvt);
  newCount.add(m->counters);
  CmiFree(msg);
  if (reported == nChildren || (nChildren==0 && parent==-1)) {
    localGvt = min(localGvt, findLeastTime());
    newCount.add(stateCounters);
    if (parent != -1) {
      sendHeartbeat(localGvt, newCount);
    }
    else {
      // I am root: broadcast
//CmiPrintf("lastGvt:%f localGvt:%f\n", lastGvt, localGvt);
      double old_gvt = gvt;
      deadlock = 0;
      if (oldCount == newCount) {
        if (localGvt == INVALIDTIME) gvt += GVT_INC;
        else {
	  gvt = localGvt;
	  if (lastGvt == gvt) {
		deadlock = 1;
		CmiPrintf("DEADLOCK detected!\n");
	  }
	}
      }
      else if (localGvt != INVALIDTIME) {
        lastGvt=localGvt;
//      gvt = max(localGvt, gvt);
        gvt = localGvt;
      }
      // compute the new heart beat interval
      {
      static int oldProcessed = -1;
      int processed = newCount.actionCount() - oldCount.actionCount();
      if (oldProcessed != -1) {
        if (processed == 0)    hearbeatInterval += 5;
        //if (processed < 10)    hearbeatInterval += 5;
        else if (processed < oldProcessed*0.6)  hearbeatInterval+=1;
        else if (processed > oldProcessed*1.2)  hearbeatInterval-=2;
      }
      hearbeatInterval=min(hearbeatInterval, HEARTBEAT_MAX);
      hearbeatInterval=max(hearbeatInterval, HEARTBEAT_MIN);
      oldProcessed = processed;
      }
      CmiPrintf("HEART BEAT %f local:%f Count:r%d p%d e%d cr%d cc%d rc%d ival:%d %d at %f\n", gvt, localGvt==INVALIDTIME?-1:localGvt, newCount.realMsgProcCnt,newCount.corrMsgProcCnt,newCount.corrMsgEnqCnt,newCount.corrMsgCRCnt,newCount.corrMsgCCCnt,newCount.corrMsgRCCnt,hearbeatInterval, programExit,CmiWallTimer());
      oldCount = newCount; 

      HeartBeatMsg *msg = (HeartBeatMsg *)CmiAlloc(sizeof(HeartBeatMsg));
      msg->gvt = gvt;
      msg->newInterval = hearbeatInterval;
      CmiSetHandler(msg, cva(heartbeatBcastHandler));
      recvGVT((char*)msg);
    }
    localGvt = INVALIDTIME;
    newCount.clear();
    reported = 0;
  }
}

// 
void recvGVT(char *msg)
{
  if (programExit == 2) return;
  HeartBeatMsg *m = (HeartBeatMsg*)msg;
  // update new gvt and heartbeat interval
  gvt = m->gvt;
  hearbeatInterval = m->newInterval;
//CmiPrintf("[%d] get gvt: %f \n", CmiMyPe(), gvt);
  if (nChildren) {
    CQdCreate(CpvAccess(cQdState), -nChildren);
    for (int i=0; i<nChildren-1; i++)
      CmiSyncSend(children[i], sizeof(HeartBeatMsg), msg);
    CmiSyncSendAndFree(children[nChildren-1], sizeof(HeartBeatMsg), msg);
  }
  else {
    CmiFree(msg);
    CcdCallFnAfter((CcdVoidFn)sendHeartbeatFunc,NULL,hearbeatInterval);
  }
  processBufferCorrectionMsgs(NULL);
}

void heartbeatBcastHandlerFunc(char *msg)
{
  CQdProcess(CpvAccess(cQdState), -1);
  recvGVT(msg);
}

void initHeartbeat()
{
  if (!correctTimeLog) return;

#if THROTTLE_WORK
  int mype = CmiMyPe();
  int nPes = CmiNumPes();
  if (mype>0) parent = (mype-1)/TREEWIDTH;
  for (int i=0; i<TREEWIDTH; i++) {
    children[i] = mype*TREEWIDTH+i+1;
    if (children[i] < nPes) nChildren ++;
  }
  if (nChildren == 0) {
    CcdCallFnAfter((CcdVoidFn)sendHeartbeatFunc,NULL,hearbeatInterval);
  }
  CpvInitialize(int, heartbeatHandler);
  cva(heartbeatHandler) = CmiRegisterHandler((CmiHandler)heartbeatHandlerFunc);
  CpvInitialize(int, heartbeatBcastHandler);
  cva(heartbeatBcastHandler) = CmiRegisterHandler((CmiHandler)heartbeatBcastHandlerFunc);
#endif
}




