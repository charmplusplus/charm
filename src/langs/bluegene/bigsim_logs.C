#include "blue.h"
#include "blue_impl.h"

#include "blue_logs.h"

int genTimeLog = 0;			// was 1 for guna 's seq correction
int correctTimeLog = 0;
int bgcorroff = 0;
extern BgTimeLineRec* currTline;
extern int currTlineIdx;

double BgGetCurTime()
{
  return tCURRTIME;
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

bgMsgEntry::bgMsgEntry(int dest,int mID,int tid,double rTime){

  msgID = mID;
  dstPe=dest;
  tID=tid;
  recvTime = rTime;
  sendMsg=NULL;
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

void BgTimeLineRec::logEntryStart(int handler, char *m) {
  if (!genTimeLog) return;
  if (tTHREADTYPE == WORK_THREAD) {
      CmiAssert(bgCurLog == NULL);
      bgCurLog = new bgTimeLog(handler, m);
      enq(bgCurLog, 1);
  }
}

void BgTimeLineRec::logEntryCommit() {
  if (!genTimeLog) return;
  if (tTHREADTYPE == WORK_THREAD) 
  {
      if(bgSkipEndFlag == 0)
	timeline[timeline.length()-1]->closeLog();
      else
        bgSkipEndFlag=0;
      if (correctTimeLog) {
	BgAdjustTimeLineInsert(*this);
	if (timeline.length()) 
          tCURRTIME = timeline[timeline.length()-1]->endTime;
	clearSendingLogs();
      }
      bgCurLog = NULL;
  }
}

void BgTimeLineRec::logEntryInsert(bgTimeLog* log)
{
  CmiAssert(bgCurLog == NULL);
  if(timeline[timeline.length()-1]->endTime == 0.0)
    CmiPrintf("\nERROR tried to insert %s after %s\n",log->name,timeline[timeline.length()-1]->name);
  enq(log, 1);
}

void BgTimeLineRec::logEntryStart(bgTimeLog* log)
{
  logEntryInsert(log);
  bgCurLog = log;
}

void BgTimeLineRec::logEntryClose() {
  if (!genTimeLog) return;
  if (tTHREADTYPE == WORK_THREAD) 
  {
    bgTimeLog *lastlog = timeline[timeline.length()-1];
    CmiAssert(bgCurLog == lastlog);
    lastlog->closeLog();
    bgCurLog = NULL;
  }
}

void BgTimeLineRec::logEntrySplit()
{
  if (!genTimeLog) return;
  CmiAssert(tTHREADTYPE == WORK_THREAD);
  CmiAssert(bgCurLog != NULL);
  bgTimeLog *rootLog = bgCurLog;
  if (bgSkipEndFlag == 0) {
    logEntryClose();
  }
  else {
    bgCurLog=NULL;
    bgSkipEndFlag = 0;
  }

  // make up a new bglog to start, setting up dependencies.
  bgTimeLog *newLog = new bgTimeLog(-1, "broadcast", BgGetTime());
  newLog->addBackwardDep(rootLog);
  logEntryInsert(newLog);
  bgCurLog = newLog;
}

int bgTimeLog::bDepExists(bgTimeLog* log){

  for(int i =0;i<backwardDeps.length();i++)
    if(backwardDeps[i] == log)
      return 1;
  return 0;
}

void bgTimeLog::pup(PUP::er &p){
    int l=0,idx;
    p|ep; 
    p|seqno; p|srcnode;p|msgID;
    p|recvTime; p|effRecvTime;p|startTime; p|execTime; p|endTime;p|index;p(name,20);
    
    /*    if(p.isUnpacking())
      CmiPrintf("Puping: %d %d %d %d %e %e %e %e %e %s\n",ep,seqno,srcnode,msgID,recvTime,effRecvTime,startTime,execTime,endTime,name);
    */

    if(p.isUnpacking()){
      threadNum = currTlineIdx;
    }

    double rTime;int destNode,msgID;CmiUInt2 tID;
    if(!p.isUnpacking()){
      l=msgs.length();
    }
    p|l;

    for(int i=0;i<l;i++){
      if(p.isUnpacking()){
	p|destNode;p|msgID;p|tID;p|rTime;
	msgs.push_back(new bgMsgEntry(destNode,msgID,tID,rTime));
      }
      else{
	p|msgs[i]->dstPe;
	p|msgs[i]->msgID;
	p|msgs[i]->tID;
	p|msgs[i]->recvTime;
     }
    }

    if(!p.isUnpacking())
      l = backwardDeps.length();
    p|l;    

    for(int i=0;i<l;i++){
      if(p.isUnpacking()){
	p|idx;
	addBackwardDep(currTline->timeline[idx]);
      }
      else{
	p|backwardDeps[i]->index;
      }
    }
 
   if(!p.isUnpacking()){
      l=forwardDeps.length();
    }
    p|l;

    for(int i=0;i<l;i++){ 
      if(p.isUnpacking())
	p|idx;
      else
	p|forwardDeps[i]->index;
      
    }

}


