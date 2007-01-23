#include "blue.h"
#include "blue_impl.h"

#include "bigsim_logs.h"

/*
 ChangeLog
 version 2 
     * add objId
*/

int bglog_version = 2;

int genTimeLog = 0;			// was 1 for guna 's seq correction
int correctTimeLog = 0;
int schedule_flag = 0;
int bgverbose = 0;
int bgcorroff = 0;

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

// null timer for using blue_logs as a sequential library
static double nullTimer() { return 0.; }

// bluegene timer call
double (*timerFunc) (void) = nullTimer;

extern "C" void BgGenerateLogs()
{
  genTimeLog = 1;;
}

// dstNode is the dest bg node, can be -1
BgMsgEntry::BgMsgEntry(char *msg, int dstNode, int tid, int local, int g)
{
  msgID = CmiBgMsgID(msg);
  sendTime = timerFunc();
  recvTime = CmiBgMsgRecvTime(msg);
  dstPe = dstNode;
  tID = tid;                   // CmiBgMsgThreadID(msg);
  msgsize = CmiBgMsgLength(msg);
  group = g;
  CmiAssert(msgsize > 0);
#if DELAY_SEND
  sendMsg = NULL;
  if (!local && correctTimeLog) sendMsg = msg;
#endif
}

BgMsgEntry::BgMsgEntry(int seqno, int _msgSize, double _sendTime, double _recvTime, int dstNode, int rank)
{
  msgID = seqno;
  sendTime = _sendTime;
  recvTime = _recvTime;
  dstPe = dstNode;
  tID = rank;                   // CmiBgMsgThreadID(msg);
  msgsize = _msgSize;
  CmiAssert(msgsize > 0);
}

/*
#if DELAY_SEND
void BgMsgEntry::send() {
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
*/

void bgEvents::print()
{
  switch (eType) {
  case BG_EVENT_PROJ:
	CmiPrintf("EVT: Projection %d\n", index);  break;
  case BG_EVENT_PRINT: {
	CmiPrintf("EVT: %s\n", data);
	break;
       }
  default: CmiAbort("bgEvents::pup(): unknown BG event type!");
  }
}

void bgEvents::write(FILE *fp)
{
  switch (eType) {
  case BG_EVENT_PROJ:
	fprintf(fp, "EVT: Projection %d\n", index);  break;
  case BG_EVENT_PRINT: {
	fprintf(fp, "EVT: %s\n", data);
	break;
       }
  default: CmiAbort("bgEvents::pup(): unknown BG event type!");
  }
}

void bgEvents::pup(PUP::er &p) 
{
  p|eType; p|rTime;
  switch (eType) {
  case BG_EVENT_PROJ:
	   p|index;  break;
  case BG_EVENT_PRINT: {
	     int slen = 0;
	     if (p.isPacking()) slen = strlen((char *)data)+1;
	     p|slen;
	     if (p.isUnpacking()) data=malloc(sizeof(char)*slen);
	     p((char *)data,slen); 
	     break;
	   }
  default: CmiAbort("bgEvents::pup(): unknown BG event type!");
  }
}

BgTimeLog::BgTimeLog(BgTimeLog *log)
{
  strncpy(name,log->name,20);
  ep = log->ep;
  startTime = log->startTime;
  recvTime = log->recvTime;
  endTime = 0.0;
  execTime = 0.0;
  msgId = log->msgId;

  seqno = 0;
  effRecvTime = recvTime;
  doCorrect = 1;
  flag = 0;
}

BgTimeLog::BgTimeLog(const BgMsgID &msgID)
{
  msgId = msgID;
  strcpy(name, "msgep");
  ep = -1;
  startTime = -1.0;
  recvTime = -1.0;
  endTime = execTime = 0.0;

  oldStartTime= startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
}

/*
BgTimeLog::BgTimeLog(int _ep, int srcpe, int msgid, double _startTime, double _endTime, double _recvTime, char *str)
{
  if (str)
    strcpy(name,str);
  else
    strcpy(name,"msgep");
  startTime = _startTime;
  endTime = _endTime;
  execTime = endTime - startTime;
  ep = _ep;
  recvTime = _recvTime;
  msgId = BgMsgID(srcpe, msgid);

  recvTime = effRecvTime = startTime;
}
*/

BgTimeLog::BgTimeLog(int epc, char* namestr,double sTime)
{ 
  if(namestr == NULL)
    namestr = "dummyname1";
  strncpy(name,namestr,20);
  ep = epc;
  startTime = sTime;
  recvTime = -1.0;//stime;
  endTime = execTime = 0.0;

  oldStartTime= startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
}

// for SDAG, somewhere else will set the effective recv time.
BgTimeLog::BgTimeLog(int epc, char* namestr, double sTime, double eTime)
{
  if(namestr == NULL)
    namestr = "dummyname2";
  strncpy(name,namestr, 20);
  ep = epc;
  startTime = sTime;
  recvTime = -1.0; //sTime;
  endTime = eTime;
  setExecTime();

  oldStartTime = startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
}

// create a new log from a message
BgTimeLog::BgTimeLog(char *msg, char *str)
{
  if (str)
    strcpy(name,str);
  else
    strcpy(name,"msgep");
  startTime = timerFunc();
  endTime = 0.0;
  execTime = 0.0;
  recvTime = 0.0;
  ep = -1;
  if (msg) {
    ep = CmiBgMsgHandle(msg);
    recvTime = CmiBgMsgRecvTime(msg);  //startTime;
    msgId = BgMsgID(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg));
  }

  oldStartTime=startTime;
  effRecvTime = recvTime;
  seqno = 0;
//  doCorrect = msg?CkMsgDoCorrect(msg):1;
  doCorrect = 1;
  flag = 0;

  if (genTimeLog && !doCorrect) {
      recvTime = effRecvTime = startTime;
  }
}

BgTimeLog::~BgTimeLog()
{
  int i;
  for (i=0; i<msgs.length(); i++)
    delete msgs[i];
  for (i=0; i<evts.length(); i++)
    delete evts[i];
}


void BgTimeLog::closeLog() 
{ 
    endTime = timerFunc();
    setExecTime();
    
//    if (correctTimeLog) BgAdjustTimeLineInsert(tTIMELINEREC);
}


void BgTimeLog::print(int node, int th)
{
  int i;
  CmiPrintf("<<== [%d th:%d] ep:%d name:%s startTime:%f endTime:%f srcnode:%d msgID:%d\n", node, th, ep, name,startTime, endTime, msgId.node(), msgId.msgID());
  for (i=0; i<msgs.length(); i++)
    msgs[i]->print();
  for (i=0; i<evts.length(); i++)
    evts[i]->print();
  CmiPrintf("==>>\n");
}


void BgTimeLog::write(FILE *fp)
{ 
  int i;
//  fprintf(fp,"%p ep:%d name:%s (srcnode:%d msgID:%d) startTime:%f endTime:%f recvime:%f effRecvTime:%e seqno:%d startevent:%d\n", this, ep, name, msgId.node(), msgId.msgID(), startTime, endTime, recvTime, effRecvTime, seqno, isStartEvent());
  fprintf(fp,"%p name:%s (srcnode:%d msgID:%d) ep:%d %s\n", this, name, msgId.node(), msgId.msgID(), ep, isStartEvent()?"STARTEVENT":"");
  fprintf(fp," recvtime:%f startTime:%f endTime:%f \n", recvTime, startTime, endTime);
  if (bglog_version >= 2) {
    if (!objId.isNull())
      fprintf(fp," ObjID: %d %d %d\n", objId.id[0], objId.id[1], objId.id[2]);
  }
  for (i=0; i<msgs.length(); i++)
    msgs[i]->write(fp);
  for (i=0; i<evts.length(); i++)
    evts[i]->write(fp);
  // fprintf(fp,"\nbackwardDeps [%d]:\n",backwardDeps.length());
  fprintf(fp, "backward: ");
  for (i=0; i<backwardDeps.length(); i++)
    fprintf(fp,"[%p %d] ",backwardDeps[i], backwardDeps[i]->seqno);
  fprintf(fp, "\n");
  fprintf(fp, "forward: ");
  for (i=0; i<forwardDeps.length(); i++)
    fprintf(fp,"[%p %d] ",forwardDeps[i], forwardDeps[i]->seqno);
  fprintf(fp, "\n");
  fprintf(fp, "\n");
}

void BgTimeLog::addMsgBackwardDep(BgTimeLineRec &tlinerec, void* msg){
  
  CmiAssert(recvTime < 0.);
  int idx;
  BgTimeLog *msglog = tlinerec.getTimeLogOnThread(BgMsgID(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg)), &idx);
  //CmiAssert(msglog != NULL);
  if (msglog != NULL)
  addBackwardDep(msglog);
}

// log  => this
void BgTimeLog::addBackwardDep(BgTimeLog* log)
{
  //CmiAssert(recvTime < 0.);
  if(log == NULL) return;
  for (int i=0; i<backwardDeps.length(); i++)
    if (backwardDeps[i] == log) return;	// already exist
  backwardDeps.insertAtEnd(log);
  log->forwardDeps.insertAtEnd(this);
  effRecvTime = BG_MAX(effRecvTime, log->effRecvTime);
}

void BgTimeLog::addBackwardDeps(CkVec<BgTimeLog*> logs){

  /*put backward and forward dependents*/
  for(int i=0;i<logs.length();i++)
    addBackwardDep(logs[i]);
}

void BgTimeLog::addBackwardDeps(CkVec<void*> logs){

  /*put backward and forward dependents*/
  for(int i=0;i<logs.length();i++)
    addBackwardDep((BgTimeLog*)(logs[i]));
}

int BgTimeLog::bDepExists(BgTimeLog* log){

  for(int i =0;i<backwardDeps.length();i++)
    if(backwardDeps[i] == log)
      return 1;
  return 0;
}

void BgTimeLog::pup(PUP::er &p){
    int l=0,idx;
    int i;

    if(p.isPacking()) {           // sanity check
      if (!strcasecmp(name, "BgSchedulerEnd")) {       // exit event
        if (endTime == 0.0) {
          endTime = startTime;
          if (msgs.length() > 0 && msgs[msgs.length()-1]->sendTime > endTime)
            endTime = msgs[msgs.length()-1]->sendTime;
        }
      }
    }

    p|ep; 
    p|seqno; p|msgId;
    p|recvTime; p|effRecvTime;p|startTime; p|execTime; p|endTime; 
    p|flag; p(name,20);
    if (bglog_version >= 2)
      p((int *)&objId, sizeof(CmiObjId)/sizeof(int));
    
    /*    if(p.isUnpacking())
      CmiPrintf("Puping: %d %d %d %d %e %e %e %e %e %s\n",ep,seqno,srcnode,msgID,recvTime,effRecvTime,startTime,execTime,endTime,name);
    */

/*
    if(p.isUnpacking()){
      threadNum = currTlineIdx;
    }
*/

    // pup for BgMsgEntry
    if(!p.isUnpacking()) l=msgs.length();
    p|l;

    for(i=0;i<l;i++) {
      if (p.isUnpacking()) msgs.push_back(new BgMsgEntry);
      msgs[i]->pup(p);
    }

    // pup events list for projections
    if(!p.isUnpacking()) l=evts.length();
    p|l;

    for(i=0;i<l;i++) {
      if (p.isUnpacking()) evts.push_back(new bgEvents);
      evts[i]->pup(p);
    }

    // pup for backwardDeps
    if(!p.isUnpacking()) l = backwardDeps.length();
    p|l;    

    for(i=0;i<l;i++){
      if(p.isUnpacking()){
	p|idx;
	addBackwardDep(currTline->timeline[idx]);
      }
      else{
	p|backwardDeps[i]->seqno;
      }
    }
 
    if(!p.isUnpacking()) l=forwardDeps.length();
    p|l;

    for(i=0;i<l;i++){ 
      if(p.isUnpacking())
	p|idx;
      else
	p|forwardDeps[i]->seqno;
    }

    // a sanity check for floatable events
    if (msgId == BgMsgID(-1,-1) && backwardDeps.length() == 0 && recvTime == -1.0) {
      CmiPrintf("Potential error in log: (a floating event) \n");
      print(-1,-1);
    }
}

// create a log with msg and insert into timeline
void BgTimeLineRec::logEntryStart(char *msg) {
//CmiPrintf("[%d] BgTimeLineRec::logEntryStart\n", BgGetGlobalWorkerThreadID());
  CmiAssert(genTimeLog);
  if (!genTimeLog) return;
  CmiAssert(bgCurLog == NULL);
  bgCurLog = new BgTimeLog(msg);
  enq(bgCurLog, 1);
}

// insert an log into timeline
void BgTimeLineRec::logEntryInsert(BgTimeLog* log)
{
  CmiAssert(genTimeLog);
  if (!genTimeLog) return;
//CmiPrintf("[%d] BgTimeLineRec::logEntryInsert\n", BgGetGlobalWorkerThreadID());
  CmiAssert(bgCurLog == NULL);
  if(timeline.length() > 0 && timeline[timeline.length()-1]->endTime == 0.0)
    CmiPrintf("\nERROR tried to insert %s after %s\n",log->name,timeline[timeline.length()-1]->name);
  enq(log, 1);
  if (bgPrevLog) {
    log->addBackwardDep(bgPrevLog);
    bgPrevLog = NULL;
  }
}

void BgTimeLineRec::logEntryStart(BgTimeLog* log)
{
//CmiPrintf("[%d] BgTimeLineRec::logEntryStart with log\n", BgGetGlobalWorkerThreadID());
  logEntryInsert(log);
  bgCurLog = log;
}

void BgTimeLineRec::logEntryClose() {
  CmiAssert(genTimeLog);
  if (!genTimeLog) return;
//CmiPrintf("[%d] BgTimeLineRec::logEntryClose\n", BgGetGlobalWorkerThreadID());
  BgTimeLog *lastlog = timeline.length()?timeline[timeline.length()-1]:NULL;
  CmiAssert(bgCurLog == lastlog);
  lastlog->closeLog();
  bgCurLog = NULL;
}

void BgTimeLineRec::logEntrySplit()
{
//CmiPrintf("BgTimeLineRec::logEntrySplit\n");
  CmiAssert(genTimeLog);
  if (!genTimeLog) return;
  CmiAssert(bgCurLog != NULL);
  BgTimeLog *rootLog = bgCurLog;
  logEntryClose();

  // make up a new bglog to start, setting up dependencies.
  BgTimeLog *newLog = new BgTimeLog(-1, "split-broadcast", timerFunc());
  newLog->addBackwardDep(rootLog);
  logEntryInsert(newLog);
  bgCurLog = newLog;
}

BgTimeLog *
BgTimeLineRec::getTimeLogOnThread(const BgMsgID &msgId, int *index)
{
  int idxOld = timeline.length()-1;
  while (idxOld >= 0)  {
    if (timeline[idxOld]->msgId == msgId) break;
    idxOld--;
  }
                                                                                
  *index = idxOld;
  if (idxOld == -1) return NULL;
  return timeline[idxOld];
}

void BgTimeLineRec::pup(PUP::er &p)
{
    int l=length();
    p|l;
    //    CmiPrintf("Puped len: %d\n",l);
    if(!p.isUnpacking()){
      // reorder the seqno
      for(int i=0;i<l;i++)
        timeline[i]->seqno = i;
    }
    else{
      //Timeline is empty when unpacking pup is called
      //timeline.removeFrom(0);
    }

    for (int i=0;i<l;i++) {
        if (p.isUnpacking()) {
                BgTimeLog* t = new BgTimeLog();
                t->pup(p);
                timeline.enq(t);
        }
        else {
          timeline[i]->pup(p);
        }
    }
}

