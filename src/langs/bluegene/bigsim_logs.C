#include "blue.h"
#include "blue_impl.h"

#include "blue_logs.h"

int genTimeLog = 0;			// was 1 for guna 's seq correction
int correctTimeLog = 0;
int bgcorroff = 0;

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

// null timer for using blue_logs as a sequential library
static double nullTimer() { return 0.; }

// bluegene timer call
double (*timerFunc) (void) = nullTimer;

// dstNode is the dest bg node, can be -1
bgMsgEntry::bgMsgEntry(char *msg, int dstNode, int tid, int local)
{
  msgID = CmiBgMsgID(msg);
//  sendtime = BgGetCurTime();
  recvTime = CmiBgMsgRecvTime(msg);
  dstPe = dstNode;
  tID = tid;                   // CmiBgMsgThreadID(msg);
  msgsize = CmiBgMsgLength(msg);
  CmiAssert(msgsize > 0);
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

bgTimeLog::bgTimeLog(char *msg, char *str)
{
  if (str)
    strcpy(name,str);
  else
    strcpy(name,"msgep");
  ep = msg?CmiBgMsgHandle(msg):-1;
  startTime = timerFunc();
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
    endTime = timerFunc();
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
    fprintf(fp,"[%p %d] ",backwardDeps[i], backwardDeps[i]->index);
  fprintf(fp, "\n");
  fprintf(fp, "forward: ");
  for (i=0; i<forwardDeps.length(); i++)
    fprintf(fp,"[%p %d] ",forwardDeps[i], forwardDeps[i]->index);
  fprintf(fp, "\n");
  fprintf(fp, "==>>\n");
}

void bgTimeLog::addMsgBackwardDep(BgTimeLineRec &tlinerec, void* msg){
  
  CmiAssert(recvTime < 0.);
  int idx;
  bgTimeLog *msglog = tlinerec.getTimeLogOnThread(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg), &idx);
  CmiAssert(msglog != NULL);
  addBackwardDep(msglog);
}

void bgTimeLog::addBackwardDep(bgTimeLog* log){
  
  CmiAssert(recvTime < 0.);
  if(log != NULL){
    for (int i=0; i<backwardDeps.length(); i++)
      if (backwardDeps[i] == log) return;	// already exist
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

// create a log with msg and insert into timeline
void BgTimeLineRec::logEntryStart(char *msg) {
//CmiPrintf("[%d] BgTimeLineRec::logEntryStart\n", BgGetGlobalWorkerThreadID());
  if (!genTimeLog) return;
  CmiAssert(bgCurLog == NULL);
  bgCurLog = new bgTimeLog(msg);
  enq(bgCurLog, 1);
}

// insert an log into timeline
void BgTimeLineRec::logEntryInsert(bgTimeLog* log)
{
  if (!genTimeLog) return;
//CmiPrintf("[%d] BgTimeLineRec::logEntryInsert\n", BgGetGlobalWorkerThreadID());
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
//CmiPrintf("[%d] BgTimeLineRec::logEntryClose\n", BgGetGlobalWorkerThreadID());
  bgTimeLog *lastlog = timeline[timeline.length()-1];
  CmiAssert(bgCurLog == lastlog);
  lastlog->closeLog();
  bgCurLog = NULL;
}

void BgTimeLineRec::logEntrySplit()
{
//CmiPrintf("BgTimeLineRec::logEntrySplit\n");
  if (!genTimeLog) return;
  CmiAssert(bgCurLog != NULL);
  bgTimeLog *rootLog = bgCurLog;
  logEntryClose();

  // make up a new bglog to start, setting up dependencies.
  bgTimeLog *newLog = new bgTimeLog(-1, "broadcast", timerFunc());
  newLog->addBackwardDep(rootLog);
  logEntryInsert(newLog);
  bgCurLog = newLog;
}

bgTimeLog *
BgTimeLineRec::getTimeLogOnThread(int srcnode, int msgID, int *index)
{
  int idxOld = timeline.length()-1;
  while (idxOld >= 0)  {
    if (timeline[idxOld]->msgID == msgID && timeline[idxOld]->srcnode == srcnode) break;
    idxOld--;
  }
                                                                                
  *index = idxOld;
  if (idxOld == -1) return NULL;
  return timeline[idxOld];
}

int bgTimeLog::bDepExists(bgTimeLog* log){

  for(int i =0;i<backwardDeps.length();i++)
    if(backwardDeps[i] == log)
      return 1;
  return 0;
}

void bgTimeLog::pup(PUP::er &p){
    int l=0,idx;
    int i;
    p|ep; 
    p|seqno; p|srcnode;p|msgID;
    p|recvTime; p|effRecvTime;p|startTime; p|execTime; p|endTime;p|index;p(name,20);
    
    /*    if(p.isUnpacking())
      CmiPrintf("Puping: %d %d %d %d %e %e %e %e %e %s\n",ep,seqno,srcnode,msgID,recvTime,effRecvTime,startTime,execTime,endTime,name);
    */

    if(p.isUnpacking()){
      threadNum = currTlineIdx;
    }

    // pup for bgMsgEntry
    if(!p.isUnpacking()) l=msgs.length();
    p|l;

    for(i=0;i<l;i++) {
      if (p.isUnpacking()) msgs.push_back(new bgMsgEntry);
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
	p|backwardDeps[i]->index;
      }
    }
 
    if(!p.isUnpacking()) l=forwardDeps.length();
    p|l;

    for(i=0;i<l;i++){ 
      if(p.isUnpacking())
	p|idx;
      else
	p|forwardDeps[i]->index;
    }
}

void BgWriteThreadTimeLine(char *pgm, int x, int y, int z, int th, BgTimeLine &tline)
{
  char *fname = (char *)malloc(strlen(pgm)+100);
  sprintf(fname, "%s-%d-%d-%d.%d.log", pgm, x,y,z,th);
  FILE *fp = fopen(fname, "w");
  CmiAssert(fp!=NULL);
  for (int i=0; i<tline.length(); i++) {
    fprintf(fp, "[%d] ", i);
    tline[i]->write(fp);
  }
  fclose(fp);
  free(fname);
}

