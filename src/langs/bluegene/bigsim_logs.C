#include "blue.h"
#include "blue_impl.h"

#include "bigsim_logs.h"

/*
 ChangeLog
 version 2 
     * add objId
 version 3
     * objId changed to 4 ints 

 versions 4, 5
 - ???

 version 6
 - MPI Record added
*/

int bglog_version = 6;

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
BgMsgEntry::BgMsgEntry(char *msg, int _dstNode, int tid, double sendT, int local, int g)
{
  msgID = CmiBgMsgID(msg);
  sendTime = sendT;
  recvTime = CmiBgMsgRecvTime(msg);
  dstNode = _dstNode;
  tID = tid;                   // CmiBgMsgThreadID(msg);
  msgsize = CmiBgMsgLength(msg);
  CmiAssert(msgsize > 0);
  group = g;
  CmiAssert(group!=0);
#if DELAY_SEND
  sendMsg = NULL;
  if (!local && correctTimeLog) sendMsg = msg;
#endif
}

BgMsgEntry::BgMsgEntry(int seqno, int _msgSize, double _sendTime, double _recvTime, int _dstNode, int rank)
{
  msgID = seqno;
  sendTime = _sendTime;
  recvTime = _recvTime;
  dstNode = _dstNode;
  tID = rank;                   // CmiBgMsgThreadID(msg);
  msgsize = _msgSize;
  CmiAssert(msgsize > 0);
  group = 1;
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

void BgMsgEntry::pup(PUP::er &p)
{
    p|msgID; p|dstNode; p|sendTime; p|recvTime; p|tID; p|msgsize;
    CmiAssert(recvTime>=sendTime);
    CmiAssert(msgsize >= 0);
    if (p.isUnpacking()) group = 1;    // default value
    if (bglog_version>0) p|group;
}

void bgEvents::print()
{
  switch (eType) {
  case BG_EVENT_PROJ:
	CmiPrintf("EVT: Projection %d\n", index);  break;
  case BG_EVENT_PRINT: {
	CmiPrintf("EVT: time:%f string:%s\n", rTime, data);
	break;
       }
  case BG_EVENT_MARK: {
	CmiPrintf("EVT: time:%f marker:%s\n", rTime, data);
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
        fprintf(fp, "EVT: time:%f string:%s\n", rTime, (const char*)data);
        break;
       }
  case BG_EVENT_MARK: {
	fprintf(fp, "EVT: time:%f marker:%s\n", rTime, (const char*)data);
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
  case BG_EVENT_MARK:
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

BgTimeLog::BgTimeLog()
    : ep(-1), charm_ep(-1), recvTime(.0), startTime(.0), endTime(.0),
      execTime(.0), effRecvTime(INVALIDTIME), seqno(0), doCorrect(1),
      flag(0), mpiOp(MPI_NONE)
{
    strcpy(name,"dummyname");
}


BgTimeLog::BgTimeLog(BgTimeLog *log)
{
  strncpy(name,log->name,BGLOG_NAMELEN-1);
  name[BGLOG_NAMELEN-1] = 0;
  ep = log->ep;
  charm_ep = -1;
  startTime = log->startTime;
  recvTime = log->recvTime;
  endTime = -1.0;
  execTime = 0.0;
  msgId = log->msgId;

  seqno = 0;
  effRecvTime = recvTime;
  doCorrect = 1;
  flag = 0;
  mpiOp = log->mpiOp;
  mpiSize = log->mpiSize;
}

BgTimeLog::BgTimeLog(const BgMsgID &msgID)
{
  msgId = msgID;
  strcpy(name, "msgep");
  ep = -1;
  charm_ep = -1;
  startTime = -1.0;
  recvTime = -1.0;
  endTime = execTime = -1.0;

  oldStartTime= startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
  mpiOp = MPI_NONE;
}

BgTimeLog::BgTimeLog(int epc, const char* namestr,double sTime)
{ 
  if(namestr == NULL)
    namestr = (char*)"dummyname1";
  strncpy(name,namestr,BGLOG_NAMELEN-1);
  name[BGLOG_NAMELEN-1] = 0;
  ep = epc;
  charm_ep = -1;
  startTime = sTime;
  recvTime = -1.0;//stime;
  endTime = execTime = -1.0;

  oldStartTime= startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
  mpiOp = MPI_NONE;
}

// for SDAG, somewhere else will set the effective recv time.
BgTimeLog::BgTimeLog(int epc, const char* namestr, double sTime, double eTime)
{
  if(namestr == NULL)
    namestr = (char*)"dummyname2";
  strncpy(name,namestr, BGLOG_NAMELEN-1);
  name[BGLOG_NAMELEN-1] = 0;
  ep = epc;
  charm_ep = -1;
  startTime = sTime;
  recvTime = -1.0; //sTime;
  endTime = eTime;
  setExecTime();

  oldStartTime = startTime;
  effRecvTime = -1.0;
  seqno = 0;
  doCorrect = 1;
  flag = 0;
  mpiOp = MPI_NONE;
}

// create a new log from a message
BgTimeLog::BgTimeLog(char *msg, char *str)
{
  if (str)
    strcpy(name,str);
  else
    strcpy(name,"msgep");
  startTime = timerFunc();
  endTime = -1.0;
  execTime = 0.0;
  recvTime = 0.0;
  ep = -1;
  charm_ep = -1;
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
  mpiOp = MPI_NONE;
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
    CmiAssert(endTime >= 0.0);
    
//    if (correctTimeLog) BgAdjustTimeLineInsert(tTIMELINEREC);
}


void BgTimeLog::print(int node, int th)
{
  int i;
  CmiPrintf("<<== [%d th:%d] ep:%d name:%s startTime:%f endTime:%f execTIme:%f srcpe:%d msgID:%d\n", node, th, ep, name,startTime, endTime, execTime, msgId.pe(), msgId.msgID());
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
  fprintf(fp,"%p name:%s (srcpe:%d msgID:%d) ep:%d ", this, name, msgId.pe(), msgId.msgID(), ep);
  if (ep == BgLogGetThreadEP() && ep!=-1) fprintf(fp, "(thread resume) ");
  if (bglog_version >= 4) fprintf(fp, "charm_ep:%d ", charm_ep);
  if (isStartEvent()) fprintf(fp, "STARTEVENT");
  if (isQDEvent()) fprintf(fp, "QDEVENT");
  fprintf(fp, "\n");

  fprintf(fp," recvtime:%f startTime:%f endTime:%f execTime:%f\n", recvTime, startTime, endTime, execTime);
  if (bglog_version >= 2) {
    if (!objId.isNull())
      fprintf(fp," ObjID: %d %d %d %d\n", objId.id[0], objId.id[1], objId.id[2], objId.id[3]);
  }
  if (bglog_version >= 6) {
    if (mpiOp!=MPI_NONE) {
      fprintf(fp, "MPI collective: ");
      switch (mpiOp) {
      case MPI_BARRIER:   fprintf(fp, "MPI Barrier"); break;
      case MPI_ALLREDUCE: fprintf(fp, "MPI_Allreduce"); break;
      case MPI_ALLTOALL:  fprintf(fp, "MPI_Alltoall"); break;
      }
      fprintf(fp, " mpiSize: %d\n", mpiSize);
    }
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

void BgTimeLog::addMsgBackwardDep(BgTimeLineRec &tlinerec, void* msg)
{
  // CmiAssert(recvTime < 0.);
  int idx;
  BgTimeLog *msglog = tlinerec.getTimeLogOnThread(BgMsgID(CmiBgMsgSrcPe(msg), CmiBgMsgID(msg)), &idx);
  //CmiAssert(msglog != NULL);
  if (msglog != NULL) {
    CmiAssert(msglog != this);
    addBackwardDep(msglog);
  }
}

// log  => this
void BgTimeLog::addBackwardDep(BgTimeLog* log)
{
  //CmiAssert(recvTime < 0.);
  if(log == NULL || log == this) return;
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

#if defined(_WIN32)
#define strcasecmp stricmp
#endif

// pup data common to both BgTimeLog::pup and BgTimeLog::winPup
void BgTimeLog::pupCommon(PUP::er &p) {

  int l, i;

  if (p.isPacking()) {           // sanity check
    if (!strcasecmp(name, "BgSchedulerEnd")) {       // exit event
      if (endTime < 0.0) {
	endTime = startTime;
	if (msgs.length() > 0 && msgs[msgs.length()-1]->sendTime > endTime)
	  endTime = msgs[msgs.length()-1]->sendTime;
	execTime = endTime - startTime;
      }
    }
  }

  p|ep; 
  p|seqno; p|msgId;

  if (bglog_version >= 6)
  {
      p|mpiOp;
      //if (MPI_NONE != mpiOp)
	  p|mpiSize;
  }

  if (bglog_version >= 4) p(charm_ep);
  p|recvTime; p|effRecvTime; p|startTime; p|execTime; p|endTime; 
  p|flag; p(name,BGLOG_NAMELEN);
  if (bglog_version >= 3)
    p((int *)&objId, sizeof(CmiObjId)/sizeof(int));
  else if (bglog_version == 2)
    p((int *)&objId, 3);           // only 3 ints before

/*
  CmiPrintf("            *** BgTimeLog::pup: ep=%d seqno=%d msgId:[node=%d msgID=%d] name=%s\n", ep, seqno, msgId.node(), msgId.msgID(), name);

  if (p.isUnpacking())
    CmiPrintf("Puping: %d %d %d %d %e %e %e %e %e %s\n",ep,seqno,srcnode,msgID,recvTime,effRecvTime,startTime,execTime,endTime,name);
*/

/*
  if (p.isUnpacking()) {
    threadNum = currTlineIdx;
  }
*/

  // pup for BgMsgEntry
  if (!p.isUnpacking()) l=msgs.length();
  //cppcheck-suppress uninitvar
  p|l;

  // CmiPrintf("               *** number of messages: %d\n", l);

  for (i = 0; i < l; i++) {
    if (p.isUnpacking()) msgs.push_back(new BgMsgEntry);
    msgs[i]->pup(p);
  }

  // pup events list for projections
  if (!p.isUnpacking()) l=evts.length();
  p|l;

  for (i = 0; i < l; i++) {
    if (p.isUnpacking()) evts.push_back(new bgEvents);
    evts[i]->pup(p);
  }

}

void BgTimeLog::pup(PUP::er &p) {
    int l=0,idx;
    int i;

    // pup data common to both BgTimeLog::pup and BgTimeLog::winPup
    pupCommon(p);

    // pup for backwardDeps
    if(!p.isUnpacking()) l = backwardDeps.length();
    p|l;    

    for(i=0;i<l;i++){
      if(p.isUnpacking()){
	p|idx;
        CmiAssert(currTline != NULL);
	addBackwardDep(currTline->timeline[idx]);
      }
      else{
	p|backwardDeps[i]->seqno;
        CmiAssert(backwardDeps[i]->seqno < seqno);
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

void BgTimeLog::winPup(PUP::er &p, int& firstLogToRead, int& numLogsToRead) {

  int l=0, idx;
  int i, j;

  // pup data common to both BgTimeLog::pup and BgTimeLog::winPup
  pupCommon(p);

  // pup for backwardDeps
  if (!p.isUnpacking()) l = backwardDeps.length();
  p|l;    

  // CmiPrintf("               *** number of bDeps: %d\n", l);

  for (i = 0; i < l; i++) {
    if (p.isUnpacking()) {
      p|idx;
      CmiAssert(currTline != NULL);
      if (idx >= firstLogToRead) {
	// bDep is in the current window
	addBackwardDep(currTline->timeline[idx - firstLogToRead]);
      } else {
	// bDep is in a previous window -> create placeholder for
	// linking later when converted to a Task
	BgTimeLog* emptyLog = new BgTimeLog();
	emptyLog->seqno = -idx;
	for (j = 0; j<backwardDeps.length(); j++)
	  if (backwardDeps[j]->seqno == emptyLog->seqno) continue;  // already exists
	backwardDeps.insertAtEnd(emptyLog);
      }
    }
    else{
      p|backwardDeps[i]->seqno;
    }
  }
 
  if (!p.isUnpacking()) l=forwardDeps.length();
  p|l;

  // CmiPrintf("               *** number of fDeps: %d\n", l);

  for (i = 0; i < l; i++) {
    if (p.isUnpacking()) {
      p|idx;
      if (idx >= (firstLogToRead + numLogsToRead)) {
	// fDep is not in this window -> create placeholder for
	// linking later when the correct window is loaded
	BgTimeLog* emptyLog = new BgTimeLog();
	emptyLog->seqno = -idx;
	for (j = 0; j<forwardDeps.length(); j++)
	  if (forwardDeps[j]->seqno == emptyLog->seqno) continue;  // already exists
	forwardDeps.insertAtEnd(emptyLog);
      }
      // ignore fDep if it's in this window -> it will be linked
      // when its corresponding bDep is PUPed
    }
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
  if(timeline.length() > 0 && timeline[timeline.length()-1]->endTime < 0.0) {
    CmiPrintf("\nERROR tried to insert %s after %s\n",log->name,timeline[timeline.length()-1]->name);
  }
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

void BgTimeLineRec::logEntrySplit(const char *name)
{
//CmiPrintf("BgTimeLineRec::logEntrySplit\n");
  CmiAssert(genTimeLog);
  if (!genTimeLog) return;
  CmiAssert(bgCurLog != NULL);
  BgTimeLog *rootLog = bgCurLog;
  logEntryClose();

  // make up a new bglog to start, setting up dependencies.
  BgTimeLog *newLog = new BgTimeLog(-1, (char*)name, timerFunc());
  newLog->addBackwardDep(rootLog);
  logEntryInsert(newLog);
  bgCurLog = newLog;
}

// split log, insert backdeps in array parentlogs, size of n
BgTimeLog * BgTimeLineRec::logSplit(const char *name, BgTimeLog **parentlogs, int n)
{
  CmiAssert(genTimeLog);
  if (!genTimeLog) return NULL;
  BgTimeLog *curLog;
  if (n == 0)  {
    curLog = timeline.length()?timeline[timeline.length()-1]:NULL;
    if (curLog != NULL) {
      parentlogs = &curLog;
      n = 1;
    }
  }
  logEntryClose();

  // make up a new bglog to start, setting up dependencies.
  BgTimeLog *newLog = new BgTimeLog(-1, (char*)name, timerFunc());
  for (int i=0; i<n; i++)
    newLog->addBackwardDep(parentlogs[i]);
  logEntryInsert(newLog);
  bgCurLog = newLog;
  return newLog;
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
        // create all logs, so that back/forward dep can be linked correctly
      for (int i=0;i<l;i++) {
        BgTimeLog* t = new BgTimeLog();
        timeline.enq(t);
      }
    }

    for (int i=0;i<l;i++) {
        if (p.isUnpacking()) {
                BgTimeLog* t = timeline[i];
                t->pup(p);
        }
        else {
          timeline[i]->pup(p);
        }
    }
}

void BgTimeLineRec::winPup(PUP::er &p, int& firstLogToRead, int& numLogsToRead, int& tLineLength) {

  int l=0;

  if (!p.isUnpacking()) {

    // if we're packing, pack the whole thing
    l = length();
    p|l;
    // ensure the seqno has been assigned
    for (int i = 0; i < l; i++)
      timeline[i]->seqno = i;
    for (int i = 0; i < l; i++)
      timeline[i]->winPup(p, firstLogToRead, numLogsToRead);

  } else {

    // unpack based on the values of the global variables
    if (firstLogToRead == 0) {
      p|l;
      tLineLength = l;
    }
    // ensure that we don't read more logs than there are in the time
    // line
    if ((firstLogToRead + numLogsToRead) > tLineLength) {
      numLogsToRead = tLineLength - firstLogToRead;
    }
    // CmiPrintf("         *** BgTimeLineRec::winPup: tLineLength=%d firstLogToRead=%d numLogsToRead=%d\n", tLineLength, firstLogToRead, numLogsToRead);
    // unpack and enqueue the logs in the time line
    for (int i = 0; i < numLogsToRead; i++) {
      BgTimeLog* t = new BgTimeLog();
      t->winPup(p, firstLogToRead, numLogsToRead);
      timeline.enq(t);
    }

  }

}
