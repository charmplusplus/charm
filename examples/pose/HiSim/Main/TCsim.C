#include "BgSim_sim.h"
extern BgTimeLineRec* currTline;
extern int currTlineIdx;

#define ALLTHREAD -1

// the applicaiton name to simulate
extern roarray<char, 1024>  appname;

int Task::convertToInt(double inp) 
{
  int out = (int)(inp*factor);
  if (out <0 && inp != -1.0) {
    CmiPrintf("Invalid value in convertToInt() - %d %f\n", out, inp);
    CmiPrintf("Considering changing factor %e to a smaller value. \n", factor);
    CmiAssert(out >= 0 || inp == -1);
  }
  return (out<0?0:out);
}


void Task::convertFrom(BgTimeLog* bglog, int procNum, int numWth) 
{
  int i;

  taskID.srcNode = bglog->msgId.node();
  taskID.msgID = bglog->msgId.msgID();
  taskID.index = bglog->seqno;

  strncpy(name,bglog->name,19);  name[19]=0;
  startTime = convertToInt(bglog->startTime);
  receiveTime = convertToInt(bglog->recvTime);
  execTime = convertToInt(bglog->execTime);
  done = 0;

  //CkPrintf("LOG rcv=%d start=%d\n", receiveTime, startTime);
  bDepsLen = bglog->backwardDeps.length();
  backwardDeps = NULL;
  if (bDepsLen) {
    backwardDeps = new TaskID[bDepsLen];
    for(i=0;i<bDepsLen;i++){
      backwardDeps[i].srcNode = -1;//bglog->backwardDeps[i]->srcnode;
      backwardDeps[i].msgID = -1;//bglog->backwardDeps[i]->msgID;
      backwardDeps[i].index = bglog->backwardDeps[i]->seqno;
    }
  }

  fDepsLen = bglog->forwardDeps.length();
  forwardDeps = NULL;
  if (fDepsLen) {
    forwardDeps = new TaskID[fDepsLen];
    for(i=0;i<fDepsLen;i++){
      forwardDeps[i].srcNode = -1;//bglog->forwardDeps[i]->srcnode;
      forwardDeps[i].msgID = -1;//bglog->forwardDeps[i]->msgID;
      forwardDeps[i].index = bglog->forwardDeps[i]->seqno;
    }
  }
  
  // these are the generated tasks?
  taskMsgsLen = bglog->msgs.length();
  taskmsgs = new TaskMsg[taskMsgsLen];
  for(i=0; i< taskMsgsLen;i++){
    taskmsgs[i].taskID.srcNode = procNum/numWth;
    taskmsgs[i].taskID.msgID = bglog->msgs[i]->msgID;
    taskmsgs[i].taskID.index = -1; 
    taskmsgs[i].destNode = bglog->msgs[i]->dstPe;
    taskmsgs[i].desttID = bglog->msgs[i]->tID;
    taskmsgs[i].receiveTime = convertToInt(bglog->msgs[i]->recvTime);
    taskmsgs[i].msgsize = bglog->msgs[i]->msgsize;
    if (taskmsgs[i].desttID >= 0) CmiAssert(taskmsgs[i].desttID < numWth);
    CmiAssert(taskmsgs[i].msgsize > 0);
  }
  
  // trace projections events and bgPrint Events
  printevtLen = projevtLen = 0;
  projevts = NULL;
  printevts = NULL;
  for(i=0; i< bglog->evts.length();i++){
    if (bglog->evts[i]->eType == BG_EVENT_PROJ)  projevtLen++;
    if (bglog->evts[i]->eType == BG_EVENT_PRINT)  printevtLen++;
  }
  if (projevtLen)  projevts = new ProjEvent[projevtLen];
  if (printevtLen)  printevts = new bgPrintEvent[printevtLen];

  printevtLen = projevtLen = 0;
  for(i=0; i< bglog->evts.length();i++){
    // CmiPrintf("[%d] %d %d \n", procNum, bglog->evts[i]->eType, bglog->evts[i]->index);
      if (bglog->evts[i]->eType == BG_EVENT_PROJ) {
        projevts[projevtLen].index = bglog->evts[i]->index;
        projevts[projevtLen].startTime = convertToInt(bglog->evts[i]->rTime);
        projevtLen++;
      }
      if (bglog->evts[i]->eType == BG_EVENT_PRINT) {
        printevts[printevtLen].data = (char *)bglog->evts[i]->data;
        printevts[printevtLen].sTime = convertToInt(bglog->evts[i]->rTime);
        printevtLen++;
      }
  }

}


void Task::pup(PUP::er &p) 
{
  taskID.pup(p);
  p|receiveTime; p|execTime; p|startTime;
  p(name,20);
  p|bDepsLen; p|fDepsLen;p|taskMsgsLen; p|printevtLen;p|projevtLen;
  p|done;

  if(p.isUnpacking()){
    backwardDeps = new TaskID[bDepsLen];
    forwardDeps = new TaskID[fDepsLen];
    taskmsgs = new TaskMsg[taskMsgsLen];
    projevts = new ProjEvent[projevtLen];
    printevts = new bgPrintEvent[printevtLen];
  }
  for(int i=0;i<bDepsLen;i++)
    backwardDeps[i].pup(p);
  for(int i=0;i<fDepsLen;i++)
    forwardDeps[i].pup(p);
  for(int i=0;i<taskMsgsLen;i++)
    taskmsgs[i].pup(p);
  for(int i=0;i<printevtLen;i++)
    p|printevts[i];
  for(int i=0;i<projevtLen;i++)
    p|projevts[i];
}

BGproc::BGproc() 
{
  // implement as you wish
  proj = NULL;
  done = NULL;
  taskList = NULL;
}

// read trace projections logs if there is any
void BGproc::loadProjections() 
{
  int i;

  proj = NULL;
  binary = 0;
  char str[1024];
  char app[128];
  for (i=0; i<128; i++) app[i] = appname[i];
  sprintf(str, "%s.%d.log", app, procNum);
  FILE *f = fopen(str, "r");
  if (f) {
    CmiPrintf("Loading projections file: %s\n", str);
    // figure out the log size
    if (!binary) {
      fgets(str, 1024, f);		// skip header
      sscanf(str, "PROJECTIONS-RECORD %d", &numLogs);
    }
    else {
      fread(&numLogs, sizeof(int), 1, f);
    }
    CmiAssert(numLogs);
    logs = new LogEntry[numLogs]; 
    fromProjectionsFile p(f);
//    PUP::fromDisk p(f);
    for (i=0; i<numLogs; i++) {
      logs[i].pup(p);
      // CmiPrintf("%d %d\n", i, logs[i].type);
    }
    fclose(f);
    // new log
    sprintf(str, "%s-bg.%d.log", app, procNum);
    proj = fopen(str, "w");
    if (!binary) {
      fprintf(proj, "PROJECTIONS-RECORD %d\n", numLogs);
    }
    CmiAssert(proj);
  }
}

BGproc::BGproc(BGprocMsg *m)
{
  int i;

  nodePID = m->nodePID;		// node pose index
  useAntimethods();
  BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
  
  int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);
  BgTimeLineRec tline;
  procNum = myHandle;
  currTline = &tline;
  currTlineIdx = procNum;
  BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tline);

  delete [] allNodeOffsets;

#if 0    // detail
  // dump bg timeline log to disk
  BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tline.timeline);
#endif

  //CkPrintf("bgProc %d has %d tasks\n", myHandle, tline.length());
  taskList = new Task[tline.length()];
  numTasks = tline.length();
  int startEvt = 0;		// start event, to skip startup
  for(i=0;i<tline.length();i++) {
    taskList[i].convertFrom(tline[i],myHandle,numWth);
    if (config.skip_on && tline[i]->isStartEvent()) { 
	CmiAssert(procNum==0); 
	startEvt = i; 
    }
  }

  buildHash();

  // store the status of done tasks
  done = new int[numTasks];
  for (i = 0; i<numTasks; i++) done[i] = 0;

  //  if (myHandle == 0) { 
  // this is processor 0 therefore the one to start main
  // get the taskID for the main task to start and invoke it

  // read trace projections logs if there is any
  loadProjections();

  // only pe 0 should start the sim
  if (procNum == 0) {
    TaskMsg *tm = new TaskMsg(-1,-1,startEvt);
    POSE_invoke(executeTask(tm), BGproc, myHandle, taskList[startEvt].startTime);
    CkPrintf("Info> timing factor %e ...\n", factor);
    CkPrintf("Info> invoking startup task from proc 0 ...\n");
    if (config.skip_on)
      CkPrintf("Info> Skipping startup to %d/%d\n", startEvt, tline.length());
  }
#if 1
    for (int i=startEvt; i<numTasks; i++) {
      if (!strcmp(taskList[i].name, "addMsg") ||
          !strcmp(taskList[i].name, "AMPI_START"))  {
        TaskMsg *tm = new TaskMsg(taskList[i].taskID.srcNode,
		       taskList[i].taskID.msgID,
		       taskList[i].taskID.index,
		       taskList[i].receiveTime,
		       0,
		       procNum/numWth,  procNum/numWth,
		       procNum%numWth
		       );
        POSE_invoke(executeTask(tm), BGproc, myHandle, taskList[i].receiveTime);
      }
    }
#endif
  //}
  // FIXME:  free tline memory
}

BGproc::~BGproc()
{
  // implement as you wish
  if (done) delete [] done;
}

void BGproc::pup(PUP::er &p)
{ 
  chpt<state_BGproc>::pup(p); 
  // pup rest of data fields here
  p|numX;p|numY;p|numZ;p|numCth;p|numWth;p|numPes;p|totalProcs;p|procNum;
  p|nodePID;
  p|numTasks;
  if(p.isUnpacking())
    taskList = new Task[numTasks];
  for(int i=0;i<numTasks;i++)
    taskList[i].pup(p);
}

// Event methods
void BGproc::executeTask(TaskMsg *m)
{
  /*
  CkPrintf("[%d Received TaskID: %d %d %d]\n", procNum, m->taskID.srcNode,
	   m->taskID.msgID, m->taskID.index, parent->thisIndex);
  parent->CommitPrintf("Received %d from %d on %d\n", m->taskID.msgID, 
		       m->taskID.srcNode, parent->thisIndex);
  */
//  CmiAssert(done[locateTask(m->taskID)] == 0);
  int taskLoc = locateTask(m->taskID);
  if (taskLoc == -1) return; // user was warned in locateTask
  if (done[taskLoc] == 1) {
    char str[1024];
    sprintf(str, "[%d] Event %d '%s' already done!\n", procNum, taskLoc, taskList[taskLoc].name);
    parent->CommitError(str);
    //    CkPrintf("POTENTIALLY: [%d] Event %d '%s' already done! %d %d %d at %d evID=", procNum, taskLoc, taskList[taskLoc].name, m->taskID.srcNode, m->taskID.msgID, m->taskID.index, m->timestamp); m->evID.dump(); CkPrintf("\n");
    return;
  }

  int oldRT = getReceiveTime(m->taskID);
  int newRT = m->timestamp;
  //if(!(m->isNull()))
  //updateReceiveTime(m);
  if (dependenciesMet(&(m->taskID))) { // dependencies met; we can execute this
    Task &task = taskList[taskLoc];
    int oldStartTime = task.startTime;
    int newStartTime = ovt;
    task.newStartTime = newStartTime;   // store new start time and used later
    //updateStartTime(&m->taskID, newStartTime);
    markTask(&m->taskID);  // what is this doing?
    //CmiPrintf("[%d] executeTask %d for %d %d %d at %d evID=", procNum, taskLoc, m->taskID.srcNode, m->taskID.msgID, m->taskID.index, m->timestamp); m->evID.dump(); CkPrintf("\n");
    enableGenTasks(&m->taskID,oldStartTime,newStartTime);
    elapse(getDuration((m->taskID)));
    enableDependents(&m->taskID); // invoke dependent tasks
    /*
      parent->CommitPrintf("[%d:%s] IDX=%d, RT1=%d RT2=%d ST1=%d ST2=%d e=%d t=%d.\n",
      procNum, taskList[taskLoc].name, taskLoc, oldRT, newRT, oldStartTime, 
      newStartTime, getDuration(m->taskID), 
      newStartTime+getDuration(m->taskID));
    */
    for(int i=0;i<task.printevtLen;i++){
    	    char str[1000];
	    strcpy(str, "[%d:%s] ");
	    strcat(str, task.printevts[i].data);
	    parent->CommitPrintf(str,procNum, taskList[taskLoc].name, (double)(task.printevts[i].sTime+newStartTime)/factor); 
    }
    // HACK:
    // look forward and see if there is any standalone or addMsg events
return;
    for (int i=taskLoc+1; i<numTasks; i++) {
      if (!strcmp(taskList[i].name, "standalone")) continue; 
      if (!strcmp(taskList[i].name, "addMsg"))  {
        TaskMsg *tm = new TaskMsg(taskList[i].taskID.srcNode,
		       taskList[i].taskID.msgID,
		       taskList[i].taskID.index,
		       ovt + taskList[i].execTime,
		       0,
		       procNum/numWth,  procNum/numWth,
		       procNum%numWth
		       );
        POSE_invoke(executeTask(tm), BGproc, myHandle, taskList[i].execTime);
        continue;
      }
      break;
    }
  }
}

void BGproc::enableGenTasks(TaskID* taskID, int oldStartTime, int newStartTime)
{
  TaskMsg* generatedTasks = getGeneratedTasks(taskID);
  int numGenTasks = getNumGeneratedTasks(taskID);
  TaskMsg *tm;

  // Given the following inputs for destNode and destTID, we need to 
  // send out messages either directly to the local node, or through
  // the network.  To do this we have to specify correct information
  // about the destination BGproc and the source and destination
  // switch in the network

  // destNode   destTID   Behavior
  // ========== ========= ==============================================
  // -1         -1        Broadcast to ALL worker threads of ALL nodes
  // -1         K         SHOULD NOT HAPPEN???
  // N          -1        Send to ALL worker threads of node N
  // N          K         Send to worker thread K of node N
  // -100-N     -1        Broadcast to all worker threads of all nodes
  //                      except for N (no worker threads of N receive)
  // -100-N     K         Broadcast to all worker threads of all nodes
  //                      except worker K of node N

  for (int i=0; i<numGenTasks; i++) {
    // time units offset in future at which task i is generated
    int taskOffset = generatedTasks[i].receiveTime - oldStartTime;
    int myNode = parent->thisIndex/numWth;
    int srcSwitch = myNode;
    
    CmiAssert(taskOffset >= 0);
          
    //generatedTasks[i].receiveTime = newStartTime + taskOffset;
    //if(generatedTasks[i].receiveTime < 0) abort();  // Sanity check
    CmiAssert(newStartTime + taskOffset >= 0);  // Sanity check

    int destNodeCode = generatedTasks[i].getDestNode();
    int destTID = generatedTasks[i].desttID; 

    if (destNodeCode >= 0) { // send a msg to a specific node
      if (destTID < -1) {
	CkPrintf("ERROR: enableGenTasks: bad destTID %d destNodeCode %d\n",
		 destTID, destNodeCode);
	CmiAbort("");
      }
    }
    else if (destNodeCode == -1) { // broadcast to all nodes
      if (destTID != -1) {
	CkPrintf("ERROR: enableGenTasks: bad destTID %d destNodeCode %d\n",
		 destTID, destNodeCode);
	CmiAbort("");
      }
    }
    else if (destNodeCode <= -100) { // broadcast to all nodes with exceptions
      if (destTID < -1) {
	CkPrintf("ERROR: enableGenTasks: bad destTID %d destNodeCode %d\n",
		 destTID, destNodeCode);
	CmiAbort("");
      }
    }
    else CkPrintf("ERROR: enableGenTasks: bad destNodeCode %d\n",destNodeCode);
    tm = new TaskMsg();
    *tm = generatedTasks[i];
    tm->receiveTime = newStartTime + taskOffset;
    SendMessage(tm, myNode, srcSwitch, destNodeCode, destTID, taskOffset);
    delete tm;
  }
}

void BGproc::SendMessage(TaskMsg *inMsg, int myNode, int srcSwitch,
			 int destNodeCode, int destTID, int taskOffset)
{
  int destSwitch, destNode, j;

  if (destNodeCode >= 0) { // send msg to one node
    destNode = destNodeCode;
    destSwitch = destNode;
    GeneralSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
	       destTID, destNode, taskOffset);
  }
  else if (destNodeCode == -1) { // send msg to all nodes
    CmiAssert(destTID == -1);
    for (j=0; j<totalProcs/numWth; j++) {
      destNode = j;
      destSwitch = destNode;
      GeneralSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
		 destTID, destNode, taskOffset);
    }
  }
  else { // send msg to all nodes except destNodeExcept when destTID=-1
    int destNodeExcept = -100 - destNodeCode;
    for (j=0; j<totalProcs/numWth; j++) {
      destNode = j;
      destSwitch = destNode;
      if ((destNode == destNodeExcept) && (destTID == -1))
        continue; // exclude the whole node
      // no fiddling with destTID necessary; receiver will handle
      GeneralSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
		 destTID, destNode, taskOffset);
    }
  }
}

void BGproc::GeneralSend(TaskMsg *inMsg, int myNode, int srcSwitch,
			int destSwitch, int destNodeCode, int destTID, 
			int destNode, int taskOffset)
{
  if (config.netsim_on) {
    if (srcSwitch == destSwitch) 
      DirectSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
               destTID, destNode, taskOffset);
    else 
      NetSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
	       destTID, destNode, taskOffset);
  }
  else 
    DirectSend(inMsg, myNode, srcSwitch, destSwitch, destNodeCode,
               destTID, destNode, taskOffset);
}

void BGproc::DirectSend(TaskMsg *inMsg, int myNode, int srcSwitch,
		     int destSwitch, int destNodeCode, int destTID, 
		     int destNode, int taskOffset)
{
    TaskMsg *tm = new TaskMsg(inMsg->taskID.srcNode, inMsg->taskID.msgID,
		     inMsg->taskID.index, inMsg->receiveTime,
		     inMsg->msgsize, destNode, destNodeCode, destTID);
    CmiAssert(destNode >= 0);
    int destNodePID = destNode + totalProcs;    // to node PID
    POSE_invoke(recvIncomingMsg(tm), BGnode, destNodePID, taskOffset);
}

void BGproc::NetSend(TaskMsg *inMsg, int myNode, int srcSwitch,
		     int destSwitch, int destNodeCode, int destTID, 
		     int destNode, int taskOffset)
{
  NicMsg *m = new NicMsg;
  m->src = inMsg->taskID.srcNode;
  m->routeInfo.dst = destSwitch;
  m->msgId = inMsg->taskID.msgID;
  m->index = inMsg->taskID.index;
  m->destNodeCode = destNodeCode;
  m->destTID = destTID;
  m->recvTime = inMsg->receiveTime;
  m->totalLen = inMsg->msgsize;
  m->origovt = ovt+taskOffset;
  //CkPrintf("%d BgSim : Sent %d -> %d msgid %d len %d \n",ovt,m->src,m->dst,m->msgId,m->totalLen);
  elapse(START_LATENCY);
  POSE_invoke(recvMsg(m), NetInterface, config.nicStart+m->src, taskOffset);
  elapse(m->totalLen/10);

  //CkPrintf("[NETWORK: BGproc:%d BGnode:%d srcSwitch:%d destSwitch:%d destNodeCode:%d destTID:%d]\n", parent->thisIndex, myNode, srcSwitch, destSwitch, destNodeCode, destTID);
  //parent->CommitPrintf("[NETWORK: BGproc:%d BGnode:%d srcSwitch:%d destSwitch:%d destNodeCode:%d destTID:%d]\n", parent->thisIndex, myNode, srcSwitch, destSwitch, destNodeCode, destTID);
}

void BGproc::executeTask_anti(TaskMsg *m)
{
  restore(this);
  if (usesAntimethods()) {
    int taskLoc = locateTask(m->taskID);
    //    CmiPrintf("[%d] executeTask_anti %d for %d %d %d\n", procNum, taskLoc, m->taskID.srcNode, m->taskID.msgID, m->taskID.index);
    done[taskLoc] = 0;
  }
}

void BGproc::executeTask_commit(TaskMsg *m)
{
  if (proj) {
    Task &task = taskList[locateTask(m->taskID)];
    // pup projections to logs
    if (!binary)  {
      toProjectionsFile p(proj);
      for (int i=0; i<task.projevtLen; i++) {
        int idx = task.projevts[i].index;
        int newStart = task.newStartTime +  task.projevts[i].startTime;
	CmiAssert(idx < numLogs);
        logs[idx].time = newStart/1e9;
        logs[idx].pup(p);
      }
    }
  }
}

void BGproc::terminus()
{
  if (proj && !binary)  {
    toProjectionsFile p(proj);
    logs[numLogs-1].time = ovt/1e9;
    logs[numLogs-1].pup(p);
  }
}


// local methods
void BGproc::buildHash(){
  // make index starting from 1 so that we can use 0 to represent absence
  for(int i=0;i<numTasks;i++){
    msgTable.put(taskList[i].taskID) = i+1;
  }
}

int BGproc::locateTask(TaskID* t){

//  if the index has been already set, use it directly
  if (t->index >= 0) return t->index;
#if 1
  t->index = msgTable.get(*t);
  if (t->index>0) {
    return --t->index;
  }
#else
  for(int i=0;i<numTasks;i++){

    if(*t == taskList[i].taskID) {
      t->index = i;
      return i;
    }
  }
#endif
 
  parent->CommitPrintf("WARNING: TASK NOT FOUND src:%d msg:%d on:%d\n",
	       t->srcNode, t->msgID, parent->thisIndex);
  //abort();
  return -1;
}

void BGproc::markTask(TaskID* t) {
  done[locateTask(t)] = 1;
}

// not used
void BGproc::updateReceiveTime(TaskMsg* m)
{
  // update the task's receive time to timestamp, and set a flag to
  // indicate the update was made. ignore subsequent receive time
  // updates to this taskID

  taskList[locateTask(&(m->taskID))].receiveTime = m->timestamp;
}

int BGproc::dependenciesMet(yourUniqueTaskIDtype taskID)
{
  // tests to see if this taskID is dependent on other tasks that have
  // not yet been executed.  If succeed, mark this task executed.

  int idx = locateTask(taskID);

  int bidx;
  for(int i=0;i<taskList[idx].bDepsLen;i++){
    bidx = locateTask(&taskList[idx].backwardDeps[i]);
//    if(!taskList[bidx].done)
    if(!done[bidx])
      return 0;
  }
  return 1;
}

// not used
void BGproc::updateStartTime(TaskID* taskID, int newTime)
{
  // update the task's start time to newTime
  taskList[locateTask(taskID)].startTime = newTime;
}

void BGproc::enableDependents(TaskID* taskID)
{
  TaskMsg *tm;
  TaskID* ftid;
  int taskIdx = locateTask(taskID);
  int execT = taskList[taskIdx].execTime;
  
  int fDepsLen = taskList[taskIdx].fDepsLen;
  for(int i=0;i<fDepsLen;i++){

    ftid = &taskList[taskIdx].forwardDeps[i];
    // CkPrintf("[Checking dependent task %d %d on %d\n", ftid->index, ftid->msgID, ftid->srcNode); 
    if(dependenciesMet(ftid)){
      // should not have done here
      CmiAssert(done[locateTask(ftid)]==0);

      tm = new TaskMsg(ftid->srcNode,
		       ftid->msgID,
		       ftid->index,
		       ovt + execT,
		       0,
		       procNum/numWth,  procNum/numWth,
		       procNum%numWth
		       );
      
      //CkPrintf("[%d] dependent: executeTask call %d, for %d %d %d\n", procNum, locateTask(ftid), ftid->srcNode, ftid->msgID, ftid->index); 
      POSE_invoke(executeTask(tm), BGproc, myHandle, 0);
    }
  }
}


TaskMsg* BGproc::getGeneratedTasks(yourUniqueTaskIDtype taskID)
{
  return taskList[locateTask(taskID)].taskmsgs;
}


int BGproc::getNumGeneratedTasks(yourUniqueTaskIDtype taskID)
{
  // get the number of generated tasks in the list of tasks generated
  // by this one
  return taskList[locateTask(taskID)].taskMsgsLen;
}

int BGproc::getDuration(TaskID taskID)
{
  // get the duration of this task
  return taskList[locateTask(taskID)].execTime;
}

 
int BGproc::getReceiveTime(TaskID taskID)
{
  // get the receive time of this task
  return taskList[locateTask(taskID)].receiveTime;
}

BGnode::BGnode(BGnodeMsg *m)
{
  procsPerNode = m->procsPerNode;
  switchPID = m->switchID;
  nodePID = parent->thisIndex;		// pose index
  myNodeIndex = m->nodeIdx;		// index in node array
  firstProcPID = myNodeIndex*procsPerNode;  // first BGproc PID
  // CmiPrintf("BGnode (%d %d) with switchPID: %d firstPE: %d\n", nodePID, myNodeIndex, switchPID, firstProcPID);
}

void BGnode::pup(PUP::er &p)
{
  p|procsPerNode;
  p|switchPID;
  p|nodePID;
  p|myNodeIndex;
  p|firstProcPID;
}

void BGnode::recvOutgoingMsg(TaskMsg *m)
{
#if 0
  int destNodeCode = m->destNodeCode;
  int destTID
  int destNode;
  int destSwitch, destProc = -1, j;

  if (destNodeCode >= 0) { // send msg to one node
    destNode = destNodeCode;
    destSwitch = nodeToSwitchPID(destNode);
    if (destTID >= 0)
      destProc = destNode*numWth + destTID;
    SimpleSend(inMsg, myProc, myNode, srcSwitch, destSwitch, destNodeCode,
	       destTID, destNode, destProc, taskOffset);
  }
  else if (destNodeCode == -1) { // send msg to all nodes
    CmiAssert(destTID == -1);
    for (j=0; j<totalProcs/numWth; j++) {
      destNode = j;
      destSwitch = totalProcs + destNode;
      SimpleSend(inMsg, myProc, myNode, srcSwitch, destSwitch, destNodeCode,
		 destTID, destNode, destProc, taskOffset);
    }
  }
  else { // send msg to all nodes except destNodeExcept when destTID=-1
    int destNodeExcept = -100 - destNodeCode;
    for (j=0; j<totalProcs/numWth; j++) {
      destNode = j;
      destSwitch = totalProcs + destNode;
      if ((destNode == destNodeExcept) && (destTID == -1))
        continue; // exclude the whole node
      // no fiddling with destTID necessary; receiver will handle
      SimpleSend(inMsg, myProc, myNode, srcSwitch, destSwitch, destNodeCode,
		 destTID, destNode, destProc, taskOffset);
    }
  }


//???
  if (destNode>=0 && destNode == myNodeIndex) {
    // slef
  }
  else {
    // network
  }
#endif
}
                                                                                
// message send to this node
// called by Switch or local node
void BGnode::recvIncomingMsg(TaskMsg *m)
{ // this function should do ONE of the following:
  // 1) send message to one worker thread on this node
  // 2) send message to all worker threads on this node
  // 3) send messages to all but one worker threads on this node
  CmiAssert(m->destNode == myNodeIndex);

  int destNodeCode = m->destNodeCode;
  int destTID = m->desttID;

  TaskMsg *tm;
  if ((destNodeCode >= 0) && (destTID >= 0)) { // case 1
    int destNode = destNodeCode;
    tm = new TaskMsg(m->taskID.srcNode, m->taskID.msgID,
		     m->taskID.index, m->receiveTime,
		     m->msgsize, destNode, destNodeCode, destTID);
    int destProc = firstProcPID + destTID;
    //    CkPrintf("[%d] incoming: executeTask call %d %d %d\n", destProc, m->taskID.srcNode, m->taskID.msgID, m->taskID.index); 
    POSE_invoke(executeTask(tm), BGproc, destProc, 0);
    //CkPrintf("[Incoming NODE: %d %d destProc:%d]\n", nodePID, myNodeIndex, destProc);
    //parent->CommitPrintf("[Incoming NODE: %d %d destProc:%d]\n", nodePID, myNodeIndex, destProc);
  }
  else {	// case 2 and 3
    int destProc = firstProcPID;
    int destNodeExcept = -1;
    if (destNodeCode < -1) destNodeExcept = -100 - destNodeCode;
    CmiAssert(!(destNodeExcept == myNodeIndex && destTID == -1));
    for (int j=0; j<procsPerNode; j++, destProc++) {
      if (destTID != -1) 
        if (myNodeIndex == destNodeExcept && j == destTID) continue;
      tm = new TaskMsg(m->taskID.srcNode, m->taskID.msgID,
		       m->taskID.index, m->receiveTime,
		       m->msgsize, destNodeCode, destNodeCode, j);
      //    CkPrintf("[%d] incoming: executeTask call %d %d %d\n", destProc, m->taskID.srcNode, m->taskID.msgID, m->taskID.index); 
      POSE_invoke(executeTask(tm), BGproc, destProc, 0);
      //CkPrintf("[Incoming NODE: %d %d bcast destNodeCode:%d destTID:%d destProc:%d]\n", nodePID, myNodeIndex, destNodeCode, destTID, destProc);
      //parent->CommitPrintf("[Incoming NODE: %d %d bcast destNodeCode:%d destTID:%d destProc:%d]\n", nodePID, myNodeIndex, destNodeCode, destTID, destProc);
      //myProc, myNode, destProc);
    }
  }
}

