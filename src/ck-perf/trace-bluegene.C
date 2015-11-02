
/**
 * \addtogroup CkPerf
*/
/*@{*/


#include "charm++.h"
#include "envelope.h"
#include "trace-common.h"
#include "trace-bluegene.h"
#include "blue.h"
#include "blue_impl.h"

#undef DEBUGF
#define DEBUGF(x)  // CmiPrintf x

void _createTracebluegene(char** argv)
{
  DEBUGF(("%d createTraceBluegene\n", CkMyPe()));
  CkpvInitialize(TraceBluegene*, _tracebg);
  CkpvAccess(_tracebg) = new  TraceBluegene(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_tracebg));
  traceBluegeneLinked = 1;
}


// this PE must be trace-enabled(in trace-common.C) to be able to do bgPrint()
static void writeData(void *data, double t, double recvT, void *ptr)
{
  TraceBluegene *traceBluegene = (TraceBluegene *)ptr;
  CmiAssert(traceBluegene);
  traceBluegene->writePrint((char*)data, t);
}

void TraceBluegene::writePrint(char* str, double t){
  if (pfp == NULL)
    creatFiles();
  fprintf(pfp,"[%d] ", CkMyPe());
  fprintf(pfp,str,t);
}

TraceBluegene::TraceBluegene(char** argv): pfp(NULL)
{
}

void TraceBluegene::traceClose() {
  DEBUGF(("%d TraceBluegene::traceClose\n", CkMyPe()));
  bgUpdateProj(2);
  if(pfp != 0)  fclose(pfp);
  pfp = NULL;
  CkpvAccess(_traces)->removeTrace(this);
}

TraceBluegene::~TraceBluegene(){
}

CpvExtern(BGMach, bgMach);
void TraceBluegene::creatFiles()
{
  char* fname = new char[1024];
  sprintf(fname, "%sbgPrintFile.%d", cva(bgMach).traceroot?cva(bgMach).traceroot:"", CkMyPe()); 
  pfp = fopen(fname,"w");     
  if(pfp==NULL)
    CmiAbort("Cannot open Bluegene print file for writing.\n");
  delete[] fname;
}

void TraceBluegene::tlineEnd(void** parentLogPtr){
  if(genTimeLog)
    *parentLogPtr = (void*)BgLastLog(tTIMELINEREC);
  else
    *parentLogPtr = NULL;
}

void TraceBluegene::bgAddTag(const char* str){
  if (!genTimeLog) return;
  BgTimeLog * log = BgLastLog(tTIMELINEREC);
  CmiAssert(log != NULL);
  log->setName(str);
}

void TraceBluegene::bgDummyBeginExec(const char* name,void** parentLogPtr, int split)
{
  if (genTimeLog) {
  CmiAssert(parentLogPtr!=NULL);
  double startTime = BgGetCurTime();
  BgTimeLog* newLog = BgStartLogByName(tTIMELINEREC, _threadEP, name, startTime, *(BgTimeLog**)parentLogPtr);
  // if event's mesgID is (-1:-1) and there is no backward dependence
  // to avoid timestamp correction, set a fake recv time so that it stays here
  if (*parentLogPtr == NULL)
    newLog->recvTime = startTime;
  else {
    if (split) {
      newLog->objId = (*(BgTimeLog**)parentLogPtr)->objId;
      newLog->charm_ep = (*(BgTimeLog**)parentLogPtr)->charm_ep;
    }
  }
  *parentLogPtr = newLog;
  }
  startVTimer();
}

void TraceBluegene::bgBeginExec(char* msg, char *name)
{
  startVTimer();
  if (!genTimeLog) return;
  BgTimeLog* newLog = new BgTimeLog(msg, name);
  tTIMELINEREC.logEntryStart(newLog);
}

// mark a new log, which depends on log
void TraceBluegene::bgSetInfo(char *msg, const char *name, void **logs, int count)
{
  if (!genTimeLog) return;
  BgTimeLog * curlog = BgLastLog(tTIMELINEREC);
  if (name != NULL) curlog->setName(name);
  for (int i=0; i<count; i++)
      curlog->addBackwardDep((BgTimeLog*)logs[i]);
  if (msg) curlog->addMsgBackwardDep(tTIMELINEREC, msg);
}

// mark a new log, which depends on log
void TraceBluegene::bgAmpiBeginExec(char *msg, char *name, void **logs, int count)
{
  startVTimer();
  if (!genTimeLog) return;
  BgTimeLog * curlog = BgLastLog(tTIMELINEREC);
  curlog->setName(name);
  for (int i=0; i<count; i++)
      curlog->addBackwardDep((BgTimeLog*)logs[i]);
  if (msg) curlog->addMsgBackwardDep(tTIMELINEREC, msg);
}

void TraceBluegene::bgAmpiLog(unsigned short op, unsigned int dataSize)
{
    if (!genTimeLog) return;
    BgTimeLog *curlog = BgLastLog(tTIMELINEREC);
    curlog->mpiOp = op;
    curlog->mpiSize = dataSize;
}

void TraceBluegene::bgEndExec(int commit)
{
  stopVTimer();
  if (!genTimeLog) return;
  if (commit) 
    BgLogEntryCommit(tTIMELINEREC);
  else
    BgEndLastLog(tTIMELINEREC);
}

void TraceBluegene::beginExecute(envelope *e, void *obj)
{
  if (e==NULL || !genTimeLog) return;
  BgTimeLog* log = tTIMELINE[tTIMELINE.length()-1];
  CmiAssert(log!=NULL);
  log->setCharmEP(e->getEpIdx());
}

void TraceBluegene::beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx, void *obj)
{
  //printf("SET OBJ ID\n");
  BgTimeLog* log;
  if(genTimeLog)
    log = tTIMELINE[tTIMELINE.length()-1];
  else
    return;
  if (idx!=NULL) log->setObjId(idx);
  log->setCharmEP(ep);
}

void TraceBluegene::getForwardDep(void* log, void** fDepPtr){

  BgTimeLog* cLog = (BgTimeLog*) log;
  
  if(cLog->forwardDeps.length() !=1) {
    cLog->write(stdout);
    CkAbort("Quitting\n");
  }
  *fDepPtr = (void*)(cLog->forwardDeps[0]);
}

void TraceBluegene::getForwardDepForAll(void** logs1, void** logs2, int logsize,void* fDepPtr){
  if(!genTimeLog) return;

  CmiAssert(logsize>0);
  BgTimeLog* cLog = (BgTimeLog*)fDepPtr;

  int i=0;

  // find the valid sdag overlap pointer
  for(i=0;i< logsize+1;i++)
    if(logs2[i])
      break;    
  
  if (i<logsize+1) {
    cLog->addBackwardDep((BgTimeLog*)logs2[i]);
  }
  // CmiAssert(i<logsize+1);
  
  cLog->objId = ((BgTimeLog*)logs1[0])->objId;    // sdag objID
  for(int j=0;j<logsize;j++)  {
      cLog->addBackwardDep((BgTimeLog*)(logs1[j]));
      CmiAssert(cLog->objId == ((BgTimeLog*)logs1[j])->objId);
  }
}

void TraceBluegene::addBackwardDep(void *log)
{
  if(!genTimeLog || log==NULL) return;
  BgTimeLog  *parentLogPtr = BgLastLog(tTIMELINEREC);
  CmiAssert(parentLogPtr);
  BgAddBackwardDep(parentLogPtr, (BgTimeLog*)log);
}

void TraceBluegene::userBracketEvent(const char* name, double bt, double et, void** parentLogPtr){

  if (!genTimeLog) return;

  BgTimeLog* newLog = new BgTimeLog(_threadEP,name,bt,et);
  if(*parentLogPtr) {
    newLog->addBackwardDep(*(BgTimeLog**)parentLogPtr);
    newLog->objId = (*(BgTimeLog**)parentLogPtr)->objId;        // sdag objID
  }
  *parentLogPtr = newLog;
  CmiAssert(*parentLogPtr != NULL);
  tTIMELINEREC.logEntryInsert(newLog);
}

void TraceBluegene::userBracketEvent(const char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList){
   
  if (!genTimeLog) return;

  BgTimeLog* newLog = new BgTimeLog(_threadEP,name,bt,et);
  newLog->addBackwardDeps(bgLogList);
  CmiAssert(bgLogList.size()>0);
  newLog->objId = ((BgTimeLog*)bgLogList[0])->objId;   // for sdag
  *parentLogPtr = newLog;
  tTIMELINEREC.logEntryInsert(newLog);
}

void TraceBluegene::bgPrint(const char* str){
  if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)) CmiDisableIsomalloc();
  double curT = BgGetTime();
  if (genTimeLog)
    bgAddProjEvent(strdup(str), -1, curT, writeData, this, BG_EVENT_PRINT);
  CmiPrintf(str, curT);
  if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)) CmiEnableIsomalloc();
}

extern "C" void BgPrintf(const char *str)
{
  BgPrint(str);
}

void TraceBluegene::bgMark(const char* str){
  double curT = BgGetTime();
  if (genTimeLog)
    bgAddProjEvent(strdup(str), -1, curT, writeData, this, BG_EVENT_MARK);
}

extern "C" void BgMark(const char *str)
{
  BgMark_(str);
}

extern "C" void BgSetStartEvent()
{
  BgTimeLog* log;
  if(genTimeLog)
    log = tTIMELINE[tTIMELINE.length()-1];
  else
    return;
  log->setStartEvent();
}

/*@}*/

