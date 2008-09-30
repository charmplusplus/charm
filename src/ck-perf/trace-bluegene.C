/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


/**
 * \addtogroup CkPerf
*/
/*@{*/


#include "charm++.h"
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
  FILE *fp = (FILE *)ptr;
  TraceBluegene *traceBluegene = (TraceBluegene *)ptr;
  CmiAssert(fp && traceBluegene);
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
}

void TraceBluegene::tlineEnd(void** parentLogPtr){
  if(genTimeLog)
    *parentLogPtr = (void*)BgLastLog(tTIMELINEREC);
  else
    *parentLogPtr = NULL;
}

void TraceBluegene::bgAddTag(char* str){
  if (!genTimeLog) return;
  BgTimeLog * log = BgLastLog(tTIMELINEREC);
  CmiAssert(log != NULL);
  log->setName(str);
}

void TraceBluegene::bgDummyBeginExec(char* name,void** parentLogPtr)
{
  startVTimer();
  if (!genTimeLog) return;
  double startTime = BgGetCurTime();
  BgTimeLog* newLog = BgStartLogByName(tTIMELINEREC, _threadEP, name, startTime, *(BgTimeLog**)parentLogPtr);
  // if event's mesgID is (-1:-1) and there is no backward dependence
  // to avoid timestamp correction, set a fake recv time so that it stays here
  if (*parentLogPtr == NULL) newLog->recvTime = startTime;
  *parentLogPtr = newLog;
}

void TraceBluegene::bgBeginExec(char* msg, char *name)
{
  startVTimer();
  if (!genTimeLog) return;
  BgTimeLog* newLog = new BgTimeLog(msg, name);
  tTIMELINEREC.logEntryStart(newLog);
  // bypass
  resetVTime();
}

// mark a new log, which depends on log
void TraceBluegene::bgSetInfo(char *msg, char *name, void **logs, int count)
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

void TraceBluegene::bgEndExec(int commit)
{
  stopVTimer();
  if (!genTimeLog) return;
  if (commit) 
    BgLogEntryCommit(tTIMELINEREC);
  else
    BgEndLastLog(tTIMELINEREC);
}

void TraceBluegene::beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx)
{
  //printf("SET OBJ ID\n");
  if (idx == NULL) return;
  BgTimeLog* log;
  if(genTimeLog)
    log = tTIMELINE[tTIMELINE.length()-1];
  else
    return;
  log->setObjId(idx);
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
  
  for(int j=0;j<logsize;j++)   
      cLog->addBackwardDep((BgTimeLog*)(logs1[j]));
}

void TraceBluegene::addBackwardDep(void *log)
{
  if(!genTimeLog || log==NULL) return;
  BgTimeLog  *parentLogPtr = BgLastLog(tTIMELINEREC);
  CmiAssert(parentLogPtr);
  BgAddBackwardDep(parentLogPtr, (BgTimeLog*)log);
}

void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr){

  if (!genTimeLog) return;

  BgTimeLog* newLog = new BgTimeLog(_threadEP,name,bt,et);
  if(*parentLogPtr)
    newLog->addBackwardDep(*(BgTimeLog**)parentLogPtr);
  *parentLogPtr = newLog;
  CmiAssert(*parentLogPtr != NULL);
  tTIMELINEREC.logEntryInsert(newLog);
}


void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList){
   
  if (!genTimeLog) return;

  BgTimeLog* newLog = new BgTimeLog(_threadEP,name,bt,et);
  newLog->addBackwardDeps(bgLogList);
  *parentLogPtr = newLog;
  tTIMELINEREC.logEntryInsert(newLog);
}

extern "C" void isomalloc_push();
extern "C" void isomalloc_pop();

void TraceBluegene::bgPrint(char* str){
  if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)) isomalloc_push();
  double curT = BgGetTime();
  if (genTimeLog)
    bgAddProjEvent(strdup(str), -1, curT, writeData, this, BG_EVENT_PRINT);
  CmiPrintf(str, curT);
  // bypass
  resetVTime();
  if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)) isomalloc_pop();
}

extern "C" void BgPrintf(char *str)
{
  BgPrint(str);
}

void TraceBluegene::bgMark(char* str){
  double curT = BgGetTime();
  if (genTimeLog)
    bgAddProjEvent(strdup(str), -1, curT, writeData, this, BG_EVENT_MARK);
  // bypass
  resetVTime();
}

extern "C" void BgMark(char *str)
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

