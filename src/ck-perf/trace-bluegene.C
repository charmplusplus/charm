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
    *parentLogPtr = (void*)tTIMELINE[tTIMELINE.length()-1];
  else
    *parentLogPtr = NULL;
}

void TraceBluegene::bgDummyBeginExec(char* name,void** parentLogPtr)
{
  if (!genTimeLog) return;
  startVTimer();
  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,BgGetCurTime());
  if(*parentLogPtr)
    newLog->addBackwardDep(*(bgTimeLog**)parentLogPtr);
  tTIMELINEREC.logEntryStart(newLog);
  *parentLogPtr = newLog;
}

void TraceBluegene::bgBeginExec(char* msg, char *name)
{
  if (!genTimeLog) return;
  startVTimer();
  bgTimeLog* newLog = new bgTimeLog(msg, name);
  tTIMELINEREC.logEntryStart(newLog);
}

// create a new log, which depends on log
void TraceBluegene::bgAmpiBeginExec(char *msg, char *name, void *log)
{
  if (!genTimeLog) return;
  startVTimer();
  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,BgGetCurTime());
  tTIMELINEREC.logEntryStart(newLog);
  newLog->addBackwardDep((bgTimeLog*)log);
  newLog->addMsgBackwardDep(tTIMELINEREC, msg);
}

void TraceBluegene::bgEndExec(int commit)
{
  if (!genTimeLog) return;
  if (commit) 
    BgLogEntryCommit(tTIMELINEREC);
  else
    tTIMELINEREC.logEntryClose();
  stopVTimer();
}


void TraceBluegene::getForwardDep(void* log, void** fDepPtr){

  bgTimeLog* cLog = (bgTimeLog*) log;
  
  if(cLog->forwardDeps.length() !=1) {
    cLog->write(stdout);
    CkAbort("Quitting\n");
  }
  *fDepPtr = (void*)(cLog->forwardDeps[0]);
}

void TraceBluegene::getForwardDepForAll(void** logs1, void** logs2, int logsize,void* fDepPtr){
  if(!genTimeLog) return;

  bgTimeLog* cLog = (bgTimeLog*)fDepPtr;

  int i=0;

  // find the valid sdag overlap pointer
  for(i=0;i< logsize+1;i++)
    if(logs2[i])
      break;    
  
  if (i<logsize+1) {
    cLog->addBackwardDep((bgTimeLog*)logs2[i]);
  }
  // CmiAssert(i<logsize+1);
  
  for(int j=0;j<logsize;j++)   
      cLog->addBackwardDep((bgTimeLog*)(logs1[j]));
}

void TraceBluegene::addBackwardDep(void *log)
{
  if(!genTimeLog || log==NULL) return;
  CmiAssert(tTIMELINE.length() > 0);
  bgTimeLog  *parentLogPtr = (bgTimeLog*)tTIMELINE[tTIMELINE.length()-1];
  CmiAssert(parentLogPtr);
  parentLogPtr->addBackwardDep((bgTimeLog*)log);
}

void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr){

  if (!genTimeLog) return;

  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,bt,et);
  if(*parentLogPtr)
    newLog->addBackwardDep(*(bgTimeLog**)parentLogPtr);
  *parentLogPtr = newLog;
  tTIMELINEREC.logEntryInsert(newLog);
}


void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList){
   
  if (!genTimeLog) return;

  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,bt,et);
  newLog->addBackwardDeps(bgLogList);
  *parentLogPtr = newLog;
  tTIMELINEREC.logEntryInsert(newLog);
}


void TraceBluegene::bgPrint(char* str){
  if (!genTimeLog) return;
  bgAddProjEvent(strdup(str), -1, BgGetTime(), writeData, this, BG_EVENT_PRINT);

}

extern "C" void BgPrintf(char *str)
{
  BgPrint(str);
}

/*@}*/

