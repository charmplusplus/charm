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

CkpvStaticDeclare(Trace*, _tracebg);

extern int traceBluegeneLinked;

void _createTracebluegene(char** argv)
{
  //DEBUGF(("%d createTraceBluegene\n", CkMyPe()));
  CkpvInitialize(Trace*, _tracebg);
  CkpvAccess(_tracebg) = new  TraceBluegene(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_tracebg));
  traceBluegeneLinked = 1;
}


// this PE must be trace-enabled(in trace-common.C) to be able to do bgPrint()
static void writeData(void *data, double t, double recvT, void *ptr)
{
  FILE *fp = (FILE *)ptr;
#if 0
  if(fp !=0)
    fprintf(fp,(char*)data,t);
#endif
  TraceBluegene *traceBluegene = (TraceBluegene *)ptr;
  traceBluegene->writePrint((char*)data, t);
}

void TraceBluegene::writePrint(char* str, double t){
  if (pfp == NULL)
    creatFiles();
  fprintf(pfp,str,t);
}

TraceBluegene::TraceBluegene(char** argv)
{
  if(CkMyPe() == 0){
    stsfp = fopen("bgTraceFile", "w");
    if(stsfp==0)
      CmiAbort("Cannot open Bluegene sts file for writing.\n");
  }
  
  pfp = NULL;
}

void TraceBluegene::traceClose() {
  bgUpdateProj(2);
  if(pfp != 0)
    fclose(pfp);
  if((CkMyPe() == 0)&&(stsfp !=0))
    fclose(stsfp);
}

TraceBluegene::~TraceBluegene(){
/*
  bgUpdateProj();
  if(pfp != 0)
    fclose(pfp);
  if((CkMyPe() == 0)&&(stsfp !=0))
    fclose(stsfp);
*/
}

void TraceBluegene::creatFiles()
{
  char* fname = new char[15];
  sprintf(fname,"bgPrintFile%d",CkMyPe());
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


void TraceBluegene::bgBeginExec(char* name,void** parentLogPtr){


  if (!genTimeLog) return;

  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,BgGetCurTime());
  if(*parentLogPtr)
    newLog->addBackwardDep(*(bgTimeLog**)parentLogPtr);
  currLog = newLog;
  BgInsertLog((void*)newLog);
  *parentLogPtr = newLog;
}


void TraceBluegene::bgEndExec(){

  if (!genTimeLog) return;
  currLog->closeLog();
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

void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr){

  if (!genTimeLog) return;

  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,bt,et);
  if(*parentLogPtr)
    newLog->addBackwardDep(*(bgTimeLog**)parentLogPtr);
  *parentLogPtr = newLog;
  currLog = newLog;
  BgInsertLog((void*)newLog);
}


void TraceBluegene::userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList){
   
  if (!genTimeLog) return;

  bgTimeLog* newLog = new bgTimeLog(_threadEP,name,bt,et);
  newLog->addBackwardDeps(bgLogList);
  *parentLogPtr = newLog;
  currLog = newLog;
  BgInsertLog((void*)newLog);
}


void TraceBluegene::traceWriteSts(){
  if (!genTimeLog) return;
  //  CmiPrintf("\n\n\n[%d]In the traceWriteSts before printing logs\n\n\n\n",CkMyPe());
  //if(CkMyPe() == 0)
  // currLog->write(stsfp);
  return;
}

void TraceBluegene::bgPrint(char* str){
  if (!genTimeLog) return;
  bgAddProjEvent(strdup(str), BgGetTime(), writeData, this, 2);

}

/*@}*/

