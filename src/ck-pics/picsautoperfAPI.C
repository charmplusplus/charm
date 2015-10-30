/*
 * =====================================================================================
 *
 *       Filename:  autoPerfAPI.C
 *
 *    Description: API for users to use Control Points 
 *
 *        Version:  1.0
 *        Created:  03/03/2013 05:25:52 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yanhua Sun(), 
 *   Organization:  uiuc
 *
 * =====================================================================================
 */
#include "picsdefs.h"
#include "picsdefscpp.h"
#include "picsautoperf.h"
#include "picsautoperfAPI.h"
#include "picsautoperfAPIC.h"
#define PERF_FREQUENCY 1
#define   CP_PERIOD  100

extern int WARMUP_STEP;
extern int PAUSE_STEP;
CkpvDeclare(int, currentStep);
CkpvDeclare(int, availAnalyzeCP);
CksvDeclare(int, availAnalyzeNodeCP);
CkpvDeclare(int, hasPendingAnalysis);
CkpvDeclare(CkCallback, callBackAutoPerfDone);

void PICS_registerAutoPerfDone(CkCallback cb, int frameworkShouldAdvancePhase){
  CkAssert(CkMyPe() == 0);
  autoPerfProxy.setAutoPerfDoneCallback(cb);
}

void PICS_setNumOfPhases(int fromGlobal, int num, char *names[]) {
  char seqNames[num*40];
  for(int i=0; i<num; i++)
  {
    strcpy(seqNames+i*40, names[i]); 
  }
  if(fromGlobal)
    autoPerfProxy.setNumOfPhases(num, seqNames);
  else
    autoPerfProxy.ckLocalBranch()->setNumOfPhases(num, seqNames);
}

void PICS_startPhase( int fromGlobal, int phaseId)
{
  if(fromGlobal)
    autoPerfProxy.startPhase(phaseId);
  else
    autoPerfProxy.ckLocalBranch()->startPhase(phaseId);
}

void PICS_endPhase( int fromGlobal)
{
  if(fromGlobal)
    autoPerfProxy.endPhase();
  else
    autoPerfProxy.ckLocalBranch()->endPhase();
}

void PICS_startStep(int fromGlobal)
{
  if(fromGlobal)
    autoPerfProxy.startStep();
  else
    autoPerfProxy.ckLocalBranch()->startStep();
}

void PICS_endStep(int fromGlobal )
{
  if(fromGlobal)
    autoPerfProxy.endStep(fromGlobal, CkMyPe(), 1);
  else
    autoPerfProxy.ckLocalBranch()->endStep(fromGlobal, CkMyPe(), 1);
}

void PICS_endStepInc(int fromGlobal, int incSteps  )
{
  if(fromGlobal)
    autoPerfProxy.endStep(fromGlobal, CkMyPe(), incSteps);
  else
    autoPerfProxy.ckLocalBranch()->endStep(fromGlobal, CkMyPe(), incSteps);
}



void PICS_endStepResumeCb( int fromGlobal, CkCallback cb)
{
  if(fromGlobal) {
    autoPerfProxy.endStepResumeCb(true, CkMyPe(), cb);
  }
  else
  {
    autoPerfProxy.ckLocalBranch()->endStepResumeCb(false, CkMyPe(), cb);
  }
}

void PICS_autoPerfRun( )
{
  autoPerfProxy.run(true, CkMyPe());
}

void PICS_autoPerfRunResumeCb(CkCallback cb )
{
  autoPerfProxy.setCbAndRun(true, CkMyPe(), cb);
}

void PICS_localAutoPerfRun( )
{
  autoPerfProxy.ckLocalBranch()->run(false,CkMyPe());
}

//called by PE0
void startAnalysis()
{
  autoPerfProxy.endPhaseAndStep(true, CkMyPe());
}

void PICS_SetAutoTimer(){
  CcdCallFnAfterOnPE((CcdVoidFn)startAnalysis, NULL, 100, CkMyPe());
}

void startAnalysisonIdle()
{
  if(traceAutoPerfGID.idx !=0 && ((CkGroupID)autoPerfProxy).idx != 0 && CksvAccess(availAnalyzeNodeCP) == 1 && CkpvAccess(hasPendingAnalysis) == 0 )
  {
    CksvAccess(availAnalyzeNodeCP) = 0;
    CcdCallFnAfterOnPE((CcdVoidFn)autoPerfReset, NULL, CP_PERIOD, CkMyPe());
  }
}

void autoPerfReset()
{
  CksvAccess(availAnalyzeNodeCP) = 1;
}

void setNoPendingAnalysis()
{
  CkpvAccess(hasPendingAnalysis) = 0;
}

void registerPerfGoal(int goalIndex) 
{
  autoPerfProxy.registerPerfGoal(goalIndex);
}

void setUserDefinedGoal(double value)
{
  autoPerfProxy.setUserDefinedGoal(value);
}

void PICS_setCollectionMode(int m) {
   setCollectionMode(m); 
}

void PICS_setEvaluationMode(int m) {
  setEvaluationMode(m);
}

void PICS_markLDBStart(int appStep) {
  autoPerfProxy.PICS_markLDBStart(appStep);
}

void PICS_markLDBEnd(){
  autoPerfProxy.PICS_markLDBEnd();
}

void PICS_setWarmUpSteps(int steps){
  WARMUP_STEP = steps;
}


void PICS_setPauseSteps(int steps){
  PAUSE_STEP = steps;
}
