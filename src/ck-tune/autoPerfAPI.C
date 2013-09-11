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

#include "trace-autoPerf.h"
#include "autoPerfAPI.h"
#define PERF_FREQUENCY 1
#define   CP_PERIOD  100

CkpvDeclare(int, availAnalyzeCP);
CksvDeclare(int, availAnalyzeNodeCP);
CkpvDeclare(int, hasPendingAnalysis);
CkpvDeclare(int, currentStep);
CkpvDeclare(CkCallback, callBackAutoPerfDone);

void autoPerfGlobalNextStep( )
{
    CkpvAccess(currentStep)++;
    if(CkpvAccess(currentStep) % PERF_FREQUENCY == 0)
        autoPerfProxy.timeStep(CkMyPe());
    else
        CkpvAccess(callBackAutoPerfDone).send(); 
}

void autoPerfLocalNextStep( )
{
    CkpvAccess(currentStep)++;
    if(CkpvAccess(currentStep) % PERF_FREQUENCY == 0)
        autoPerfProxy.ckLocalBranch()->timeStep(CkMyPe());
    else
        CkpvAccess(callBackAutoPerfDone).send(); 
}

void startAnalysisonIdle()
{
    if(traceAutoPerfGID.idx !=0 && ((CkGroupID)autoPerfProxy).idx != 0 && CksvAccess(availAnalyzeNodeCP) == 1 && CkpvAccess(hasPendingAnalysis) == 0 )
    {
        CksvAccess(availAnalyzeNodeCP) = 0;
        CcdCallFnAfterOnPE((CcdVoidFn)autoPerfReset, NULL, CP_PERIOD, CkMyPe());
        autoPerfProxy.ckLocalBranch()->localPerfQuery();
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

void registerAutoPerfDone(CkCallback cb, bool frameworkShouldAdvancePhase){
    CkAssert(CkMyPe() == 0);
    autoPerfProxy.setAutoPerfDoneCallback(cb, frameworkShouldAdvancePhase);
}

