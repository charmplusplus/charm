#include "charm++.h"
#include "TraceAutoPerf.decl.h"
#include "trace-autoPerf.h"
#include <algorithm>
#include <math.h>
#define TRIGGER_PERF_IDLE_PERCENTAGE 0.1 

#define SMP_ANALYSIS  0 
#define DEBUG_LEVEL 0
#define   CP_PERIOD  100

#define TIMESTEP_RATIO_THRESHOLD 0

#define UTIL_PERCENTAGE   0.95

#if 0 
#define DEBUG_PRINT(x) x  
#else
#define DEBUG_PRINT(x) 
#endif

// trace functions here
#include "trace-perf.C"
CkpvDeclare(savedPerfDatabase*, perfDatabase);
CkpvExtern(int, availAnalyzeCP);
CksvExtern(int, availAnalyzeNodeCP);
CkpvExtern(int, hasPendingAnalysis);
CkpvExtern(int, currentStep);
CkpvExtern(CkCallback, callBackAutoPerfDone);
CkGroupID traceAutoPerfGID;
CProxy_TraceAutoPerfBOC autoPerfProxy;
CProxy_TraceNodeAutoPerfBOC autoPerfNodeProxy;
extern void setNoPendingAnalysis();
extern void startAnalysisonIdle();
extern void autoPerfReset();
//-----------------------utility functions ----------------------
//Reduce summary data
CkReductionMsg *perfDataReduction(int nMsg,CkReductionMsg **msgs){
    perfData *ret;
    if(nMsg > 0){
        ret=(perfData*)msgs[0]->getData();
    }
    for (int i=1;i<nMsg;i++) {
        perfData *m=(perfData*)(msgs[i]->getData());
        // idle time (min/s$um/max)
        ret->idleMin = min(ret->idleMin, m->idleMin);
        ret->idleTotalTime += m->idleTotalTime; 
        ret->idleMax = max(ret->idleMax, m->idleMax);
        // overhead time (min/sum/max)
        ret->overheadMin = min(ret->overheadMin, m->overheadMin);
        ret->overheadTotalTime += m->overheadTotalTime; 
        ret->overheadMax = max(ret->overheadMax, m->overheadMax);
        // util time (min/sum/max)
        ret->utilMin = min(ret->utilMin, m->utilMin);
        ret->utilTotalTime += m->utilTotalTime; 
        ret->utilMax = max(ret->utilMax, m->utilMax);
        // mem usage (max)
        ret->mem =max(ret->mem,m->mem);
        // bytes per invocation for two types of entry methods
        ret->numMsgs += m->numMsgs; 
        ret->numBytes += m->numBytes; 
        ret->commTime += m->commTime; 
        // Grain size (avg, max)
        ret->grainsizeAvg += m->grainsizeAvg;
        ret->grainsizeMax = max(ret->grainsizeMax, m->grainsizeMax);
        //Total invocations
        ret->numInvocations += m->numInvocations;
        ret->objLoadMax = max(ret->objLoadMax, m->objLoadMax);
    }  
    CkReductionMsg *msg= CkReductionMsg::buildNew(sizeof(perfData),ret); 
    return msg;
}

TraceAutoPerfInit::TraceAutoPerfInit(CkArgMsg* args)
{
    traceAutoPerfGID = CProxy_TraceAutoPerfBOC::ckNew();
    autoPerfProxy = CProxy_TraceAutoPerfBOC::ckNew();
    autoPerfNodeProxy = CProxy_TraceNodeAutoPerfBOC::ckNew();
    bool isIdleAnalysis = CmiGetArgFlagDesc(args->argv,"+idleAnalysis","start performance analysis when idle");
    if(isIdleAnalysis){
        CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)startAnalysisonIdle, NULL);
        CcdCallFnAfterOnPE((CcdVoidFn)autoPerfReset, NULL, 10, CmiMyPe());
    }
}

// set the call back function, which is invoked after auto perf is done
void TraceAutoPerfBOC::setAutoPerfDoneCallback(CkCallback cb, bool frameworkShouldAdvancePhase)
{
    CkpvAccess(callBackAutoPerfDone) = cb;
}

//mark time step
void TraceNodeAutoPerfBOC::timeStep(int reductionPE)
{
    getPerfData(reductionPE, CkCallback::ignore );
}

CkReduction::reducerType perfDataReductionType;
void TraceNodeAutoPerfBOC::getPerfData(int reductionPE, CkCallback cb)
{
}

void TraceAutoPerfBOC::timeStep(int reductionPE)
{
    getPerfData(reductionPE, CkCallback::ignore );
}

// Collect local perf data and send results to reductionPE
void TraceAutoPerfBOC::getPerfData(int reductionPE, CkCallback cb)
{
    TraceAutoPerf *t = localAutoPerfTracingInstance();
    t->markStep();
    perfData * data = t->getSummary();
    DEBUG_PRINT (
        t->printSummary();
        )
    CkCallback *cb1 = new CkCallback(CkIndex_TraceAutoPerfBOC::globalPerfAnalyze(NULL), thisProxy[reductionPE]);
    contribute(sizeof(perfData),data,perfDataReductionType, *cb1);
    t->resetAll();
    CkpvAccess(hasPendingAnalysis) = 1;
    CcdCallFnAfterOnPE((CcdVoidFn)setNoPendingAnalysis, NULL, CP_PERIOD, CkMyPe());
}

//check local idle percentage to decide whether trigger global analysis
void TraceAutoPerfBOC::localPerfQuery()
{

    TraceAutoPerf *t = localAutoPerfTracingInstance();
    double idlePercent = t->checkIdleRatioDuringIdle();
    CkpvAccess(currentStep)++;
    if( idlePercent > TRIGGER_PERF_IDLE_PERCENTAGE ) //TUNABLE  
    {
        //CkPrintf("\nTIMER:%f PE:%d idle percentage is HIGH start analysis  %.3f\n", TraceTimer(), CkMyPe(),   idlePercent);
#if SMP_ANALYSIS
        {
            for(int i=0; i<CkNumNodes(); i++)
                autoPerfNodeProxy[i].getPerfData(CkMyNode(), CkCallback::ignore);
        }
#else
        autoPerfProxy.getPerfData(0, CkCallback::ignore);
        //autoPerfProxy.getPerfData(CkMyPe(), CkCallback::ignore);
#endif
    }else if(idlePercent < 0)
    {
        TraceAutoPerf *t = localAutoPerfTracingInstance();
        t->markStep();
        //CkPrintf("%f PE:%d idle percentage is negative %f\n", TraceTimer(), CkMyPe(), idlePercent);
    } else
    {
        //CkPrintf("%f PE:%d idle percentage is okay  %f\n", TraceTimer(), CkMyPe(),idlePercent);
    }
}

//perf data from all processors are collected on one PE, perform analysis based on global data
void TraceAutoPerfBOC::globalPerfAnalyze(CkReductionMsg *msg )
{
    static int counters = 0;
    int level = 0;
    //CkPrintf("\n-------------------------global %d  Timer:%f analyzing------- %d \n\n", CkMyPe(), CkWallTimer(), counters++);
    int size=msg->getSize() / sizeof(double);
    perfData *data=(perfData*) msg->getData();
    double totalTime = data->utilTotalTime  + data->idleTotalTime + data->overheadTotalTime ;
    double idlePercentage = data->idleTotalTime/totalTime;
    double overheadPercentage = data->overheadTotalTime/totalTime;
    double utilPercentage = data->utilTotalTime/totalTime;
    //DEBUG_PRINT ( 
    CkPrintf("Utilization(%):  \t(min:max:avg):(%.1f:\t  %.1f:\t  %.1f)\n", data->utilMin*100, data->utilMax*100, utilPercentage*100 );
    CkPrintf("Idle(%):         \t(min:max:avg):(%.1f:\t  %.1f:\t  %.1f) \n", data->idleMin*100,  data->idleMax*100, idlePercentage*100);
    CkPrintf("Overhead(%):     \t(min:max:avg):(%.1f:\t  %.1f:\t  %.1f) \n", data->overheadMin*100, data->overheadMax*100, overheadPercentage*100);
    CkPrintf("Grainsize(ms):\t(avg:max)\t: (%.3f:    %.3f) \n", data->utilTotalTime/data->numInvocations*1000, data->grainsizeMax*1000);
    CkPrintf("Invocations:  \t%lld\n", data->numInvocations);
    //)
   
    // --- time step measurement 
    double timeElapse = CkWallTimer() - startStepTimer;
    double avgTimeStep = timeElapse/(CkpvAccess(currentStep) - lastAnalyzeStep);
    CkpvAccess(perfDatabase)->insert(avgTimeStep, utilPercentage,  idlePercentage, overheadPercentage); 
    DEBUG_PRINT ( 
        CkPrintf("-------------- timestep --%d:%d--- \n", CkpvAccess(currentStep),  lastAnalyzeStep);
        )
    startStepTimer = CkWallTimer();
    lastAnalyzeStep = CkpvAccess(currentStep);
    //check the performance, and decide whether to tune
    //
    CkpvAccess(callBackAutoPerfDone).send(); 
}

/*
 *  based on the history data, (tunnable parameter values, performance metrics)
 *  generate a performance model using curve fitting.
 */

enum  functionType { LINEAR, SECOND_ORDER, THIRD_ORDER };

void TraceAutoPerfBOC::generatePerfModel()
{
    // a set of performance results is the function value
    // a set of tunable parameter values  is the function variable
    // linear,  second degree polynomial , third degree polynomial 
    // exponential polynomial fit
    // GNU scientific library has tools to do this

    int modelType;
    modelType = LINEAR;
    switch( modelType)
    {
        case LINEAR:

            break;

        case SECOND_ORDER:
            break;

        case THIRD_ORDER:
            break;

        default:
            break;
    }
}

extern "C" void traceAutoPerfExitFunction() {
    CkPrintf("calling before exiting............................\n");
    autoPerfProxy.timeStep(CkMyPe());
    //CkExit();
}
void _initTraceAutoPerfBOC()
{
    perfDataReductionType=CkReduction::addReducer(perfDataReduction);

    CkpvInitialize(int, currentStep);
    CkpvAccess(currentStep) = 0;
    CkpvInitialize(int, hasPendingAnalysis);
    CkpvAccess(hasPendingAnalysis) = 0;
    CkpvInitialize(CkCallback, callBackAutoPerfDone);
    CkpvAccess(callBackAutoPerfDone) = CkCallback::ckExit; 
    CkpvInitialize(savedPerfDatabase*, perfDatabase);
    CkpvAccess(perfDatabase) = new savedPerfDatabase();
#ifdef __BIGSIM__
    if (BgNodeRank()==0) {
#else               
    if (CkMyRank() == 0) {
#endif
            registerExitFn(traceAutoPerfExitFunction);
        }
}

void _initTraceNodeAutoPerfBOC()
{
    CksvInitialize(int, availAnalyzeNodeCP);
    CksvAccess(availAnalyzeNodeCP) = 1;
}
#include "TraceAutoPerf.def.h"
