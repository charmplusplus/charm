#ifndef  TRACE__AUTOPERF__H__
#define  TRACE__AUTOPERF__H__
#define _VERBOSE_H

#include <stdio.h>
#include <errno.h>
#include "charm++.h"
#include "trace.h"
#include "envelope.h"
#include "register.h"
#include "trace-common.h"
#include "TraceAutoPerf.decl.h"
#include "trace-projections.h"
#include <vector>
#include <map>
#include <list>

using namespace std;

extern CkGroupID traceAutoPerfGID;
extern CProxy_TraceAutoPerfBOC autoPerfProxy;
extern CProxy_TraceNodeAutoPerfBOC autoPerfNodeProxy;
// class to store performance data on each PE


class perfMetric
{
public:
    double timeStep;
    double utilPercentage;
    double overheadPercentage;
    double idlePercentage;

    perfMetric(double step, double util, double idle, double overhead)
    {
        timeStep = step;
        idlePercentage = idle;
        overheadPercentage = overhead;
        utilPercentage = util;
    }
};


class savedPerfDatabase
{
private:
    std::list<perfMetric*> perfList;
    perfMetric *previous;
    perfMetric *current;

public:
    savedPerfDatabase() {}

    void insert(double timestep, double idleP, double utilP, double overheadP) {
        if(perfList.size() ==0)
        {
            previous = current= new perfMetric(timestep, utilP, idleP, overheadP);
        }
        else if(perfList.size() < 10)
        {
            //only save 10 iterations to save memory
            previous = (perfMetric*)perfList.back();
            current = new perfMetric(timestep, utilP, idleP, overheadP);
        }
        else
        {
            previous = (perfMetric*)perfList.back();
            current = (perfMetric*) perfList.front();
            perfList.pop_front();
            current->timeStep = timestep;
            current->utilPercentage = utilP;
            current->idlePercentage = idleP;
            current->overheadPercentage = overheadP;
        }
        perfList.push_back(current);
    }

    void getData(int i)
    {

    }

    bool timeStepLonger()
    {
        return current->timeStep > previous->timeStep;
    }

    double getCurrentTimestep()
    {
        return current->timeStep; 
    }

    double getPreviousTimestep()
    {
        return previous->timeStep;
    }

    double getTimestepRatio()
    {
        CkPrintf("Time step changes from %f to %f \n", previous->timeStep, current->timeStep);
        return current->timeStep/previous->timeStep;
    }
    
    double getUtilRatio()
    {
       return current->utilPercentage/ previous->utilPercentage; 
    }

    double getCurrentIdlePercentage()
    {
        return current->idlePercentage;
    }
    
    double getPreviousIdlePercentage()
    {
        return previous->idlePercentage;
    }

    double getIdleRatio()
    {
        return  current->idlePercentage/previous->idlePercentage;
    }
    double getCurrentOverheadPercentage()
    {
        return current->overheadPercentage;
    }
    
    double getPreviousOverheadPercentage()
    {
        return previous->overheadPercentage;
    }
    
    double getOverheadRatio()
    {
        return current->overheadPercentage/previous->overheadPercentage;
    }

    void getAllTimeSteps(double *y)
    {
       int i=0; 

       for(std::list<perfMetric*>::iterator it=perfList.begin(); it != perfList.end(); it++,i++)
       {
           y[i] = (*it)->timeStep;
       }
    }
};


class perfData 
{
public:
    double idleMin;
    double idleTotalTime;
    double idleMax;
    
    double utilMin;
    double utilTotalTime;
    double utilMax;
   
    double overheadMin;
    double overheadTotalTime;
    double overheadMax;

    double mem;
    
    double grainsizeAvg;
    double grainsizeMax;
    
    long   numInvocations;
    
    // communication related data 
    long    numMsgs;
    long    numBytes;
    double  commTime;
    double  objLoadMax;

#if CMK_HAS_COUNTER_PAPI
    LONG_LONG_PAPI papiValues[NUMPAPIEVENTS];
#endif
    // functions
    perfData(){}
};


typedef struct {
    double packing;
    double unpacking;

} sideSummary_t;

typedef struct{
    double beginTimer;
    double endTimer;
}timerPair;

//map<int, double> ObjectLoadTime;

class TraceAutoPerfInit : public Chare {

public:
    TraceAutoPerfInit(CkArgMsg*);

    TraceAutoPerfInit(CkMigrateMessage *m):Chare(m) {}
};


class TraceAutoPerfBOC : public CBase_TraceAutoPerfBOC {
private:
    int         lastAnalyzeStep;   
    double      startStepTimer;
public:
    TraceAutoPerfBOC() {
        startStepTimer = CkWallTimer();
        lastAnalyzeStep = 0;
    }

    TraceAutoPerfBOC(CkMigrateMessage *m) : CBase_TraceAutoPerfBOC(m) {};

    void pup(PUP::er &p)
    {
        CBase_TraceAutoPerfBOC::pup(p);
    }

    void setAutoPerfDoneCallback(CkCallback cb, bool frameworkShouldAdvancePhase); 
    void timeStep(int);
    void getPerfData(int reductionPE, CkCallback cb);
    void globalPerfAnalyze(CkReductionMsg *msg);
    void localPerfQuery();
    void generatePerfModel();

};

//SMP mode
class TraceNodeAutoPerfBOC : public CBase_TraceNodeAutoPerfBOC {

public:
    TraceNodeAutoPerfBOC(void) {}
    TraceNodeAutoPerfBOC(CkMigrateMessage *m) : CBase_TraceNodeAutoPerfBOC(m) {};

    void timeStep(int);
    void getPerfData(int reductionPE, CkCallback cb);

};


class TraceAutoPerf : public Trace {

    friend class TraceAutoPerfBOC;

public:

#if CMK_HAS_COUNTER_PAPI
    LONG_LONG_PAPI previous_papiValues[NUMPAPIEVENTS];
#endif
    double  lastBeginExecuteTime;
    int     lastbeginMessageSize;
    int     lastEvent;
    /** The start of the idle region */
    double  lastBeginIdle;

    /** Amount of time spent so far in untraced regions */
    double totalUntracedTime;

    /** When tracing was suspended (0 if not currently suspended) */
    double whenStoppedTracing;

    /** The amount of time spent executing entry methods since we last reset the counters */
    double totalEntryMethodTime;

    /** The amount of time spent idle since we last reset the counters */
    double totalIdleTime;

    /* * maximum excution time of a single entry method */
    double maxEntryMethodTime;

    /** The highest seen memory usage  since we last reset the counters */
    double memUsage;

    /** The number of entry method invocations since we last reset the counters */
    long totalEntryMethodInvocations;

    /** The time we last rest the counters */
    double lastResetTime;

    double phaseEndTime;

    /* * summary data */
    perfData *currentSummary; 

    vector<timerPair> phasesTimers;

    int currentGroupID;
    CkArrayIndex currentIndex;
    map<int, map<CkArrayIndex, double> > ObjectLoadTime;

    // In some programs like Changa, entry methods may be nested, and hence we only want to consider the outermost one
    int nesting_level;

    TraceAutoPerf(char **argv);
  
    //begin/end tracing
    void traceBegin(void);
    void traceEnd(void);


  // a user event has just occured
  void userEvent(int eventID);
  // a pair of begin/end user event has just occured
  void userBracketEvent(int eventID, double bt, double et);
  
  // "creation" of message(s) - message Sends
  void creation(envelope *, int epIdx, int num=1);
  void creationMulticast(envelope *, int epIdx, int num=1, int *pelist=NULL);
  void creationDone(int num=1);
  
  void messageRecv(char *env, int pe);
  
  void beginExecute(envelope *);
  void beginExecute(CmiObjId *tid);
  
  void beginExecute(
            envelope* env,
            int event,   // event type defined in trace-common.h
		    int msgType, // message type
		    int ep,      // Charm++ entry point id
		    int srcPe,   // Which PE originated the call
		    int ml,      // message size
		    CmiObjId* idx);    // index

  
  void beginExecute(
		    int event,   // event type defined in trace-common.h
		    int msgType, // message type
		    int ep,      // Charm++ entry point id
		    int srcPe,   // Which PE originated the call
		    int ml,      // message size
		    CmiObjId* idx);    // index
  void endExecute(void);
  
  // begin/end idle time for this pe
  void beginIdle(double curWallTime);
  void endIdle(double curWallTime);
  
  // begin/end of execution
  void beginComputation(void);
  void endComputation(void);
  
  /* Memory tracing */
  void malloc(void *where, int size, void **stack, int stackSize);
  void free(void *where, int size);
  
  // do any clean-up necessary for tracing
  void traceClose();

  // ==================================================================
  /** reset the idle time and entry method execution time accumulators */
  void resetTimings();
  /** Reset the idle, overhead, and memory measurements */
  void resetAll();

  /*  mark one phase (to record begin and end timer ) */
  void markStep();

  /** Fraction of the time spent idle since resetting the counters */
  inline double checkIdleRatioDuringIdle() 
  {
      if(lastEvent == BEGIN_IDLE)
          return (totalIdleTime + TraceTimer() - lastBeginIdle ) / ( TraceTimer()-lastResetTime);  
      else
          return (totalIdleTime) / ( TraceTimer()-lastResetTime);  

  }

  inline double idleRatio(){
      if(lastEvent == BEGIN_IDLE)
          totalIdleTime += (TraceTimer() - lastBeginIdle);
      return (totalIdleTime) / totalTraceTime();
  }

  inline double idleTime()
  {
      if(lastEvent == BEGIN_IDLE)
          totalIdleTime += (TraceTimer() - lastBeginIdle);
      return totalIdleTime;
  }

  inline double untracedTime(){
    if(whenStoppedTracing <= 0){
      return totalUntracedTime;     
    } else {
      return totalUntracedTime + (phaseEndTime -whenStoppedTracing);
    }

  }

  inline double totalTraceTime()
  {
      return phaseEndTime - lastResetTime ;
      //return phaseEndTime - lastResetTime - untracedTime();
  }
  /** Fraction of time spent as overhead since resetting the counters */
  inline double overheadRatio(){
    double t = totalTraceTime(); 
    return (t - totalIdleTime - totalEntryMethodTime)/t; 
  } 

  inline double overheadTime(){
    double t = totalTraceTime(); 
    return (t - totalIdleTime - totalEntryMethodTime); 
  } 

  inline double utilRatio() {
      if(lastEvent == BEGIN_PROCESSING)
          totalEntryMethodTime += (TraceTimer() - lastBeginExecuteTime);
      return totalEntryMethodTime/ totalTraceTime(); 
  }

  inline double utilTime() {
      if(lastEvent == BEGIN_PROCESSING)
          totalEntryMethodTime += (TraceTimer() - lastBeginExecuteTime);
      return totalEntryMethodTime; 
  }
  /** Highest memory usage (in MB) value we've seen since resetting the counters */
  inline double memoryUsageMB(){
    return ((double)memUsage) / 1024.0 / 1024.0;
  }

  /** Determine the average grain size since last reset of counters */
  inline double grainSize(){
    return (double)totalEntryMethodTime / totalEntryMethodInvocations;
  }

  inline double maxGrainSize() {
    return maxEntryMethodTime;
  }

  inline long bytesPerEntry() {
    return currentSummary->numBytes / currentSummary->numMsgs;
  }
   
  inline long numInvocations() {
      return totalEntryMethodInvocations;
  }

#if CMK_HAS_COUNTER_PAPI
  inline void readPAPI()
  {
      if (PAPI_read(CkpvAccess(papiEventSet), CkpvAccess(papiValues)) != PAPI_OK) {
          CmiAbort("PAPI failed to read at begin execute!\n");
      }
  }
#endif

  perfData* getSummary()
  {
      currentSummary->idleMin = currentSummary->idleMax= idleRatio(); 
      currentSummary->idleTotalTime = idleTime();
      currentSummary->utilMin = currentSummary->utilMax = utilRatio(); 
      currentSummary->utilTotalTime= utilTime();
      currentSummary->overheadMin = currentSummary->overheadMax = overheadRatio();
      currentSummary->overheadTotalTime = overheadTime();
      currentSummary->grainsizeAvg = grainSize();
      currentSummary->grainsizeMax = maxGrainSize();
      currentSummary->numInvocations = totalEntryMethodInvocations;
#if CMK_HAS_COUNTER_PAPI
      readPAPI();
      for(int i=0; i<NUMPAPIEVENTS; i++)
      {
          currentSummary->papiValues[i] = (CkpvAccess(papiValues)[i] - previous_papiValues[i]);
      }
#endif
      return currentSummary;
  }

  void printSummary()
  {
      CkPrintf("################\n");
      CkPrintf("\t-------%d local data idle:util:overhead %f:%f:%f\n", CkMyPe(), currentSummary->idleMin, currentSummary->utilMin, currentSummary->overheadMin);
      CkPrintf("################\n");
  }
};


TraceAutoPerf *localControlPointTracingInstance();

#endif

