#include "charm++.h"
#include "trace-controlPoints.h"
#include "trace-controlPointsBOC.h"


/**
 *   \addtogroup ControlPointFramework
 *   @{
 */


// Charm++ "processor"(user thread)-private global variable
CkpvStaticDeclare(TraceControlPoints*, _trace);

// This global variable is required for any post-execution 
// parallel analysis or parallel activities the trace module 
// might wish to perform.
CkGroupID traceControlPointsGID;

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C

  This module is special in that it is always included in charm, but sometimes it does nothing.
  This is called on all processors in SMP version.
*/
void _createTracecontrolPoints(char **argv)
{
  CkpvInitialize(TraceControlPoints*, _trace);
  CkpvAccess(_trace) = new TraceControlPoints(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}

TraceControlPoints::TraceControlPoints(char **argv)
{
  resetTimings();

  nesting_level = 0;

  whenStoppedTracing = 0;

  b1=0;
  b2=0;
  b3=0;

  if (CkpvAccess(traceOnPe) == 0) return;

  // Process runtime arguments intended for the module
  // CmiGetArgIntDesc(argv,"+ControlPointsPar0", &par0, "Fake integer parameter 0");
}


void TraceControlPoints::traceBegin(void){
  if(whenStoppedTracing != 0)
    totalUntracedTime += CmiWallTimer() - whenStoppedTracing;
  whenStoppedTracing = 0;
  CkPrintf("[%d] TraceControlPoints::traceBegin() totalUntracedTime=%f\n", CkMyPe(), totalUntracedTime);
}


void TraceControlPoints::traceEnd(void){
  CkPrintf("[%d] TraceControlPoints::traceEnd()\n", CkMyPe());
  CkAssert(whenStoppedTracing == 0); // can't support nested traceEnds on one processor yet...
  whenStoppedTracing = CmiWallTimer();
}



void TraceControlPoints::userEvent(int eventID) 
{
  //  CkPrintf("[%d] User Point Event id %d encountered\n", CkMyPe(), eventID);
}

void TraceControlPoints::userBracketEvent(int eventID, double bt, double et) {
  //  CkPrintf("[%d] User Bracket Event id %d encountered\n", CkMyPe(), eventID);
}

void TraceControlPoints::creation(envelope *, int epIdx, int num) {
  //  CkPrintf("[%d] Point-to-Point Message for Entry Method id %d sent\n",  CkMyPe(), epIdx);
}

void TraceControlPoints::creationMulticast(envelope *, int epIdx, int num, 
				    int *pelist) {
  //  CkPrintf("[%d] Multicast Message for Entry Method id %d sent to %d pes\n", CkMyPe(), epIdx, num);
}

void TraceControlPoints::creationDone(int num) {
  //  CkPrintf("[%d] Last initiated send completes\n", CkMyPe());
}
  
void TraceControlPoints::messageRecv(char *env, int pe) {
  // CkPrintf("[%d] Message from pe %d received by scheduler\n", CkMyPe(), pe);
}
  
void TraceControlPoints::beginExecute(CmiObjId *tid)
{
  nesting_level++;
  if(nesting_level == 1){
    // CmiObjId is a 4-integer tuple uniquely identifying a migratable
    //   Charm++ object. Note that there are other non-migratable Charm++
    //   objects that CmiObjId will not identify.
    b1++;
    lastBeginExecuteTime = CmiWallTimer();
    lastbeginMessageSize = -1;
  }
}



void TraceControlPoints::beginExecute(envelope *e)
{
  nesting_level++;
  if(nesting_level == 1){
    lastBeginExecuteTime = CmiWallTimer();
    lastbeginMessageSize = e->getTotalsize();
    b2++;
    b2mlen += lastbeginMessageSize;
  }
}
 
void TraceControlPoints::beginExecute(int event,int msgType,int ep,int srcPe, 
			       int mlen, CmiObjId *idx)
{
  nesting_level++;
  if(nesting_level == 1){
    b3++;
    b3mlen += mlen;
    lastBeginExecuteTime = CmiWallTimer();
    lastbeginMessageSize = mlen;
  }
}

void TraceControlPoints::endExecute(void)
{
  //  CkPrintf("TraceControlPoints::endExecute\n");
  nesting_level--;
  if(nesting_level == 0){
    
    double executionTime = CmiWallTimer() - lastBeginExecuteTime;
    totalEntryMethodTime += executionTime;
    totalEntryMethodInvocations ++;
    
    double m = (double)CmiMemoryUsage();
    if(memUsage < m){
      memUsage = m;
    }    
  }
}

void TraceControlPoints::beginIdle(double curWallTime) {
  lastBeginIdle = CmiWallTimer();
  // CkPrintf("[%d] Scheduler has no useful user-work\n", CkMyPe());

  double m = (double)CmiMemoryUsage();
  if(memUsage < m){
    memUsage = m;
  }
}

void TraceControlPoints::endIdle(double curWallTime) {
  totalIdleTime += CmiWallTimer() - lastBeginIdle;
  //  CkPrintf("[%d] Scheduler now has useful user-work\n", CkMyPe());
}

void TraceControlPoints::beginComputation(void)
{
  CkPrintf("[%d] TraceControlPoints::beginComputation\n", CkMyPe());
  // Code Below shows what trace-summary would do.
  // initialze arrays because now the number of entries is known.
  // _logPool->initMem();
}

void TraceControlPoints::endComputation(void)
{
  CkPrintf("[%d] TraceControlPoints::endComputationn", CkMyPe());
}

void TraceControlPoints::malloc(void *where, int size, void **stack, int stackSize)
{
  // CkPrintf("[%d] Memory allocation of size %d occurred\n", CkMyPe(), size);
  double m = (double)CmiMemoryUsage();
  if(memUsage < m){
    memUsage = m;
  }
}

void TraceControlPoints::free(void *where, int size) {
  //  CkPrintf("[%d] %d-byte Memory block freed\n", CkMyPe(), size);
}

void TraceControlPoints::traceClose(void)
{
  // Print out some performance counters on BG/P
  CProxy_TraceControlPointsBOC myProxy(traceControlPointsGID);

    
  CkpvAccess(_trace)->endComputation();
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
}




void TraceControlPoints::resetTimings(){
  totalIdleTime = 0.0;
  totalEntryMethodTime = 0.0;
  totalEntryMethodInvocations = 0;
  lastResetTime = CmiWallTimer();
  totalUntracedTime = 0;
  if(whenStoppedTracing !=0){
    whenStoppedTracing = CmiWallTimer();
  }
}

void TraceControlPoints::resetAll(){
  totalIdleTime = 0.0;
  totalEntryMethodTime = 0.0;
  memUsage = 0;
  totalEntryMethodInvocations = 0;
  b2mlen=0;
  b3mlen=0;
  b2=0;
  b3=0;
  lastResetTime = CmiWallTimer();
  totalUntracedTime = 0;
  if(whenStoppedTracing !=0){
    whenStoppedTracing = CmiWallTimer();
  }
}






TraceControlPoints *localControlPointTracingInstance(){
  return CkpvAccess(_trace);
}



extern "C" void traceControlPointsExitFunction() {
  // The exit function of any Charm++ module must call CkExit() or
  // the entire exit process will hang if multiple modules are linked.
  // FIXME: This is NOT a feature. Something needs to be done about this.
  CkExit();
}

// Initialization of the parallel trace module.
void initTraceControlPointsBOC() {
/*
#ifdef __BIGSIM__
  if (BgNodeRank()==0) {
#else
    if (CkMyRank() == 0) {
#endif
      registerExitFn(traceControlPointsExitFunction);
    }
*/
}



#include "TraceControlPoints.def.h"


/*@}*/
