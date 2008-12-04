#include "charm++.h"
#include "trace-controlPoints.h"
#include "trace-controlPointsBOC.h"

// Charm++ "processor"(user thread)-private global variable
CkpvStaticDeclare(TraceControlPoints*, _trace);

// This global variable is required for any post-execution 
// parallel analysis or activities the trace module might wish to perform.
CkGroupID traceControlPointsGID;

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
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

  if (CkpvAccess(traceOnPe) == 0) return;

  // Process runtime arguments intended for the module
  // CmiGetArgIntDesc(argv,"+ControlPointsPar0", &par0, "Fake integer parameter 0");
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
  // CmiObjId is a 4-integer tuple uniquely identifying a migratable
  //   Charm++ object. Note that there are other non-migratable Charm++
  //   objects that CmiObjId will not identify.

  lastBeginExecuteTime = CmiWallTimer();
  lastbeginMessageSize = -1;

  //  CkPrintf("[%d] TraceControlPoints::beginExecute(CmiObjId *tid)\n", CkMyPe());
}

void TraceControlPoints::beginExecute(envelope *e)
{
  lastBeginExecuteTime = CmiWallTimer();
  lastbeginMessageSize = e->getTotalsize();

  //  CkPrintf("[%d] WARNING ignoring TraceControlPoints::beginExecute(envelope *e)\n", CkMyPe());

  // no message means thread execution
  //  if (e != NULL) {
  //  CkPrintf("[%d] TraceControlPoints::beginExecute  Method=%d, type=%d, source pe=%d, size=%d\n", 
  //	     CkMyPe(), e->getEpIdx(), e->getMsgtype(), e->getSrcPe(), e->getTotalsize() );
  //}  

  //  CkPrintf("[%d] TraceControlPoints::beginExecute(envelope *e=%p)\n", CkMyPe(), e);

}

void TraceControlPoints::beginExecute(int event,int msgType,int ep,int srcPe, 
			       int mlen, CmiObjId *idx)
{
  lastBeginExecuteTime = CmiWallTimer();
  lastbeginMessageSize = mlen;
  //  CkPrintf("[%d] TraceControlPoints::beginExecute event=%d, msgType=%d, ep=%d, srcPe=%d, mlen=%d CmiObjId is also avaliable\n", CkMyPe(), event, msgType, ep, srcPe, mlen);
  //  CkPrintf("[%d] TraceControlPoints::beginExecute(int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx)\n", CkMyPe());

}

void TraceControlPoints::endExecute(void)
{
  double executionTime = CmiWallTimer() - lastBeginExecuteTime;
  totalEntryMethodTime += executionTime;
  
  //  CkPrintf("[%d] Previously executing Entry Method completes. lastbeginMessageSize=%d executionTime=%lf\n", CkMyPe(), lastbeginMessageSize, executionTime);
}

void TraceControlPoints::beginIdle(double curWallTime) {
  lastBeginIdle = CmiWallTimer();
  // CkPrintf("[%d] Scheduler has no useful user-work\n", CkMyPe());
}

void TraceControlPoints::endIdle(double curWallTime) {
  totalIdleTime += CmiWallTimer() - lastBeginIdle;
  //  CkPrintf("[%d] Scheduler now has useful user-work\n", CkMyPe());
}

void TraceControlPoints::beginComputation(void)
{
  //  CkPrintf("[%d] Computation Begins\n", CkMyPe());
  // Code Below shows what trace-summary would do.
  // initialze arrays because now the number of entries is known.
  // _logPool->initMem();
}

void TraceControlPoints::endComputation(void)
{
  //  CkPrintf("[%d] Computation Ends\n", CkMyPe());
}

void TraceControlPoints::malloc(void *where, int size, void **stack, int stackSize)
{
  // CkPrintf("[%d] Memory allocation of size %d occurred\n", CkMyPe(), size);
}

void TraceControlPoints::free(void *where, int size) {
  //  CkPrintf("[%d] %d-byte Memory block freed\n", CkMyPe(), size);
}

void TraceControlPoints::traceClose(void)
{
  CkpvAccess(_trace)->endComputation();
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
}

void TraceControlPoints::resetTimings(){
  totalIdleTime = 0.0;
  totalEntryMethodTime = 0.0;
  lastResetTime = CmiWallTimer();
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
#ifdef __BLUEGENE__
  if (BgNodeRank()==0) {
#else
    if (CkMyRank() == 0) {
#endif
      registerExitFn(traceControlPointsExitFunction);
    }
}

#include "TraceControlPoints.def.h"


/*@}*/
