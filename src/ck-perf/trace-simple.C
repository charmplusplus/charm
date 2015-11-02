#include "charm++.h"
#include "trace-simple.h"
#include "trace-simpleBOC.h"

// Charm++ "processor"(user thread)-private global variable
CkpvStaticDeclare(TraceSimple*, _trace);

// This global variable is required for any post-execution 
// parallel analysis or activities the trace module might wish to perform.
CkGroupID traceSimpleGID;

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTracesimple(char **argv)
{
  CkpvInitialize(TraceSimple*, _trace);
  CkpvAccess(_trace) = new TraceSimple(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}

TraceSimple::TraceSimple(char **argv)
{
  if (CkpvAccess(traceOnPe) == 0) return;

  // Process runtime arguments intended for the module
  CmiGetArgIntDesc(argv,"+SimplePar0", &par0, "Fake integer parameter 0");
  CmiGetArgDoubleDesc(argv,"+SimplePar1", &par1, "Fake double parameter 1");
}

void TraceSimple::userEvent(int eventID) 
{
  CkPrintf("[%d] User Point Event id %d encountered\n", CkMyPe(), eventID);
}

void TraceSimple::userBracketEvent(int eventID, double bt, double et) {
  CkPrintf("[%d] User Bracket Event id %d encountered\n", CkMyPe(), eventID);
}

void TraceSimple::creation(envelope *, int epIdx, int num) {
  CkPrintf("[%d] Point-to-Point Message for Entry Method id %d sent\n",
	   CkMyPe(), epIdx);
}

void TraceSimple::creationMulticast(envelope *, int epIdx, int num, 
				    int *pelist) {
  CkPrintf("[%d] Multicast Message for Entry Method id %d sent to %d pes\n",
	   CkMyPe(), epIdx, num);
}

void TraceSimple::creationDone(int num) {
  CkPrintf("[%d] Last initiated send completes\n", CkMyPe());
}
  
void TraceSimple::messageRecv(char *env, int pe) {
  CkPrintf("[%d] Message from pe %d received by scheduler\n", 
	   CkMyPe(), pe);
}
  
void TraceSimple::beginExecute(CmiObjId *tid)
{
  // CmiObjId is a 4-integer tuple uniquely identifying a migratable
  //   Charm++ object. Note that there are other non-migratable Charm++
  //   objects that CmiObjId will not identify.
  CkPrintf("[%d] Entry Method invoked using object id\n", CkMyPe());
}

void TraceSimple::beginExecute(envelope *e, void *obj)
{
  // no message means thread execution
  if (e == NULL) {
    CkPrintf("[%d] Entry Method invoked via thread id %d\n", CkMyPe(),
	     _threadEP);
    // Below is what is found in trace-summary.
    // beginExecute(-1,-1,_threadEP,-1);
  } else {
    CkPrintf("[%d] Entry Method %d invoked via message envelope\n", 
	     CkMyPe(), e->getEpIdx());
    // Below is what is found in trace-summary.
    // beginExecute(-1,-1,e->getEpIdx(),-1);
  }  
}

void TraceSimple::beginExecute(int event,int msgType,int ep,int srcPe, 
			       int mlen, CmiObjId *idx, void *obj)
{
  CkPrintf("[%d] Entry Method %d invoked by parameters\n", CkMyPe(),
	   ep);
}

void TraceSimple::endExecute(void)
{
  CkPrintf("[%d] Previously executing Entry Method completes\n", CkMyPe());
}

void TraceSimple::beginIdle(double curWallTime) {
  CkPrintf("[%d] Scheduler has no useful user-work\n", CkMyPe());
}

void TraceSimple::endIdle(double curWallTime) {
  CkPrintf("[%d] Scheduler now has useful user-work\n", CkMyPe());
}
  
void TraceSimple::beginComputation(void)
{
  CkPrintf("[%d] Computation Begins\n", CkMyPe());
  // Code Below shows what trace-summary would do.
  // initialze arrays because now the number of entries is known.
  // _logPool->initMem();
}

void TraceSimple::endComputation(void)
{
  CkPrintf("[%d] Computation Ends\n", CkMyPe());
}

void TraceSimple::malloc(void *where, int size, void **stack, int stackSize)
{
  CkPrintf("[%d] Memory allocation of size %d occurred\n", CkMyPe(), size);
}

void TraceSimple::free(void *where, int size) {
  CkPrintf("[%d] %d-byte Memory block freed\n", CkMyPe(), size);
}

void TraceSimple::traceClose(void)
{
  CkpvAccess(_trace)->endComputation();
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
}

extern "C" void traceSimpleExitFunction() {
  // The exit function of any Charm++ module must call CkExit() or
  // the entire exit process will hang if multiple modules are linked.
  // FIXME: This is NOT a feature. Something needs to be done about this.
  CkExit();
}

// Initialization of the parallel trace module.
void initTraceSimpleBOC() {
#ifdef __BIGSIM__
  if (BgNodeRank()==0) {
#else
    if (CkMyRank() == 0) {
#endif
      registerExitFn(traceSimpleExitFunction);
    }
}

#include "TraceSimple.def.h"


/*@}*/
