CkpvStaticDeclare(TraceAutoPerf*, _trace);
//-------- group information ---------------------------

TraceAutoPerf *localAutoPerfTracingInstance()
{
  return CkpvAccess(_trace);
}

// instrumentation and analysis 
TraceAutoPerf::TraceAutoPerf(char **argv) 
{
    DEBUG_PRINT( CkPrintf("trace control point resetting %f\n", TraceTimer()); ) 
    currentSummary = new perfData();  
    resetTimings();
    nesting_level = 0;
    whenStoppedTracing = 0; 
    if (CkpvAccess(traceOnPe) == 0) return;
}

void TraceAutoPerf::resetTimings(){
    totalIdleTime = 0.0;
    totalEntryMethodTime = 0.0;
    totalEntryMethodInvocations = 0;
    lastBeginIdle = lastBeginExecuteTime = lastResetTime = TraceTimer();
    totalUntracedTime = 0;
    maxEntryMethodTime = 0;
    if(whenStoppedTracing !=0){
        whenStoppedTracing = TraceTimer();
    }

    currentSummary->numMsgs = 0;
    currentSummary->numBytes = 0;
    currentSummary->commTime = 0;
    currentSummary->objLoadMax = 0;
}

void TraceAutoPerf::resetAll(){
    totalIdleTime = 0.0;
    totalEntryMethodTime = 0.0;
    memUsage = 0;
    totalEntryMethodInvocations = 0;
    lastBeginIdle = lastBeginExecuteTime = lastResetTime = TraceTimer();
    totalUntracedTime = 0;
    if(whenStoppedTracing !=0){
        whenStoppedTracing = TraceTimer();
    }
    currentSummary->numMsgs = 0;
    currentSummary->numBytes = 0;
    currentSummary->commTime = 0;
    currentSummary->objLoadMax = 0;
}

void TraceAutoPerf::traceBegin(void){
    if(whenStoppedTracing != 0)
        totalUntracedTime += (TraceTimer() - whenStoppedTracing);
    whenStoppedTracing = 0;
}

void TraceAutoPerf::traceEnd(void){
  CkAssert(whenStoppedTracing == 0); // can't support nested traceEnds on one processor yet...
  whenStoppedTracing = TraceTimer();
}

void TraceAutoPerf::userEvent(int eventID) { }
void TraceAutoPerf::userBracketEvent(int eventID, double bt, double et) { }
void TraceAutoPerf::creation(envelope *, int epIdx, int num) { } 
void TraceAutoPerf::creationMulticast(envelope *, int epIdx, int num, int *pelist) { }
void TraceAutoPerf::creationDone(int num) { }
void TraceAutoPerf::messageRecv(char *env, int pe) { }

void TraceAutoPerf::beginExecute(CmiObjId *tid)
{
    //nesting_level++;
    lastBeginExecuteTime = TraceTimer();
    lastEvent =  BEGIN_PROCESSING;
    lastbeginMessageSize = -1;
    DEBUG_PRINT( CkPrintf("begin Executing tid   %d  msg(%d:%d) time:%d\n", nesting_level, currentSummary->numMsgs, currentSummary->numBytes, (int)(lastBeginExecuteTime*1000000)); )
}

void TraceAutoPerf::beginExecute(envelope *env)
{
    //nesting_level++;
    //if(nesting_level == 1){
    lastBeginExecuteTime = TraceTimer();
    lastEvent =  BEGIN_PROCESSING;
    lastbeginMessageSize = env->getTotalsize();
    currentSummary->numMsgs++;
    currentSummary->numBytes += lastbeginMessageSize;
    DEBUG_PRINT( CkPrintf("begin Executing env   %d  msg(%d:%d) time:%d\n", nesting_level, currentSummary->numMsgs, currentSummary->numBytes, (int)(lastBeginExecuteTime*1000000)); )
}

void TraceAutoPerf::beginExecute(envelope *env, int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx)
{
    //nesting_level++;
    //if(nesting_level == 1){
    lastbeginMessageSize = mlen;
    currentSummary->numMsgs++;
    currentSummary->numBytes += lastbeginMessageSize;
    //`currentSummary->commTime += (env->getRecvTime() - env->getSentTime());
    lastBeginExecuteTime = TraceTimer();
    lastEvent =  BEGIN_PROCESSING;
    DEBUG_PRINT( CkPrintf("begin Executing env  6  %d  msg(%d:%d) time:%d\n", nesting_level, currentSummary->numMsgs, currentSummary->numBytes, (int)(lastBeginExecuteTime*1000000)); )
}

void TraceAutoPerf::beginExecute(int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx)
{
    //nesting_level++;
    //if(nesting_level == 1){
    lastbeginMessageSize = mlen;
    lastBeginExecuteTime = TraceTimer();
    lastEvent =  BEGIN_PROCESSING;
    DEBUG_PRINT( CkPrintf("begin Executing 6 no env %d  msg(%d:%d) time:%d\n", nesting_level, currentSummary->numMsgs, currentSummary->numBytes, (int)(lastBeginExecuteTime*1000000)); )
}

void TraceAutoPerf::endExecute(void)
{
    //MAYBE a bug
    //nesting_level--;
    nesting_level = 0;
    if(nesting_level == 0){
        double endTime = TraceTimer() ;
        double executionTime = endTime - lastBeginExecuteTime;
        lastEvent =  -1;
        DEBUG_PRINT( CkPrintf("end executing %d, duration %d\n", (int)(1000000*endTime), (int)(executionTime*1000000)); )
        totalEntryMethodTime += executionTime;
        totalEntryMethodInvocations ++;
        if(executionTime > maxEntryMethodTime)
            maxEntryMethodTime = executionTime;
        double m = (double)CmiMemoryUsage();
        if(memUsage < m){
            memUsage = m;
        }    
    }
}

void TraceAutoPerf::beginIdle(double curWallTime) {
    lastBeginIdle =  curWallTime; 
    lastEvent =  BEGIN_IDLE;
    double m = (double)CmiMemoryUsage();
    if(memUsage < m){
        memUsage = m;
    }
}

void TraceAutoPerf::endIdle(double curWallTime) {
    totalIdleTime += (curWallTime - lastBeginIdle) ;
    lastEvent =  -1;
}

void TraceAutoPerf::beginComputation(void) { }
void TraceAutoPerf::endComputation(void) { }

void TraceAutoPerf::malloc(void *where, int size, void **stack, int stackSize)
{
    double m = (double)CmiMemoryUsage();
    if(memUsage < m){
        memUsage = m;
    }
}

void TraceAutoPerf::free(void *where, int size) { }

void TraceAutoPerf::traceClose(void)
{
    CkpvAccess(_traces)->endComputation();
    CkpvAccess(_traces)->removeTrace(this);
}

void TraceAutoPerf::markStep()
{
    double now = TraceTimer();
    timerPair newpairs;
    newpairs.beginTimer = lastResetTime;
    newpairs.endTimer = now; 
    phasesTimers.push_back(newpairs);
    phaseEndTime = now;
    DEBUG_PRINT ( CkPrintf(" PE %d marking phase  %d at timer:%f traceTimer:%f (%f:%f) \n", CmiMyPe(), phasesTimers.size(), now, TraceTimer(), newpairs.beginTimer,  newpairs.endTimer); )

}

void _createTraceautoPerf(char **argv)
{
    CkpvInitialize(TraceAutoPerf*, _trace);
    CkpvAccess(_trace) = new TraceAutoPerf(argv);
    CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
    //CkPrintf("##### init ####\n");
}
