#if CMK_HAS_COUNTER_PAPI
#ifdef USE_SPP_PAPI
int papiEvents[NUMPAPIEVENTS];
#else
int papiEvents[NUMPAPIEVENTS] = { PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_DCM, PAPI_TLB_DM, PAPI_L1_DCH, PAPI_L2_DCH, PAPI_L3_DCH, PAPI_FP_OPS, PAPI_TOT_CYC };
#endif
#endif // CMK_HAS_COUNTER_PAPI


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

    //PAPI related 
#if CMK_HAS_COUNTER_PAPI
  // We initialize and create the event sets for use with PAPI here.
  int papiRetValue;
  if(CkMyRank()==0){
    papiRetValue = PAPI_library_init(PAPI_VER_CURRENT);
    if (papiRetValue != PAPI_VER_CURRENT) {
      CmiAbort("PAPI Library initialization failure!\n");
    }
#if CMK_SMP
    if(PAPI_thread_init(pthread_self) != PAPI_OK){
      CmiAbort("PAPI could not be initialized in SMP mode!\n");
    }
#endif
  }

#if CMK_SMP
  //PAPI_thread_init has to finish before calling PAPI_create_eventset
  #if CMK_SMP_TRACE_COMMTHREAD
      CmiNodeAllBarrier();
  #else
      CmiNodeBarrier();
  #endif
#endif
  // PAPI 3 mandates the initialization of the set to PAPI_NULL
  papiEventSet = PAPI_NULL; 
  if (PAPI_create_eventset(&papiEventSet) != PAPI_OK) {
    CmiAbort("PAPI failed to create event set!\n");
  }
#ifdef USE_SPP_PAPI
  //  CmiPrintf("Using SPP counters for PAPI\n");
  if(PAPI_query_event(PAPI_FP_OPS)==PAPI_OK) {
    papiEvents[0] = PAPI_FP_OPS;
  }else{
    if(CmiMyPe()==0){
      CmiAbort("WARNING: PAPI_FP_OPS doesn't exist on this platform!");
    }
  }
  if(PAPI_query_event(PAPI_TOT_INS)==PAPI_OK) {
    papiEvents[1] = PAPI_TOT_INS;
  }else{
    CmiAbort("WARNING: PAPI_TOT_INS doesn't exist on this platform!");
  }
  int EventCode;
  int ret;
  ret=PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_LL:ACCESS",&EventCode);
  if(PAPI_query_event(EventCode)==PAPI_OK) {
    papiEvents[2] = EventCode;
  }else{
    CmiAbort("WARNING: perf::PERF_COUNT_HW_CACHE_LL:ACCESS doesn't exist on this platform!");
  }
  ret=PAPI_event_name_to_code("DATA_PREFETCHER:ALL",&EventCode);
  if(PAPI_query_event(EventCode)==PAPI_OK) {
    papiEvents[3] = EventCode;
  }else{
    CmiAbort("WARNING: DATA_PREFETCHER:ALL doesn't exist on this platform!");
  }
  if(PAPI_query_event(PAPI_L1_DCM)==PAPI_OK) {
    papiEvents[4] = PAPI_L1_DCM;
  }else{
    CmiAbort("WARNING: PAPI_L1_DCM doesn't exist on this platform!");
  }
  if(PAPI_query_event(PAPI_TOT_CYC)==PAPI_OK) {
    papiEvents[5] = PAPI_TOT_CYC;
  }else{
    CmiAbort("WARNING: PAPI_TOT_CYC doesn't exist on this platform!");
  }
  if(PAPI_query_event(PAPI_L2_DCM)==PAPI_OK) {
    papiEvents[6] = PAPI_L2_DCM;
  }else{
    CmiAbort("WARNING: PAPI_L2_DCM doesn't exist on this platform!");
  }
  if(PAPI_query_event(PAPI_L1_DCA)==PAPI_OK) {
    papiEvents[7] = PAPI_L1_DCA;
  }else{
    CmiAbort("WARNING: PAPI_L1_DCA doesn't exist on this platform!");
  }
#else
  // just uses { PAPI_L2_DCM, PAPI_FP_OPS } the 2 initialized PAPI_EVENTS
#endif
  papiRetValue = PAPI_add_events(papiEventSet, papiEvents, NUMPAPIEVENTS);
  if (papiRetValue < 0) {
    if (papiRetValue == PAPI_ECNFLCT) {
      CmiAbort("PAPI events conflict! Please re-assign event types!\n");
    } else {
      char error_str[PAPI_MAX_STR_LEN];
      PAPI_perror(error_str);
      //PAPI_perror(papiRetValue,error_str,PAPI_MAX_STR_LEN);
      CmiPrintf("PAPI failed with error %s val %d\n",error_str,papiRetValue);
      CmiAbort("PAPI failed to add designated events!\n");
    }
  }
  if(CkMyPe()==0)
    {
      CmiPrintf("Registered %d PAPI counters:",NUMPAPIEVENTS);
      char nameBuf[PAPI_MAX_STR_LEN];
      for(int i=0;i<NUMPAPIEVENTS;i++)
	{
	  PAPI_event_code_to_name(papiEvents[i], nameBuf);
	  CmiPrintf("%s ",nameBuf);
	}
      CmiPrintf("\n");
    }
  memset(papiValues, 0, NUMPAPIEVENTS*sizeof(LONG_LONG_PAPI));
  memset(previous_papiValues, 0, NUMPAPIEVENTS*sizeof(LONG_LONG_PAPI));
#endif


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
#if CMK_HAS_COUNTER_PAPI
    memcpy(previous_papiValues, papiValues, sizeof(LONG_LONG_PAPI)*NUMPAPIEVENTS);
#endif
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

void TraceAutoPerf::beginComputation(void) {
#if CMK_HAS_COUNTER_PAPI
  // we start the counters here
  if (PAPI_start(papiEventSet) != PAPI_OK) {
    CmiAbort("PAPI failed to start designated counters!\n");
  }
#endif

}

void TraceAutoPerf::endComputation(void) { 
#if CMK_HAS_COUNTER_PAPI
  // we stop the counters here. A silent failure is alright since we
  // are already at the end of the program.
  if (PAPI_stop(papiEventSet, papiValues) != PAPI_OK) {
    CkPrintf("Warning: PAPI failed to stop correctly!\n");
  }
  //else 
  //{
  //    char eventName[PAPI_MAX_STR_LEN];
  //    for (int i=0;i<NUMPAPIEVENTS;i++) {
  //        PAPI_event_code_to_name(papiEvents[i], eventName);
  //        CkPrintf(" EVENT  %s   counter   %lld \n", eventName, papiValues[i]);
  //    }
  //}
  // NOTE: We should not do a complete close of PAPI until after the
  // sts writer is done.
#endif

}

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
