/// Global POSE data and functions; includes and dependencies handled here
#include "pose.h"
#include "pose.def.h"

CpvDeclare(int, stateRecovery);

#ifdef POSE_COMM_ON
extern int comm_debug;
#endif
double busyWait;
double sim_timer;
int POSE_inactDetect;
POSE_TimeType POSE_endtime;
POSE_TimeType POSE_GlobalClock;
POSE_TimeType POSE_GlobalTS;
ComlibInstanceHandle POSE_commlib_insthndl;

// Main initialization for all of POSE
void POSE_init() // use inactivity detection by default
{
  POSE_init(1, POSE_UnsetTS);
}

void POSE_init(int ET) // a single parameter specifies endtime
{
  POSE_init(0, ET);
}

void POSE_init(int IDflag, int ET) // can specify both
{
  CkPrintf("Initializing POSE...  \n");
  POSE_inactDetect = IDflag;
  POSE_endtime = ET;
#ifdef TRACE_DETAIL
  traceRegisterUserEvent("Forward Execution", 10);
  traceRegisterUserEvent("Cancellation", 20);
  traceRegisterUserEvent("Cancel Spawn", 30);
  traceRegisterUserEvent("Rollback", 40);
  traceRegisterUserEvent("Commit", 50);
  traceRegisterUserEvent("OptSync", 60);
#endif
#ifndef SEQUENTIAL_POSE
#ifdef POSE_COMM_ON
  // Create the communication library for POSE
  POSE_commlib_insthndl = CkGetComlibInstance();
  // Create the communication strategy for POSE
  StreamingStrategy *strategy = new StreamingStrategy(COMM_TIMEOUT,COMM_MAXMSG);
  //MeshStreamingStrategy *strategy = new MeshStreamingStrategy(COMM_TIMEOUT,COMM_MAXMSG);
  //PrioStreaming *strategy = new PrioStreaming(COMM_TIMEOUT,COMM_MAXMSG);
  //Register the strategy
  POSE_commlib_insthndl.setStrategy(strategy);
  //comm_debug=1;
  //CkPrintf("Simulation run with PrioStreaming(%d,%d) for communication optimization...\n", COMM_TIMEOUT, COMM_MAXMSG);
  CkPrintf("Simulation run with StreamingStrategy(%d,%d) for communication optimization...\n", COMM_TIMEOUT, COMM_MAXMSG);
  //CkPrintf("Simulation run with MeshStreaming(%d,%d) for communication optimization...\n", COMM_TIMEOUT, COMM_MAXMSG);
#endif
  // Create a MemoryPool with global handle for memory recycling 
  MemPoolID = CProxy_MemoryPool::ckNew();
#endif
  // Create array to hold all POSE objects
#ifdef POSE_COMM_ON  
  POSE_Objects_RO = CProxy_sim::ckNew(); 
  POSE_Objects = POSE_Objects_RO;
#else
  POSE_Objects = CProxy_sim::ckNew(); 
#endif
  //#ifndef SEQUENTIAL_POSE
  //#ifdef POSE_COMM_ON
  // Make POSE_Objects use the comm lib
  //  ComlibDelegateProxy(&POSE_Objects);
  //#endif
  //#endif
  // Initialize statistics collection if desired
#ifdef POSE_STATS_ON
  theLocalStats = CProxy_localStat::ckNew();
  CProxy_globalStat::ckNew(&theGlobalStats);
#endif
#ifndef SEQUENTIAL_POSE
  // Initialize global handles to GVT and PVT
  ThePVT = CProxy_PVT::ckNew(); 
  TheGVT = CProxy_GVT::ckNew();
  // Start off using normal forward execution
  CpvInitialize(int, stateRecovery);
  CpvAccess(stateRecovery) = 0;
#ifdef LB_ON
  // Initialize the load balancer
  TheLBG = CProxy_LBgroup::ckNew();
  TheLBstrategy = CProxy_LBstrategy::ckNew();
  CkPrintf("Load balancing is ON.\n");
#endif
#endif
  CProxy_pose::ckNew(&POSE_Coordinator_ID, 0);
#ifdef SEQUENTIAL_POSE
  if (CkNumPes() > 1) CkAbort("ERROR: Cannot run a sequential simulation on more than one processor!\n");
  CkPrintf("NOTE: POSE running in sequential simulation mode!\n");
  int fnIdx = CkIndex_pose::stop();
  CkStartQD(fnIdx, &POSE_Coordinator_ID);
  POSE_GlobalClock = 0;
  POSE_GlobalTS = 0;
#endif  
  CkPrintf("POSE initialization complete.\n");
  if (POSE_inactDetect) CkPrintf("Using Inactivity Detection for termination.\n");
  else 
#if USE_LONG_TIMESTAMPS
    CkPrintf("Using endTime of %lld for termination.\n", POSE_endtime);
#else
    CkPrintf("Using endTime of %d for termination.\n", POSE_endtime);
#endif
  sim_timer = CmiWallTimer(); 
}

void POSE_startTimer() {
  CkPrintf("Starting simulation...\n"); 
  sim_timer = CmiWallTimer(); 
}

/// Use Inactivity Detection to terminate program
void POSE_useID() 
{
  CkPrintf("WARNING: POSE_useID obsolete. See POSE_init params.\n");
}

/// Use a user-specified end time to terminate program
void POSE_useET(POSE_TimeType et) 
{
  CkPrintf("WARNING: POSE_useET obsolete. See POSE_init params.\n");
}

/// Specify an optional callback to be called when simulation terminates
void POSE_registerCallBack(CkCallback cb)
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  callBack *cbm = new callBack;
  cbm->callback = cb;
  POSE_Coordinator.registerCallBack(cbm);
}

/// Stop POSE simulation
void POSE_stop()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.stop();
}

/// Exit simulation program
void POSE_exit()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.exit();
}

/// Set busy wait time
void POSE_set_busy_wait(double n) { busyWait = n; }

/// Busy wait for busyWait
void POSE_busy_wait()
{
  double start = CmiWallTimer();
  while (CmiWallTimer() - start < busyWait) ;
}

/// Busy wait for n
void POSE_busy_wait(double n)
{
  double start = CmiWallTimer();
  while (CmiWallTimer() - start < n) ;
}

/// Register the callback with POSE
void pose::registerCallBack(callBack *cbm) 
{
  callBackSet = 1;
  cb = cbm->callback;
}

/// Stop the simulation
void pose::stop(void) 
{ 
#ifdef POSE_STATS_ON
  CProxy_localStat stats(theLocalStats);
#endif
#ifdef SEQUENTIAL_POSE
#if USE_LONG_TIMESTAMPS
  CkPrintf("Sequential Endtime Approximation: %lld\n", POSE_GlobalClock);
#else
  CkPrintf("Sequential Endtime Approximation: %d\n", POSE_GlobalClock);
#endif
  // Call sequential termination here...
  POSE_Objects.Terminate();
#endif
#ifdef POSE_STATS_ON
  CkPrintf("%d PE Simulation finished at %f. Gathering stats...\n", 
	   CkNumPes(), CmiWallTimer() - sim_timer);
  stats.SendStats();
#else
  CkPrintf("%d PE Simulation finished at %f.\n", CkNumPes(), 
	   CmiWallTimer() - sim_timer);
  POSE_exit();
#endif  
}

/// Exit the simulation
void pose::exit(void) 
{ 
  if (callBackSet)
    cb.send(); // need to make callback here
  else
    CkExit();
}

// this is a HACK to get module seqpose working
void _registerseqpose(void)
{
  _registerpose();
}

