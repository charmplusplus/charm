/// Global POSE data and functions; includes and dependencies handled here

#if ! defined(_WIN32) || defined(__CYGWIN__)
#include "unistd.h"
#endif

#include "pose.h"
#include "pose.def.h"

CpvDeclare(int, stateRecovery);
CpvDeclare(eventID, theEventID);

void POSEreadCmdLine();
double busyWait;
double sim_timer;
int POSE_inactDetect;
int totalNumPosers;
POSE_TimeType POSE_endtime;
POSE_TimeType POSE_GlobalClock;
POSE_TimeType POSE_GlobalTS;
POSE_Config pose_config;
int _POSE_SEQUENTIAL;
int seqCheckpointInProgress;
POSE_TimeType seqLastCheckpointGVT;
double seqLastCheckpointTime;
double seqStartTime;
CkQ<Skipped_Event> POSE_Skipped_Events;
int poseIndexOfStopEvent;

const eventID& GetEventID() {
  //CpvStaticDeclare(eventID, theEventID);  // initializes to [0.pe]
  //  for each pe called on
  CpvAccess(theEventID).incEventID();
  CkAssert(CpvAccess(theEventID).getPE()>=0);
  return(CpvAccess(theEventID));
 }

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
  POSEreadCmdLine();
  if (pose_config.checkpoint_gvt_interval) {
    CkPrintf("POSE checkpointing interval set to %lld GVT ticks\n", pose_config.checkpoint_gvt_interval);
  }
  if (pose_config.checkpoint_time_interval) {
    CkPrintf("POSE checkpointing interval set to %d seconds\n", pose_config.checkpoint_time_interval);
  }
  if (pose_config.dop) {
    CkPrintf("POSE DOP analysis enabled...deleting dop log files...\n");
    char fName[32];
    for (int i = 0; i < CkNumPes(); i++) {
      sprintf(fName, "dop%d.log", i);
      unlink(fName);
    }
    sprintf(fName, "dop_mod.out");
    unlink(fName);
    sprintf(fName, "dop_sim.out");
    unlink(fName);
  }
  POSE_inactDetect = IDflag;
  totalNumPosers = 0;
  POSE_endtime = ET;
#ifdef SEQUENTIAL_POSE
  _POSE_SEQUENTIAL = 1;
#else
  _POSE_SEQUENTIAL = 0;
#endif
#if !CMK_TRACE_DISABLED
  traceRegisterUserEvent("Forward Execution", 10);
  traceRegisterUserEvent("Cancellation", 20);
  traceRegisterUserEvent("Cancel Spawn", 30);
  traceRegisterUserEvent("Rollback", 40);
  traceRegisterUserEvent("Commit", 50);
  traceRegisterUserEvent("OptSync", 60);
#endif
#ifndef SEQUENTIAL_POSE
  // Create a MemoryPool with global handle for memory recycling 
  MemPoolID = CProxy_MemoryPool::ckNew();
  // Create a Temporal Memory Manager
  TempMemID = CProxy_TimePool::ckNew();
#endif
  // Initialize statistics collection if desired
#if !CMK_TRACE_DISABLED
  theLocalStats = CProxy_localStat::ckNew();
  CProxy_globalStat::ckNew(&theGlobalStats);
#endif
#ifndef SEQUENTIAL_POSE
  // Initialize global handles to GVT and PVT
  ThePVT = CProxy_PVT::ckNew(); 
  TheGVT = CProxy_GVT::ckNew();
  // Start off using normal forward execution
  if(pose_config.lb_on)
    {
      // Initialize the load balancer
      TheLBG = CProxy_LBgroup::ckNew();
      TheLBstrategy = CProxy_LBstrategy::ckNew();
      CkPrintf("Load balancing is ON.\n");
    }
#endif
  CProxy_pose::ckNew(&POSE_Coordinator_ID, 0);
  // Create array to hold all POSE objects
  POSE_Objects = CProxy_sim::ckNew(); 

#ifdef SEQUENTIAL_POSE
  if (CkNumPes() > 1) CkAbort("ERROR: Cannot run a sequential simulation on more than one processor!\n");
  CkPrintf("NOTE: POSE running in sequential simulation mode!\n");
  int fnIdx = CkIndex_pose::stop();
  CkStartQD(fnIdx, &POSE_Coordinator_ID);
  POSE_GlobalClock = 0;
  POSE_GlobalTS = 0;
  seqCheckpointInProgress = 0;
  seqLastCheckpointGVT = 0;
  seqLastCheckpointTime = seqStartTime = 0.0;
  poseIndexOfStopEvent = -1;
#else
  /*  CkPrintf("WARNING: Charm Quiescence termination enabled!\n");
  int fnIdx = CkIndex_pose::stop();
  CkStartQD(fnIdx, &POSE_Coordinator_ID);
  */
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

/// Set the poser index for an event to be executed when POSE detects quiescence
void setPoseIndexOfStopEvent(int index) {
  poseIndexOfStopEvent = index;
}

/// Exit simulation program after terminus reduction
void POSE_prepExit(void *param, void *msg)
{
  CkReductionMsg *m = (CkReductionMsg *)msg;
  long long *finalBasicStats = ((long long*)m->getData());
  CkPrintf("Final basic stats: Commits: %lld  Rollbacks: %lld\n", finalBasicStats[0], finalBasicStats[1]);
  delete m;
#ifdef SEQUENTIAL_POSE
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.prepExit();
#else
  CProxy_GVT g(TheGVT);
  g.sumGVTIterationCounts();
#endif
}

/// Collect GVT iteration counts
void POSE_sumGVTIterations(void *param, void *msg) {
  CkReductionMsg *m = (CkReductionMsg *)msg;
  CkPrintf("Final basic stats: GVT iterations: %d\n", *((int*)m->getData()));
  delete m;
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.prepExit();
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
#ifdef SEQUENTIAL_POSE
  // invoke any registered stop events and restart quiescence detection
  if (poseIndexOfStopEvent >= 0) {
    POSE_Objects[poseIndexOfStopEvent].invokeStopEvent();
    CkStartQD(CkIndex_pose::stop(), &POSE_Coordinator_ID);
  // don't stop if quiescence was reached for a checkpoint operation
  } else if (seqCheckpointInProgress) {
    POSE_Objects[0].SeqBeginCheckpoint();
  } else {
#if USE_LONG_TIMESTAMPS
    CkPrintf("Sequential Endtime Approximation: %lld\n", POSE_GlobalClock);
#else
    CkPrintf("Sequential Endtime Approximation: %d\n", POSE_GlobalClock);
#endif
    // Call sequential termination here, when done it calls prepExit
    POSE_Objects.Terminate();
  }
#endif
  // prepExit();
}

//! dump stats if enabled and exit
void pose::prepExit(void) 
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    {
      CProxy_localStat stats(theLocalStats);
      CkPrintf("%d PE Simulation finished at %f. Gathering stats...\n", 
	       CkNumPes(), CmiWallTimer() - sim_timer);
      stats.SendStats();
    }
  else
    {
      CkPrintf("%d PE Simulation finished at %f.\n", CkNumPes(), 
	       CmiWallTimer() - sim_timer);
      POSE_exit();
    }
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

void POSEreadCmdLine()
{
  char **argv = CkGetArgv();
  CmiArgGroup("Charm++","POSE");
  pose_config.stats=CmiGetArgFlagDesc(argv, "+stats_pose",
                        "Gather timing information and other statistics");
  /*  semantic meaning for these still to be determined
  CmiGetArgIntDesc(argv, "+start_proj_pose",&pose_config.start_proj,
                        "GVT to initiate projections tracing");
  CmiGetArgIntDesc(argv, "+end_proj_pose",&pose_config.end_proj,
                        "GVT to end projections tracing");
  */
  pose_config.trace=CmiGetArgFlagDesc(argv, "+trace_pose",
                        "Traces key POSE operations like Forward Execution, Rollback, Cancellation, Fossil Collection, etc. via user events for display in projections");

  /* DOP command-line parameter truth table:
     |---- Input ---|   |------------ Output --------------|
     dop dopSkipCalcs   DOP logs written DOP calcs performed
     --- ------------   ---------------- -------------------
      F       F                 No                No
      F       T                 Yes               No
      T       F                 Yes               Yes
      T       T                 Yes               No
  */
  pose_config.dop=CmiGetArgFlagDesc(argv, "+dop_pose",
                        "Critical path analysis by measuring degree of parallelism");
  pose_config.dopSkipCalcs=CmiGetArgFlagDesc(argv, "+dop_pose_skip_calcs",
                        "Records degree of parallelism logs but doesn't perform end-of-simulation calculations");
  if (pose_config.dopSkipCalcs) {
    pose_config.dop = true;
  }

  CmiGetArgIntDesc(argv, "+memman_pose", &pose_config.max_usage , "Coarse memory management: Restricts forward execution of objects with over <max_usage>/<checkpoint store_rate> checkpoints; default to 10");
  /*
  pose_config.msg_pool=CmiGetArgFlagDesc(argv, "+pose_msgpool",  "Store and reuse pools of messages under a certain size default 1000");
  CmiGetArgIntDesc(argv, "+msgpoolsize_pose", &pose_config.msg_pool_size , "Store and reuse pools of messages under a certain size default 1000");

  CmiGetArgIntDesc(argv, "+msgpoolmax_pose", &pose_config.max_pool_msg_size , "Store and reuse pools of messages under a certain size");
  */
  pose_config.lb_on=CmiGetArgFlagDesc(argv, "+lb_on_pose", "Use load balancing");
  CmiGetArgIntDesc(argv, "+lb_skip_pose", &pose_config.lb_skip , "Load balancing skip N; default 51");
  CmiGetArgIntDesc(argv, "+lb_threshold_pose", &pose_config.lb_threshold , "Load balancing threshold N; default 4000");
  CmiGetArgIntDesc(argv, "+lb_diff_pose", &pose_config.lb_diff , "Load balancing  min diff between min and max load PEs; default 2000");
  CmiGetArgIntDesc(argv, "+checkpoint_rate_pose", &pose_config.store_rate , "Sets checkpoint to 1 for every <rate> events. Default to 1. ");
  CmiGetArgIntDesc(argv, "+checkpoint_gvt_pose", &pose_config.checkpoint_gvt_interval, 
		   "Checkpoint approximately every <gvt #> of GVT ticks; default = 0 = no checkpointing; overrides +checkpoint_time_pose");
  if (pose_config.checkpoint_gvt_interval < 0) {
    CmiAbort("+checkpoint_gvt_pose value must be >= 0; 0 = no checkpointing\n");
  }
  CmiGetArgIntDesc(argv, "+checkpoint_time_pose", &pose_config.checkpoint_time_interval, 
		   "Checkpoint approximately every <time> seconds; default = 0 = no checkpointing; overridden by checkpoint_gvt_pose");
  if (pose_config.checkpoint_time_interval < 0) {
    CmiAbort("+checkpoint_time_pose value must be >= 0; 0 = no checkpointing\n");
  }
  if ((pose_config.checkpoint_gvt_interval > 0) && (pose_config.checkpoint_time_interval > 0)) {
    CmiPrintf("WARNING: checkpoint GVT and time values both set; ignoring time value\n");
    pose_config.checkpoint_time_interval = 0;
  }
  /* load balancing */
  CmiGetArgIntDesc(argv, "+lb_gvt_pose", &pose_config.lb_gvt_interval, 
		   "Load balancing approximately every <gvt #> of GVT ticks; default = 0 = no lb");
  if (pose_config.lb_gvt_interval < 0) {
    CmiAbort("+lb_gvt_pose value must be >= 0; 0 = no load balancing\n");
  }
  /* max_iteration seems to be defunct */
  //  CmiGetArgIntDesc(argv, "+FEmax_pose", &pose_config.max_iter , "Sets max events executed in single forward execution step.  Default to 100.");
  CmiGetArgIntDesc(argv, "+leash_specwindow_pose", &pose_config.spec_window , "Sets speculative window behavior.");
  CmiGetArgIntDesc(argv, "+leash_min_pose", &pose_config.min_leash , "Sets speculative window behavior minimum leash. Default 10.");
  CmiGetArgIntDesc(argv, "+leash_max_pose", &pose_config.max_leash , "Sets speculative window behavior maximum leash. Default 100.");
  CmiGetArgIntDesc(argv, "+leash_flex_pose", &pose_config.max_leash , "Sets speculative window behavior leash flex. Default 10.");
  if(pose_config.deterministic= CmiGetArgFlagDesc(argv, "+deterministic_pose",  "sorts events of same timestamp by event id for repeatable behavior "))
    {
      CkPrintf("WARNING: deterministic_pose: enter at your own risk, though this feature is hopefully not broken anymore\n");
    }
}  
