/// Global POSE data and functions; includes and dependencies handled here
#include "pose.h"
#include "pose.def.h"

CpvDeclare(int, stateRecovery);

#ifdef POSE_COMM_ON
extern int comm_debug;
#endif
double busyWait;
int POSE_endtime;

// Main initialization for all of POSE
void POSE_init()
{
  CkPrintf("Initializing POSE...  \n");
#ifdef POSE_COMM_ON
  // Create the communication library for POSE
  ComlibInstanceHandle cinst = CkGetComlibInstance();
  // Create the communication strategy for POSE
  //DummyStrategy *strategy = new DummyStrategy();
  StreamingStrategy *strategy =new StreamingStrategy(COMM_TIMEOUT,COMM_MAXMSG);
  strategy->enableShortArrayMessagePacking();
  //Register the strategy
  cinst.setStrategy(strategy);
  //comm_debug=1;
  CkPrintf("Simulation run with StreamingStrategy(%d,%d) for communication optimization...\n", COMM_TIMEOUT, COMM_MAXMSG);
#endif
  // Create an EventMsgPool with global handle for message recycling 
  PoolInitMsg *m = new PoolInitMsg;
  m->numPools = 1; //MapSizeToIdx(-1);
  EvmPoolID = CProxy_EventMsgPool::ckNew(m);
  // Create array to hold all POSE objects
  POSE_Objects = CProxy_sim::ckNew(); 
#ifdef POSE_COMM_ON
  // Make POSE_Objects use the comm lib
  ComlibDelegateProxy(&POSE_Objects);
#endif
  CProxy_pose::ckNew(&POSE_Coordinator_ID);
  // Initialize statistics collection if desired
#ifdef POSE_STATS_ON
  theLocalStats = CProxy_localStat::ckNew();
  CProxy_globalStat::ckNew(&theGlobalStats);
#endif
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
  CkPrintf("POSE initialization complete.\n");
}

/// Use Inactivity Detection to terminate program
void POSE_useID() 
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.IDon();
  POSE_endtime = -1;
}

/// Use a user-specified end time to terminate program
void POSE_useET(int et) 
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.ETon();
  POSE_endtime = et;
}

/// Start POSE simulation timer and event processing
void POSE_start()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.start();
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
#ifdef POSE_STATS_ON
  CkPrintf("%d PE Simulation finished at %f.  Gathering stats...\n", 
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

