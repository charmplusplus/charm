// File: pose.C
// Global POSE data and functions; includes and dependencies handled here
// Last Modified: 25.11.03 by Terry L. Wilmarth

#include "pose.h"
#include "pose.def.h"

CpvDeclare(int, stateRecovery);
#ifdef POSE_COMM_ON
CkGroupID dmid;
#endif
double busyWait;
int POSE_endtime;

// Debugging statement indentation for pretty printing of data dumps
void pdb_indent(int pdb_level)
{
  int i, indent=pdb_level * DEBUG_INDENT_INC;
  for (i=0; i<indent; i++)
    CkPrintf(" ");
}

// Main initialization function for all of POSE
void POSE_init()
{
  CkPrintf("Initializing POSE...  \n");

#ifdef POSE_COMM_ON
  // Create the communication library for POSE
  ComlibInstanceHandle cinst = CkGetComlibInstance();
  // Create the communication strategy for POSE
  Strategy *strategy = new StreamingStrategy(10,2);
  //Register the strategy
  cinst.setStrategy(strategy);
#endif

  // Create an EventMsgPool with global handle for message recycling 
  PoolInitMsg *m = new PoolInitMsg;
  m->numPools = 1; //MapSizeToIdx(-1);
  EvmPoolID = CProxy_EventMsgPool::ckNew(m);

  POSE_Objects = CProxy_sim::ckNew();  // Create array to hold all POSE objects
#ifdef POSE_COMM_ON
  ComlibDelegateProxy(&POSE_Objects);  // Make POSE_Objects use the comm lib
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

// use QD to terminate program
void POSE_useQD() 
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.QDon();
  POSE_endtime = -1;
}

// use QD to terminate program
void POSE_useID() 
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.IDon();
  POSE_endtime = -1;
}

// start POSE simulation timer and other behaviors
void POSE_start()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.start();
}

void POSE_registerCallBack(CkCallback cb)
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  callBack *cbm = new callBack;
  cbm->callback = cb;
  POSE_Coordinator.registerCallBack(cbm);
}

// stop POSE simulation timer
void POSE_stop()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.stop();
}

// stop POSE simulation timer
void POSE_exit()
{
  CProxy_pose POSE_Coordinator(POSE_Coordinator_ID);
  POSE_Coordinator.exit();
}

void POSE_set_busy_wait(double n)
{
  busyWait = n;
}

void POSE_busy_wait()
{
  double start = CmiWallTimer();
  while (CmiWallTimer() - start < busyWait) ;
}

void POSE_busy_wait(double n)
{
  double start = CmiWallTimer();
  while (CmiWallTimer() - start < n) ;
}

void pose::registerCallBack(callBack *cbm) 
{
  callBackSet = 1;
  cb = cbm->callback;
}

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

void pose::exit(void) 
{ 
  if (callBackSet)
    cb.send(); // need to make callback here
  else
    CkExit();
}

