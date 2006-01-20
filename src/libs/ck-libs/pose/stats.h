/// Modest statistics gathering facility for POSE
/** Counters for: rollbacks, undos, commits, computes, speculative computes,
    and checkpointed bytes; Timers for: rollback, speculative
    computation, total computation, checkpointing time, simulation
    overhead, and gvt overhead, checkpointing overhead and
    cancellation overhead */
#ifndef STATS_H
#define STATS_H
#include "stats.decl.h"
#include "pose_config.h"

// Timer flags
#define DO_TIMER 1
#define RB_TIMER 2
#define GVT_TIMER 3
#define SIM_TIMER 4
#define CAN_TIMER 5
#define CP_TIMER 6
#define LB_TIMER 7
#define FC_TIMER 8
#define COMM_TIMER 9

// Global readonly variables to access stats facility from all PEs
extern CkChareID theGlobalStats;
extern CkGroupID theLocalStats;
// our configuration bundle
extern POSE_Config pose_config;

/// Message to gather local stats from all PEs for printing
class localStatSummary : public CMessage_localStatSummary {
public:
  double doTime, rbTime, gvtTime, simTime, cpTime, canTime, lbTime, fcTime, 
    commTime, maxDo, minDo, maxGRT;
  long cpBytes;
  int pe, dos, undos, commits, loops, gvts, maxChkPts, maxGVT;
};

/// Group to gather stats on a each PE separately
class localStat : public Group {
private:
  /// Current active timer
  short int whichStat;
  /// Counters for various occurrences
  int rollbacks, dos, undos, commits, loops, gvts, chkPts, maxChkPts;  
  /// Count of bytes checkpointed
  long cpBytes;
  /// Timer start values
  double dot, rbt, gvtt, simt, cpt, cant, lbt, fct, commt;
  /// Time accumulators
  double rollbackTime, totalTime, gvtTime, simTime, cpTime, canTime, 
    lbTime, fcTime, commTime, maxDo, minDo; 
  /// Maximum values for GVT and real time taken by events
  /* For degree of parallelism calculations */
  int maxGVT;
  double maxGRT;
public:
  /// Basic Constructor
  localStat(void) {
    whichStat=rollbacks=dos=undos=commits=loops=gvts=cpBytes=chkPts=maxChkPts=
      maxGVT = 0;
    rollbackTime=totalTime=gvtTime=simTime=cpTime=canTime=lbTime=fcTime=
      commTime=maxGRT = 0.0;
    maxDo = minDo = -1.0;
  }
  localStat(CkMigrateMessage *) { };
  /// Start the specified timer
  void TimerStart(int timer);  
  /// Stop the currently active timer
  void TimerStop();            
  /// Switch to different timer, stopping active timer
  void SwitchTimer(int timer); 
  /// Increment event forward execution count
  void Do() { dos++; }         
  /// Increment event rollback count
  void Undo() { undos++; }    
  /// Increment commit count
  void Commit() { commits++; }    
  /// Increment event loop count
  void Loop() { loops++; }    
  /// Increment GVT estimation count     
  void GvtInc() { gvts++; }   
  /// Increment rollback count
  void Rollback() { rollbacks++; }  
  /// Increment checkpoint count and adjust max
  void Checkpoint() { chkPts++; if (chkPts > maxChkPts) maxChkPts = chkPts; }
  /// Decrement checkpoint count
  void Reclaim() { chkPts--; }
  /// Add to checkpointed bytes count
  void CPbytes(int n) { cpBytes += n; }  
  /// Send local stats to global collector
  void SendStats();
  /// Query which timer is active
  int TimerRunning() { return (whichStat); }
  /// Set maximum times
  void SetMaximums(int gvt, double grt) { 
    if (gvt > maxGVT) maxGVT = gvt; 
    if (grt > maxGRT) maxGRT = grt;
  }
};

/// Entity to gather stats from each PE and prepare final report
class globalStat : public Chare {
private:
  double doAvg, doMax, rbAvg, rbMax, gvtAvg, gvtMax, simAvg, simMax, 
    cpAvg, cpMax, canAvg, canMax, lbAvg, lbMax, fcAvg, fcMax, commAvg, commMax,
    maxTime;
  double minDo, maxDo, avgDo, GvtTime, maxGRT;
  long cpBytes;
  int reporting, totalDos, totalUndos, totalCommits, totalLoops, totalGvts, 
    maxChkPts, maxGVT;
public:
  /// Basic Constructor
  globalStat(void);
  globalStat(CkMigrateMessage *) { };
  /// Receive, calculate and print statistics
  void localStatReport(localStatSummary *m); 
  void DOPcalc(int gvt, double grt);
};


// All timer functions are inlined below
// Start the specified timer.  If one is active, print warning, switch timer
inline void localStat::TimerStart(int timer) 
{ 
  double now;
  if (whichStat > 0) { // there is an active timer
    CkPrintf("WARNING: timer %d active.  Switching to timer %d.\n",
	     whichStat, timer);
    SwitchTimer(timer);
    return;
  }
  now = CmiWallTimer();
  whichStat = timer;
  switch (timer) { // initialize the appropriate start value
  case DO_TIMER: dot = now; break;
  case RB_TIMER: rbt = now; break;
  case GVT_TIMER: gvtt = now; break;
  case SIM_TIMER: simt = now; break;
  case CP_TIMER: cpt = now; break;
  case CAN_TIMER: cant = now; break;
  case LB_TIMER: lbt = now; break;
  case FC_TIMER: fct = now; break;
  case COMM_TIMER: commt = now; break;
  default: CkPrintf("ERROR: Invalid timer %d\n", timer);
  }
}

// Stop the currently active timer.  Print error msg if no active timer.
inline void localStat::TimerStop() 
{ 
  double now;
  now = CmiWallTimer();
  switch (whichStat) {
  case DO_TIMER: 
    {
      double eventTime = now - dot;
      totalTime += eventTime; 
      if (maxDo < eventTime)
	maxDo = eventTime;
      if ((minDo < 0.0) || (minDo > eventTime))
	minDo = eventTime;
      break;
    }
  case RB_TIMER: rollbackTime += now - rbt; break;
  case GVT_TIMER: gvtTime += now - gvtt; break;
  case SIM_TIMER: simTime += now - simt; break;
  case CP_TIMER: cpTime += now - cpt; break;
  case CAN_TIMER: canTime += now - cant; break;
  case LB_TIMER: lbTime += now - lbt; break;
  case FC_TIMER: fcTime += now - fct; break;
  case COMM_TIMER: commTime += now - commt; break;
  default: CkPrintf("ERROR: No timer active.\n");
  }
  whichStat = 0;
}

// Switch to different timer, stopping currently active timer.  If no active
// timer, print error msg and activate requested timer.
inline void localStat::SwitchTimer(int timer)
{
  double now;
  if (timer == whichStat)
    return;
  now = CmiWallTimer();
  switch (whichStat) { // deactivate previous timer
  case DO_TIMER:
    {
      double eventTime = now - dot;
      totalTime += eventTime; 
      if (maxDo < eventTime)
	maxDo = eventTime;
      if ((minDo < 0.0) || (minDo > eventTime))
	minDo = eventTime;
      break;
    }    
  case RB_TIMER: rollbackTime += now - rbt; break;
  case GVT_TIMER: gvtTime += now - gvtt; break;
  case SIM_TIMER: simTime += now - simt; break;
  case CP_TIMER: cpTime += now - cpt; break;
  case CAN_TIMER: canTime += now - cant; break;
  case LB_TIMER: lbTime += now - lbt; break;
  case FC_TIMER: fcTime += now - fct; break;
  case COMM_TIMER: commTime += now - commt; break;
  default: CkPrintf("ERROR: No active timer.\n");
  }
  whichStat = timer;
  switch (timer) { // activate new timer
  case DO_TIMER: dot = now; break;
  case RB_TIMER: rbt = now; break;
  case GVT_TIMER: gvtt = now; break;
  case SIM_TIMER: simt = now; break;
  case CP_TIMER: cpt = now; break;
  case CAN_TIMER: cant = now; break;
  case LB_TIMER: lbt = now; break;
  case FC_TIMER: fct = now; break;
  case COMM_TIMER: commt = now; break;
  default: CkPrintf("ERROR: Invalid timer %d\n", timer);
  }    
}

#endif
