/// Modest statistics gathering facility for POSE
/** Counters for: rollbacks, undos, computes, speculative computes,
    and checkpointed bytes; Timers for: rollback, speculative
    computation, total computation, checkpointing time, simulation
    overhead, and gvt overhead, checkpointing overhead and
    cancellation overhead */
#ifndef STATS_H
#define STATS_H
#include "stats.decl.h"

// #define POSE_STATS_ON 1  // turn this on in pose.h

// Timer flags
#define DO_TIMER 1
#define RB_TIMER 2
#define GVT_TIMER 3
#define SIM_TIMER 4
#define CAN_TIMER 5
#define CP_TIMER 6
#define LB_TIMER 7
#define MISC_TIMER 8

// Global readonly variables to access stats facility from all PEs
extern CkChareID theGlobalStats;
extern CkGroupID theLocalStats;

/// Message to gather local stats from all PEs for printing
class localStatSummary : public CMessage_localStatSummary {
public:
  double doTime, rbTime, gvtTime, simTime, cpTime, canTime, lbTime, miscTime, 
    maxDo, minDo;
  long cpBytes;
  int pe, dos, undos, loops, gvts, maxChkPts;
};

/// Group to gather stats on a each PE separately
class localStat : public Group {
private:
  /// Current active timer
  short int whichStat;
  /// Counters for various occurrences
  int rollbacks, dos, undos, loops, gvts, chkPts, maxChkPts;  
  /// Count of bytes checkpointed
  long cpBytes;
  /// Timer start values
  double dot, rbt, gvtt, simt, cpt, cant, lbt, misct;
  /// Time accumulators
  double rollbackTime, totalTime, gvtTime, simTime, cpTime, canTime, 
    lbTime, miscTime, maxDo, minDo; 
public:
  /// Basic Constructor
  localStat(void) {
    whichStat=rollbacks=dos=undos=loops=gvts=cpBytes=chkPts=maxChkPts = 0;
    rollbackTime=totalTime=gvtTime=simTime=cpTime=canTime=lbTime=miscTime= 0.0;
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
};

/// Entity to gather stats from each PE and prepare final report
class globalStat : public Chare {
private:
  double doAvg, doMax, rbAvg, rbMax, gvtAvg, gvtMax, simAvg, simMax, 
    cpAvg, cpMax, canAvg, canMax, lbAvg, lbMax, miscAvg, miscMax, maxTime;
  double minDo, maxDo, avgDo, GvtTime;
  long cpBytes;
  int reporting, totalDos, totalUndos, totalLoops, totalGvts, maxChkPts;
public:
  /// Basic Constructor
  globalStat(void);
  globalStat(CkMigrateMessage *) { };
  /// Receive, calculate and print statistics
  void localStatReport(localStatSummary *m); 
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
  case MISC_TIMER: misct = now; break;
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
  case MISC_TIMER: miscTime += now - misct; break;
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
  case MISC_TIMER: miscTime += now - misct; break;
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
  case MISC_TIMER: misct = now; break;
  default: CkPrintf("ERROR: Invalid timer %d\n", timer);
  }    
}

#endif
