// File: stats.h
// Modest statistics gathering facility.
// Counters for: rollbacks, undos, computes, speculative computes, 
//    and checkpointed bytes
// Timers for: rollback, speculative computation, total computation, 
//    checkpointing time, simulation overhead, and gvt overhead, checkpointing 
//    overhead and cancellation overhead.
// Last Modified: 7.31.01 by Terry L. Wilmarth

#ifndef STATS_H
#define STATS_H
#include "stats.decl.h"

// Define POSE_STATS_ON to turn stats on. Off by default
// #define POSE_STATS_ON

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

// Message to gather local stats from all PEs for printing
class localStatSummary : public CMessage_localStatSummary {
public:
  double doTime, rbTime, gvtTime, simTime, cpTime, canTime, lbTime, miscTime, 
    maxDo, minDo;
  long cpBytes;
  int pe, dos, undos, gvts, maxChkPts;
};

// Gathers stats on a single PE
class localStat : public Group {
private:
  short int whichStat; // current active timer: 0=do 1=rb 2=gvt 3=sim
  int rollbacks, dos, undos, gvts, chkPts, maxChkPts;  
  long cpBytes;        // counts bytes checkpointed
  double dot, rbt, gvtt, simt, cpt, cant, lbt, misct;  // timer start values
  // time accumulators
  double rollbackTime, totalTime, gvtTime, simTime, cpTime, canTime, 
    lbTime, miscTime, maxDo, minDo; 
public:
  localStat(void) {
    whichStat = 0;
    rollbacks = dos = undos = gvts = cpBytes = chkPts = maxChkPts = 0;
    rollbackTime=totalTime=gvtTime=simTime=cpTime=canTime=lbTime=miscTime= 0.0;
    maxDo = minDo = -1.0;
  }
  localStat(CkMigrateMessage *) { };
  void TimerStart(int timer);  // start the specified timer
  void TimerStop();            // stop the currently active timer
  void SwitchTimer(int timer); // switch to different timer, stopping active
  void Do() { dos++; }         // increment event execution count
  void Undo() { undos++; }         // increment event execution count
  void GvtInc() { gvts++; }         // increment event execution count
  void Rollback() { rollbacks++; }  // increment rollback count
  void Checkpoint() { chkPts++; if (chkPts > maxChkPts) maxChkPts = chkPts; }
  void Reclaim() { chkPts--; }
  void CPbytes(int n) { cpBytes += n; }  // add to checkpointed bytes count
  void SendStats();           // send local stats to global collector
  int TimerRunning() { return (whichStat); }
  
};

class globalStat : public Chare {
private:
  double doAvg, doMax, rbAvg, rbMax, gvtAvg, gvtMax, simAvg, simMax, 
    cpAvg, cpMax, canAvg, canMax, lbAvg, lbMax, miscAvg, miscMax, maxTime;
  double minDo, maxDo, avgDo, GvtTime;
  long cpBytes;
  int reporting, totalDos, totalUndos, totalGvts, maxChkPts;
public:
  globalStat(void);
  globalStat(CkMigrateMessage *) { };
  void localStatReport(localStatSummary *m); // receive, calc & print stats
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
