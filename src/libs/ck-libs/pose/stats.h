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
  int pe, dos, undos, commits, loops, gvts, maxChkPts;
  POSE_TimeType maxGVT;
};

/// Group to gather stats on each PE separately
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
  POSE_TimeType maxGVT;
  double maxGRT;
  /// Output file name for stats for DOP calculation
  char dopFileName[20];
  /// Output file pointer for DOP calculation
  FILE *dopFilePtr;
public:
  /// Basic Constructor
  localStat(void) : whichStat(0),rollbacks(0),dos(0),undos(0),commits(0),loops(0),gvts(0),cpBytes(0),chkPts(0),maxChkPts(0),   maxGVT (0),
    rollbackTime(0.0),totalTime(0.0),gvtTime(0.0),simTime(0.0),cpTime(0.0),canTime(0.0),lbTime(0.0),fcTime(0.0),
    commTime(0.0),maxGRT(0.0),
    maxDo(-1.0),minDo(-1.0)
  {
#ifdef VERBOSE_DEBUG
    CkPrintf("[%d] constructing localStat\n",CkMyPe());
#endif
    if (pose_config.dop) {
      sprintf(dopFileName, "dop%d.log", CkMyPe());
      dopFilePtr = fopen(dopFileName, "w");
      if (dopFilePtr == NULL) {
	CkPrintf("ERROR: unable to open DOP file %s for writing\n", dopFileName);
	CkAbort("Error opening file");
      }
    }
  }
  /// Migration constructor
  localStat(CkMigrateMessage *msg) : Group(msg) { };
  /// Destructor
  ~localStat() {
    fclose(dopFilePtr);
  }
  /// Start the specified timer
  void TimerStart(int timer);
  /// Stop the currently active timer
  void TimerStop();
  /// Switch to different timer, stopping active timer
  void SwitchTimer(int timer);
  /// Increment event forward execution count
  inline void Do() { dos++; }
  /// Increment event rollback count
  inline void Undo() { undos++; }
  /// Increment commit count
  inline void Commit() { commits++; }
  /// Increment event loop count
  inline void Loop() { loops++; }
  /// Increment GVT estimation count
  inline void GvtInc() { gvts++; }
  /// Increment rollback count
  inline void Rollback() { rollbacks++; }
  /// Increment checkpoint count and adjust max
  inline void Checkpoint() { chkPts++; if (chkPts > maxChkPts) maxChkPts = chkPts; }
  /// Decrement checkpoint count
  inline void Reclaim() { chkPts--; }
  /// Add to checkpointed bytes count
  inline void CPbytes(int n) { cpBytes += n; }
  /// Send local stats to global collector
  void SendStats();
  /// Query which timer is active
  inline int TimerRunning() { return (whichStat); }
  /// Set maximum times
  inline void SetMaximums(POSE_TimeType gvt, double grt) {
    if (gvt > maxGVT) maxGVT = gvt;
    if (grt > maxGRT) maxGRT = grt;
  }
  /// Write data to this PE's DOP log file
  inline void WriteDopData(double srt, double ert, POSE_TimeType svt, POSE_TimeType evt) {
#if USE_LONG_TIMESTAMPS
    const char* format = "%f %f %lld %lld\n";
#else
    const char* format = "%f %f %d %d\n";
#endif
    // fprintf returns the number of characters written, or a negative
    // number if something went wrong
    if (fprintf(dopFilePtr, format, srt, ert, svt, evt) <= 0) {
      CkPrintf("WARNING: DOP data not written to %s\n", dopFileName);
    }
  }
};
PUPbytes(localStat)

/// Entity to gather stats from each PE and prepare final report
class globalStat : public Chare {
private:
  double doAvg, doMax, rbAvg, rbMax, gvtAvg, gvtMax, simAvg, simMax, cpAvg, 
    cpMax, canAvg, canMax, lbAvg, lbMax, fcAvg, fcMax, commAvg, commMax, maxTime;
  double minDo, maxDo, avgDo, GvtTime, maxGRT;
  long cpBytes;
  int reporting, totalDos, totalUndos, totalCommits, totalLoops, totalGvts, maxChkPts;
  POSE_TimeType maxGVT;
public:
  /// Basic Constructor
  globalStat(void);
  /// Migration constructor
  globalStat(CkMigrateMessage *msg) { };
  /// Receive, calculate and print statistics
  void localStatReport(localStatSummary *m); 
  void DOPcalc(POSE_TimeType gvt, double grt);
};
PUPbytes(globalStat)


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
