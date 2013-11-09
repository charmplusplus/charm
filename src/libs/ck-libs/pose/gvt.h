/// Global Virtual Time estimation for POSE
/** Implements the Global Virtual Time (GVT) algorithm; provides chare
    groups PVT and GVT. Objects interact with the local PVT branch.
    PVT branches summarize object info and report to the single
    floating GVT object, which broadcasts results to all PVT branches. */

#ifndef GVT_H
#define GVT_H
#include "gvt.decl.h"

/// synchronization strategies
#define OPTIMISTIC 0
#define CONSERVATIVE 1

/// Global handles ThePVT and TheGVT are declared in gvt.C, used everywhere
extern CkGroupID ThePVT;  
extern CkGroupID TheGVT;

class SRtable;  // from srtable.h
class SRentry;  // from srtable.h

void POSE_sumGVTIterations(void *param, void *msg); 

/// Message to send info to GVT 
/** PVT sends processor virtual time and send/recv information to GVT.  
    GVT also sends info to next GVT index for next GVT invocation. */
class UpdateMsg : public CMessage_UpdateMsg {
public:
  /// PVT of local optimistic objects
  /** Used to send estimated GVT from one GVT invocation to next */
  POSE_TimeType optPVT;
  /// PVT of local conservative objects
  POSE_TimeType conPVT;
  /// Max timestamp in SR table
  POSE_TimeType maxSR;
  /// # sends/recvs at particular timestamps <= PVT
  SRentry *SRs;
  /// Count of entries in SRs
  int numEntries;
  /// Inactive status (GVT only)
  int inactive;
  /// Inactive time (GVT only)
  POSE_TimeType inactiveTime;
  /// Iterations of GVT since last LB (GVT only)
  int nextLB;
  /// Flag used by runGVT to call computeGVT indicating readiness
  int runGVTflag;
};

/// Message to send GVT estimate back to PVT
class GVTMsg : public CMessage_GVTMsg {
public:
  /// GVT estimate
  POSE_TimeType estGVT;
  /// Termination flag
  int done;
};

/// Prioritized int msg; used to force GVT calculations
class prioBcMsg : public CMessage_prioBcMsg {
public:
  int bc;
};

/// PVT chare group for computing local processor virtual time 
/** Keeps track of local sends/recvs and computes processor virtual time.
    Interacts with GVT to obtain new estimate and invokes fossil 
    collection and forward execution on objects with new estimate. */
class PVT : public Group {
 private:
#if !CMK_TRACE_DISABLED
  localStat *localStats;
#endif
  /// PVT of local optimistic posers
  POSE_TimeType optPVT;
  /// PVT of local conservative posers
  POSE_TimeType conPVT;
  /// Last GVT estimate
  POSE_TimeType estGVT;       
  /// Last reported PVT estimate
  POSE_TimeType repPVT;       
  /// Simulation termination flag
  int simdone;
  /// Minimum send/recv timestamp in this iteration
  POSE_TimeType iterMin;
  /// Flag to indicate waiting for first send/recv of next iteration
  /** Used to indicate when to restructure the SendsAndRecvs table */
  int waitForFirst;
  /// Table to store send/recv timestamps
  SRtable *SendsAndRecvs;            
  /// List of objects registered with this PVT branch
  pvtObjects objs;           
  /// reduction-related vars
  int reportTo, reportsExpected, reportReduceTo, reportEnd;
  /// where the centralized GVT goes
  int gvtTurn;
  int specEventCount, eventCount;
  /// startPhase active flag
  int startPhaseActive;
  /// starting time of the simulation
  double parStartTime;
  /// indicates if checkpointing is in progress
  int parCheckpointInProgress;
  /// GVT at which the last checkpoint was performed
  POSE_TimeType parLastCheckpointGVT;
  /// indicates if load balancing is in progress
  int parLBInProgress;
  /// GVT at which the last load balancing was performed
  POSE_TimeType parLastLBGVT;
  /// Time at which the last checkpoint was performed
  double parLastCheckpointTime;
  /* things which used to be member function statics */
  /// optimistic and coservative GVTs
  POSE_TimeType optGVT, conGVT;
  int rdone;
  /// used in PVT report reduction
  SRentry *SRs;
#ifdef MEM_TEMPORAL
  TimePool *localTimePool;
#endif
  /// Circular buffer for storing debug prints
  //  int debugBufferLoc, debugBufferWrapped, debugBufferDumped;
  //  char debugBuffer[NUM_PVT_DEBUG_BUFFER_LINES][PVT_DEBUG_BUFFER_LINE_LENGTH];

 public:
  /// Basic Constructor
  PVT(void);
  /// Migration Constructor
  PVT(CkMigrateMessage *msg) : Group(msg) { };
  /// PUP routine
  void pup(PUP::er &p);
  /// Destructor
  ~PVT() { }
  /// ENTRY: runs the PVT calculation and reports to GVT
  void startPhase(prioBcMsg *m);             
  /// ENTRY: runs the expedited PVT calculation and reports to GVT
  void startPhaseExp(prioBcMsg *m);             
  /// ENTRY: receive GVT estimate; wake up objects
  /** Receives the new GVT estimate and termination flag; wakes up objects
      for fossil collection and forward execution with new GVT estimate. */
  void setGVT(GVTMsg *m);            
  /// ENTRY: begin checkpoint now that quiescence has been reached
  void beginCheckpoint(eventMsg *m);
  /// ENTRY: resume after checkpointing, restarting, or if checkpointing doesn't occur
  void resumeAfterCheckpoint(eventMsg *m);
  /// ENTRY: 
  void beginLoadbalancing(eventMsg *m);
  /// ENTRY: 
  void resumeAfterLB(eventMsg *m);
  /// ENTRY: 
  void callAtSync();
  void doneLB();

  /// Returns GVT estimate
  inline POSE_TimeType getGVT() { return estGVT; }    

  inline int getSpecEventCount() { return specEventCount; }    
  inline int getEventCount() { return eventCount; }    
  inline void incSpecEventCount() { specEventCount++; }    
  inline void incEventCount() { eventCount++; }
  inline void decEventCount() { eventCount--; }
  /// Returns termination flag
  inline int done() { return simdone; }
  /// Register poser with PVT
  int objRegister(int arrIdx, POSE_TimeType safeTime, int sync, sim *myPtr);
  /// Unregister poser from PVT
  void objRemove(int pvtIdx);
  /// Update send/recv table at timestamp
  void objUpdate(POSE_TimeType timestamp, int sr); 
  /// Update PVT with safeTime and send/recv table at timestamp
  void objUpdateOVT(int pvtIdx, POSE_TimeType safeTime, POSE_TimeType ovt);
  /// ENTRY: Reduction point for PVT reports
  void reportReduce(UpdateMsg *);
  /// Adds incoming send/recv information to a list
  void addSR(SRentry **SRs, SRentry *e, POSE_TimeType og, int ne);
  inline int getNumObjs() { return objs.getNumObjs(); }
/*
  /// ENTRY: Dump the debug buffer
  inline void dumpDebugBuffer() {
    int endLoc;
    if (!debugBufferDumped) {
      debugBufferDumped = 1;
      if (debugBufferLoc == 0) {
	endLoc = NUM_PVT_DEBUG_BUFFER_LINES - 1;
      } else {
	endLoc = debugBufferLoc - 1;
      }
      if (debugBufferWrapped) {
	int j = 0;
	for (int i = debugBufferLoc; i < NUM_PVT_DEBUG_BUFFER_LINES; i++) {
	  CkPrintf("{%5d} %s", j, debugBuffer[i]);
	  j++;
	}
	if (debugBufferLoc != 0) {
	  for (int i = 0; i <= endLoc; i++) {
	    CkPrintf("{%5d} %s", j, debugBuffer[i]);
	    j++;
	  }
	}
      } else {
	for (int i = 0; i <= endLoc; i++) {
	  CkPrintf("{%5d} %s", i, debugBuffer[i]);
	}
      }
    }
  }
  /// Write to the debug buffer
  inline void printDebug(char *str) {
    strcpy(debugBuffer[debugBufferLoc], str);
    debugBufferLoc++;
    if (debugBufferLoc >= NUM_PVT_DEBUG_BUFFER_LINES) {
      debugBufferLoc = 0;
      debugBufferWrapped = 1;
    }
  }
*/
};

/// GVT chare group for estimating GVT
/** Responsibility for GVT estimation shifts between branches after each
    GVT invocation. */
class GVT : public Group { 
private:
#if !CMK_TRACE_DISABLED
  localStat *localStats;
#endif
  /// Latest GVT estimate
  POSE_TimeType estGVT; 
  /// Inactivity status: number of iterations since GVT has changed
  int inactive;
  /// Time at which GVT last went inactive
  POSE_TimeType inactiveTime;
  /// Number of GVT iterations since last LB run
  int nextLBstart; 
  /// Earliest send/recv timestamp in previous GVT invocation
  POSE_TimeType lastEarliest;
  /// Number of sends at lastEarliest
  int lastSends;
  /// Number of receives at lastEarliest
  int lastRecvs;
  /// Number of PVT reports expected (1 or 2)
  int reportsExpected;
  /* things which used to be member function static */
  /// optimistic and coservative GVTs
  POSE_TimeType optGVT, conGVT;
  int done;
  /// used to calculate GVT from PVT reports
  SRentry *SRs;
  int startOffset;
public:
  /// Counts the number of GVT interations
  /* This count is contributed to a summation reduction upon
     simulation completion for printout during one of the POSE exit
     functions */
  int gvtIterationCount;

  /// Basic Constructor
  GVT(void);
  /// Migration Constructor
  GVT(CkMigrateMessage *msg) : Group(msg) { };
  /// PUP routine
  void pup(PUP::er &p);
  /// ENTRY: return the number of GVT iterations so far
  inline int getGVTIterationCount() { return gvtIterationCount; }
  //Use this for Ccd calls
  //static void _runGVT(UpdateMsg *);
  /// ENTRY: Contribute the current GVT iteration count to a summation reduction at the end of the simulation
  inline void sumGVTIterationCounts() {
    contribute(sizeof(int), &gvtIterationCount, CkReduction::sum_int, CkCallback(POSE_sumGVTIterations, NULL));
  }
  /// ENTRY: Run the GVT
  /** Updates data fields with info from previous GVT estimation.  Fires
      off PVT calculations on all PVT branches. */
  void runGVT(UpdateMsg *);
  /// ENTRY: Gathers PVT reports; calculates and broadcasts GVT to PVTs
  void computeGVT(UpdateMsg *); 
  /// Adds incoming send/recv information to a list
  void addSR(SRentry **SRs, SRentry *e, POSE_TimeType og, int ne);
};

#endif
