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

/// Message to send info to GVT 
/** PVT sends processor virtual time and send/recv information to GVT.  
    GVT also sends info to next GVT index for next GVT invocation. */
class UpdateMsg : public CMessage_UpdateMsg {
public:
  /// PVT of local optimistic objects
  /** Used to send estimated GVT from one GVT invocation to next */
  int optPVT;
  /// PVT of local conservative objects
  int conPVT;
  /// # sends/recvs at particular timestamps <= PVT
  SRentry *SRs;
  /// Count of entries in SRs
  int numEntries;
  /// Inactive status (GVT only)
  int inactive;
  /// Inactive time (GVT only)
  int inactiveTime;
  /// Iterations of GVT since last LB (GVT only)
  int nextLB;
  /// Flag used by runGVT to call computeGVT indicating readiness
  int runGVTflag;
};

/// Message to send GVT estimate back to PVT
class GVTMsg : public CMessage_GVTMsg {
public:
  /// GVT estimate
  int estGVT;
  /// Termination flag
  int done;
};

/// PVT chare group for computing local processor virtual time 
/** Keeps track of local sends/recvs and computes processor virtual time.
    Interacts with GVT to obtain new estimate and invokes fossil 
    collection and forward execution on objects with new estimate. */
class PVT : public Group {  
 private:
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  /// PVT of local optimistic posers
  int optPVT;
  /// PVT of local conservative posers
  int conPVT;
  /// Last GVT estimate
  int estGVT;       
  /// Simulation termination flag
  int simdone;
  /// Minimum send/recv timestamp in this iteration
  int iterMin;
  /// Flag to indicate waiting for first send/recv of next iteration
  /** Used to indicate when to restructure the SendsAndRecvs table */
  int waitForFirst;
  /// Table to store send/recv timestamps
  SRtable *SendsAndRecvs;            
  /// List of objects registered with this PVT branch
  pvtObjects objs;                   
 public:
  /// Basic Constructor
  PVT(void);
  PVT(CkMigrateMessage *) { };
  /// ENTRY: runs the PVT calculation and reports to GVT
  void startPhase(void);             
  /// ENTRY: receive GVT estimate; wake up objects
  /** Receives the new GVT estimate and termination flag; wakes up objects
      for fossil collection and forward execution with new GVT estimate. */
  void setGVT(GVTMsg *m);            
  /// Returns GVT estimate
  int getGVT() { return estGVT; }    
  /// Returns termination flag
  int done() { return simdone; }
  /// Register poser with PVT
  int objRegister(int arrIdx, int safeTime, int sync, sim *myPtr);
  /// Unregister poser from PVT
  void objRemove(int pvtIdx);
  /// Update send/recv table at timestamp
  void objUpdate(int timestamp, int sr); 
  /// Update PVT with safeTime and send/recv table at timestamp
  void objUpdate(int pvtIdx, int safeTime, int timestamp, int sr);
};

/// GVT chare group for estimating GVT
/** Responsibility for GVT estimation shifts between branches after each
    GVT invocation. */
class GVT : public Group { 
private:
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  /// Latest GVT estimate
  int estGVT; 
  /// Inactivity status: number of iterations since GVT has changed
  int inactive;
  /// Time at which GVT last went inactive
  int inactiveTime;
  /// Number of GVT iterations since last LB run
  int nextLBstart; 
  /// Earliest send/recv timestamp in previous GVT invocation
  int lastEarliest;
  /// Number of sends at lastEarliest
  int lastSends;
  /// Number of receives at lastEarliest
  int lastRecvs;
public:
  /// Basic Constructor
  GVT(void);
  GVT(CkMigrateMessage *) { };
  //Use this for Ccd calls
  //static void _runGVT(UpdateMsg *);
  /// ENTRY: Run the GVT
  /** Updates data fields with info from previous GVT estimation.  Fires
      off PVT calculations on all PVT branches. */
  void runGVT(UpdateMsg *);
  /// ENTRY: Gathers PVT reports; calculates and broadcasts GVT to PVTs
  void computeGVT(UpdateMsg *); 
  /// Adds incoming send/recv information to a list
  void addSR(SRentry **SRs, SRentry e);
};

#endif
