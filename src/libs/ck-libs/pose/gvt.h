// File: gvt.h

// Implements the Global Virtual Time (GVT) algorithm; provides chare
// groups PVT and GVT. Objects interact with the local PVT branch.
// PVT branches summarize object info and report to the single
// floating GVT object, which broadcasts results to all PVT branches.

#ifndef GVT_H
#define GVT_H

// synchronization strategies
#define OPTIMISTIC 0
#define CONSERVATIVE 1

#include "gvt.decl.h"

// Global handles ThePVT and TheGVT are declared in gvt.C, used everywhere
extern CkGroupID ThePVT;  
extern CkGroupID TheGVT;

class SRentry;
class SRtable;

// data structure used to send info to GVT
class UpdateMsg : public CMessage_UpdateMsg {
public:
  int optPVT, conPVT;         // pvts of local optimistic/conservative objects
  int earlyTS, earlySends, earlyRecvs;
  int nextTS, nextSends, nextRecvs;
  int inactive, inactiveTime;
  int nextLB;                 // iterations of GVT since last LB
};

// Message for the exchange of basic info between PVT & GVT
class GVTMsg : public CMessage_GVTMsg {
public:
  int estGVT, done;
};

class PVT : public Group {  // PVT chare group
private:
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  int optPVT, conPVT, estGVT;        // PVT and GVT estimates
  int simdone;                       // simulation done flag
  int waitingForGVT;                 // flag to synchronize PVTs with GVT
  SRtable *SendsAndRecvs;            // Send and Recv events
  pvtObjects objs;                   // list of registered objects
public:
  PVT(void);
  PVT(CkMigrateMessage *) { };
  void startPhase(void);             // starts off a PVT cycle
  void setGVT(GVTMsg *m);            // set gvt on local branch (used by GVT)
  // non-entry methods
  int getGVT() { return estGVT; }    // objects get the current gvt estimate
  int done() { return simdone; }     // objects check if the simulation is done
  int objRegister(int arrIdx, int safeTime, int sync, sim *myPtr);
  void objRemove(int pvtIdx);        // unregister object
  void objUpdate(int timestamp, int sr); // update send/recv, ovt
  void objUpdate(int pvtIdx, int safeTime, int timestamp, int sr);
};

class GVT : public Group { // GVT chare
private:
  int estGVT;                        // GVT estimates
  int inactive, inactiveTime;        // #iterations since change in state
  int nextLBstart;                   // #iterations since last LB run
  int lastEarliest, lastSends, lastRecvs;
  int lastNextEarliest, lastNextSends, lastNextRecvs;
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  int maxEndtime(int Pend, int optEst, int conEst);
public:
  GVT(void);
  GVT(CkMigrateMessage *) { };
  static void _runGVT(UpdateMsg *);
  void runGVT(void);            // starts a GVT cycle locally
  void runGVT(UpdateMsg *);     // starts a GVT cycle remotely
  void computeGVT(UpdateMsg *); // gathers PVT reports; sends GVT estimate
};

#endif
