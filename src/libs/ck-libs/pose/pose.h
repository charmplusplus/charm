// File: pose.h
// Global POSE data and functions; includes and dependencies handled here
// Last Modified: 5.29.01 by Terry L. Wilmarth

#ifndef POSE_H
#define POSE_H

// Primary versions
#define POSE_STATS_ON 1
//#define POSE_COMM_ON 1
//#define LB_ON 1

#ifdef POSE_COMM_ON
#include <StreamingStrategy.h>
#endif 
#include "pose.decl.h"

// Load balancer variables
#define LB_SKIP 51           // LB done 1/LB_SKIP times GVT iterations
#define LB_THRESHOLD 2000    // 20 heavy objects
#define LB_DIFF 10000        // min diff between min and max load PEs

// Strategy variables
#define MAX_FUTURE_OFFSET 10000 // CancelList GetItem gets cancels w/ts < gvt+this
#define STORE_RATE 1         // default store rate: 1 for every n events
#define SPEC_WINDOW 30     // speculative event window
#define GVT_WINDOW 200      // GVT improvement limit; sets s/r table size
                             // GVT window should not be more than leash so
                             // upper bound is SPEC_WINDOW or MIN_LEASH 
                             // depending on synch strategy in use
#define GVT_bucket 25       // number of buckets to sort sends/recvs into
#define TBL_THRESHOLD 2000
#define MIN_LEASH 100
#define MAX_LEASH 20000

// MISC
#define MAX_POOL_SIZE 20     // maximum size of an eventMsg pool
#define DEBUG_INDENT_INC 3   // debug indentation increment

#define SEND 0
#define RECV 1

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include "charm++.h"
#include "eventID.h"
#include "evmpool.h"
#include "srtable.h"
#include "stats.h"
#include "cancel.h"

class eventMsg;  // defined later in sim.h
class rep;       // defined later in rep.h
#include "event.h"
#include "eqheap.h"

class sim;  // defined later in sim.h
#include "pvtobj.h"
#include "lbObject.h"
#include "ldbal.h"
#include "gvt.h"
#include "evq.h"

class strat; // defined later in strat.h
#include "rep.h"
#include "strat.h"
#include "sim.h"
#include "opt.h"
#include "adapt.h"
#include "cons.h"
#include "chpt.h"

void POSE_init();  // Main initialization for all of POSE
void POSE_start(); // start POSE simulation timer and other behaviors
void POSE_useQD(); // use QD to terminate program
void POSE_useID(); // use Inactivity Detection to terminate program
void POSE_registerCallBack(CkCallback cb);
void POSE_stop();  // stop POSE simulation timer
void POSE_exit();  // exit program

void POSE_set_busy_wait(double n);
void POSE_busy_wait();
void POSE_busy_wait(double n);

CpvExtern(int, stateRecovery);
extern double busyWait;
extern int POSE_endtime;

class callBack : public CMessage_callBack
{
 public:
  CkCallback callback;
};

class pose : public Chare {
 private:
  double sim_timer;
  CkCallback cb;
  int callBackSet, useQD, useID;
 public:
  pose(void) { callBackSet = 0; useQD = useID = 0; }
  pose(CkMigrateMessage *) { }
  void QDon() { useQD = 1; }
  void IDon() { useID = 1; }
  void start() { 
    if (useQD) {
      CkPrintf("Using Quiescence Detection for termination.\n");
      CkStartQD(CkIndex_pose::quiesce(), &thishandle);
    }
    else if (useID) {
      CkPrintf("Using Inactivity Detection for termination.\n");
    }
    else {
      CkPrintf("Using endTime of %d for termination.\n", POSE_endtime);
    }
    CkPrintf("Starting simulation...\n"); 
    sim_timer = CmiWallTimer(); 
  }
  void registerCallBack(callBack *);
  void stop();
  void exit();
  void quiesce() { CkPrintf("Your program has quiesced!\n"); POSE_stop(); }
};

void pdb_indent(int pdb_level);  // debug indentation

#endif
