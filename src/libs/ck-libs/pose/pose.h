// File: pose.h
// Global POSE data and functions; includes and dependencies handled here
// Last Modified: 5.29.01 by Terry L. Wilmarth

#ifndef POSE_H
#define POSE_H

extern int cancelNodeCount;
extern int heapNodeCount;
extern int spawnedEventCount;
extern int eventCount;
extern int srEntryCount;
extern int repCount;
extern int eventMsgCount;
extern int eventMsgsRecvd;
extern int eventMsgsDiscarded;

// Primary versions
#define POSE_STATS_ON 1
//#define POSE_COMM_ON 1
//#define LB_ON 1

#ifdef POSE_COMM_ON
#include <StreamingStrategy.h>
#define COMM_TIMEOUT 20
#define COMM_MAXMSG 100
#endif 
#include "pose.decl.h"

// Strategy variables
#define STORE_RATE 100           // default store rate: 1 for every n events
#define SPEC_WINDOW 100         // speculative event window
#define MIN_LEASH 10            // min spec window for adaptive strategy
#define MAX_LEASH 500         // max  "     "     "     "        " 
#define LEASH_FLEX 1           // leash increment
#define GVT_WINDOW 500          // Maximum time GVT can advance

// Load balancer variables
#define LB_SKIP 51           // LB done 1/LB_SKIP times GVT iterations
#define LB_THRESHOLD 2000    // 20 heavy objects
#define LB_DIFF 10000        // min diff between min and max load PEs

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
#include "opt2.h"
#include "opt3.h"
#include "spec.h"
#include "adapt.h"
#include "adapt2.h"
#include "cons.h"
#include "chpt.h"

void POSE_init();  // Main initialization for all of POSE
void POSE_start(); // start POSE simulation timer and other behaviors
void POSE_useQD(); // use QD to terminate program
void POSE_useID(); // use Inactivity Detection to terminate program
void POSE_useET(int et); // use end time to terminate program
void POSE_registerCallBack(CkCallback cb);
void POSE_stop();  // stop POSE simulation timer
void POSE_exit();  // exit program

void POSE_set_busy_wait(double n);
void POSE_busy_wait();
void POSE_busy_wait(double n);

CpvExtern(int, stateRecovery);
extern double busyWait;
extern int POSE_endtime;
#ifdef POSE_COMM_ON
extern int comm_debug;
#endif

class callBack : public CMessage_callBack
{
 public:
  CkCallback callback;
};

class pose : public Chare {
 private:
  double sim_timer;
  CkCallback cb;
  int callBackSet, useQD, useID, useET;
 public:
  pose(void) { callBackSet = 0; useQD = useID = useET = 0; }
  pose(CkMigrateMessage *) { }
  void QDon() { useQD = 1; if (!useET) POSE_endtime = -1; }
  void IDon() { useID = 1; if (!useET) POSE_endtime = -1; }
  void ETon() { useET = 1; }
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
