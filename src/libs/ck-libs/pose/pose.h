/// Global POSE data and functions; includes and dependencies handled here
/** This code provides all the major global data structures plus a 
    coordination entity that handles initialization and termination of the
    simulation. */
#ifndef POSE_H
#define POSE_H

#include "pose_config.h"
#include "pose.decl.h"

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

class eventMsg; // defined later in sim.h
class rep; // defined later in rep.h
#include "event.h"
#include "eqheap.h"

class sim; // defined later in sim.h
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

/// Main initialization for all of POSE
void POSE_init(); 
/// Start POSE simulation timer and event processing
void POSE_start(); 
/// Use Inactivity Detection to terminate program
void POSE_useID();
/// Simulation end time
extern int POSE_endtime;
/// Use a user-specified end time to terminate program
/** Also uses inactivity detection in conjunction with end time */
void POSE_useET(int et); 
/// Specify an optional callback to be called when simulation terminates
void POSE_registerCallBack(CkCallback cb);
/// Stop POSE simulation
/** Stops timer so statistics collection, callback, final output, etc. 
    are not counted in simulation time. */
void POSE_stop(); 
/// Exit simulation program
void POSE_exit(); 

/// User specified busy wait time (for grainsize testing)
extern double busyWait;
/// Set busy wait time
void POSE_set_busy_wait(double n);
/// Busy wait for busyWait
void POSE_busy_wait();
/// Busy wait for n
void POSE_busy_wait(double n);

/// Flag to indicate how foward execution should proceed
/** 0 for normal forward execution; 1 for state recovery only */
CpvExtern(int, stateRecovery);

#ifdef POSE_COMM_ON
extern int comm_debug;
#endif

/// Class for user-specified callback
class callBack : public CMessage_callBack
{
 public:
  CkCallback callback;
};

/// Coordinator of simulation initialization, start and termination
class pose : public Chare {
 private:
  /// The simulation timer
  double sim_timer;
  /// A callback to execute on termination
  /** If this is used, control is turned over to this at the very end of the
      simulation. */
  CkCallback cb;
  /// Flag to indicate if a callback will be used
  int callBackSet;
  /// Flag to indicate if inactivity detection will be used
  int useID;
  /// Flag to indicate if an end time will be used
  int useET;
 public:
  /// Basic Constructor
  pose(void) { callBackSet = 0; useID = useET = 0; }
  pose(CkMigrateMessage *) { }
  /// Turn on inactivity detection
  void IDon() { useID = 1; }
  /// Turn on end time termination
  void ETon() { useET = 1; }
  /// Start the simulation timer
  void start() { 
    if (useID) CkPrintf("Using Inactivity Detection for termination.\n");
    else CkPrintf("Using endTime of %d for termination.\n", POSE_endtime);
    CkPrintf("Starting simulation...\n"); 
    sim_timer = CmiWallTimer(); 
  }
  /// Register the callback with POSE
  void registerCallBack(callBack *);
  /// Stop the simulation
  /** Stops timer and gathers POSE statistics and proceeds to exit. */
  void stop();
  /// Exit the simulation
  /** Executes callback before terminating program. */
  void exit();
};

#endif
