/// Adaptive Synchronization Strategy No. 5
/** This strategy consists of 4 different algorithmic attempts at
    improving performance by adjusting the time leash in different
    ways.  ALGORITHM_TO_USE chooses the algorithm.  See
    SyncStrategies.txt for more info. */

#ifndef ADAPT5_H
#define ADAPT5_H

#define ALGORITHM_TO_USE 4

class adapt5 : public opt3 {
 public:
  int iter, objUsage;
  int recentAvgRBLeash, recentTotalRBLeash, recentAvgRBLeashCount, avgEventsPerRB;
  int avgRBsPerGVTIter, recentAvgEventSparsity;
  bool initialAvgRBLeashCalc;  // true until the first rollback occurs
  adapt5() { 
    // divide maximums by 64 to give some room in case POSE_TimeType is an int
    //    timeLeashTotal = 0LL;
    //    stepCalls = 0;
    iter = 0;
    objUsage = pose_config.max_usage * pose_config.store_rate;
    STRAT_T = ADAPT5_T; 
    //timeLeash = POSE_TimeMax/2;
    timeLeash = POSE_TimeMax / 64;
    avgRBsPerGVTIter = 0;
    recentAvgEventSparsity = 1;
    recentAvgRBLeash = 1000;
    recentTotalRBLeash = 0;
    recentAvgRBLeashCount = 0;
    avgEventsPerRB = 1000;
    initialAvgRBLeashCalc = true;
  }
  virtual void Step();
  /// Set the average number of rollbacks per GVT iteration
  inline void setAvgRBsPerGVTIter(int avgRBs) { avgRBsPerGVTIter = avgRBs; }
  /// Set the recent event sparsity average (in GVT ticks / event)
  inline void setRecentAvgEventSparsity(int avgSparsity) { recentAvgEventSparsity = avgSparsity; }
  /// Set the value of the time leash
  inline void setTimeLeash(POSE_TimeType tl) {
#if ALGORITHM_TO_USE == 4
    timeLeash = tl;
    if (timeLeash > 50000) {
      timeLeash = 50000;
    }
    if (timeLeash < 1) {
      timeLeash = 1;
    }
#endif
  }
};

#endif
