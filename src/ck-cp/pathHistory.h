
/**     @defgroup CriticalPathFramework Critical Path Detection
 *
 *      Usable for determining the critical paths in Charm++, Charisma, and SDAG programs.
 */

/**
 *
 *    @addtogroup CriticalPathFramework
 *    @{
 *
 */
#ifndef __PATH_HISTORY_H__
#define __PATH_HISTORY_H__

#include <vector>
#include <map>
#include <utility>
#include <cmath>
#include "PathHistory.decl.h"
#include "envelope.h"

#include <pup_stl.h>

void initializeCriticalPath(void);
void useThisCriticalPathForPriorities();
void automaticallySetMessagePriority(envelope *env);

class pathHistoryManager : public CBase_pathHistoryManager {
private:
    std::map< int, int> criticalPathForPriorityCounts;

    pathInformationMsg *pathForUser; // A place to store the path while we perform a broadcast and reduction to save projections user events.

public:

    pathHistoryManager();

    pathHistoryManager(CkMigrateMessage *m) : CBase_pathHistoryManager(m) {
        CkAbort("pathHistoryManager does not have a working migration constructor.");
    }

    void pup(PUP::er &p) {
        CkAbort("pathHistoryManager cannot be pupped.");
    }

    /** Trace perform a traversal backwards over the critical path specified as a
     * table index for the processor upon which this is called.
     *
     * * If msg->saveAsProjectionsUserEvents is true then the resulting path will be
     * broadcast to broadcastCriticalPathProjections() which will then call a
     * reduction to criticalPathProjectionsDone() which will call the user supplied
     * callback.
     *
     * Otherwise, the callback cb will be called with the resulting msg after the path has
     * been traversed to its origin.
     *
     */
 void traceCriticalPathBackStepByStep(pathInformationMsg *msg);

 void broadcastCriticalPathProjections(pathInformationMsg *msg);

 void criticalPathProjectionsDone(CkReductionMsg *msg);

 void saveCriticalPathForPriorities(pathInformationMsg *msg);

 /// Traverse back and aquire the critical path to be used for automatic message prioritization
 void useCriticalPathForPriories();

 const std::map< int, int> & getCriticalPathForPriorityCounts() const {
     return criticalPathForPriorityCounts;
 }

};

/**
 * Augments the PathHistory in the envelope with the other necessary information
 * from the envelope. These should be derived from an incoming message.
 *
 * These objects can then be used for storing of information about a path
 * outside of the incoming message's header.
 *
   These are used for:
   1) information about the currently executing message
   2) combining the maximum paths along a reduction
   3) combining multiple paths in Charisma and SDAG programs

   This can be constructed from an envelope.
   */

class MergeablePathHistory {
public:
    double timeEntryMethodStarted;

    double preceding_path_time;

    int sender_pe;  // for traversing back over the PAG
    int sender_history_table_idx; // for traversing back over the PAG

    int local_ep;  // The locally executing EP

    int hops;

    MergeablePathHistory(const envelope *env)
    {
        sender_pe = env->getSrcPe();
#if USE_CRITICAL_PATH_HEADER_ARRAY
        sender_history_table_idx = env->pathHistory.get_sender_history_table_idx();
        preceding_path_time = env->pathHistory.getTime();
        hops = env->pathHistory.getHops() + 1;
#endif
        local_ep = env->getEpIdx();
        timeEntryMethodStarted = 0.0;
    }

    MergeablePathHistory() {
        reset();
    }

    MergeablePathHistory(const MergeablePathHistory &current) {
        sender_pe = current.sender_pe;
        sender_history_table_idx = current.sender_history_table_idx;
        local_ep = current.local_ep;
        preceding_path_time = get_entry_method_time(current);
        hops = current.hops;
    }

    double get_entry_method_time(const MergeablePathHistory &current) {
      return (current.preceding_path_time +
        CmiWallTimer() - current.timeEntryMethodStarted);
    }

    void reset(){
        sender_pe = -1;
        sender_history_table_idx = -1;
        local_ep = -1;
        preceding_path_time = 0.0;
        timeEntryMethodStarted = -1;
        hops = 0;
    }

    void sanity_check(){
        if(sender_history_table_idx > -1){
            CkAssert(sender_pe < CkNumPes());
            CkAssert(sender_pe >= -1);
        }
    }

    void updateMax(const MergeablePathHistory& other){
        double now = CmiWallTimer();
        //CkPrintf("====== merging history table  current (%d:%d:%d:%d:%f) \n", sender_pe, sender_history_table_idx, local_ep, local_arr, preceding_path_time+now - timeEntryMethodStarted);
        //CkPrintf("====== merging history table  old (%d:%d:%d:%d:%f)\n", other.sender_pe, other.sender_history_table_idx, other.local_ep, other.local_arr, other.preceding_path_time);
        if(preceding_path_time + now - timeEntryMethodStarted < other.preceding_path_time)
        {
            //CkPrintf(" switch current to the other \n");
            *this = other;
            timeEntryMethodStarted = now;
        }
    }

    void updateMax(const envelope* env){
        updateMax(MergeablePathHistory(env));
    }

    double getTotalTime() const{
        return preceding_path_time;
    }

    /// Write a description of the path into the beginning of the provided buffer. The buffer ought to be large enough.
  void printHTMLToString(char* buf) const{
      buf[0] = '\0';
      sprintf(buf+strlen(buf), "MergeablePathHistory time=%lf send pe=%d idx=%d timeEntryStarted=%lf", (double)preceding_path_time, (int)sender_pe, (int)sender_history_table_idx, (double)timeEntryMethodStarted);
  }

  void setDebug100(){
      preceding_path_time = 100.0;
      CkPrintf("Setting path length to 100\n");
  }

};

/** 
    Stores information about the critical path in the table on each PE. 
    The PAG can be constructed by merging these together.
    These can be constructed from a MergeablePathHistory, and are assumed to refer to the local PE.
*/
class PathHistoryTableEntry {
public:
    int sender_pe;
    int sender_history_table_idx;
    int local_ep;
    int local_pe;
private:
    double start_time;
    double local_path_time;
    double preceding_path_time;

public:

 PathHistoryTableEntry() 
   : sender_pe(-1), 
    sender_history_table_idx(-1), 
    local_ep(-1),
    local_pe(CkMyPe()),
    start_time(0.0),
    local_path_time(0.0), 
    preceding_path_time(0.0)
      {
	// No body
      }
  

  /// Construct an entry for the table assuming the start time in input path is correct
 PathHistoryTableEntry(const MergeablePathHistory& p, double localContribution = 0.0)
    : sender_pe(p.sender_pe), 
    sender_history_table_idx(p.sender_history_table_idx), 
    local_ep(p.local_ep),
    local_pe(CkMyPe()),
    start_time(p.timeEntryMethodStarted),
    local_path_time(localContribution), 
    preceding_path_time(p.preceding_path_time)
      {
	// No body
      }

  /// Construct an entry with the actual start and end times for it.
  PathHistoryTableEntry(const MergeablePathHistory& p, double start, double finish)
    : sender_pe(p.sender_pe), 
    sender_history_table_idx(p.sender_history_table_idx), 
    local_ep(p.local_ep),
    local_pe(CkMyPe()),
    start_time(start),
    local_path_time(finish-start), 
    preceding_path_time(p.preceding_path_time)
      {
	// No body
      }

  void printInfo(const char *prefix = ""){
      CkPrintf("%s [sender pe=%d table idx=%d] [local path contribution=%lf ep=%d] [Time= %lf + %lf]\n", prefix,
          sender_pe, sender_history_table_idx, local_path_time, local_ep, preceding_path_time, local_path_time);
  }


  /// Add an entry for this path history into the table, and write the corresponding information into the provided header
  int addToTableAndEnvelope(envelope *env);
  
  /// Add an entry for this path history into the table. Returns the new index in the table.
  int addToTable();
  
  /// Return the length of the path up to and including this entry
  double getTotalTime(){
      return local_path_time + preceding_path_time;
  }

  double get_start_time() const {return start_time;}
  double get_local_path_time() const {return local_path_time;}
  double get_preceding_path_time() const {return preceding_path_time;}
};


/// A debugging routine that outputs critical path info as Projections user events.
void  saveCurrentPathAsUserEvent(const char* prefix="");

/// A debugging helper routine that sets the values in the currently executing message's path to 100
void setCurrentlyExecutingPathTo100(void);

/// A debugging routine that outputs critical path info as Projections user events.
void  printPathInMsg(void* msg);

/// A debugging routine that outputs critical path info as Projections user events.
void  printEPInfo();

/// Acquire the critical path and deliver it to the user supplied callback
void traceCriticalPathBack(CkCallback cb, bool saveToProjectionsTraces = false);


/// A message containing information about a path of entry method invocations. This contains an array of PathHistoryTableEntry objects
class pathInformationMsg : public CMessage_pathInformationMsg {
public:
    PathHistoryTableEntry *history;
    int historySize;
    int saveAsProjectionsUserEvents;
    CkCallback cb;
    int table_idx;
    int hops;

    void printme(){
        CkPrintf("Path contains %d entries\n", historySize);
        for(int i=historySize-1;i>=0;i--){
            CkPrintf("\tPath Step %d: local_path_time=%lf ep=%d starttime=%lf preceding path time=%lf pe=%d\n",i, history[i].get_local_path_time(),  history[i].local_ep, history[i].get_start_time(), history[i].get_preceding_path_time(), history[i].local_pe);
        }
    }
};


CkpvExtern(MergeablePathHistory, currentlyExecutingPath); // The maximal incoming path for the node
CkpvExtern(double, timeEntryMethodStarted);

/** A table to store all the local nodes in the parallel dependency graph */
typedef std::map<int, PathHistoryTableEntry> PathHistoryTableType;
CkpvExtern(PathHistoryTableType, pathHistoryTable);
/** A counter that defines the new keys for the entries in the pathHistoryTable */
CkpvExtern(int, pathHistoryTableLastIdx);

// Reset the counts for the currently executing message. Cut the incoming path
extern void resetThisEntryPath();

#if USE_CRITICAL_PATH_HEADER_ARRAY
extern void criticalPath_start(envelope * env); 
extern void criticalPath_setep(int epIdx);
extern void criticalPath_send(envelope * sendingEnv);
extern void criticalPath_end();
extern void criticalPath_split(); 

extern   MergeablePathHistory* saveCurrentPath();
extern   void mergePathHistory(MergeablePathHistory*);
#define CK_AUTOMATE_PRIORITY(x)     automaticallySetMessagePriority(x);
#define CK_CRITICALPATH_SEND(x)   criticalPath_send(x);
#define CK_CRITICALPATH_START(x)  criticalPath_start(x);
#define CK_CRITICALPATH_SETEP(x)  criticalPath_setep(x);
#define CK_CRITICALPATH_END()     criticalPath_end();
#define CK_CRITICALPATH_SPLIT()   criticalPath_split();

/// Declare a MergeablePathHistory variable, whose name is mangled with the supplied parameter
#define MERGE_PATH_DECLARE(x) MergeablePathHistory merge_path_##x

/// Reset the merge_path variable
#define MERGE_PATH_RESET(x) merge_path_##x.reset()

/// Take the maximal path from the stored merge_path variable and the currently executing path. Put the result in currently executing path.
#define MERGE_PATH_MAX(x) merge_path_##x.updateMax(CkpvAccess(currentlyExecutingPath)); CkpvAccess(currentlyExecutingPath) = merge_path_##x; 

/// Declare a dynamic MergeablePathHistory variable. Each object can have many merge points stored in this single DECLARE.
#define MERGE_PATH_DECLARE_D(x) std::map<int,MergeablePathHistory> merge_path_D_##x

/// Reset the merge_path variable
#define MERGE_PATH_RESET_D(x,n) merge_path_D_##x[n].reset()

/// Delete the merge_path variable
#define MERGE_PATH_DELETE_D(x,n) merge_path_D_##x.erase(n)

/// Delete all entries in the merge_path variable
#define MERGE_PATH_DELETE_ALL_D(x) merge_path_D_##x.clear()

/// Take the maximal path from the stored merge_path variable and the currently executing path. Put the result in currently executing path.
#define MERGE_PATH_MAX_D(x,n) merge_path_D_##x[n].updateMax(CkpvAccess(currentlyExecutingPath)); CkpvAccess(currentlyExecutingPath) = merge_path_D_##x[n]; 


#else

/// Empty no-op version for when critical path history is not compiled in

#define CK_AUTOMATE_PRIORITY(x)
#define CK_CRITICALPATH_SETEP(x)
#define CK_CRITICALPATH_SEND(x)
#define CK_CRITICALPATH_START(x)
#define CK_CRITICALPATH_END()
#define CK_CRITICALPATH_SPLIT()

#define MERGE_PATH_DECLARE(x) ;
#define MERGE_PATH_RESET(x) ;
#define MERGE_PATH_MAX(x) ;
#define MERGE_PATH_DECLARE_D(x) ;
#define MERGE_PATH_RESET_D(x,n) ;
#define MERGE_PATH_DELETE_D(x,n) ;
#define MERGE_PATH_DELETE_ALL_D(x) ;
#define MERGE_PATH_MAX_D(x,n) ;

#endif

/** @} */
#endif
