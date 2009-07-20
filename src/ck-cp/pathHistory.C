#include <charm++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <utility>

#include "PathHistory.decl.h"
#include "LBDatabase.h"
//#include "controlPoints.h"
#include "pathHistory.h"
#include "arrayRedistributor.h"
#include <register.h> // for _entryTable


/**
 *  \addtogroup CriticalPathFramework
 *   @{
 *
 */



/*readonly*/ CProxy_pathHistoryManager pathHistoryManagerProxy;

CkpvDeclare(MergeablePathHistory, currentlyExecutingPath); // The maximal incoming path for the node
CkpvDeclare(double, timeEntryMethodStarted);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
/** A table to store all the local nodes in the parallel dependency graph */
CkpvDeclare(PathHistoryTableType, pathHistoryTable);
/** A counter that defines the new keys for the entries in the pathHistoryTable */
CkpvDeclare(int, pathHistoryTableLastIdx);
#endif


/// A mainchare that is used just to create a group at startup
class pathHistoryMain : public CBase_pathHistoryMain {
public:
  pathHistoryMain(CkArgMsg* args){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    pathHistoryManagerProxy = CProxy_pathHistoryManager::ckNew();
#endif
  }
  ~pathHistoryMain(){}
};


pathHistoryManager::pathHistoryManager(){

}


  /** Trace perform a traversal backwards over the critical path specified as a 
      table index for the processor upon which this is called.
      
      The callback cb will be called with the resulting msg after the path has 
      been traversed to its origin.  
  */
 void pathHistoryManager::traceCriticalPathBack(pathInformationMsg *msg){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
   int count = CkpvAccess(pathHistoryTable).count(msg->table_idx);

#if DEBUGPRINT > 2
   CkPrintf("Table entry %d on pe %d occurs %d times in table\n", msg->table_idx, CkMyPe(), count);
#endif
   CkAssert(count==0 || count==1);

    if(count > 0){ 
      PathHistoryTableEntry & path = CkpvAccess(pathHistoryTable)[msg->table_idx];
      int idx = path.sender_history_table_idx;
      int pe = path.sender_pe;

#if DEBUGPRINT > 2
      CkPrintf("Table entry %d on pe %d points to pe=%d idx=%d\n", msg->table_idx, CkMyPe(), pe, idx);
#endif

      // Make a copy of the message for broadcasting to all PEs
      pathInformationMsg *newmsg = new(msg->historySize+1) pathInformationMsg;
      for(int i=0;i<msg->historySize;i++){
	newmsg->history[i] = msg->history[i];
      }
      newmsg->history[msg->historySize] = path;
      newmsg->historySize = msg->historySize+1;
      newmsg->cb = msg->cb;
      newmsg->table_idx = idx;
      
      // Keep a message for returning to the user's callback
      pathForUser = new(msg->historySize+1) pathInformationMsg;
      for(int i=0;i<msg->historySize;i++){
	pathForUser->history[i] = msg->history[i];
      }
      pathForUser->history[msg->historySize] = path;
      pathForUser->historySize = msg->historySize+1;
      pathForUser->cb = msg->cb;
      pathForUser->table_idx = idx;
      
      
      if(idx > -1 && pe > -1){
	CkAssert(pe < CkNumPes() && pe >= 0);
	thisProxy[pe].traceCriticalPathBack(newmsg);
      } else {
	CkPrintf("Traced critical path back to its origin.\n");
	CkPrintf("Broadcasting it to all PE\n");
	thisProxy.broadcastCriticalPathResult(newmsg);
      }
    } else {
      CkAbort("ERROR: Traced critical path back to a nonexistent table entry.\n");
    }

    delete msg;
#else
    CkAbort("Shouldn't call pathHistoryManager::traceCriticalPathBack when critical path detection is not enabled");
#endif
  }


void pathHistoryManager::broadcastCriticalPathResult(pathInformationMsg *msg){

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  CkPrintf("[%d] Received broadcast of critical path\n", CkMyPe());
  int me = CkMyPe();
  int intersectsLocalPE = false;

  // Create user events for critical path

  for(int i=msg->historySize-1;i>=0;i--){
    if(CkMyPe() == msg->history[i].local_pe){
      // Part of critical path is local
      // Create user event for it

      //      CkPrintf("\t[%d] Path Step %d: local_path_time=%lf arr=%d ep=%d starttime=%lf preceding path time=%lf pe=%d\n",CkMyPe(), i, msg->history[i].get_local_path_time(), msg-> history[i].local_arr, msg->history[i].local_ep, msg->history[i].get_start_time(), msg->history[i].get_preceding_path_time(), msg->history[i].local_pe);
      traceUserBracketEvent(5900, msg->history[i].get_start_time(), msg->history[i].get_start_time() + msg->history[i].get_local_path_time());
      intersectsLocalPE = true;
    }

  }
  

#if PRUNE_CRITICAL_PATH_LOGS
  // Tell projections tracing to only output log entries if I contain part of the critical path
  if(! intersectsLocalPE){
    disableTraceLogOutput();
    CkPrintf("PE %d doesn't intersect the critical path, so its log files won't be created\n", CkMyPe() );
  }
#endif  
  
#if TRACE_ALL_PATH_TABLE_ENTRIES
  // Create user events for all table entries
  std::map< int, PathHistoryTableEntry >::iterator iter;
  for(iter=pathHistoryTable.begin(); iter != pathHistoryTable.end(); iter++){
    double startTime = iter->second.get_start_time();
    double endTime = iter->second.get_start_time() + iter->second.get_local_path_time();
    traceUserBracketEvent(5901, startTime, endTime);
  }
#endif

  int data=1;
  CkCallback cb(CkIndex_pathHistoryManager::criticalPathDone(NULL),thisProxy[0]); 
  contribute(sizeof(int), &data, CkReduction::sum_int, cb);

#endif

}

void pathHistoryManager::criticalPathDone(CkReductionMsg *msg){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  CkPrintf("[%d] All PEs have received the critical path information. Sending critical path to user supplied callback.\n", CkMyPe());
  pathForUser->cb.send(pathForUser);
  pathForUser = NULL;
#endif
}




void initializeCriticalPath(void){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  CkpvInitialize(MergeablePathHistory, currentlyExecutingPath); // The maximal incoming path for the node
  CkpvInitialize(double, timeEntryMethodStarted);
  CkpvAccess(timeEntryMethodStarted) = 0.0;


  CkpvInitialize(PathHistoryTableType, pathHistoryTable);
  CkpvInitialize(int, pathHistoryTableLastIdx);
  CkpvAccess(pathHistoryTableLastIdx) = 0;
#endif
}










#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
// The PathHistoryEnvelope class is disabled if USE_CRITICAL_PATH_HEADER_ARRAY isn't defined
void PathHistoryEnvelope::reset(){
  totalTime = 0.0;
  sender_history_table_idx = -1;
}

  
void PathHistoryEnvelope::print() const {
  CkPrintf("print() is not implemented\n");
}

/// Write a description of the path into the beginning of the provided buffer. The buffer ought to be large enough.

  
void PathHistoryEnvelope::incrementTotalTime(double time){
  totalTime += time;
}



void PathHistoryEnvelope::setDebug100(){
  totalTime = 100.0;   
}


/// Add an entry for this path history into the table, and write the corresponding information into the outgoing envelope
int PathHistoryTableEntry::addToTableAndEnvelope(envelope *env){
  // Add to table
  int new_idx = addToTable();

  // Fill in outgoing envelope
  CkAssert(env != NULL);
  env->pathHistory.set_sender_history_table_idx(new_idx);
  env->pathHistory.setTime(local_path_time + preceding_path_time);

  // Create a user event for projections
  char *note = new char[4096];
  sprintf(note, "addToTableAndEnvelope<br> ");
  env->pathHistory.printHTMLToString(note+strlen(note));
  traceUserSuppliedNote(note); // stores a copy of the string
  delete[] note;
  
  return new_idx;
}
  

/// Add an entry for this path history into the table. Returns the index in the table for it.
int PathHistoryTableEntry::addToTable(){
  int new_idx = CkpvAccess(pathHistoryTableLastIdx) ++;
  CkpvAccess(pathHistoryTable)[new_idx] = *this;
  return new_idx;
}
#endif


  
void resetThisEntryPath(void) {
  CkpvAccess(currentlyExecutingPath).reset();
}


/// A debugging routine that outputs critical path info as user events.
void  saveCurrentPathAsUserEvent(char* prefix){
  if(CkpvAccess(currentlyExecutingPath).getTotalTime() > 0.0){
    //traceUserEvent(5020);

    char *note = new char[4096];
    sprintf(note, "%s<br> saveCurrentPathAsUserEvent()<br> ", prefix);
    CkpvAccess(currentlyExecutingPath).printHTMLToString(note+strlen(note));
    traceUserSuppliedNote(note); // stores a copy of the string
    delete[] note;

  } else {
    traceUserEvent(5010);
  }
 
}


void setCurrentlyExecutingPathTo100(void){
  CkpvAccess(currentlyExecutingPath).setDebug100();
}



/// A routine for printing out information along the critical path.
void traceCriticalPathBack(CkCallback cb){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY

  pathInformationMsg *newmsg = new(0) pathInformationMsg;
  newmsg->historySize = 0;
  newmsg->cb = cb;
  newmsg->table_idx = CkpvAccess(currentlyExecutingPath).sender_history_table_idx;
  int pe = CkpvAccess(currentlyExecutingPath).sender_pe;
  CkPrintf("Starting tracing of critical path from pe=%d table_idx=%d\n", pe,  CkpvAccess(currentlyExecutingPath).sender_history_table_idx);
  CkAssert(pe < CkNumPes() && pe >= 0);
  pathHistoryManagerProxy[pe].traceCriticalPathBack(newmsg);
#else

  pathInformationMsg * pathForUser = new(0) pathInformationMsg;  
  pathForUser->historySize = 0;                                                                                        
  pathForUser->cb = CkCallback();                                                                                                      
  pathForUser->table_idx = -1;      
  cb.send(pathForUser);    

#endif
}



// void  printPathInMsg(void* msg){
//   envelope *env = UsrToEnv(msg);
//   env->printPath();
// }



/// A debugging routine that prints the number of EPs for the program, and the size of the envelope's path fields
void  printEPInfo(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  CkPrintf("printEPInfo():\n");
  CkPrintf("There are %d EPs\n", (int)_entryTable.size());
  for (int epIdx=0;epIdx<_entryTable.size();epIdx++)
    CkPrintf("EP %d is %s\n", epIdx, _entryTable[epIdx]->name);
#endif
}




/// Save information about the critical path contained in the message that is about to execute.
void criticalPath_start(envelope * env){ 
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
#if DEBUG
  CkPrintf("criticalPath_start(envelope * env) srcpe=%d sender table idx=%d  time=%lf\n", env->getSrcPe(),  env->pathHistory.get_sender_history_table_idx(), env->pathHistory.getTotalTime() );
#endif

  CkpvAccess(currentlyExecutingPath).sender_pe = env->getSrcPe();
  CkpvAccess(currentlyExecutingPath).sender_history_table_idx = env->pathHistory.get_sender_history_table_idx();
  CkpvAccess(currentlyExecutingPath).preceding_path_time = env->pathHistory.getTotalTime();

  CkpvAccess(currentlyExecutingPath).sanity_check();
  
  CkpvAccess(currentlyExecutingPath).local_ep  = -1;
  CkpvAccess(currentlyExecutingPath).local_arr = -1;

  double now = CmiWallTimer();
  CkpvAccess(currentlyExecutingPath).timeEntryMethodStarted = now;
  CkpvAccess(timeEntryMethodStarted) = now;

  switch(env->getMsgtype()) {
  case ForArrayEltMsg:
    //    CkPrintf("Critical Path Detection handling a ForArrayEltMsg\n");    
    CkpvAccess(currentlyExecutingPath).local_ep = env->getsetArrayEp();
    CkpvAccess(currentlyExecutingPath).local_arr = env->getArrayMgrIdx();
    
    break;

  case ForNodeBocMsg:
    //CkPrintf("Critical Path Detection handling a ForNodeBocMsg\n");    
    break;

  case ForChareMsg:
    //CkPrintf("Critical Path Detection handling a ForChareMsg\n");        
    break;

  case ForBocMsg:
    //CkPrintf("Critical Path Detection handling a ForBocMsg\n");    
    break;

  case ArrayEltInitMsg:
    // Don't do anything special with these
    break;

  default:
    CkPrintf("Critical Path Detection can't yet handle message type %d\n", (int)env->getMsgtype());
  }
  
  

  saveCurrentPathAsUserEvent("criticalPath_start()<br> ");


#endif
}


/// Modify the envelope of a message that is being sent for critical path detection and store an entry in a table on this PE.
void criticalPath_send(envelope * sendingEnv){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
#if DEBUG
  CkPrintf("criticalPath_send(envelope * sendingEnv)\n");
#endif
  double now = CmiWallTimer();
  PathHistoryTableEntry entry(CkpvAccess(currentlyExecutingPath), CkpvAccess(timeEntryMethodStarted), now);
  entry.addToTableAndEnvelope(sendingEnv);
  
#endif
}


/// Handle the end of the entry method in the critical path detection processes. This should create a forward dependency for the object.
void criticalPath_end(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  saveCurrentPathAsUserEvent("criticalPath_end()<br> ");

  CkpvAccess(currentlyExecutingPath).reset();

#if DEBUG
  CkPrintf("criticalPath_end()\n");
#endif

#endif
}



/// Split an entry method invocation into multiple logical tasks for the critical path analysis.
/// SDAG doen's break up the code in useful ways, so I'll make it add calls to this in the generated code.
void criticalPath_split(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  saveCurrentPathAsUserEvent("criticalPath_split()<br> ");

  // save an entry in the table
  double now = CmiWallTimer();
  PathHistoryTableEntry entry(CkpvAccess(currentlyExecutingPath), now-CkpvAccess(timeEntryMethodStarted) );
  int tableidx = entry.addToTable();

  // end the old task

  
  // start the new task
  CkpvAccess(currentlyExecutingPath).sender_pe = CkMyPe();
  CkpvAccess(currentlyExecutingPath).sender_history_table_idx = tableidx;
  CkpvAccess(currentlyExecutingPath).preceding_path_time = entry.getTotalTime();
  CkpvAccess(currentlyExecutingPath).timeEntryMethodStarted = now;
  CkpvAccess(timeEntryMethodStarted) = now;
#endif
}


#include "PathHistory.def.h"

/*! @} */

