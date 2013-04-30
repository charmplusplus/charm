#include <charm++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <utility>


#if CMK_WITH_CONTROLPOINT


#include "PathHistory.decl.h"
#include "LBDatabase.h"
#include "pathHistory.h"
//#include "controlPoints.h"
//#include "arrayRedistributor.h"
#include <register.h> // for _entryTable

#include "trace-projections.h"


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
 void pathHistoryManager::traceCriticalPathBackStepByStep(pathInformationMsg *msg){
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

      // Make a copy of the message as we forward it along
      pathInformationMsg *newmsg = new(msg->historySize+1) pathInformationMsg;
      for(int i=0;i<msg->historySize;i++){
	newmsg->history[i] = msg->history[i];
      }
      newmsg->history[msg->historySize] = path;
      newmsg->historySize = msg->historySize+1;
      newmsg->saveAsProjectionsUserEvents = msg->saveAsProjectionsUserEvents;
      newmsg->cb = msg->cb;
      newmsg->table_idx = idx;
        
      if(idx > -1 && pe > -1){
	// Not yet at origin, keep tracing the path back
	CkAssert(pe < CkNumPes() && pe >= 0);
	thisProxy[pe].traceCriticalPathBackStepByStep(newmsg);
      } else {
	CkPrintf("Traced critical path back to its origin.\n");
	if(msg->saveAsProjectionsUserEvents){

	  // Keep a message for returning to the user's callback
	  pathForUser = new(msg->historySize+1) pathInformationMsg;
	  for(int i=0;i<msg->historySize;i++){
	    pathForUser->history[i] = msg->history[i];
	  }
	  pathForUser->history[msg->historySize] = path;
	  pathForUser->historySize = msg->historySize+1;
	  pathForUser->saveAsProjectionsUserEvents = msg->saveAsProjectionsUserEvents;
	  pathForUser->cb = msg->cb;
	  pathForUser->table_idx = idx;
	  
	  CkPrintf("Broadcasting it to all PE\n");
	  thisProxy.broadcastCriticalPathProjections(newmsg);
	} else {
	  newmsg->cb.send(newmsg);
	}

      }
    } else {
      CkAbort("ERROR: Traced critical path back to a nonexistent table entry.\n");
    }

    delete msg;
#else
    CkAbort("Shouldn't call pathHistoryManager::traceCriticalPathBack when critical path detection is not enabled");
#endif
  }


void pathHistoryManager::broadcastCriticalPathProjections(pathInformationMsg *msg){

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
      traceUserBracketEvent(32000, msg->history[i].get_start_time(), msg->history[i].get_start_time() + msg->history[i].get_local_path_time());

      intersectsLocalPE = true;
    }

  }

  traceRegisterUserEvent("Critical Path", 32000);
  
  
#define PRUNE_CRITICAL_PATH_LOGS 0

#if PRUNE_CRITICAL_PATH_LOGS
  // Tell projections tracing to only output log entries if I contain part of the critical path

  enableTraceLogOutput();
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
    traceUserBracketEvent(32001, startTime, endTime);
  }
#endif

  int data=1;
  CkCallback cb(CkIndex_pathHistoryManager::criticalPathProjectionsDone(NULL),thisProxy[0]); 
  contribute(sizeof(int), &data, CkReduction::sum_int, cb);

#endif

}

void pathHistoryManager::criticalPathProjectionsDone(CkReductionMsg *msg){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  CkPrintf("[%d] All PEs have received the critical path information. Sending critical path to user supplied callback.\n", CkMyPe());
  pathForUser->cb.send(pathForUser);
  pathForUser = NULL;
#endif
}




/// An interface callable by the application.
void useThisCriticalPathForPriorities(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  pathHistoryManagerProxy.ckLocalBranch()->useCriticalPathForPriories();
#endif
}


/// Callable from inside charm++ delivery mechanisms (after envelope contains epIdx):
void automaticallySetMessagePriority(envelope *env){
  #ifdef USE_CRITICAL_PATH_HEADER_ARRAY

#if DEBUG
  if(env->getPriobits() == 8*sizeof(int)){
    CkPrintf("[%d] priorities for env=%p are integers\n", CkMyPe(), env);
  } else if(env->getPriobits() == 0) {
    CkPrintf("[%d] priorities for env=%p are not allocated in message\n", CkMyPe(), env);
  } else {
    CkPrintf("[%d] priorities for env=%p are not integers: %d priobits\n", CkMyPe(), env, env->getPriobits());
  }
#endif
  
  
  const std::map< std::pair<int,int>, int> & criticalPathForPriorityCounts = pathHistoryManagerProxy.ckLocalBranch()->getCriticalPathForPriorityCounts();
  
  if(criticalPathForPriorityCounts.size() > 0 && env->getPriobits() == 8*sizeof(int)) {
    
    switch(env->getMsgtype()) {
    case ForArrayEltMsg:
    case ForIDedObjMsg:
    case ForChareMsg:
      {        
	const int ep = env->getsetArrayEp();
	const int arr = env->getArrayMgrIdx();
	
	const std::pair<int,int> k = std::make_pair(arr, ep);
	const int count = criticalPathForPriorityCounts.count(k);
	
#if DEBUG
	CkPrintf("[%d] destination array,ep occurs %d times along stored critical path\n", CkMyPe(), count);
#endif
      	
	if(count > 0){
	  // Set the integer priority to high
#if DEBUG
	  CkPrintf("Prio auto high\n");
#endif
	  *(int*)(env->getPrioPtr()) = -5;
	} else {
	  // Set the integer priority to low
#if DEBUG
	  CkPrintf("Prio auto low: %d,%d\n", arr, ep);
#endif
	  *(int*)(env->getPrioPtr()) = 5;
	}
	
      }
      break;
      
    case ForNodeBocMsg:
      CkPrintf("Can't Critical Path Autoprioritize a ForNodeBocMsg\n");    
      break;
      
    case ForBocMsg:
      CkPrintf("Can't Critical Path Autoprioritize a ForBocMsg\n");
      break;
      
    case ArrayEltInitMsg:
      // Don't do anything special with these
      CkPrintf("Can't Critical Path Autoprioritize a ArrayEltInitMsg\n");
      break;
      
    default:
      CkPrintf("Can't Critical Path Autoprioritize messages of [unknown type]\n");
      break;
    }
      
  }

#endif
}



void pathHistoryManager::useCriticalPathForPriories(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY

  // Request a critical path that will be stored everywhere for future use in autotuning message priorities
  
  // The resulting critical path should be broadcast to saveCriticalPathForPriorities() on all PEs
  CkCallback cb(CkIndex_pathHistoryManager::saveCriticalPathForPriorities(NULL),thisProxy); 
  traceCriticalPathBack(cb, false);
#endif  
}



void pathHistoryManager::saveCriticalPathForPriorities(pathInformationMsg *msg){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY

  CkPrintf("[%d] saveCriticalPathForPriorities() Receiving critical paths\n", CkMyPe());
  fflush(stdout);
  
  criticalPathForPriorityCounts.clear();
  
  // Save a list of which entries are along the critical path
  for(int i=msg->historySize-1;i>=0;i--){
    
    PathHistoryTableEntry &e = msg->history[i];
    
#if DEBUG
    if(CkMyPe() == 0){
      CkPrintf("\t[%d] Path Step %d: local_path_time=%lf arr=%d ep=%d starttime=%lf preceding path time=%lf pe=%d\n",CkMyPe(), i, e.get_local_path_time(), msg-> history[i].local_arr, e.local_ep, e.get_start_time(), e.get_preceding_path_time(), e.local_pe);
    }
#endif
    
    const std::pair<int,int> k = std::make_pair(e.local_arr, e.local_ep);
    if(criticalPathForPriorityCounts.count(k) == 1)
      criticalPathForPriorityCounts[k]++;
    else
      criticalPathForPriorityCounts[k] = 1;  
  }
  



  // print out the list just for debugging purposes
  if(CkMyPe() == 0){
    std::map< std::pair<int,int>, int>::iterator iter;
    for(iter=criticalPathForPriorityCounts.begin();iter!=criticalPathForPriorityCounts.end();++iter){
      const std::pair<int,int> k = iter->first;
      const int c = iter->second;

      CkPrintf("[%d] On critical path EP %d,%d occurs %d times\n", CkMyPe(), k.first, k.second, c);

    }
  }

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

#if 0
  // Create a user event for projections
  char *note = new char[4096];
  sprintf(note, "addToTableAndEnvelope<br> ");
  env->pathHistory.printHTMLToString(note+strlen(note));
  traceUserSuppliedNote(note); // stores a copy of the string
  delete[] note;
#endif  

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
void  saveCurrentPathAsUserEvent(const char* prefix){
  if(CkpvAccess(currentlyExecutingPath).getTotalTime() > 0.0){
    //traceUserEvent(5020);

#if 0
    char *note = new char[4096];
    sprintf(note, "%s<br> saveCurrentPathAsUserEvent()<br> ", prefix);
    CkpvAccess(currentlyExecutingPath).printHTMLToString(note+strlen(note));
    traceUserSuppliedNote(note); // stores a copy of the string
    delete[] note;
#endif

  } else {
#if 0
    traceUserEvent(5010);
#endif
  }
 
}


void setCurrentlyExecutingPathTo100(void){
  CkpvAccess(currentlyExecutingPath).setDebug100();
}


/// Acquire the critical path and deliver it to the user supplied callback
void traceCriticalPathBack(CkCallback cb, bool saveToProjectionsTraces){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  pathInformationMsg *newmsg = new(0) pathInformationMsg;
  newmsg->historySize = 0;
  newmsg->cb = cb;
  newmsg->saveAsProjectionsUserEvents = saveToProjectionsTraces;
  newmsg->table_idx = CkpvAccess(currentlyExecutingPath).sender_history_table_idx;
  int pe = CkpvAccess(currentlyExecutingPath).sender_pe;
  CkPrintf("Starting tracing of critical path from pe=%d table_idx=%d\n", pe,  CkpvAccess(currentlyExecutingPath).sender_history_table_idx);
  CkAssert(pe < CkNumPes() && pe >= 0);
  pathHistoryManagerProxy[pe].traceCriticalPathBackStepByStep(newmsg);
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

#endif
