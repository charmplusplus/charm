#include <charm++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <utility>

#include "ControlPoints.decl.h"
#include "trace-controlPoints.h"
#include "LBDatabase.h"
#include "controlPoints.h"
#include "pathHistory.h"
#include "arrayRedistributor.h"
#include <register.h> // for _entryTable


/**
 *  \addtogroup CriticalPathFramework
 *   @{
 *
 */

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY


CkpvDeclare(MergeablePathHistory, currentlyExecutingPath); // The maximal incoming path for the node
CkpvDeclare(double, timeEntryMethodStarted);

void initializeCriticalPath(void){
  CkpvInitialize(MergeablePathHistory, currentlyExecutingPath); // The maximal incoming path for the node
  CkpvInitialize(double, timeEntryMethodStarted);
  CkpvAccess(timeEntryMethodStarted) = 0.0;
}

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
  std::map<int, PathHistoryTableEntry> &table = controlPointManagerProxy.ckLocalBranch()->pathHistoryTable;
  int &table_last_idx = controlPointManagerProxy.ckLocalBranch()->pathHistoryTableLastIdx;
  int new_idx = table_last_idx++;
  table[new_idx] = *this;
  return new_idx;
}



  
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

  pathInformationMsg *newmsg = new(0) pathInformationMsg;
  newmsg->historySize = 0;
  newmsg->cb = cb;
  newmsg->table_idx = CkpvAccess(currentlyExecutingPath).sender_history_table_idx;
  int pe = CkpvAccess(currentlyExecutingPath).sender_pe;
  CkPrintf("Starting tracing of critical path from pe=%d table_idx=%d\n", pe,  CkpvAccess(currentlyExecutingPath).sender_history_table_idx);
  CkAssert(pe < CkNumPes() && pe >= 0);
  controlPointManagerProxy[pe].traceCriticalPathBack(newmsg);
}



// void  printPathInMsg(void* msg){
//   envelope *env = UsrToEnv(msg);
//   env->printPath();
// }



/// A debugging routine that prints the number of EPs for the program, and the size of the envelope's path fields
void  printEPInfo(){
  CkPrintf("printEPInfo():\n");
  CkPrintf("There are %d EPs\n", (int)_entryTable.size());
  for (int epIdx=0;epIdx<_entryTable.size();epIdx++)
    CkPrintf("EP %d is %s\n", epIdx, _entryTable[epIdx]->name);
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
}




#endif





/*! @} */

