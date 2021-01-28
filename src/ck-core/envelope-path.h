/**
\file
\addtogroup CkEnvelope
*/
#ifndef _ENVELOPE_PATH_H
#define _ENVELOPE_PATH_H

#include "pup.h"
#include "charm.h"

// #define USE_CRITICAL_PATH_HEADER_ARRAY

#if USE_CRITICAL_PATH_HEADER_ARRAY
// This critical path detection is still experimental
// Added by Isaac (Dec 2008)

// stores the pointer to the currently executing msg
// used in cklocation.C, ck.C
// TODO: convert to CkPv

extern envelope * currentlyExecutingMsg;
extern bool thisMethodSentAMessage;
extern double timeEntryMethodStarted;

// The methods provided by the control point framework to access 
// the most Critical path seen by this PE. 
// These ought not to be called by the user program.
// (defined in controlPoints.C)
extern void resetPECriticalPath();
extern void printPECriticalPath();
extern void registerTerminalEntryMethod();

// Reset the counts for the currently executing message, 
// and also reset the PE's critical path detection
// To be called by the user program
extern void resetCricitalPathDetection();

// Reset the counts for the currently executing message
extern void resetThisEntryPath();



#endif


/** static sizes for arrays in PathHistory objects */
#define numEpIdxs 150
#define numArrayIds 20

/** A class that is used to track the entry points and other information 
    about a critical path as a charm++ program executes.

    This class won't do useful things unless USE_CRITICAL_PATH_HEADER_ARRAY is defined

*/
class PathHistory {
 private:
  int epIdxCount[numEpIdxs];
  int arrayIdxCount[numArrayIds];
  double totalTime;

 public:
  
  const int* getEpIdxCount(){
    return epIdxCount;
  }
  
  const int* getArrayIdxCount(){
    return arrayIdxCount;
  }

  int getEpIdxCount(int i) const {
    return epIdxCount[i];
  }
  
  int getArrayIdxCount(int i) const {
    return arrayIdxCount[i];
  }


  double getTotalTime() const{
    return totalTime;
  }

  
  PathHistory(){
    reset();
  }

  void pup(PUP::er &p) {
    for(int i=0;i<numEpIdxs;i++)
      p|epIdxCount[i];
    for(int i=0;i<numArrayIds;i++)
      p|arrayIdxCount[i];
    p | totalTime;
  } 
  
  double getTime(){
    return totalTime;
  }
  
  void reset(){
    // CkPrintf("reset() currentlyExecutingMsg=%p\n", currentlyExecutingMsg);
    
    totalTime = 0.0;
    
    for(int i=0;i<numEpIdxs;i++){
      epIdxCount[i] = 0;
    }
    for(int i=0;i<numArrayIds;i++){
      arrayIdxCount[i]=0;
    }
  }
  
  int updateMax(const PathHistory & other){
    if(other.totalTime > totalTime){
      //	  CkPrintf("[%d] Found a longer terminal path:\n", CkMyPe());
      //	  other.print();
      
      totalTime = other.totalTime;
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = other.epIdxCount[i];
      }
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i]=other.arrayIdxCount[i];
      }
      return 1;
    }
    return 0;
  }
  
  
  void print() const {
    CkPrintf("Critical Path Time=%lf : ", (double)totalTime);
    for(int i=0;i<numEpIdxs;i++){
      if(epIdxCount[i]>0){
	CkPrintf("EP %d count=%d : ", i, (int)epIdxCount[i]);
      }
    }
    for(int i=0;i<numArrayIds;i++){
      if(arrayIdxCount[i]>0){
	CkPrintf("Array %d count=%d : ", i, (int)arrayIdxCount[i]);
      }
    }
    CkPrintf("\n");
  }

  /// Write a description of the path into the beginning of the provided buffer. The buffer ought to be large enough.
  void printHTMLToString(char* buf) const {
    buf[0] = '\0';

    sprintf(buf+strlen(buf), "Path Time=%lf<br>", (double)totalTime);
    for(int i=0;i<numEpIdxs;i++){
      if(epIdxCount[i]>0){
	sprintf(buf+strlen(buf),"EP %d count=%d<br>", i, (int)epIdxCount[i]);
      }
    }
    for(int i=0;i<numArrayIds;i++){
      if(arrayIdxCount[i]>0){
	sprintf(buf+strlen(buf), "Array %d count=%d<br>", i, (int)arrayIdxCount[i]);
      }
    }
  }
  
  void incrementTotalTime(double time){
    totalTime += time;
  }
  

#if USE_CRITICAL_PATH_HEADER_ARRAY
  void createPath(PathHistory *parentPath){
    // Note that we are sending a message
    thisMethodSentAMessage = true;
    double timeNow = CmiWallTimer();
    
    if(parentPath != NULL){
      //	  CkPrintf("createPath() totalTime = %lf + %lf\n",(double)currentlyExecutingMsg->pathHistory.totalTime, (double)timeNow-timeEntryMethodStarted);
      totalTime = parentPath->totalTime + (timeNow-timeEntryMethodStarted);
      
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = parentPath->epIdxCount[i];
      }
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i] = parentPath->arrayIdxCount[i];
      }
      
    }
    else {
      totalTime = 0.0;
      
      for(int i=0;i<numEpIdxs;i++){
	epIdxCount[i] = 0;
      } 
      for(int i=0;i<numArrayIds;i++){
	arrayIdxCount[i]=0;
      }
    }
	
  }
#endif
      
  void incrementEpIdxCount(int ep){
    epIdxCount[ep]++;
  }

  void incrementArrayIdxCount(int arr){
    arrayIdxCount[arr]++;
  }
      
};


#endif
