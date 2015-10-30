/**
 * \addtogroup CkPerf
 */
/*@{*/


#ifndef _TRACE_UTILIZATION_H
#define _TRACE_UTILIZATION_H

#include <stdio.h>
#include <errno.h>
#include <deque>

#include "charm++.h"


#include "trace.h"
#include "envelope.h"
#include "register.h"
#include "trace-common.h"
#include "ckcallback-ccs.h"

#include "TraceUtilization.decl.h"

#define INVALIDEP     -2
#define TRACEON_EP    -3
#define NUM_DUMMY_EPS 9


// initial bin size, time in seconds
#define  BIN_PER_SEC	1000
#define  BIN_SIZE	0.001
#define NUM_BINS      32768


/** Define the types used in the gathering of sum detail statistics for use with CCS */
#define numBins_T int
#define numProcs_T int
#define entriesInBin_T short
#define ep_T short
#define utilization_T unsigned char
#define other_EP 10000


/* readonly */ extern CProxy_TraceUtilizationBOC traceUtilizationGroupProxy;

void collectUtilizationData(void *data, double currT);



/** A main chare that can create the BOC/group */
class TraceUtilizationInit : public Chare {
 public:
  TraceUtilizationInit(CkArgMsg *m) {
    delete m;
    CkPrintf("[%d] TraceUtilizationInit creating traceUtilizationGroupProxy");
    fflush(stdout);
   
    traceUtilizationGroupProxy = CProxy_TraceUtilizationBOC::ckNew();
    
    CkPrintf("Trace Summary now listening in for CCS Client\n");
    CcsRegisterHandler("CkPerfSumDetail compressed", CkCallback(CkIndex_TraceUtilizationBOC::ccsRequestSumDetailCompressed(NULL), traceUtilizationGroupProxy[0])); 
    
    CkPrintf("[%d] Setting up periodic startCollectData callback\n", CkMyPe());
    CcdCallOnConditionKeep(CcdPERIODIC_1second, collectUtilizationData, (void *)NULL);

  }
};






/** 
    A class that reads/writes a buffer out of different types of data.

    This class exists because I need to get references to parts of the buffer 
    that have already been used so that I can increment counters inside the buffer.
*/

class compressedBuffer {
 public:
  char* buf;
  int pos; ///<< byte position just beyond the previously read/written data

  compressedBuffer(){
    buf = NULL;
    pos = 0;
  }

  compressedBuffer(int bytes){
    buf = (char*)malloc(bytes);
    pos = 0;
  }

  compressedBuffer(void *buffer){
    buf = (char*)buffer;
    pos = 0;
  }
  
  void init(void *buffer){
    buf = (char*)buffer;
    pos = 0;
  }
  
  inline void * currentPtr(){
    return (void*)(buf+pos);
  }
  
  template <typename T>
    T read(int offset){
    // to resolve unaligned writes causing bus errors, need memcpy
    T v;
    memcpy(&v, buf+offset, sizeof(T));
    return v;
  }
  
  template <typename T>
    void write(T v, int offset){
    T v2 = v; // on stack
    // to resolve unaligned writes causing bus errors, need memcpy
    memcpy(buf+offset, &v2, sizeof(T));
  }
    
  template <typename T>
    void increment(int offset){
    T temp;
    temp = read<T>(offset);
    temp ++;
    write<T>(temp, offset);
  }

  template <typename T>
    void accumulate(T v, int offset){
    T temp;
    temp = read<T>(offset);
    temp += v;
    write<T>(temp, offset);
  }
  
  template <typename T>
    int push(T v){
    int oldpos = pos;
    write<T>(v, pos);
    pos += sizeof(T);
    return oldpos;
  }
  
  template <typename T>
    T pop(){
    T temp = read<T>(pos);
    pos += sizeof(T);
    return temp;
  }

  template <typename T>
    T peek(){
    T temp = read<T>(pos);
    return temp;
  }

  template <typename T0, typename T>
    T peekSecond(){
    T temp;
    memcpy(&temp, buf+pos+sizeof(T0), sizeof(T));
    return temp;
  }

  int datalength(){
    return pos;
  }
     
  void * buffer(){
    return (void*) buf;
  }  

  void freeBuf(){
    free(buf);
  }

  ~compressedBuffer(){
    // don't free the buf because the user my have supplied the buffer
  }
    
};



compressedBuffer compressAvailableNewSumDetail(int max=10000);
void mergeCompressedBin(compressedBuffer *srcBufferArray, int numSrcBuf, int *numProcsRepresentedInMessage, int totalProcsAcrossAllMessages, compressedBuffer &destBuffer);
//void printSumDetailInfo(int desiredBinsToSend);
CkReductionMsg *sumDetailCompressedReduction(int nMsg,CkReductionMsg **msgs);
void printCompressedBuf(compressedBuffer b);
compressedBuffer fakeCompressedMessage();
compressedBuffer emptyCompressedBuffer();
void sanityCheckCompressedBuf(compressedBuffer b);
bool isCompressedBufferSane(compressedBuffer b);
double averageUtilizationInBuffer(compressedBuffer b);





class TraceUtilization : public Trace {
 public:
  int execEp; // the currently executing EP
  double start; // the start time for the currently executing EP

  unsigned int epInfoSize;

  double *cpuTime;     // NUM_BINS*epInfoSize
  int lastBinUsed;
  unsigned int numBinsSent;
  unsigned int previouslySentBins;


  TraceUtilization() {
    execEp = TRACEON_EP;
    cpuTime = NULL;
    lastBinUsed = -1;
    numBinsSent = 0;
  }


  /// Initialize memory after the number of EPs has been determined
  void initMem(){
    int _numEntries=_entryTable.size();
    epInfoSize = _numEntries + NUM_DUMMY_EPS + 1; // keep a spare EP
    //    CkPrintf("allocating cpuTime[%d]\n", (int) (NUM_BINS*epInfoSize));
    cpuTime = new double[NUM_BINS*epInfoSize];
    _MEMCHECK(cpuTime);

    if(CkMyPe() == 0)
      writeSts();

  }

  void writeSts(void);

  void creation(envelope *e, int epIdx, int num=1) {}

  void beginExecute(envelope *e, void *obj);
  void beginExecute(CmiObjId  *tid);
  void beginExecute(int event,int msgType,int ep,int srcPe, int mlen=0, CmiObjId *idx=NULL, void *obj=NULL);
  void endExecute(void);
  void beginIdle(double currT) {}
  void endIdle(double currT) {}
  void beginPack(void){}
  void endPack(void) {}
  void beginUnpack(void) {}
  void endUnpack(void) {}
  void beginComputation(void) {  initMem(); }
  void endComputation(void) {}
  void traceClearEps() {}
  void traceWriteSts() {}
  void traceClose() {}

  void addEventType(int eventType);
  
  int cpuTimeEntriesAvailable() const { return lastBinUsed+1; }
  int cpuTimeEntriesSentSoFar() const { return numBinsSent; }
  void incrementNumCpuTimeEntriesSent(int n) { numBinsSent += n; }


  double sumUtilization(int startBin, int endBin);


  void updateCpuTime(int epIdx, double startTime, double endTime){
    
    //    CkPrintf("updateCpuTime(startTime=%lf endTime=%lf)\n", startTime, endTime);
    
    if (epIdx >= epInfoSize) {
      CkPrintf("WARNING: epIdx=%d >=  epInfoSize=%d\n", (int)epIdx, (int)epInfoSize );
      return;
    }
    
    int startingBinIdx = (int)(startTime/BIN_SIZE);
    int endingBinIdx = (int)(endTime/BIN_SIZE);
    
    if (startingBinIdx == endingBinIdx) {
      addToCPUtime(startingBinIdx, epIdx, endTime - startTime);
    } else if (startingBinIdx < endingBinIdx) { // EP spans intervals
      addToCPUtime(startingBinIdx, epIdx, (startingBinIdx+1)*BIN_SIZE - startTime);
      while(++startingBinIdx < endingBinIdx)
	addToCPUtime(startingBinIdx, epIdx, BIN_SIZE);
      addToCPUtime(endingBinIdx, epIdx, endTime - endingBinIdx*BIN_SIZE);
    } 
  }


  /// Zero out all entries from (lastBinUsed+1) up to and including interval.
  inline void zeroIfNecessary(unsigned int interval){
    for(unsigned int i=lastBinUsed+1; i<= interval; i++){
      // zero all eps for this bin
      for(unsigned int j=0;j<epInfoSize;j++){
	cpuTime[(i%NUM_BINS)*epInfoSize+j] = 0.0;
      }
    }
    lastBinUsed = interval;
  }
    
  UInt getEpInfoSize() { return epInfoSize; }

  /// for Summary-Detail
  inline double getCPUtime(unsigned int interval, unsigned int ep) {
    CkAssert(ep < epInfoSize);
    if(cpuTime != NULL && interval <= lastBinUsed)
      return cpuTime[(interval%NUM_BINS)*epInfoSize+ep]; 
    else {
      CkPrintf("getCPUtime called with invalid options: cpuTime=%p interval=%d ep=%d\n", cpuTime, (int)interval, (int)ep);
      return 0.0;
    }
  }

  inline void addToCPUtime(unsigned int interval, unsigned int ep, double val){
    //    CkAssert(ep < epInfoSize);
    zeroIfNecessary(interval);
    //    CkPrintf("addToCPUtime interval=%d\n", (int)interval);
    cpuTime[(interval%NUM_BINS)*epInfoSize+ep] += val;
  }
   

  inline double getUtilization(int interval, int ep){
    return getCPUtime(interval, ep) * 100.0 * (double)BIN_PER_SEC; 
  }


  compressedBuffer compressNRecentSumDetail(int desiredBinsToSend);


};






class TraceUtilizationBOC : public CBase_TraceUtilizationBOC {
  
  std::deque<CkReductionMsg *> storedSumDetailResults;
  
 public:
  TraceUtilizationBOC() {}
  TraceUtilizationBOC(CkMigrateMessage* msg) {}
  ~TraceUtilizationBOC() {}
 

  /// Entry methods:
  void ccsRequestSumDetailCompressed(CkCcsRequestMsg *m);
  void collectSumDetailData();
  void sumDetailDataCollected(CkReductionMsg *);

};





#endif

/*@}*/
