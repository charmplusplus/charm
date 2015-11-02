/**
 * \addtogroup CkPerf
*/
/*@{*/

#include "trace-utilization.h"


/* readonly */ CProxy_TraceUtilizationBOC traceUtilizationGroupProxy;


/// A reduction type for merging compressed sum detail data
CkReduction::reducerType sumDetailCompressedReducer;


void collectUtilizationData(void *ignore, double currT) {
  // Only start collecting after a few seconds have passed. This way there will hopefully be at least 1000 bins to pickup each time we try.
  static int numTimesCalled = 0;
  numTimesCalled ++;
  if(numTimesCalled > 4){
    traceUtilizationGroupProxy.collectSumDetailData();
  }
}


CkpvStaticDeclare(TraceUtilization*, _trace);

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceutilization(char **argv)
{
  //  CkPrintf("[%d] _createTraceutilization\n", CkMyPe());

  // Register the reducer
  CkAssert(sizeof(short) == 2);
  sumDetailCompressedReducer=CkReduction::addReducer(sumDetailCompressedReduction);

  CkpvInitialize(TraceUtilization*, _trace);
  CkpvAccess(_trace) = new TraceUtilization();
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));

}



void TraceUtilization::beginExecute(CmiObjId *tid)
{
  beginExecute(-1,-1,_threadEP,-1);
}

void TraceUtilization::beginExecute(envelope *e, void *obj)
{
  // no message means thread execution
  if (e==NULL) {
    beginExecute(-1,-1,_threadEP,-1);
  }
  else {
    beginExecute(-1,-1,e->getEpIdx(),-1);
  }  
}

void TraceUtilization::beginExecute(int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx, void *obj)
{
  if (execEp != INVALIDEP) {
    TRACE_WARN("Warning: TraceUtilization two consecutive BEGIN_PROCESSING!\n");
    return;
  }
  
  execEp=ep;
  start = TraceTimer();
}


void TraceUtilization::endExecute(void)
{

  if (execEp == TRACEON_EP) {
    // if trace just got turned on, then one expects to see this
    // END_PROCESSING event without seeing a preceeding BEGIN_PROCESSING
    return;
  }

  double endTime = TraceTimer();
 
  updateCpuTime(execEp, start, endTime);


  execEp = INVALIDEP;
}



void TraceUtilization::addEventType(int eventType)
{
  CkPrintf("FIXME handle TraceUtilization::addEventType(%d)\n", eventType);
}




/**

Send back to the client compressed sum-detail style measurements about the 
utilization for each active PE combined across all PEs.

The data format sent by this handler is a bunch of records(one for each bin) of the following format:
   #samples (EP,utilization)* 

One example record for two EPS that executed during the sample period. 
EP 3 used 150/200 of the time while EP 122 executed for 20/200 of the time. 
All of these would be packed as bytes into the message:
2 3 150 122 20

 */
void TraceUtilizationBOC::ccsRequestSumDetailCompressed(CkCcsRequestMsg *m) {
  CkPrintf("CCS request for compressed sum detail. (found %d stored in deque)\n",  storedSumDetailResults.size() );
  //  CkAssert(sumDetail);
  int datalength;

#if 0

  compressedBuffer fakeMessage = fakeCompressedMessage();
  CcsSendDelayedReply(m->reply, fakeMessage.datalength(), fakeMessage.buffer() );
  fakeMessage.freeBuf();

#else

  if (storedSumDetailResults.size()  == 0) {
    compressedBuffer b = emptyCompressedBuffer();
    CcsSendDelayedReply(m->reply, b.datalength(), b.buffer()); 
    b.freeBuf();
  } else {
    CkReductionMsg * msg = storedSumDetailResults.front();
    storedSumDetailResults.pop_front();

    
    void *sendBuffer = (void *)msg->getData();
    datalength = msg->getSize();
    CcsSendDelayedReply(m->reply, datalength, sendBuffer);
    
    delete msg;
  }
    
  
#endif

  //  CkPrintf("CCS response of %d bytes sent.\n", datalength);
  delete m;
}



void TraceUtilizationBOC::collectSumDetailData() {
  TraceUtilization* t = CkpvAccess(_trace);

  compressedBuffer b = t->compressNRecentSumDetail(BIN_PER_SEC);

  //  CkPrintf("[%d] contributing buffer created by compressNRecentSumDetail avg util=%lg\n", CkMyPe(), averageUtilizationInBuffer(b));
  //  printCompressedBuf(b);
  // fflush(stdout);
  
  
#if 0
  b = fakeCompressedMessage();
#endif
  
  //  CkPrintf("[%d] contributing %d bytes worth of SumDetail data\n", CkMyPe(), b.datalength());
  
  //  CProxy_TraceUtilizationBOC sumProxy(traceSummaryGID);
  CkCallback cb(CkIndex_TraceUtilizationBOC::sumDetailDataCollected(NULL), thisProxy[0]);
  contribute(b.datalength(), b.buffer(), sumDetailCompressedReducer, cb);
  
  b.freeBuf();
}


void TraceUtilizationBOC::sumDetailDataCollected(CkReductionMsg *msg) {
  CkAssert(CkMyPe() == 0);

  compressedBuffer b(msg->getData());
  CkPrintf("putting CCS reply in queue (average utilization= %lg)\n", averageUtilizationInBuffer(b));
  //if(isCompressedBufferSane(b)){
    storedSumDetailResults.push_back(msg); 
    //}

    // CkPrintf("[%d] Reduction of SumDetail completed. Result stored in storedSumDetailResults deque(size now=%d)\n", CkMyPe(), storedSumDetailResults.size() );
    //  fflush(stdout);

}



void TraceUtilization::writeSts(void) {
  // open sts file
  char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".util.sts")+1];
  sprintf(fname, "%s.util.sts", CkpvAccess(traceRoot));
  FILE* stsfp = fopen(fname, "w+");
  if (stsfp == 0) {
       CmiAbort("Cannot open summary sts file for writing.\n");
  }
  delete[] fname;

  traceWriteSTS(stsfp,0);
  fprintf(stsfp, "END\n");

  fclose(stsfp);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  
/// Compress a buffer by merging all entries in a bin that are less than the threshold into a single "other" category
  compressedBuffer moveTinyEntriesToOther(compressedBuffer src, double threshold){
    //    CkPrintf("[%d] moveTinyEntriesToOther\n", CkMyPe());
    
    // reset the src buffer to the beginning
    src.pos = 0;

    compressedBuffer dest(100000); 
    
    int numBins = src.pop<numBins_T>();
    int numProcs = src.pop<numProcs_T>();
    
    dest.push<numBins_T>(numBins);
    dest.push<numProcs_T>(numProcs);
    
    
    for(int i=0;i<numBins;i++){
      double utilizationInOther = 0.0;
      
      entriesInBin_T numEntriesInSrcBin = src.pop<entriesInBin_T>();
      int numEntriesInDestBinOffset = dest.push<entriesInBin_T>(0);
      
      CkAssert(numEntriesInSrcBin < 200);

      for(int j=0;j<numEntriesInSrcBin;j++){
	ep_T ep = src.pop<ep_T>();
	double v = src.pop<utilization_T>();
	
	if(v < threshold * 250.0){
	  // do not copy bin into destination
	  utilizationInOther += v / 250.0;
	} else {
	  // copy bin into destination
	  dest.increment<entriesInBin_T>(numEntriesInDestBinOffset);
	  dest.push<ep_T>(ep);
	  dest.push<utilization_T>(v);
	}

      }
      
      // if other category has stuff in it, add it to the destination buffer
      if(utilizationInOther > 0.0){
	dest.increment<entriesInBin_T>(numEntriesInDestBinOffset);
	dest.push<ep_T>(other_EP);
	if(utilizationInOther > 1.0)
	  utilizationInOther = 1.0;
	dest.push<utilization_T>(utilizationInOther*250.0);
      }
      
    }
   
    return dest;
  }
  
    


/// A reducer for merging compressed sum detail data
CkReductionMsg *sumDetailCompressedReduction(int nMsg,CkReductionMsg **msgs){
  // CkPrintf("[%d] sumDetailCompressedReduction(nMsgs=%d)\n", CkMyPe(), nMsg);
  
  compressedBuffer *incomingMsgs = new compressedBuffer[nMsg];
  int *numProcsRepresentedInMessage = new int[nMsg];
  
  int numBins = 0;
  int totalsize = 0;
  int totalProcsAcrossAllMessages = 0;
  
  for (int i=0;i<nMsg;i++) {
    incomingMsgs[i].init(msgs[i]->getData());
    
    //  CkPrintf("[%d] Incoming reduction message %d has average utilization %lg\n", CkMyPe(),  i, averageUtilizationInBuffer(incomingMsgs[i])); 
    //   CkPrintf("Is buffer %d sane? %s\n", i, isCompressedBufferSane(incomingMsgs[i]) ? "yes": "no" );


    totalsize += msgs[i]->getSize();
    //  CkPrintf("BEGIN MERGE MESSAGE=========================================================\n");
    //   printCompressedBuf(incomingMsgs[i]);
    
    // Read first value from message. 
    // Make sure all messages have the same number of bins
    if(i==0)
      numBins = incomingMsgs[i].pop<numBins_T>();
    else 
      CkAssert( numBins ==  incomingMsgs[i].pop<numBins_T>() );
    
    // Read second value from message. 
    numProcsRepresentedInMessage[i] = incomingMsgs[i].pop<numProcs_T>();
    totalProcsAcrossAllMessages += numProcsRepresentedInMessage[i];
    //    CkPrintf("Number of procs in message[%d] is %d\n", i,  (int)numProcsRepresentedInMessage[i]);
  }
  
  compressedBuffer dest(totalsize + 100); 
  
  // build a compressed representation of each merged bin
  dest.push<numBins_T>(numBins);
  dest.push<numProcs_T>(totalProcsAcrossAllMessages);
  
  for(int i=0; i<numBins; i++){
    mergeCompressedBin(incomingMsgs, nMsg, numProcsRepresentedInMessage, totalProcsAcrossAllMessages, dest);
  }
  
  // CkPrintf("END MERGE RESULT=========================================================\n");
  // printCompressedBuf(dest);


  //CkPrintf("[%d] Merged buffer has average utilization %lg \n", CkMyPe(), averageUtilizationInBuffer(dest));

  //CkPrintf("Is resulting merged buffer sane? %s\n", isCompressedBufferSane(dest) ? "yes": "no" );  
  
  compressedBuffer dest2 = moveTinyEntriesToOther(dest, 0.10);
  
  //  CkPrintf("Is resulting merged Filtered buffer sane? %s\n", isCompressedBufferSane(dest2) ? "yes": "no" ); 

  //  CkPrintf("[%d] Outgoing reduction (filtered) message has average utilization %lf \n", CkMyPe(), averageUtilizationInBuffer(dest2));

  
  CkReductionMsg *m = CkReductionMsg::buildNew(dest2.datalength(),dest2.buffer());   
  dest.freeBuf();
  delete[] incomingMsgs;
  delete[] numProcsRepresentedInMessage;
  return m;
}







/// Create fake sum detail data in the compressed format (for debugging)
 compressedBuffer fakeCompressedMessage(){
   CkPrintf("[%d] fakeCompressedMessage\n", CkMyPe());
   
   compressedBuffer fakeBuf(10000);
   
   int numBins = 55;
   int numProcs = 1000;

   // build a compressed representation of each merged bin
   fakeBuf.push<numBins_T>(numBins);
   fakeBuf.push<numProcs_T>(numProcs);
   for(int i=0; i<numBins; i++){
     int numRecords = 3;
     fakeBuf.push<entriesInBin_T>(numRecords);
     for(int j=0;j<numRecords;j++){
       fakeBuf.push<ep_T>(j*10+2);
       fakeBuf.push<utilization_T>(120.00);
     }  
   }
   
   //CkPrintf("Fake Compressed Message:=========================================================\n");
   //   printCompressedBuf(fakeBuf);

   CkAssert(isCompressedBufferSane(fakeBuf));

   return fakeBuf;
 }


 /// Create an empty message
 compressedBuffer emptyCompressedBuffer(){
   compressedBuffer result(sizeof(numBins_T));
   result.push<numBins_T>(0);
   return result;
 }




/** print out the compressed buffer starting from its begining*/
void printCompressedBuf(compressedBuffer b){
  // b should be passed in by value, and hence we can modify it
  b.pos = 0;
  int numEntries = b.pop<numBins_T>();
  CkPrintf("Buffer contains %d records\n", numEntries);
  int numProcs = b.pop<numProcs_T>();
  CkPrintf("Buffer represents an average over %d PEs\n", numProcs);

  for(int i=0;i<numEntries;i++){
    entriesInBin_T recordLength = b.pop<entriesInBin_T>();
    if(recordLength > 0){
      CkPrintf("    Record %d is of length %d : ", i, recordLength);
      
      for(int j=0;j<recordLength;j++){
	ep_T ep = b.pop<ep_T>();
	utilization_T v = b.pop<utilization_T>();
	CkPrintf("(%d,%f) ", ep, v);
      }
    
      CkPrintf("\n");
    }
  }
}



 bool isCompressedBufferSane(compressedBuffer b){
   // b should be passed in by value, and hence we can modify it  
   b.pos = 0;  
   numBins_T numBins = b.pop<numBins_T>();  
   numProcs_T numProcs = b.pop<numProcs_T>();  
   
   if(numBins > 2000){
     ckout << "WARNING: numBins=" << numBins << endl;
     return false;
   }
   
   for(int i=0;i<numBins;i++){  
     entriesInBin_T recordLength = b.pop<entriesInBin_T>();  
     if(recordLength > 200){
       ckout << "WARNING: recordLength=" << recordLength << endl;
       return false;
     }
     
     if(recordLength > 0){  
       
       for(int j=0;j<recordLength;j++){  
         ep_T ep = b.pop<ep_T>();  
         utilization_T v = b.pop<utilization_T>();  
         //      CkPrintf("(%d,%f) ", ep, v);  
	 if(((ep>800 || ep <0 ) && ep != other_EP) || v < 0.0 || v > 251.0){
	   ckout << "WARNING: ep=" << ep << " v=" << v << endl;
	   return false;
	 }
       }  
       
     }  
   }  
   
   return true;
 }



 double averageUtilizationInBuffer(compressedBuffer b){
   // b should be passed in by value, and hence we can modify it  
   b.pos = 0;  
   numBins_T numBins = b.pop<numBins_T>();  
   numProcs_T numProcs = b.pop<numProcs_T>();  
   
   //   CkPrintf("[%d] averageUtilizationInBuffer numProcs=%d   (grep reduction message)\n", CkMyPe(), numProcs);
   
   double totalUtilization = 0.0;
   
   for(int i=0;i<numBins;i++) {  
     entriesInBin_T entriesInBin = b.pop<entriesInBin_T>();     
     for(int j=0;j<entriesInBin;j++){  
       ep_T ep = b.pop<ep_T>();  
       totalUtilization +=  b.pop<utilization_T>();  
     }
   }
   
   return totalUtilization / numBins / 2.5;
 }
 
 

void sanityCheckCompressedBuf(compressedBuffer b){  
   CkAssert(isCompressedBufferSane(b)); 
 }  
 


double TraceUtilization::sumUtilization(int startBin, int endBin){
   int epInfoSize = getEpInfoSize();
   
   double a = 0.0;

   for(int i=startBin; i<=endBin; i++){
     for(int j=0; j<epInfoSize; j++){
       a += cpuTime[(i%NUM_BINS)*epInfoSize+j];
     }
   }
   return a;
 }

 
 /// Create a compressed buffer of the n most recent sum detail samples
 compressedBuffer TraceUtilization::compressNRecentSumDetail(int desiredBinsToSend){
   //   CkPrintf("compressNRecentSumDetail(desiredBinsToSend=%d)\n", desiredBinsToSend);

   int startBin =  cpuTimeEntriesSentSoFar();
   int numEntries = getEpInfoSize();

   int endBin = startBin + desiredBinsToSend - 1;
   int binsToSend = endBin - startBin + 1;
   CkAssert(binsToSend >= desiredBinsToSend );
   incrementNumCpuTimeEntriesSent(binsToSend);


#if 0
   bool nonePrinted = true;
   for(int i=0;i<(NUM_BINS-1000);i+=1000){
     double expectedU = sumUtilization(i, i+999);
     if(expectedU > 0.0){
          CkPrintf("[%d of %d] compressNRecentSumDetail All bins: start=%05d end=%05d values in array sum to %lg\n", CkMyPe(), CkNumPes(),  i, i+999, expectedU);
       nonePrinted = false;
     }
   }
   
   if(nonePrinted)
     CkPrintf("[%d of %d] compressNRecentSumDetail All bins are 0\n", CkMyPe(), CkNumPes() );

   fflush(stdout);
#endif

   int bufferSize = 8*(2+numEntries) * (2+binsToSend)+100;
   compressedBuffer b(bufferSize);

   b.push<numBins_T>(binsToSend);
   b.push<numProcs_T>(1); // number of processors along reduction subtree. I am just one processor.
   //   double myu = 0.0;
   
   for(int i=0; i<binsToSend; i++) {
     // Create a record for bin i
     //  CkPrintf("Adding record for bin %d\n", i);
     int numEntriesInRecordOffset = b.push<entriesInBin_T>(0); // The number of entries in this record
     
     for(int e=0; e<numEntries; e++) {
       double scaledUtilization = getUtilization(i+startBin,e) * 2.5; // use range of 0 to 250 for the utilization, so it can fit in an unsigned char
       if(scaledUtilization > 0.0) {
	 //CkPrintf("scaledUtilization=%lg !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", scaledUtilization);
	 if(scaledUtilization > 250.0)
	   scaledUtilization = 250.0;
	 
	 b.push<ep_T>(e);
	 b.push<utilization_T>(scaledUtilization);
	 //	 myu += scaledUtilization;
	 b.increment<entriesInBin_T>(numEntriesInRecordOffset);
       }
     }
   }
   
   // CkPrintf("[%d] compressNRecentSumDetail resulting buffer: averageUtilizationInBuffer()=%lg myu=%lg\n", CkMyPe(), averageUtilizationInBuffer(b), myu);
   // fflush(stdout);
   
   return b;
 }
 




/** Merge the compressed entries from the first bin in each of the srcBuf buffers.
     
*/
 void mergeCompressedBin(compressedBuffer *srcBufferArray, int numSrcBuf, int *numProcsRepresentedInMessage, int totalProcsAcrossAllMessages, compressedBuffer &destBuf){
  // put a counter at the beginning of destBuf
  int numEntriesInDestRecordOffset = destBuf.push<entriesInBin_T>(0);
  
  //  CkPrintf("BEGIN MERGE------------------------------------------------------------------\n");
  
  // Read off the number of bins in each buffer
  int *remainingEntriesToRead = new int[numSrcBuf];
  for(int i=0;i<numSrcBuf;i++){
    remainingEntriesToRead[i] = srcBufferArray[i].pop<entriesInBin_T>();
  }

  int count = 0;
  // Count remaining entries to process
  for(int i=0;i<numSrcBuf;i++){
    count += remainingEntriesToRead[i];
  }
  
  while (count>0) {
    // find first EP from all buffers (these are sorted by EP already)
    int minEp = 10000;
    for(int i=0;i<numSrcBuf;i++){
      if(remainingEntriesToRead[i]>0){
	int ep = srcBufferArray[i].peek<ep_T>();
	if(ep < minEp){
	  minEp = ep;
	}
      }
    }
    
    //   CkPrintf("[%d] mergeCompressedBin minEp found was %d   totalProcsAcrossAllMessages=%d\n", CkMyPe(), minEp, (int)totalProcsAcrossAllMessages);
    
    destBuf.increment<entriesInBin_T>(numEntriesInDestRecordOffset);

    // Merge contributions from all buffers that list the EP
    double v = 0.0;
    for(int i=0;i<numSrcBuf;i++){
      if(remainingEntriesToRead[i]>0){
	int ep = srcBufferArray[i].peek<ep_T>(); 
	if(ep == minEp){
	  srcBufferArray[i].pop<ep_T>();  // pop ep
	  double util = srcBufferArray[i].pop<utilization_T>();
	  v += util * numProcsRepresentedInMessage[i];
	  remainingEntriesToRead[i]--;
	  count --;
	}
      }
    }

    // create a new entry in the output for this EP.
    destBuf.push<ep_T>(minEp);
    destBuf.push<utilization_T>(v / (double)totalProcsAcrossAllMessages);

  }


  delete [] remainingEntriesToRead;
  // CkPrintf("[%d] End of mergeCompressedBin:\n", CkMyPe() );
  // CkPrintf("END MERGE ------------------------------------------------------------------\n");
 }



#include "TraceUtilization.def.h"


/*@}*/




















