#ifndef NDMESH_STREAMER_H
#define NDMESH_STREAMER_H

#include <algorithm>
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"
#include "completion.h"
#include "ckarray.h"

// limit total number of buffered data items to
// maxNumDataItemsBuffered_ (flush when limit is reached) but allow
// allocation of up to a factor of OVERALLOCATION_FACTOR more space to
// take advantage of nonuniform filling of buffers
#define OVERALLOCATION_FACTOR 4

// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH
// #define CACHE_ARRAY_METADATA // only works for 1D array clients
// #define STREAMER_VERBOSE_OUTPUT

#define TRAM_BROADCAST (-100)

struct MeshLocation {
  int dimension; 
  int bufferIndex; 
}; 

template<class dtype>
class MeshStreamerMessage : public CMessage_MeshStreamerMessage<dtype> {

public:

  int finalMsgCount; 
  int dimension; 
  int numDataItems;
  int *destinationPes;
  dtype *dataItems;

  MeshStreamerMessage(int dim): numDataItems(0), dimension(dim) {
    finalMsgCount = -1; 
  }

  int addDataItem(const dtype& dataItem) {
    dataItems[numDataItems] = dataItem;
    return ++numDataItems; 
  }

  void markDestination(const int index, const int destinationPe) {
    destinationPes[index] = destinationPe;
  }

  const dtype& getDataItem(const int index) {
    return dataItems[index];
  }

};

template <class dtype>
class MeshStreamerArrayClient : public CBase_MeshStreamerArrayClient<dtype>{
private:
  CompletionDetector *detectorLocalObj_;
public:
  MeshStreamerArrayClient(){
    detectorLocalObj_ = NULL; 
  }
  MeshStreamerArrayClient(CkMigrateMessage *msg) {}
  // would like to make it pure virtual but charm will try to
  // instantiate the abstract class, leading to errors
  virtual void process(const dtype& data) {
    CkAbort("Error. MeshStreamerArrayClient::process() is being called. "
            "This virtual function should have been defined by the user.\n");
  };     
  void setDetector(CompletionDetector *detectorLocalObj) {
    detectorLocalObj_ = detectorLocalObj;
  }
  void receiveRedeliveredItem(dtype data) {
#ifdef STREAMER_VERBOSE_OUTPUT
    CkPrintf("[%d] redelivered to index %d\n", 
             CkMyPe(), this->thisIndex.data[0]);
#endif
    if (detectorLocalObj_ != NULL) {
      detectorLocalObj_->consume();
    }
    process(data);
  }

  void pup(PUP::er& p) {
    CBase_MeshStreamerArrayClient<dtype>::pup(p);
   }  

};

template <class dtype>
class MeshStreamerGroupClient : public CBase_MeshStreamerGroupClient<dtype>{

public:
  virtual void process(const dtype& data) = 0;
  virtual void receiveArray(dtype *data, int numItems, int sourcePe) {
    for (int i = 0; i < numItems; i++) {
      process(data[i]);
    }
  }
};

template <class dtype>
class MeshStreamer : public CBase_MeshStreamer<dtype> {

private:
  int bufferSize_; 
  int maxNumDataItemsBuffered_;
  int numDataItemsBuffered_;

  int numMembers_; 
  int *individualDimensionSizes_;
  int *combinedDimensionSizes_;

  int *startingIndexAtDimension_; 

  int myIndex_;
  int *myLocationIndex_;

  CkCallback   userCallback_;
  bool yieldFlag_;

  double progressPeriodInMs_; 
  bool isPeriodicFlushEnabled_; 
  bool hasSentRecently_;
  MeshStreamerMessage<dtype> ***dataBuffers_;

  CProxy_CompletionDetector detector_;
  int prio_;
  int yieldCount_;

#ifdef CACHE_LOCATIONS
  MeshLocation *cachedLocations_;
  bool *isCached_; 
#endif


  // only used for staged completion
  int **cntMsgSent_;
  int *cntMsgReceived_;
  int *cntMsgExpected_;
  int *cntFinished_; 
  int dimensionToFlush_;
  int numLocalDone_; 
  int numLocalContributors_;

  void storeMessage(int destinationPe, 
                    const MeshLocation& destinationCoordinates, 
                    const void *dataItem, bool copyIndirectly = false);
  virtual void localDeliver(const dtype& dataItem) = 0; 
  virtual void localBroadcast(const dtype& dataItem) = 0; 
  virtual int numElementsInClient() = 0;
  virtual int numLocalElementsInClient() = 0; 

  virtual void initLocalClients() = 0;

  void sendLargestBuffer();
  void flushToIntermediateDestinations();
  void flushDimension(int dimension, bool sendMsgCounts = false); 
  MeshLocation determineLocation(int destinationPe, 
                                 int dimensionReceivedAlong);

protected:

  int numDimensions_;
  bool useStagedCompletion_;
  CompletionDetector *detectorLocalObj_;
  virtual int copyDataItemIntoMessage(
              MeshStreamerMessage<dtype> *destinationBuffer, 
              const void *dataItemHandle, bool copyIndirectly = false);
  void insertData(const void *dataItemHandle, int destinationPe);
  void broadcast(const void *dataItemHandle, int dimension, 
                 bool copyIndirectly);

public:

  MeshStreamer(int maxNumDataItemsBuffered, int numDimensions, 
               int *dimensionSizes, int bufferSize,
               bool yieldFlag = 0, double progressPeriodInMs = -1.0);
  ~MeshStreamer();

  // entry
  void init(int numLocalContributors, CkCallback startCb, CkCallback endCb, 
            int prio, bool usePeriodicFlushing);
  void init(int numContributors, CkCallback startCb, CkCallback endCb, 
            CProxy_CompletionDetector detector,
            int prio, bool usePeriodicFlushing);
  void init(CkArrayID senderArrayID, CkCallback startCb, CkCallback endCb, 
            int prio, bool usePeriodicFlushing);

  void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
  virtual void receiveAtDestination(MeshStreamerMessage<dtype> *msg) = 0;
  void flushIfIdle();
  void finish();

  // non entry
  bool isPeriodicFlushEnabled() {
    return isPeriodicFlushEnabled_;
  }
  virtual void insertData(const dtype& dataItem, int destinationPe); 
  virtual void broadcast(const dtype& dataItem); 
  void registerPeriodicProgressFunction();

  // flushing begins only after enablePeriodicFlushing has been invoked

  void enablePeriodicFlushing(){
    isPeriodicFlushEnabled_ = true; 
    registerPeriodicProgressFunction();
  }

  void done(int numContributorsFinished = 1) {

    if (useStagedCompletion_) {
      numLocalDone_ += numContributorsFinished; 
      if (numLocalDone_ == numLocalContributors_) {
        startStagedCompletion();
      }
    }
    else {
      detectorLocalObj_->done(numContributorsFinished);
    }
  }
  
  bool stagedCompletionStarted() {    
    return (useStagedCompletion_ && dimensionToFlush_ != numDimensions_ - 1); 
  }

  void startStagedCompletion() {          
    if (individualDimensionSizes_[dimensionToFlush_] != 1) {
      flushDimension(dimensionToFlush_, true);
    }
    dimensionToFlush_--;

    checkForCompletedStages();
  }

  void markMessageReceived(int dimension, int finalCount) {
    cntMsgReceived_[dimension]++;
    if (finalCount != -1) {
      cntFinished_[dimension]++; 
      cntMsgExpected_[dimension] += finalCount; 
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] received dimension: %d finalCount: %d cntFinished: %d "
               "cntMsgExpected: %d cntMsgReceived: %d\n", CkMyPe(), dimension, 
               finalCount, cntFinished_[dimension], cntMsgExpected_[dimension],
               cntMsgReceived_[dimension]); 
#endif
    }  
    if (dimensionToFlush_ != numDimensions_ - 1) {
      checkForCompletedStages();
    }
  }

  void checkForCompletedStages() {

    while (cntFinished_[dimensionToFlush_ + 1] == 
           individualDimensionSizes_[dimensionToFlush_ + 1] - 1 &&
           cntMsgExpected_[dimensionToFlush_ + 1] == 
           cntMsgReceived_[dimensionToFlush_ + 1]) {
      if (dimensionToFlush_ == -1) {
#ifdef STREAMER_VERBOSE_OUTPUT
        CkPrintf("[%d] contribute\n", CkMyPe()); 
#endif
        CkAssert(numDataItemsBuffered_ == 0); 
        isPeriodicFlushEnabled_ = false; 
        if (!userCallback_.isInvalid()) {
          this->contribute(userCallback_);
          userCallback_ = CkCallback();
        }
        return; 
      }
      else if (individualDimensionSizes_[dimensionToFlush_] != 1) {
        flushDimension(dimensionToFlush_, true); 
      }
      dimensionToFlush_--; 
    }
  }

};

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(
		     int maxNumDataItemsBuffered, int numDimensions, 
		     int *dimensionSizes, 
                     int bufferSize,
		     bool yieldFlag, 
                     double progressPeriodInMs)
 :numDimensions_(numDimensions), 
  maxNumDataItemsBuffered_(maxNumDataItemsBuffered), 
  yieldFlag_(yieldFlag), 
  progressPeriodInMs_(progressPeriodInMs), 
  bufferSize_(bufferSize)
{

  int sumAlongAllDimensions = 0;   
  individualDimensionSizes_ = new int[numDimensions_];
  combinedDimensionSizes_ = new int[numDimensions_ + 1];
  myLocationIndex_ = new int[numDimensions_];
  startingIndexAtDimension_ = new int[numDimensions_ + 1]; 
  memcpy(individualDimensionSizes_, dimensionSizes, 
	 numDimensions * sizeof(int)); 
  combinedDimensionSizes_[0] = 1; 
  for (int i = 0; i < numDimensions; i++) {
    sumAlongAllDimensions += individualDimensionSizes_[i];
    combinedDimensionSizes_[i + 1] = 
      combinedDimensionSizes_[i] * individualDimensionSizes_[i];
  }

  CkAssert(combinedDimensionSizes_[numDimensions] == CkNumPes()); 

  // a bufferSize input of 0 indicates it should be calculated by the library
  if (bufferSize_ == 0) {
    CkAssert(maxNumDataItemsBuffered_ > 0);
    // except for personalized messages, the buffers for dimensions with the 
    //   same index as the sender's are not used
    bufferSize_ = OVERALLOCATION_FACTOR * maxNumDataItemsBuffered_ 
      / (sumAlongAllDimensions - numDimensions_ + 1); 
  }
  else {
    maxNumDataItemsBuffered_ = 
      bufferSize_ * (sumAlongAllDimensions - numDimensions_ + 1);
  }

  if (bufferSize_ <= 0) {
    bufferSize_ = 1; 
    CkPrintf("Argument maxNumDataItemsBuffered to MeshStreamer constructor "
	     "is invalid. Defaulting to a single buffer per destination.\n");
  }
  numDataItemsBuffered_ = 0; 
  numMembers_ = CkNumPes(); 

  dataBuffers_ = new MeshStreamerMessage<dtype> **[numDimensions_]; 
  for (int i = 0; i < numDimensions; i++) {
    int numMembersAlongDimension = individualDimensionSizes_[i]; 
    dataBuffers_[i] = 
      new MeshStreamerMessage<dtype> *[numMembersAlongDimension];
    for (int j = 0; j < numMembersAlongDimension; j++) {
      dataBuffers_[i][j] = NULL;
    }
  }

  myIndex_ = CkMyPe();
  int remainder = myIndex_;
  startingIndexAtDimension_[numDimensions_] = 0;
  for (int i = numDimensions_ - 1; i >= 0; i--) {    
    myLocationIndex_[i] = remainder / combinedDimensionSizes_[i];
    int dimensionOffset = combinedDimensionSizes_[i] * myLocationIndex_[i];
    remainder -= dimensionOffset; 
    startingIndexAtDimension_[i] = 
      startingIndexAtDimension_[i+1] + dimensionOffset; 
  }

  isPeriodicFlushEnabled_ = false; 
  detectorLocalObj_ = NULL;

#ifdef CACHE_LOCATIONS
  cachedLocations_ = new MeshLocation[numMembers_];
  isCached_ = new bool[numMembers_];
  std::fill(isCached_, isCached_ + numMembers_, false);
#endif

  cntMsgSent_ = NULL; 
  cntMsgReceived_ = NULL; 
  cntMsgExpected_ = NULL; 
  cntFinished_ = NULL; 

}

template <class dtype>
MeshStreamer<dtype>::~MeshStreamer() {

  for (int i = 0; i < numDimensions_; i++) {
    for (int j=0; j < individualDimensionSizes_[i]; j++) {
      delete[] dataBuffers_[i][j]; 
    }
    delete[] dataBuffers_[i]; 
  }

  delete[] individualDimensionSizes_;
  delete[] combinedDimensionSizes_; 
  delete[] myLocationIndex_;
  delete[] startingIndexAtDimension_;

#ifdef CACHE_LOCATIONS
  delete[] cachedLocations_;
  delete[] isCached_; 
#endif

  if (cntMsgSent_ != NULL) {
    for (int i = 0; i < numDimensions_; i++) {
      delete[] cntMsgSent_[i]; 
    }
    delete[] cntMsgSent_; 
    delete[] cntMsgReceived_; 
    delete[] cntMsgExpected_; 
    delete[] cntFinished_;
  }

}

template <class dtype>
inline
MeshLocation MeshStreamer<dtype>::determineLocation(
                                  int destinationPe, 
                                  int dimensionReceivedAlong) {

#ifdef CACHE_LOCATIONS
  if (isCached_[destinationPe]) {    
    return cachedLocations_[destinationPe]; 
  }
#endif

  MeshLocation destinationLocation;
  int remainder = 
    destinationPe - startingIndexAtDimension_[dimensionReceivedAlong];
  int dimensionIndex; 
  for (int i = dimensionReceivedAlong - 1; i >= 0; i--) {        
    dimensionIndex = remainder / combinedDimensionSizes_[i];
    
    if (dimensionIndex != myLocationIndex_[i]) {
      destinationLocation.dimension = i; 
      destinationLocation.bufferIndex = dimensionIndex; 
#ifdef CACHE_LOCATIONS
      cachedLocations_[destinationPe] = destinationLocation;
      isCached_[destinationPe] = true; 
#endif
      return destinationLocation;
    }

    remainder -= combinedDimensionSizes_[i] * dimensionIndex;
  }

  CkAbort("Error. MeshStreamer::determineLocation called with destinationPe "
          "equal to sender's PE. This is unexpected and may cause errors.\n"); 
  // to prevent warnings
  return destinationLocation; 
}

template <class dtype>
inline 
int MeshStreamer<dtype>::copyDataItemIntoMessage(
			 MeshStreamerMessage<dtype> *destinationBuffer,
			 const void *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->addDataItem(*((const dtype *)dataItemHandle)); 
}

template <class dtype>
inline
void MeshStreamer<dtype>::storeMessage(
			  int destinationPe, 
			  const MeshLocation& destinationLocation,
			  const void *dataItem, bool copyIndirectly) {

  int dimension = destinationLocation.dimension;
  int bufferIndex = destinationLocation.bufferIndex; 
  MeshStreamerMessage<dtype> ** messageBuffers = dataBuffers_[dimension];   

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    if (dimension == 0) {
      // personalized messages do not require destination indices
      messageBuffers[bufferIndex] = 
        new (0, bufferSize_, sizeof(int)) MeshStreamerMessage<dtype>(dimension);
    }
    else {
      messageBuffers[bufferIndex] = 
        new (bufferSize_, bufferSize_, sizeof(int)) 
        MeshStreamerMessage<dtype>(dimension);
    }
    *(int *) CkPriorityPtr(messageBuffers[bufferIndex]) = prio_;
    CkSetQueueing(messageBuffers[bufferIndex], CK_QUEUEING_IFIFO);
    CkAssert(messageBuffers[bufferIndex] != NULL);
  }
  
  MeshStreamerMessage<dtype> *destinationBuffer = messageBuffers[bufferIndex];
  int numBuffered = 
    copyDataItemIntoMessage(destinationBuffer, dataItem, copyIndirectly);
  if (dimension != 0) {
    destinationBuffer->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;

  // send if buffer is full
  if (numBuffered == bufferSize_) {

    int destinationIndex;

    destinationIndex = myIndex_ + 
      (bufferIndex - myLocationIndex_[dimension]) * 
      combinedDimensionSizes_[dimension];

    if (dimension == 0) {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending to %d\n", CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAtDestination(destinationBuffer);
    }
    else {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending intermediate to %d\n", 
               CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }

    if (useStagedCompletion_) {
      cntMsgSent_[dimension][bufferIndex]++; 
    }

    messageBuffers[bufferIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered; 
    hasSentRecently_ = true; 

  }
  // send if total buffering capacity has been reached
  else if (numDataItemsBuffered_ == maxNumDataItemsBuffered_) {
    sendLargestBuffer();
    hasSentRecently_ = true; 
  }

}
template <class dtype>
inline
void MeshStreamer<dtype>::broadcast(const dtype& dataItem) {
  const static bool copyIndirectly = true;

  // no data items should be submitted after all local contributors call done 
  // and staged completion has begun
  CkAssert(stagedCompletionStarted() == false);

  // produce and consume once per PE
  if (!useStagedCompletion_) {
    detectorLocalObj_->produce(CkNumPes());
  }

  // deliver locally
  localBroadcast(dataItem);

  broadcast(&dataItem, numDimensions_ - 1, copyIndirectly); 
}

template <class dtype>
inline
void MeshStreamer<dtype>::broadcast(const void *dataItemHandle, int dimension, 
                                    bool copyIndirectly) {

  MeshLocation destinationLocation;
  destinationLocation.dimension = dimension; 

  while (destinationLocation.dimension != -1) {
    for (int i = 0; 
         i < individualDimensionSizes_[destinationLocation.dimension]; 
         i++) {

      if (i != myLocationIndex_[destinationLocation.dimension]) {
        destinationLocation.bufferIndex = i; 
        storeMessage(TRAM_BROADCAST, destinationLocation, 
                     dataItemHandle, copyIndirectly); 
      }
      // release control to scheduler if requested by the user, 
      //   assume caller is threaded entry
      if (yieldFlag_ && ++yieldCount_ == 1024) {
        yieldCount_ = 0; 
        CthYield();
      }
    }
    destinationLocation.dimension--; 
  }

}


template <class dtype>
inline
void MeshStreamer<dtype>::insertData(const void *dataItemHandle, 
                                     int destinationPe) {
  const static bool copyIndirectly = true;

  // treat newly inserted items as if they were received along
  // a higher dimension (e.g. for a 3D mesh, received along 4th dimension)
  MeshLocation destinationLocation = determineLocation(destinationPe, 
                                                       numDimensions_);
  storeMessage(destinationPe, destinationLocation, dataItemHandle, 
	       copyIndirectly); 
  // release control to scheduler if requested by the user, 
  //   assume caller is threaded entry
  if (yieldFlag_ && ++yieldCount_ == 1024) {
    yieldCount_ = 0; 
    CthYield();
  }

}

template <class dtype>
inline
void MeshStreamer<dtype>::insertData(const dtype& dataItem, int destinationPe) {

  // no data items should be submitted after all local contributors call done 
  // and staged completion has begun
  CkAssert(stagedCompletionStarted() == false);

  if (!useStagedCompletion_) {
    detectorLocalObj_->produce();
  }
  if (destinationPe == CkMyPe()) {
    localDeliver(dataItem);
    return;
  }

  insertData((const void *) &dataItem, destinationPe);
}

template <class dtype>
void MeshStreamer<dtype>::init(int numLocalContributors, CkCallback startCb, 
                               CkCallback endCb, int prio, 
                               bool usePeriodicFlushing) {
  useStagedCompletion_ = true; 
  // allocate memory on first use
  if (cntMsgSent_ == NULL) {
    cntMsgSent_ = new int*[numDimensions_]; 
    cntMsgReceived_ = new int[numDimensions_];
    cntMsgExpected_ = new int[numDimensions_];
    cntFinished_ = new int[numDimensions_]; 

    for (int i = 0; i < numDimensions_; i++) {
      cntMsgSent_[i] = new int[individualDimensionSizes_[i]]; 
    }
  }


  for (int i = 0; i < numDimensions_; i++) {
    std::fill(cntMsgSent_[i], 
              cntMsgSent_[i] + individualDimensionSizes_[i], 0);
    cntMsgReceived_[i] = 0;
    cntMsgExpected_[i] = 0; 
    cntFinished_[i] = 0; 
  }
  dimensionToFlush_ = numDimensions_ - 1;

  yieldCount_ = 0; 
  userCallback_ = endCb; 
  prio_ = prio;

  numLocalDone_ = 0; 
  numLocalContributors_ = numLocalContributors; 
  initLocalClients();

  if (numLocalContributors_ == 0) {
    startStagedCompletion();
  }

  hasSentRecently_ = false;
  if (usePeriodicFlushing) {
    enablePeriodicFlushing();
  }

  this->contribute(startCb);
}

template <class dtype>
void MeshStreamer<dtype>::init(int numContributors,	  
                               CkCallback startCb, CkCallback endCb,
                               CProxy_CompletionDetector detector, 
                               int prio, bool usePeriodicFlushing) {

  useStagedCompletion_ = false; 
  yieldCount_ = 0; 
  prio_ = prio;
  userCallback_ = endCb; 
  CkCallback flushCb(CkIndex_MeshStreamer<dtype>::flushIfIdle(), 
                     this->thisProxy);
  CkCallback finish(CkIndex_MeshStreamer<dtype>::finish(), 
		    this->thisProxy);
  detector_ = detector;      
  detectorLocalObj_ = detector_.ckLocalBranch();
  initLocalClients();

  detectorLocalObj_->start_detection(numContributors, startCb, flushCb, 
                                     finish , 0);
  
  if (progressPeriodInMs_ <= 0) {
    CkPrintf("Using completion detection in NDMeshStreamer requires"
	     " setting a valid periodic flush period. Defaulting"
	     " to 10 ms\n");
    progressPeriodInMs_ = 10;
  }
  
  hasSentRecently_ = false; 
  if (usePeriodicFlushing) {
    enablePeriodicFlushing();
  }

}

template <class dtype>
void MeshStreamer<dtype>::init(CkArrayID senderArrayID, CkCallback startCb, 
            CkCallback endCb, int prio, bool usePeriodicFlushing) {
    CkArray *senderArrayMgr = senderArrayID.ckLocalBranch();
    int numLocalElements = senderArrayMgr->getLocMgr()->numLocalElements();
    init(numLocalElements, startCb, endCb, prio, usePeriodicFlushing); 
  }


template <class dtype>
void MeshStreamer<dtype>::finish() {
  isPeriodicFlushEnabled_ = false; 

  if (!userCallback_.isInvalid()) {
    this->contribute(userCallback_);
    userCallback_ = CkCallback();      // nullify the current callback
  }

}

template <class dtype>
void MeshStreamer<dtype>::receiveAlongRoute(MeshStreamerMessage<dtype> *msg) {

  int destinationPe, lastDestinationPe; 
  MeshLocation destinationLocation;

  lastDestinationPe = -1;
  for (int i = 0; i < msg->numDataItems; i++) {
    destinationPe = msg->destinationPes[i];
    const dtype& dataItem = msg->getDataItem(i);
    if (destinationPe == CkMyPe()) {
      localDeliver(dataItem);
    }
    else if (destinationPe != TRAM_BROADCAST) {
      if (destinationPe != lastDestinationPe) {
        // do this once per sequence of items with the same destination
        destinationLocation = determineLocation(destinationPe, msg->dimension);
      }
      storeMessage(destinationPe, destinationLocation, &dataItem);   
    }
    else /* if (destinationPe == TRAM_BROADCAST) */ {
      localBroadcast(dataItem);       
      broadcast(&dataItem, msg->dimension - 1, false); 
    }
    lastDestinationPe = destinationPe; 
  }

  if (useStagedCompletion_) {
    markMessageReceived(msg->dimension, msg->finalMsgCount); 
  }

  delete msg;

}

template <class dtype>
void MeshStreamer<dtype>::sendLargestBuffer() {

  int flushDimension, flushIndex, maxSize, destinationIndex, numBuffers;
  MeshStreamerMessage<dtype> ** messageBuffers; 
  MeshStreamerMessage<dtype> *destinationBuffer; 

  for (int i = 0; i < numDimensions_; i++) {

    messageBuffers = dataBuffers_[i]; 
    numBuffers = individualDimensionSizes_[i]; 

    flushDimension = i; 
    maxSize = 0;    
    for (int j = 0; j < numBuffers; j++) {
      if (messageBuffers[j] != NULL && 
	  messageBuffers[j]->numDataItems > maxSize) {
	maxSize = messageBuffers[j]->numDataItems;
	flushIndex = j; 
      }
    }

    if (maxSize > 0) {

      messageBuffers = dataBuffers_[flushDimension]; 
      destinationBuffer = messageBuffers[flushIndex];
      destinationIndex = myIndex_ + 
	(flushIndex - myLocationIndex_[flushDimension]) * 
	combinedDimensionSizes_[flushDimension] ;

      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBuffer);
      env->setTotalsize(env->getTotalsize() - sizeof(dtype) *
                        (bufferSize_ - destinationBuffer->numDataItems));
      *((int *) env->getPrioPtr()) = prio_;

      numDataItemsBuffered_ -= destinationBuffer->numDataItems;

      if (flushDimension == 0) {
#ifdef STREAMER_VERBOSE_OUTPUT
        CkPrintf("[%d] sending flush to %d\n", CkMyPe(), destinationIndex); 
#endif
        this->thisProxy[destinationIndex].
          receiveAtDestination(destinationBuffer);
      }
      else {
#ifdef STREAMER_VERBOSE_OUTPUT
        CkPrintf("[%d] sending intermediate flush to %d\n", 
                 CkMyPe(), destinationIndex); 
#endif
	this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
      }

      if (useStagedCompletion_) {
        cntMsgSent_[i][flushIndex]++; 
      }

      messageBuffers[flushIndex] = NULL;

    }

  }
}

template <class dtype>
void MeshStreamer<dtype>::flushToIntermediateDestinations() {
  
  for (int i = 0; i < numDimensions_; i++) {
    flushDimension(i); 
  }
}

template <class dtype>
void MeshStreamer<dtype>::flushDimension(int dimension, bool sendMsgCounts) {
#ifdef STREAMER_VERBOSE_OUTPUT
  CkPrintf("[%d] flushDimension: %d, sendMsgCounts: %d\n", 
           CkMyPe(), dimension, sendMsgCounts); 
#endif
  MeshStreamerMessage<dtype> **messageBuffers; 
  MeshStreamerMessage<dtype> *destinationBuffer; 
  int destinationIndex, numBuffers; 

  messageBuffers = dataBuffers_[dimension]; 
  numBuffers = individualDimensionSizes_[dimension]; 

  for (int j = 0; j < numBuffers; j++) {

    if(messageBuffers[j] == NULL) {      
      if (sendMsgCounts && j != myLocationIndex_[dimension]) {
        messageBuffers[j] = 
          new (0, 0, sizeof(int)) MeshStreamerMessage<dtype>(dimension);
        *(int *) CkPriorityPtr(messageBuffers[j]) = prio_;
        CkSetQueueing(messageBuffers[j], CK_QUEUEING_IFIFO);
      }
      else {
        continue; 
      } 
    }

    destinationBuffer = messageBuffers[j];
    destinationIndex = myIndex_ + 
      (j - myLocationIndex_[dimension]) * 
      combinedDimensionSizes_[dimension] ;

    if (destinationBuffer->numDataItems != 0) {
      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBuffer);
      env->setTotalsize(env->getTotalsize() - sizeof(dtype) *
                        (bufferSize_ - destinationBuffer->numDataItems));
      *((int *) env->getPrioPtr()) = prio_;
    }
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;

    if (useStagedCompletion_) {
      cntMsgSent_[dimension][j]++;
      if (sendMsgCounts) {
        destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
      }
    }

    if (dimension == 0) {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending dimension flush to %d\n", 
               CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAtDestination(destinationBuffer);
    }
    else {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending intermediate dimension flush to %d\n", 
               CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }
    messageBuffers[j] = NULL;
  }
  
}


template <class dtype>
void MeshStreamer<dtype>::flushIfIdle(){
  // flush if (1) this is not a periodic call or 
  //          (2) this is a periodic call and no sending took place
  //              since the last time the function was invoked
  if (!isPeriodicFlushEnabled_ || !hasSentRecently_) {

    if (numDataItemsBuffered_ != 0) {
      flushToIntermediateDestinations();
    }    
    CkAssert(numDataItemsBuffered_ == 0); 
    
  }

  hasSentRecently_ = false; 

}

template <class dtype>
void periodicProgressFunction(void *MeshStreamerObj, double time) {

  MeshStreamer<dtype> *properObj = 
    static_cast<MeshStreamer<dtype>*>(MeshStreamerObj); 

  if (properObj->isPeriodicFlushEnabled()) {
    properObj->flushIfIdle();
    properObj->registerPeriodicProgressFunction();
  }
}

template <class dtype>
void MeshStreamer<dtype>::registerPeriodicProgressFunction() {
  CcdCallFnAfter(periodicProgressFunction<dtype>, (void *) this, 
		 progressPeriodInMs_); 
}


template <class dtype>
class GroupMeshStreamer : public MeshStreamer<dtype> {
private:

  CProxy_MeshStreamerGroupClient<dtype> clientProxy_;
  MeshStreamerGroupClient<dtype> *clientObj_;

  void receiveAtDestination(MeshStreamerMessage<dtype> *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      const dtype& data = msg->getDataItem(i);
      clientObj_->process(data);
    }

    if (MeshStreamer<dtype>::useStagedCompletion_) {
#ifdef STREAMER_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n", 
               CkMyPe(), env->getSrcPe(), msg->numDataItems, 
               msg->finalMsgCount);  
#endif
      markMessageReceived(msg->dimension, msg->finalMsgCount); 
    }
    else {
      this->detectorLocalObj_->consume(msg->numDataItems);    
    }

    delete msg;
  }

  void localDeliver(const dtype& dataItem) {
    clientObj_->process(dataItem);
    if (MeshStreamer<dtype>::useStagedCompletion_ == false) {
      MeshStreamer<dtype>::detectorLocalObj_->consume();
    }
  }

  void localBroadcast(const dtype& dataItem) {
    localDeliver(dataItem); 
  }

  int numElementsInClient() {
    // client is a group - there is one element per PE
    return CkNumPes();
  }

  int numLocalElementsInClient() {
    return 1; 
  }

  void initLocalClients() {
    // no action required
  }

public:

  GroupMeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
		    int *dimensionSizes, 
		    const CProxy_MeshStreamerGroupClient<dtype>& clientProxy,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0)
   :MeshStreamer<dtype>(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
                        0, yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientObj_ = 
      ((MeshStreamerGroupClient<dtype> *)CkLocalBranch(clientProxy_));
  }

  GroupMeshStreamer(int numDimensions, int *dimensionSizes, 
		    const CProxy_MeshStreamerGroupClient<dtype>& clientProxy,
		    int bufferSize, bool yieldFlag = 0, 
                    double progressPeriodInMs = -1.0)
   :MeshStreamer<dtype>(0, numDimensions, dimensionSizes, bufferSize, 
                        yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientObj_ = 
      ((MeshStreamerGroupClient<dtype> *)CkLocalBranch(clientProxy_));
  }

};

template <class dtype>
class ClientInitializer : public CkLocIterator {

public:
  
  CompletionDetector *detectorLocalObj_;
  CkArray *clientArrMgr_;
  ClientInitializer(CompletionDetector *detectorObj, 
			     CkArray *clientArrMgr) 
    : detectorLocalObj_(detectorObj), clientArrMgr_(clientArrMgr) {}

  // CkLocMgr::iterate will call addLocation on all elements local to this PE
  void addLocation(CkLocation& loc) {

    MeshStreamerArrayClient<dtype> *clientObj = 
      (MeshStreamerArrayClient<dtype> *) clientArrMgr_->lookup(loc.getIndex());

    CkAssert(clientObj != NULL); 
    clientObj->setDetector(detectorLocalObj_); 
  }

};

template <class dtype>
class LocalBroadcaster : public CkLocIterator {

public:
  CkArray *clientArrMgr_;
  const dtype *dataItem_; 

  LocalBroadcaster(CkArray *clientArrMgr, const dtype *dataItem) 
   : clientArrMgr_(clientArrMgr), dataItem_(dataItem) {}

  void addLocation(CkLocation& loc) {
    MeshStreamerArrayClient<dtype> *clientObj = 
      (MeshStreamerArrayClient<dtype> *) clientArrMgr_->lookup(loc.getIndex());

    CkAssert(clientObj != NULL); 

    clientObj->process(*dataItem_); 
  }

};

template <class dtype, class itype>
class ArrayMeshStreamer : public MeshStreamer<ArrayDataItem<dtype, itype> > {
  
private:
  
  CProxy_MeshStreamerArrayClient<dtype> clientProxy_;
  CkArray *clientArrayMgr_;
  int numArrayElements_;
  int numLocalArrayElements_;
#ifdef CACHE_ARRAY_METADATA
  MeshStreamerArrayClient<dtype> **clientObjs_;
  int *destinationPes_;
  bool *isCachedArrayMetadata_;
#endif

  void localDeliver(const ArrayDataItem<dtype, itype>& packedDataItem) {
    itype arrayId = packedDataItem.arrayIndex; 
    if (arrayId == itype(TRAM_BROADCAST)) {
      localBroadcast(packedDataItem);
      return;
    }
    MeshStreamerArrayClient<dtype> *clientObj;
#ifdef CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = clientProxy_[arrayId].ckLocal();
#endif

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
      if (MeshStreamer<ArrayDataItem<dtype, itype> >
           ::useStagedCompletion_ == false) {
        MeshStreamer<ArrayDataItem<dtype, itype> >
         ::detectorLocalObj_->consume();
      }
    }
    else { 
      // array element is no longer present locally - redeliver using proxy
      clientProxy_[arrayId].receiveRedeliveredItem(packedDataItem.dataItem);
    }
  }

  void localBroadcast(const ArrayDataItem<dtype, itype>& packedDataItem) {

    LocalBroadcaster<dtype> clientIterator(clientProxy_.ckLocalBranch(), 
                                           &packedDataItem.dataItem);
    CkLocMgr *clientLocMgr = clientProxy_.ckLocMgr(); 
    clientLocMgr->iterate(clientIterator);

    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useStagedCompletion_ == false) {
        MeshStreamer<ArrayDataItem<dtype, itype> >
         ::detectorLocalObj_->consume();      
    }

  }

  int numElementsInClient() {
    return numArrayElements_;
  }

  int numLocalElementsInClient() {
    return numLocalArrayElements_;
  }

  void initLocalClients() {
    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useStagedCompletion_ == false) {
#ifdef CACHE_ARRAY_METADATA
      std::fill(isCachedArrayMetadata_, 
                isCachedArrayMetadata_ + numArrayElements_, false);

      for (int i = 0; i < numArrayElements_; i++) {
        clientObjs_[i] = clientProxy_[i].ckLocal();
        if (clientObjs_[i] != NULL) {
          clientObjs_[i]->setDetector(
           MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_);
        }
      }
#else
      // set completion detector in local elements of the client
      CkLocMgr *clientLocMgr = clientProxy_.ckLocMgr(); 
      ClientInitializer<dtype> clientIterator(
          MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_, 
          clientProxy_.ckLocalBranch());
      clientLocMgr->iterate(clientIterator);
#endif    
    }
    else {
      numLocalArrayElements_ = clientProxy_.ckLocMgr()->numLocalElements();
    }
  }

  void commonInit() {
#ifdef CACHE_ARRAY_METADATA
    numArrayElements_ = (clientArrayMgr_->getNumInitial()).data()[0];
    clientObjs_ = new MeshStreamerArrayClient<dtype>*[numArrayElements_];
    destinationPes_ = new int[numArrayElements_];
    isCachedArrayMetadata_ = new bool[numArrayElements_];
    std::fill(isCachedArrayMetadata_, 
	      isCachedArrayMetadata_ + numArrayElements_, false);
#endif    
  }

public:

  struct DataItemHandle {
    itype arrayIndex; 
    const dtype *dataItem;
  };

  ArrayMeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
		    int *dimensionSizes, 
                    const CProxy_MeshStreamerArrayClient<dtype>& clientProxy,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0)
    :MeshStreamer<ArrayDataItem<dtype, itype> >(
                  maxNumDataItemsBuffered, numDimensions, dimensionSizes, 
                  0, yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientArrayMgr_ = clientProxy_.ckLocalBranch();
    commonInit();
  }

  ArrayMeshStreamer(int numDimensions, int *dimensionSizes, 
		    const CProxy_MeshStreamerArrayClient<dtype>& clientProxy,
		    int bufferSize, bool yieldFlag = 0, 
                    double progressPeriodInMs = -1.0)
    :MeshStreamer<ArrayDataItem<dtype,itype> >(
                  0, numDimensions, dimensionSizes, 
                  bufferSize, yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientArrayMgr_ = clientProxy_.ckLocalBranch();
    commonInit();

  }

  ~ArrayMeshStreamer() {
#ifdef CACHE_ARRAY_METADATA
    delete [] clientObjs_;
    delete [] destinationPes_;
    delete [] isCachedArrayMetadata_; 
#endif
  }

  void receiveAtDestination(
       MeshStreamerMessage<ArrayDataItem<dtype, itype> > *msg) {

    for (int i = 0; i < msg->numDataItems; i++) {
      const ArrayDataItem<dtype, itype>& packedData = msg->getDataItem(i);
      localDeliver(packedData);
    }
    if (MeshStreamer<ArrayDataItem<dtype, itype> >::useStagedCompletion_) {
      markMessageReceived(msg->dimension, msg->finalMsgCount);
    }

    delete msg;
  }

  inline void broadcast(const dtype& dataItem) {
    const static bool copyIndirectly = true;

    // no data items should be submitted after all local contributors call done
    // and staged completion has begun
    CkAssert((MeshStreamer<ArrayDataItem<dtype, itype> >
               ::stagedCompletionStarted()) == false);

    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useStagedCompletion_ == false) {
      MeshStreamer<ArrayDataItem<dtype, itype> >
        ::detectorLocalObj_->produce(CkNumPes());
    }

    // deliver locally
    ArrayDataItem<dtype, itype>& packedDataItem(TRAM_BROADCAST, dataItem);
    localBroadcast(packedDataItem);

    DataItemHandle tempHandle; 
    tempHandle.dataItem = &dataItem;
    tempHandle.arrayIndex = TRAM_BROADCAST;

    int numDimensions = 
      MeshStreamer<ArrayDataItem<dtype, itype> >::numDimensions_;
    MeshStreamer<ArrayDataItem<dtype, itype> >::
      broadcast(&tempHandle, numDimensions - 1, copyIndirectly);
  }

  inline void insertData(const dtype& dataItem, itype arrayIndex) {

    // no data items should be submitted after all local contributors call done
    // and staged completion has begun
    CkAssert((MeshStreamer<ArrayDataItem<dtype, itype> >
               ::stagedCompletionStarted()) == false);

    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useStagedCompletion_ == false) {
      MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->produce();
    }
    int destinationPe; 
#ifdef CACHE_ARRAY_METADATA
  if (isCachedArrayMetadata_[arrayIndex]) {    
    destinationPe =  destinationPes_[arrayIndex];
  }
  else {
    destinationPe = 
      clientArrayMgr_->lastKnown(clientProxy_[arrayIndex].ckGetIndex());
    isCachedArrayMetadata_[arrayIndex] = true;
    destinationPes_[arrayIndex] = destinationPe;
  }
#else 
  destinationPe = 
    clientArrayMgr_->lastKnown(clientProxy_[arrayIndex].ckGetIndex());
#endif

    if (destinationPe == CkMyPe()) {
      ArrayDataItem<dtype, itype> packedDataItem(arrayIndex, dataItem);
      localDeliver(packedDataItem);
      return;
    }

    // this implementation avoids copying an item before transfer into message
    DataItemHandle tempHandle; 
    tempHandle.arrayIndex = arrayIndex; 
    tempHandle.dataItem = &dataItem;

    MeshStreamer<ArrayDataItem<dtype, itype> >::
     insertData(&tempHandle, destinationPe);

  }

  inline int copyDataItemIntoMessage(
      MeshStreamerMessage<ArrayDataItem <dtype, itype> > *destinationBuffer, 
      const void *dataItemHandle, bool copyIndirectly) {

    if (copyIndirectly == true) {
      // newly inserted items are passed through a handle to avoid copying
      int numDataItems = destinationBuffer->numDataItems;
      const DataItemHandle *tempHandle = 
        (const DataItemHandle *) dataItemHandle;
      (destinationBuffer->dataItems)[numDataItems].dataItem = 
	*(tempHandle->dataItem);
      (destinationBuffer->dataItems)[numDataItems].arrayIndex = 
	tempHandle->arrayIndex;
      return ++destinationBuffer->numDataItems;
    }
    else {
      // this is an item received along the route to destination
      // we can copy it from the received message
      return MeshStreamer<ArrayDataItem<dtype, itype> >::
	      copyDataItemIntoMessage(destinationBuffer, dataItemHandle);
    }
  }

};

template <class dtype>
class GroupChunkMeshStreamer : 
  public GroupMeshStreamer<ChunkDataItem> {

private:
  ChunkDataItem **receiveBuffers;
  int *receivedChunks;

public:

  GroupChunkMeshStreamer(
       int maxNumDataItemsBuffered, int numDimensions,
       int *dimensionSizes, 
       const CProxy_MeshStreamerGroupClient<dtype>& clientProxy,
       bool yieldFlag = 0, double progressPeriodInMs = -1.0)
    :GroupMeshStreamer<ChunkDataItem>(maxNumDataItemsBuffered, 
                                      numDimensions, dimensionSizes,
                                      0, yieldFlag, progressPeriodInMs) {
    
    receiveBuffers = new ChunkDataItem*[CkNumPes()];
    receivedChunks = new int[CkNumPes()]; 
    memset(receivedChunks, 0, CkNumPes() * sizeof(int));
    memset(receiveBuffers, 0, CkNumPes() * sizeof(ChunkDataItem*)); 
  }

  GroupChunkMeshStreamer(
       int numDimensions, int *dimensionSizes, 
       const CProxy_MeshStreamerGroupClient<dtype>& clientProxy,
       int bufferSize, bool yieldFlag = 0, 
       double progressPeriodInMs = -1.0)
    :GroupMeshStreamer<ChunkDataItem>(0, numDimensions, dimensionSizes, 
                                      bufferSize, yieldFlag, 
                                      progressPeriodInMs) {}

  inline void insertData(dtype *dataArray, int numElements, int destinationPe) {

    char *inputData = (char *) dataArray; 
    int arraySizeInBytes = numElements * sizeof(dtype); 
    ChunkDataItem chunk;
    int chunkNumber = 0; 
    chunk.sourcePe = CkMyPe();
    chunk.chunkNumber = 0; 
    chunk.chunkSize = CHUNK_SIZE;
    chunk.numChunks = numElements * sizeof(dtype) / CHUNK_SIZE; 
    chunk.numItems = numElements; 
    for (int offset = 0; offset < arraySizeInBytes; offset += CHUNK_SIZE) {

      if (offset + CHUNK_SIZE > arraySizeInBytes) {
        chunk.chunkSize = arraySizeInBytes - offset; 
        memset(chunk.rawData, 0, CHUNK_SIZE);
        memcpy(chunk.rawData + offset, inputData + offset, chunk.chunkSize); 
      }
      else {
        memcpy(chunk.rawData, inputData + offset, CHUNK_SIZE);
      }

    }

    insertData(chunk, destinationPe); 

  }

  inline void processChunk(const ChunkDataItem& chunk) {

    if (receiveBuffers[chunk.sourcePe] == NULL) {
      receiveBuffers[chunk.sourcePe] = new dtype[chunk.numItems]; 
    }      

    char *receiveBuffer = &receiveBuffers[chunk.sourcePe];

    memcpy(receiveBuffer + chunk.chunkNumber * sizeof(dtype), 
           chunk.rawData, chunk.chunkSize);
    if (++receivedChunks[chunk.sourcePe] == chunk.numChunks) {
      clientObj_->receiveArray((dtype *) receiveBuffer, chunk.numItems, chunk.sourcePe);
      receivedChunks[chunk.sourcePe] = 0;        
      delete [] receiveBuffers[chunk.sourcePe]; 
      receiveBuffers[chunk.sourcePe] = NULL;
    }

  }

  inline void localDeliver(const ChunkDataItem& chunk) {
    processChunk(chunk);
    if (MeshStreamer<dtype>::useStagedCompletion_ == false) {
      MeshStreamer<dtype>::detectorLocalObj_->consume();
    }
  }

  inline void receiveAtDestination(
       MeshStreamerMessage<ChunkDataItem> *msg) {

    for (int i = 0; i < msg->numDataItems; i++) {
      const ChunkDataItem& chunk = msg->getDataItem(i);
      processChunk(chunk);             
    }

    if (MeshStreamer<dtype>::useStagedCompletion_) {
#ifdef STREAMER_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n", 
               CkMyPe(), env->getSrcPe(), msg->numDataItems, 
               msg->finalMsgCount);  
#endif
      markMessageReceived(msg->dimension, msg->finalMsgCount); 
    }
    else {
      this->detectorLocalObj_->consume(msg->numDataItems);    
    }

    delete msg;
    
  }

};



#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
