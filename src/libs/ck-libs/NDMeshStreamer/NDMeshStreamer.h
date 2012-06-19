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

// #define DEBUG_STREAMER
// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH
// #define CACHE_ARRAY_METADATA // only works for 1D array clients
// #define STREAMER_VERBOSE_OUTPUT
#define STAGED_COMPLETION

struct MeshLocation {
  int dimension; 
  int bufferIndex; 
}; 

template<class dtype>
class MeshStreamerMessage : public CMessage_MeshStreamerMessage<dtype> {

public:

#ifdef STAGED_COMPLETION
  int finalMsgCount; 
#endif
  int dimension; 
  int numDataItems;
  int *destinationPes;
  dtype *dataItems;

  MeshStreamerMessage(int dim): numDataItems(0), dimension(dim) {
#ifdef STAGED_COMPLETION
    finalMsgCount = -1; 
#endif
  }

  int addDataItem(const dtype &dataItem) {
    dataItems[numDataItems] = dataItem;
    return ++numDataItems; 
  }

  void markDestination(const int index, const int destinationPe) {
    destinationPes[index] = destinationPe;
  }

  dtype &getDataItem(const int index) {
    return dataItems[index];
  }

};

template <class dtype>
class MeshStreamerArrayClient : public CBase_MeshStreamerArrayClient<dtype>{
private:
  CompletionDetector *detectorLocalObj_;
public:
  MeshStreamerArrayClient(){}
  MeshStreamerArrayClient(CkMigrateMessage *msg) {}
  // would like to make it pure virtual but charm will try to
  // instantiate the abstract class, leading to errors
  virtual void process(dtype &data) {
    CkAbort("Error. MeshStreamerArrayClient::process() is being called. "
            "This virtual function should have been defined by the user.\n");
  };     
  void setDetector(CompletionDetector *detectorLocalObj) {
    detectorLocalObj_ = detectorLocalObj;
  }
  void receiveRedeliveredItem(dtype data) {
#ifdef STREAMER_VERBOSE_OUTPUT
    CkPrintf("[%d] redelivered to index %d\n", CkMyPe(), this->thisIndex.data[0]);
#endif
    detectorLocalObj_->consume();
    process(data);
  }

  void pup(PUP::er &p) {
    CBase_MeshStreamerArrayClient<dtype>::pup(p);
   }  

};

template <class dtype>
class MeshStreamerGroupClient : public CBase_MeshStreamerGroupClient<dtype>{

public:
  virtual void process(dtype &data) = 0;

};

template <class dtype>
class MeshStreamer : public CBase_MeshStreamer<dtype> {

private:
  int bufferSize_; 
  int maxNumDataItemsBuffered_;
  int numDataItemsBuffered_;

  int numMembers_; 
  int numDimensions_;
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
                    const MeshLocation &destinationCoordinates, 
                    void *dataItem, bool copyIndirectly = false);
  virtual void localDeliver(dtype &dataItem) = 0; 

  virtual int numElementsInClient() = 0;
  virtual int numLocalElementsInClient() = 0; 

  virtual void initLocalClients() = 0;

  void sendLargestBuffer();
  void flushToIntermediateDestinations();
  void flushDimension(int dimension, bool sendMsgCounts = false); 

protected:

  CompletionDetector *detectorLocalObj_;
  virtual int copyDataItemIntoMessage(
              MeshStreamerMessage<dtype> *destinationBuffer, 
              void *dataItemHandle, bool copyIndirectly = false);
  MeshLocation determineLocation(int destinationPe, 
                                 int dimensionReceivedAlong);
public:

  MeshStreamer(int maxNumDataItemsBuffered, int numDimensions, 
               int *dimensionSizes, int bufferSize,
               bool yieldFlag = 0, double progressPeriodInMs = -1.0);
  ~MeshStreamer();

  // entry
  void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
  virtual void receiveAtDestination(MeshStreamerMessage<dtype> *msg) = 0;
  void flushDirect();
  void finish();

  // non entry
  bool isPeriodicFlushEnabled() {
    return isPeriodicFlushEnabled_;
  }
  virtual void insertData(dtype &dataItem, int destinationPe); 
  void insertData(void *dataItemHandle, int destinationPe);
  void associateCallback(int numContributors, 
                         CkCallback startCb, CkCallback endCb, 
                         CProxy_CompletionDetector detector,
                         int prio);
  void flushAllBuffers();
  void registerPeriodicProgressFunction();

  // flushing begins only after enablePeriodicFlushing has been invoked

  void enablePeriodicFlushing(){
    isPeriodicFlushEnabled_ = true; 
    registerPeriodicProgressFunction();
  }

  void done(int numContributorsFinished = 1) {
#ifdef STAGED_COMPLETION
    numLocalDone_ += numContributorsFinished; 
    if (numLocalDone_ == numLocalContributors_) {
      startStagedCompletion();
    }
#else
    detectorLocalObj_->done(numContributorsFinished);
#endif
  }

  void init(int numLocalContributors, CkCallback startCb, CkCallback endCb, int prio);
  
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
#ifdef DEBUG_STREAMER
        CkAssert(numDataItemsBuffered_ == 0); 
#endif
        this->contribute(userCallback_);
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
    startingIndexAtDimension_[i] = startingIndexAtDimension_[i+1] + dimensionOffset; 
  }

  isPeriodicFlushEnabled_ = false; 
  detectorLocalObj_ = NULL;

#ifdef CACHE_LOCATIONS
  cachedLocations_ = new MeshLocation[numMembers_];
  isCached_ = new bool[numMembers_];
  std::fill(isCached_, isCached_ + numMembers_, false);
#endif

#ifdef STAGED_COMPLETION

  cntMsgSent_ = new int*[numDimensions_]; 
  cntMsgReceived_ = new int[numDimensions_];
  cntMsgExpected_ = new int[numDimensions_];
  cntFinished_ = new int[numDimensions_]; 

  for (int i = 0; i < numDimensions_; i++) {
    cntMsgSent_[i] = new int[individualDimensionSizes_[i]]; 
  }

#endif

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

#ifdef STAGED_COMPLETION
  for (int i = 0; i < numDimensions_; i++) {
    delete[] cntMsgSent_[i]; 
  }
  delete[] cntMsgSent_; 
  delete[] cntMsgReceived_; 
  delete[] cntMsgExpected_; 
  delete[] cntFinished_;
#endif

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
  int remainder = destinationPe - startingIndexAtDimension_[dimensionReceivedAlong];
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
			 void *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->addDataItem(*((dtype *)dataItemHandle)); 
}

template <class dtype>
inline
void MeshStreamer<dtype>::storeMessage(
			  int destinationPe, 
			  const MeshLocation& destinationLocation,
			  void *dataItem, bool copyIndirectly) {

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
#ifdef DEBUG_STREAMER
    CkAssert(messageBuffers[bufferIndex] != NULL);
#endif
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
      CkPrintf("[%d] sending intermediate to %d\n", CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }

#ifdef STAGED_COMPLETION
    cntMsgSent_[dimension][bufferIndex]++; 
#endif

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
void MeshStreamer<dtype>::insertData(void *dataItemHandle, int destinationPe) {
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
void MeshStreamer<dtype>::insertData(dtype &dataItem, int destinationPe) {
#ifndef STAGED_COMPLETION
  detectorLocalObj_->produce();
#endif
  if (destinationPe == CkMyPe()) {
    // copying here is necessary - user code should not be 
    // passed back a reference to the original item
    dtype dataItemCopy = dataItem;
    localDeliver(dataItemCopy);
    return;
  }

  insertData((void *) &dataItem, destinationPe);
}

template <class dtype>
void MeshStreamer<dtype>::init(int numLocalContributors, CkCallback startCb, 
                               CkCallback endCb, int prio) {

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
  this->contribute(startCb);
}

template <class dtype>
void MeshStreamer<dtype>::associateCallback(
			  int numContributors,
			  CkCallback startCb, CkCallback endCb, 
			  CProxy_CompletionDetector detector, 
			  int prio) {

  yieldCount_ = 0; 
  prio_ = prio;
  userCallback_ = endCb; 
  CkCallback flushCb(CkIndex_MeshStreamer<dtype>::flushDirect(), 
                     this->thisProxy);
  CkCallback finish(CkIndex_MeshStreamer<dtype>::finish(), 
		    this->thisProxy);
  detector_ = detector;      
  detectorLocalObj_ = detector_.ckLocalBranch();
  initLocalClients();

  detectorLocalObj_->start_detection(numContributors, startCb, flushCb, finish , 0);
  
  if (progressPeriodInMs_ <= 0) {
    CkPrintf("Using completion detection in NDMeshStreamer requires"
	     " setting a valid periodic flush period. Defaulting"
	     " to 10 ms\n");
    progressPeriodInMs_ = 10;
  }
  
  hasSentRecently_ = false; 
  enablePeriodicFlushing();
      
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
    dtype &dataItem = msg->getDataItem(i);
    if (destinationPe == CkMyPe()) {
      localDeliver(dataItem);
    }
    else {
      if (destinationPe != lastDestinationPe) {
        // do this once per sequence of items with the same destination
        destinationLocation = determineLocation(destinationPe, msg->dimension);
      }
      storeMessage(destinationPe, destinationLocation, &dataItem);   
    }
    lastDestinationPe = destinationPe; 
  }

#ifdef STAGED_COMPLETION
  markMessageReceived(msg->dimension, msg->finalMsgCount); 
#endif

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

      if (destinationBuffer->numDataItems < bufferSize_) {
	// not sending the full buffer, shrink the message size
	envelope *env = UsrToEnv(destinationBuffer);
	env->setTotalsize(env->getTotalsize() - sizeof(dtype) *
			  (bufferSize_ - destinationBuffer->numDataItems));
	*((int *) env->getPrioPtr()) = prio_;
      }
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
        CkPrintf("[%d] sending intermediate flush to %d\n", CkMyPe(), destinationIndex); 
#endif
	this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
      }

#ifdef STAGED_COMPLETION
      cntMsgSent_[i][flushIndex]++; 
#endif

      messageBuffers[flushIndex] = NULL;

    }

  }
}

template <class dtype>
void MeshStreamer<dtype>::flushAllBuffers() {

  MeshStreamerMessage<dtype> **messageBuffers; 
  int numBuffers; 

  for (int i = 0; i < numDimensions_; i++) {

    messageBuffers = dataBuffers_[i]; 
    numBuffers = individualDimensionSizes_[i]; 

    for (int j = 0; j < numBuffers; j++) {

      if(messageBuffers[j] == NULL) {
	continue;
      }

      numDataItemsBuffered_ -= messageBuffers[j]->numDataItems;

      if (i == 0) {
	int destinationPe = myIndex_ + j - myLocationIndex_[i];
        this->thisProxy[destinationPe].receiveAtDestination(messageBuffers[j]);
      }	 
      else {

	for (int k = 0; k < messageBuffers[j]->numDataItems; k++) {

	  MeshStreamerMessage<dtype> *directMsg = 
	    new (0, 1, sizeof(int)) MeshStreamerMessage<dtype>(i);
	  *(int *) CkPriorityPtr(directMsg) = prio_;
	  CkSetQueueing(directMsg, CK_QUEUEING_IFIFO);

#ifdef DEBUG_STREAMER
	  CkAssert(directMsg != NULL);
#endif
	  int destinationPe = messageBuffers[j]->destinationPes[k]; 
	  dtype &dataItem = messageBuffers[j]->getDataItem(k);   
	  directMsg->addDataItem(dataItem);
          this->thisProxy[destinationPe].receiveAtDestination(directMsg);
	}
	delete messageBuffers[j];
      }
      messageBuffers[j] = NULL;
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
  CkPrintf("[%d] flushDimension: %d, sendMsgCounts: %d\n", CkMyPe(), dimension, sendMsgCounts); 
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

    if (destinationBuffer->numDataItems < bufferSize_) {
#ifdef STAGED_COMPLETION
      if (destinationBuffer->numDataItems != 0) {
#endif
      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBuffer);
      env->setTotalsize(env->getTotalsize() - sizeof(dtype) *
                        (bufferSize_ - destinationBuffer->numDataItems));
      *((int *) env->getPrioPtr()) = prio_;
#ifdef STAGED_COMPLETION
      }
#endif
    }
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;

#ifdef STAGED_COMPLETION
    destinationBuffer->finalMsgCount = ++cntMsgSent_[dimension][j];
#endif

    if (dimension == 0) {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending dimension flush to %d\n", CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAtDestination(destinationBuffer);
    }
    else {
#ifdef STREAMER_VERBOSE_OUTPUT
      CkPrintf("[%d] sending intermediate dimension flush to %d\n", CkMyPe(), destinationIndex); 
#endif
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }
    messageBuffers[j] = NULL;
  }
  
}


template <class dtype>
void MeshStreamer<dtype>::flushDirect(){
  // flush if (1) this is not a periodic call or 
  //          (2) this is a periodic call and no sending took place
  //              since the last time the function was invoked
  if (!isPeriodicFlushEnabled_ || !hasSentRecently_) {

    if (numDataItemsBuffered_ != 0) {
      flushAllBuffers();
    }    
#ifdef DEBUG_STREAMER
    CkAssert(numDataItemsBuffered_ == 0); 
#endif
    
  }

  hasSentRecently_ = false; 

}

template <class dtype>
void periodicProgressFunction(void *MeshStreamerObj, double time) {

  MeshStreamer<dtype> *properObj = 
    static_cast<MeshStreamer<dtype>*>(MeshStreamerObj); 

  if (properObj->isPeriodicFlushEnabled()) {
    properObj->flushDirect();
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
      dtype &data = msg->getDataItem(i);
      clientObj_->process(data);
    }

#ifdef STAGED_COMPLETION
#ifdef STREAMER_VERBOSE_OUTPUT
    envelope *env = UsrToEnv(msg);
    CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n", CkMyPe(), env->getSrcPe(), msg->numDataItems, msg->finalMsgCount);  
#endif
    markMessageReceived(msg->dimension, msg->finalMsgCount); 
#else 
    this->detectorLocalObj_->consume(msg->numDataItems);    
#endif
    delete msg;
  }

  void localDeliver(dtype &dataItem) {
    clientObj_->process(dataItem);
#ifndef STAGED_COMPLETION
    MeshStreamer<dtype>::detectorLocalObj_->consume();
#endif
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
		    const CProxy_MeshStreamerGroupClient<dtype> &clientProxy,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0)
   :MeshStreamer<dtype>(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
                        0, yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientObj_ = 
      ((MeshStreamerGroupClient<dtype> *)CkLocalBranch(clientProxy_));
  }

  GroupMeshStreamer(int numDimensions, int *dimensionSizes, 
		    const CProxy_MeshStreamerGroupClient<dtype> &clientProxy,
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
class MeshStreamerClientIterator : public CkLocIterator {

public:
  
  CompletionDetector *detectorLocalObj_;
  CkArray *clientArrMgr_;
  MeshStreamerClientIterator(CompletionDetector *detectorObj, 
			     CkArray *clientArrMgr) 
    : detectorLocalObj_(detectorObj), clientArrMgr_(clientArrMgr) {}

  // CkLocMgr::iterate will call addLocation on all elements local to this PE
  void addLocation(CkLocation &loc) {

    MeshStreamerArrayClient<dtype> *clientObj = 
      (MeshStreamerArrayClient<dtype> *) clientArrMgr_->lookup(loc.getIndex());

#ifdef DEBUG_STREAMER
    CkAssert(clientObj != NULL); 
#endif
    clientObj->setDetector(detectorLocalObj_); 
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

  void localDeliver(ArrayDataItem<dtype, itype> &packedDataItem) {
    itype arrayId = packedDataItem.arrayIndex; 

    MeshStreamerArrayClient<dtype> *clientObj;
#ifdef CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = clientProxy_[arrayId].ckLocal();
#endif

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
#ifndef STAGED_COMPLETION
      MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->consume();
#endif
    }
    else { 
      // array element is no longer present locally - redeliver using proxy
      clientProxy_[arrayId].receiveRedeliveredItem(packedDataItem.dataItem);
    }
  }

  int numElementsInClient() {
    return numArrayElements_;
  }

  int numLocalElementsInClient() {
    return numLocalArrayElements_;
  }

  void initLocalClients() {
#ifndef STAGED_COMPLETION

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
    MeshStreamerClientIterator<dtype> clientIterator(
     MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_, 
     clientProxy_.ckLocalBranch());
    clientLocMgr->iterate(clientIterator);
  #endif    

#else 
    numLocalArrayElements_ = clientProxy_.ckLocMgr()->numLocalElements();
#endif
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
    dtype *dataItem;
  };

  ArrayMeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
		    int *dimensionSizes, 
                    const CProxy_MeshStreamerArrayClient<dtype> &clientProxy,
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
		    const CProxy_MeshStreamerArrayClient<dtype> &clientProxy,
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
      ArrayDataItem<dtype, itype> &packedData = msg->getDataItem(i);
      localDeliver(packedData);
    }
#ifdef STAGED_COMPLETION
    markMessageReceived(msg->dimension, msg->finalMsgCount);
#endif

    delete msg;
  }

  void insertData(dtype &dataItem, itype arrayIndex) {
#ifndef STAGED_COMPLETION
    MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->produce();
#endif
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

  ArrayDataItem<dtype, itype> packedDataItem;
    if (destinationPe == CkMyPe()) {
      // copying here is necessary - user code should not be 
      // passed back a reference to the original item
      packedDataItem.arrayIndex = arrayIndex; 
      packedDataItem.dataItem = dataItem;
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

  int copyDataItemIntoMessage(
      MeshStreamerMessage<ArrayDataItem <dtype, itype> > *destinationBuffer, 
      void *dataItemHandle, bool copyIndirectly) {

    if (copyIndirectly == true) {
      // newly inserted items are passed through a handle to avoid copying
      int numDataItems = destinationBuffer->numDataItems;
      DataItemHandle *tempHandle = (DataItemHandle *) dataItemHandle;
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

#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
