#ifndef NDMESH_STREAMER_H
#define NDMESH_STREAMER_H

#include <algorithm>
#include <list>
#include <map>
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"
#include "completion.h"
#include "ckarray.h"

// limit total number of buffered data items to
// maxNumDataItemsBuffered_ (flush when limit is reached) but allow
// allocation of up to a factor of CMK_TRAM_OVERALLOCATION_FACTOR more space to
// take advantage of nonuniform filling of buffers
#define CMK_TRAM_OVERALLOCATION_FACTOR 4

// The "intranode arbitration" scheme partitions PEs into teams where each team
//  is responsible for sending messages to the peers along one dimension of the
//  topology; items that need to be sent along the non-assigned dimension are
//  first forwarded to one of the PEs in the team responsible for sending along
//  that dimension. Such forwarding messages will always be delivered intranode
//  as long as the last dimension in the topology comprises PEs within the same
//  node. The scheme improves aggregation at the cost of additional intranode
//  traffic.
// #define CMK_TRAM_INTRANODE_ARBITRATION
// #define CMK_TRAM_CACHE_LOCATIONS
// #define CMK_TRAM_CACHE_ARRAY_METADATA // only works for 1D array clients
// #define CMK_TRAM_VERBOSE_OUTPUT

#define TRAM_BROADCAST (-100)

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
static const int personalizedMsgType = 0;
static const int forwardMsgType = 1;
#endif

extern void QdCreate(int n);
extern void QdProcess(int n);

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

  inline int addDataItem(const dtype& dataItem) {
    dataItems[numDataItems] = dataItem;
    return ++numDataItems;
  }

  inline void markDestination(const int index, const int destinationPe) {
    destinationPes[index] = destinationPe;
  }

  inline const dtype& getDataItem(const int index) {
    return dataItems[index];
  }
};

template <class dtype>
class MeshStreamer : public CBase_MeshStreamer<dtype> {

private:
  int bufferSize_;
  int maxNumDataItemsBuffered_;
  int numDataItemsBuffered_;

  int *individualDimensionSizes_;
  int *combinedDimensionSizes_;

  int *myLocationIndex_;

  CkCallback   userCallback_;
  bool yieldFlag_;

  double progressPeriodInMs_;
  bool isPeriodicFlushEnabled_;
  bool hasSentRecently_;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // the number of PEs per team assigned to buffer messages for each dimension
  int teamSize_;
  int myAssignedDim_;
  int *forwardingDestinations_;

  int numSendersToWaitFor_;
  int dimensionOfArrivingMsgs_;
  bool finishedAssignedDim_;

#endif
  MeshStreamerMessage<dtype> ***dataBuffers_;

  CProxy_CompletionDetector detector_;
  int prio_;
  int yieldCount_;

#ifdef CMK_TRAM_CACHE_LOCATIONS
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

  virtual void localDeliver(const dtype& dataItem) = 0;
  virtual void localBroadcast(const dtype& dataItem) = 0;

  virtual void initLocalClients() = 0;

  void sendLargestBuffer();
  void flushToIntermediateDestinations();
  void flushDimension(int dimension, bool sendMsgCounts = false);

  int determineDestinationIndex(int bufferIndex, int dimension);
protected:

  int numMembers_;
  int myIndex_;
  int numDimensions_;
  bool useStagedCompletion_;
  bool stagedCompletionStarted_;
  bool useCompletionDetection_;
  CompletionDetector *detectorLocalObj_;
  virtual int copyDataItemIntoMessage(
              MeshStreamerMessage<dtype> *destinationBuffer,
              const void *dataItemHandle, bool copyIndirectly = false);
  void insertData(const void *dataItemHandle, int destinationPe);
  void broadcast(const void *dataItemHandle, int dimension,
                 bool copyIndirectly);
  MeshLocation determineLocation(int destinationPe,
                                 int dimensionReceivedAlong);
  void storeMessage(int destinationPe,
                    const MeshLocation& destinationCoordinates,
                    const void *dataItem, bool copyIndirectly = false);

  void ctorHelper(int maxNumDataItemsBuffered, int numDimensions,
                  int *dimensionSizes, int bufferSize,
                  bool yieldFlag, double progressPeriodInMs);

public:

  MeshStreamer() {}
  MeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
               int *dimensionSizes, int bufferSize,
               bool yieldFlag = 0, double progressPeriodInMs = -1.0);
  ~MeshStreamer();

  // entry

  void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
  void enablePeriodicFlushing(){
    if (progressPeriodInMs_ <= 0) {
      if (myIndex_ == 0) {
        CkPrintf("Using periodic flushing for NDMeshStreamer requires"
                 " setting a valid periodic flush period. Defaulting"
                 " to 10 ms.\n");
      }
      progressPeriodInMs_ = 10;
    }

    isPeriodicFlushEnabled_ = true;
    registerPeriodicProgressFunction();
  }
  void finish();
  void init(int numLocalContributors, CkCallback startCb, CkCallback endCb,
            int prio, bool usePeriodicFlushing);
  void init(int numContributors, CkCallback startCb, CkCallback endCb,
            CProxy_CompletionDetector detector,
            int prio, bool usePeriodicFlushing);
  void init(CkArrayID senderArrayID, CkCallback startCb, CkCallback endCb,
            int prio, bool usePeriodicFlushing);
  void init(CkCallback startCb, int prio);

  virtual void receiveAtDestination(MeshStreamerMessage<dtype> *msg) = 0;

  // non entry
  void flushIfIdle();
  inline bool isPeriodicFlushEnabled() {
    return isPeriodicFlushEnabled_;
  }
  virtual void insertData(const dtype& dataItem, int destinationPe);
  virtual void broadcast(const dtype& dataItem);

  void sendMeshStreamerMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                               int dimension, int destinationIndex);

  void registerPeriodicProgressFunction();

  inline void done(int numContributorsFinished = 1) {

    if (useStagedCompletion_) {
      numLocalDone_ += numContributorsFinished;
      CkAssert(numLocalDone_ <= numLocalContributors_);
      if (numLocalDone_ == numLocalContributors_) {
        startStagedCompletion();
      }
    }
    else if (useCompletionDetection_){
      detectorLocalObj_->done(numContributorsFinished);
    }
  }

  inline void startStagedCompletion() {
    stagedCompletionStarted_ = true;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
    if (myAssignedDim_ > numDimensions_ - 1) {
      // one of the left over PEs that do not have an assigned dimension
      numSendersToWaitFor_ = 0;
    }
    else {

      for (int i = numDimensions_ - 2; i >= dimensionOfArrivingMsgs_; i--) {
        flushDimension(i, true);
      }

      // contributors forwarding messages along the last dimension
      numSendersToWaitFor_ = numDimensions_ - 1;
      // contributors sending along their assigned dimension
      if (myAssignedDim_ != numDimensions_ - 2) {
        numSendersToWaitFor_ +=
          individualDimensionSizes_[dimensionOfArrivingMsgs_] - 1;
      }
      // contributors that were not assigned a team
      // for any receiving PE, there can be at most one such contributor
      if (myLocationIndex_[numDimensions_ - 1]
          + (numDimensions_ - myAssignedDim_) * teamSize_
          < individualDimensionSizes_[numDimensions_ - 1]) {
        numSendersToWaitFor_++;
      }
    }
#ifdef CMK_TRAM_VERBOSE_OUTPUT
    CkPrintf("[%d] Initiating staged completion. dimensionOfArrivingMsgs: %d "
             "numSendersToWaitFor: %d\n", myIndex_, dimensionOfArrivingMsgs_,
             numSendersToWaitFor_);
#endif
#else
    dimensionToFlush_ = numDimensions_ - 1;
    flushDimension(dimensionToFlush_, true);
    dimensionToFlush_--;
#endif
    checkForCompletedStages();
  }

  inline void markMessageReceived(int dimension, int finalCount) {
    cntMsgReceived_[dimension]++;
    if (finalCount != -1) {
      cntFinished_[dimension]++;
      cntMsgExpected_[dimension] += finalCount;
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      CkPrintf("[%d] received dimension: %d finalCount: %d cntFinished: %d "
               "cntMsgExpected: %d cntMsgReceived: %d\n", myIndex_, dimension,
               finalCount, cntFinished_[dimension], cntMsgExpected_[dimension],
               cntMsgReceived_[dimension]);
#endif
    }
    if (stagedCompletionStarted_) {
      checkForCompletedStages();
    }
  }

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  inline void checkForCompletedStages() {
    if (!finishedAssignedDim_ &&
        cntFinished_[forwardMsgType] == numSendersToWaitFor_ &&
        cntMsgExpected_[forwardMsgType] == cntMsgReceived_[forwardMsgType]) {

#ifdef CMK_TRAM_VERBOSE_OUTPUT
      CkPrintf("[%d] stage completion ready to flush assigned dimension %d, "
               "received contributions from %d PEs, cntMsgExpected: %d "
               "cntMsgReceived: %d\n", myIndex_,  myAssignedDim_,
               cntFinished_[forwardMsgType], cntMsgExpected_[forwardMsgType],
               cntMsgReceived_[forwardMsgType]);
#endif

      for (int i = dimensionOfArrivingMsgs_ - 1; i >= 0; i --) {
        flushDimension(i, true);
      }
      flushDimension(numDimensions_ - 1, true);
      finishedAssignedDim_ = true;
      numSendersToWaitFor_ = teamSize_;
      if (myAssignedDim_ == numDimensions_ - 1) {
        numSendersToWaitFor_--;
      }
    }

    if (finishedAssignedDim_ &&
        cntFinished_[personalizedMsgType] == numSendersToWaitFor_ &&
        cntMsgExpected_[personalizedMsgType]
        == cntMsgReceived_[personalizedMsgType]) {
      CkAssert(numDataItemsBuffered_ == 0);
      isPeriodicFlushEnabled_ = false;
      if (!userCallback_.isInvalid()) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
        CkPrintf("[%d] All done. Reducing to final callback ...\n", myIndex_);
#endif
        this->contribute(userCallback_);
        userCallback_ = CkCallback();
      }
    }
  }

#else
  inline void checkForCompletedStages() {

    while (cntFinished_[dimensionToFlush_ + 1] ==
           individualDimensionSizes_[dimensionToFlush_ + 1] - 1 &&
           cntMsgExpected_[dimensionToFlush_ + 1] ==
           cntMsgReceived_[dimensionToFlush_ + 1]) {
      if (dimensionToFlush_ == -1) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
        CkPrintf("[%d] contribute\n", myIndex_);
#endif
        CkAssert(numDataItemsBuffered_ == 0);
        isPeriodicFlushEnabled_ = false;
        if (!userCallback_.isInvalid()) {
          this->contribute(userCallback_);
          userCallback_ = CkCallback();
        }
        return;
      }
      else {
        flushDimension(dimensionToFlush_, true);
      }
      dimensionToFlush_--;
    }
  }
#endif
};

template <class dtype>
MeshStreamer<dtype>::
MeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
             int *dimensionSizes, int bufferSize, bool yieldFlag,
             double progressPeriodInMs) {
  ctorHelper(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
             bufferSize, yieldFlag, progressPeriodInMs);
}

template <class dtype>
void MeshStreamer<dtype>::
ctorHelper(int maxNumDataItemsBuffered, int numDimensions,
           int *dimensionSizes, int bufferSize,
           bool yieldFlag, double progressPeriodInMs) {

  numDimensions_ = numDimensions;
  maxNumDataItemsBuffered_ = maxNumDataItemsBuffered;
  yieldFlag_ = yieldFlag;
  progressPeriodInMs_ = progressPeriodInMs;
  bufferSize_ = bufferSize;
  numDataItemsBuffered_ = 0;
  numMembers_ = CkNumPes();
  myIndex_ = CkMyPe();

  int sumAlongAllDimensions = 0;
  individualDimensionSizes_ = new int[numDimensions_];
  combinedDimensionSizes_ = new int[numDimensions_];
  myLocationIndex_ = new int[numDimensions_];
  memcpy(individualDimensionSizes_, dimensionSizes,
	 numDimensions_ * sizeof(int));
  combinedDimensionSizes_[numDimensions - 1] = 1;
  sumAlongAllDimensions += individualDimensionSizes_[numDimensions_ - 1];
  for (int i = numDimensions_ - 2; i >= 0; i--) {
    sumAlongAllDimensions += individualDimensionSizes_[i];
    combinedDimensionSizes_[i] =
      combinedDimensionSizes_[i + 1] * individualDimensionSizes_[i + 1];
  }
  if (combinedDimensionSizes_[0] * individualDimensionSizes_[0]
      != numMembers_) {
    CkAbort("Error: number of elements in virtual topology must be equal to "
            "total number of PEs.");
  }

  int remainder = myIndex_;
  for (int i = 0; i < numDimensions_; i++) {
    myLocationIndex_[i] = remainder / combinedDimensionSizes_[i];
    remainder -= combinedDimensionSizes_[i] * myLocationIndex_[i];
  }

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // we need at least as many PEs per node as dimensions in the virtual topology
  if (individualDimensionSizes_[numDimensions_ - 1] < numDimensions_) {
    CkAbort("Error: Last dimension in virtual topology must have size greater "
            "than or equal to number of dimensions in the topology.\n");
  }
  teamSize_ = individualDimensionSizes_[numDimensions_ - 1] / numDimensions_;
  myAssignedDim_ = myLocationIndex_[numDimensions_ - 1] / teamSize_;
  int numBuffers;
  // if the number of PEs per node does not divide evenly by the number of
  //  dimensions, some PEs will be left with an invalid dimension assignment;
  //  this is fine - just use them to forward data locally
  if (myAssignedDim_ > numDimensions_ - 1) {
    numBuffers = numDimensions_;
    dimensionOfArrivingMsgs_ = numDimensions_ - 1;
  }
  else {
    // sum of number of remote and local buffers
    numBuffers = individualDimensionSizes_[myAssignedDim_] + numDimensions_ - 2;
    dimensionOfArrivingMsgs_ =
      myAssignedDim_ == numDimensions_ - 1 ? 0 : myAssignedDim_ + 1;
  }
  forwardingDestinations_ = new int[numDimensions_];

  int baseIndex = myIndex_ - myLocationIndex_[numDimensions_ - 1]
    + myIndex_ % teamSize_;
  for (int i = 0; i < numDimensions_; i++) {
    forwardingDestinations_[i] = baseIndex + i * teamSize_;
  }
#else
  // buffers for dimensions with the
  //   same index as the sender's are not allocated/used
  int numBuffers = sumAlongAllDimensions - numDimensions_ + 1;
#endif

  dataBuffers_ = new MeshStreamerMessage<dtype> **[numDimensions_];
  for (int i = 0; i < numDimensions; i++) {
    int numMembersAlongDimension = individualDimensionSizes_[i];
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
    if (i != myAssignedDim_) {
      numMembersAlongDimension = 1;
    }
#endif
    dataBuffers_[i] =
      new MeshStreamerMessage<dtype> *[numMembersAlongDimension];
    for (int j = 0; j < numMembersAlongDimension; j++) {
      dataBuffers_[i][j] = NULL;
    }
  }

  // a bufferSize input of 0 indicates it should be calculated by the library
  if (bufferSize_ == 0) {
    CkAssert(maxNumDataItemsBuffered_ > 0);
    bufferSize_ = CMK_TRAM_OVERALLOCATION_FACTOR * maxNumDataItemsBuffered_
      / numBuffers;
  }
  else {
    maxNumDataItemsBuffered_ = bufferSize_ * numBuffers
      / CMK_TRAM_OVERALLOCATION_FACTOR;
  }

  if (bufferSize_ <= 0) {
    bufferSize_ = 1;
    if (myIndex_ == 0) {
      CkPrintf("Warning: Argument maxNumDataItemsBuffered to MeshStreamer "
               "constructor is set very low, translating to less than a single "
               "item per aggregation buffer. Defaulting to a single item "
               "per buffer.\n");
    }
  }

  isPeriodicFlushEnabled_ = false;
  detectorLocalObj_ = NULL;

#ifdef CMK_TRAM_CACHE_LOCATIONS
  cachedLocations_ = new MeshLocation[numMembers_];
  isCached_ = new bool[numMembers_];
  std::fill(isCached_, isCached_ + numMembers_, false);
#endif

  cntMsgSent_ = NULL;
  cntMsgReceived_ = NULL;
  cntMsgExpected_ = NULL;
  cntFinished_ = NULL;

#ifdef CMK_TRAM_VERBOSE_OUTPUT
  CkPrintf("[%d] Instance initialized. Buffer size: %d, Capacity: %d, "
           "Yield: %d, Flush period: %f, Number of buffers: %d\n",
           myIndex_, bufferSize_, maxNumDataItemsBuffered_, yieldFlag_,
           progressPeriodInMs_, numBuffers);
#endif
}

template <class dtype>
MeshStreamer<dtype>::~MeshStreamer() {

  for (int i = 0; i < numDimensions_; i++) {
    int numBuffers = individualDimensionSizes_[i];
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
    if (i != myAssignedDim_) {
      numBuffers = 1;
    }
#endif
    for (int j = 0; j < numBuffers; j++) {
      delete[] dataBuffers_[i][j];
    }
    delete[] dataBuffers_[i];
  }
  delete[] dataBuffers_;

  delete[] individualDimensionSizes_;
  delete[] combinedDimensionSizes_;
  delete[] myLocationIndex_;

#ifdef CMK_TRAM_CACHE_LOCATIONS
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

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  delete[] forwardingDestinations_;
#endif

}

template <class dtype>
inline MeshLocation MeshStreamer<dtype>::
determineLocation(int destinationPe, int dimensionReceivedAlong) {

#ifdef CMK_TRAM_CACHE_LOCATIONS
  if (isCached_[destinationPe]) {
    return cachedLocations_[destinationPe];
  }
#endif

  MeshLocation destinationLocation;
  for (int i = dimensionReceivedAlong - 1; i >= 0; i--) {
    int blockIndex = destinationPe / combinedDimensionSizes_[i];

    int dimensionIndex =
      blockIndex - blockIndex / individualDimensionSizes_[i]
      * individualDimensionSizes_[i];

    if (dimensionIndex != myLocationIndex_[i]) {
      destinationLocation.dimension = i;
      destinationLocation.bufferIndex = dimensionIndex;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
      // items to be sent along dimensions other than my assigned dimension
      // are batched into a single message per dimension, to be forwarded
      // intranode to a responsible PE
      if (myAssignedDim_ != i) {
        destinationLocation.bufferIndex = 0;
      }
#endif
#ifdef CMK_TRAM_CACHE_LOCATIONS
      cachedLocations_[destinationPe] = destinationLocation;
      isCached_[destinationPe] = true;
#endif
      return destinationLocation;
    }
  }
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // routing along the intranode dimension is delayed until the end
  destinationLocation.dimension = numDimensions_ - 1;
  if (myAssignedDim_ != numDimensions_ - 1) {
    destinationLocation.bufferIndex = 0;
  }
  else {
    destinationLocation.bufferIndex = destinationPe - destinationPe /
      individualDimensionSizes_[numDimensions_ - 1]
      * individualDimensionSizes_[numDimensions_ - 1];
  }
#ifdef CMK_TRAM_CACHE_LOCATIONS
  cachedLocations_[destinationPe] = destinationLocation;
  isCached_[destinationPe] = true;
#endif

#else
  CkAbort("Error. MeshStreamer::determineLocation called with destinationPe "
          "equal to sender's PE. This is unexpected and may cause errors.\n");
#endif
  return destinationLocation;
}

template <class dtype>
inline int MeshStreamer<dtype>::
copyDataItemIntoMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        const void *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->addDataItem(*((const dtype *)dataItemHandle));
}

template <class dtype>
inline int MeshStreamer<dtype>::
determineDestinationIndex(int dimension, int bufferIndex) {
  int destinationIndex;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  if (dimension != myAssignedDim_) {
    destinationIndex = forwardingDestinations_[dimension];
    return destinationIndex;
  }
#endif

  destinationIndex = myIndex_ + (bufferIndex - myLocationIndex_[dimension]) *
    combinedDimensionSizes_[dimension];

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // adjust the destination index so that the message will arrive at
  //  a PE responsible for sending along the proper dimension
  if (dimension == 0) {
    destinationIndex += (numDimensions_ - 1) * teamSize_;
  }
  else if (dimension != numDimensions_ - 1) {
    destinationIndex -= teamSize_;
  }
#endif

  return destinationIndex;
}

template <class dtype>
inline void MeshStreamer<dtype>::
sendMeshStreamerMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        int dimension, int destinationIndex) {

  bool personalizedMessage;

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  personalizedMessage =
    dimension == numDimensions_ - 1 && myAssignedDim_ == numDimensions_ - 1;
  destinationBuffer->dimension =
    personalizedMessage ? personalizedMsgType : forwardMsgType;
#else
  personalizedMessage = dimension == 0;
#endif

  if (personalizedMessage) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
    CkPrintf("[%d] sending to %d\n", myIndex_, destinationIndex);
#endif
    this->thisProxy[destinationIndex].receiveAtDestination(destinationBuffer);
  }
  else {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
    CkPrintf("[%d] sending intermediate to %d\n",
             myIndex_, destinationIndex);
#endif
    this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
  }
}

template <class dtype>
inline void MeshStreamer<dtype>::
storeMessage(int destinationPe, const MeshLocation& destinationLocation,
             const void *dataItem, bool copyIndirectly) {

  int dimension = destinationLocation.dimension;
  int bufferIndex = destinationLocation.bufferIndex;
  MeshStreamerMessage<dtype> ** messageBuffers = dataBuffers_[dimension];

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  bool personalizedMessage =
    dimension == numDimensions_ - 1 && myAssignedDim_ == numDimensions_ - 1;
#else
  bool personalizedMessage = dimension == 0;
#endif

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    int numDestIndices = bufferSize_;
    // personalized messages do not require destination indices
    if (personalizedMessage) {
      numDestIndices = 0;
    }
    messageBuffers[bufferIndex] =
      new (numDestIndices, bufferSize_, 8 * sizeof(int))
      MeshStreamerMessage<dtype>(dimension);

    *(int *) CkPriorityPtr(messageBuffers[bufferIndex]) = prio_;
    CkSetQueueing(messageBuffers[bufferIndex], CK_QUEUEING_IFIFO);
    CkAssert(messageBuffers[bufferIndex] != NULL);
  }

  MeshStreamerMessage<dtype> *destinationBuffer = messageBuffers[bufferIndex];
  int numBuffered =
    copyDataItemIntoMessage(destinationBuffer, dataItem, copyIndirectly);
  if (!personalizedMessage) {
    destinationBuffer->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;

  // send if buffer is full
  if (numBuffered == bufferSize_) {

    int destinationIndex = determineDestinationIndex(dimension, bufferIndex);
    sendMeshStreamerMessage(destinationBuffer, dimension, destinationIndex);

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
inline void MeshStreamer<dtype>::broadcast(const dtype& dataItem) {

  const static bool copyIndirectly = true;

  // no data items should be submitted after all local contributors call done
  // and staged completion has begun
  CkAssert(stagedCompletionStarted_ == false);

  // produce and consume once per PE
  if (useCompletionDetection_) {
    detectorLocalObj_->produce(numMembers_);
  }
  QdCreate(numMembers_);

  // deliver locally
  localBroadcast(dataItem);

  broadcast(&dataItem, numDimensions_ - 1, copyIndirectly);
}

template <class dtype>
inline void MeshStreamer<dtype>::
broadcast(const void *dataItemHandle, int dimension, bool copyIndirectly) {

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  CkAbort("Broadcast is currently incompatible with intranode arbitration\n");
#endif

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
inline void MeshStreamer<dtype>::
insertData(const void *dataItemHandle, int destinationPe) {

  const static bool copyIndirectly = true;

  int initialRoutingDimension = numDimensions_ - 1;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // in this scheme skip routing along the last (intranode) dimension
  // until the last step
  initialRoutingDimension = numDimensions_ - 2;
#endif

  // treat newly inserted items as if they were received along
  // a higher dimension (e.g. for a 3D mesh, received along 4th dimension)
  MeshLocation destinationLocation =
    determineLocation(destinationPe, initialRoutingDimension + 1);
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
inline void MeshStreamer<dtype>::
insertData(const dtype& dataItem, int destinationPe) {

  // no data items should be submitted after all local contributors call done
  // and staged completion has begun
  CkAssert(stagedCompletionStarted_ == false);

  if (useCompletionDetection_) {
    detectorLocalObj_->produce();
  }
  QdCreate(1);
  if (destinationPe == myIndex_) {
    localDeliver(dataItem);
    return;
  }

  insertData((const void *) &dataItem, destinationPe);
}

template <class dtype>
void MeshStreamer<dtype>::init(CkCallback startCb, int prio) {

  useStagedCompletion_ = false;
  stagedCompletionStarted_ = false;
  useCompletionDetection_ = false;

  yieldCount_ = 0;
  userCallback_ = CkCallback();
  prio_ = prio;

  initLocalClients();

  hasSentRecently_ = false;
  enablePeriodicFlushing();

  this->contribute(startCb);
}

template <class dtype>
void MeshStreamer<dtype>::
init(int numLocalContributors, CkCallback startCb, CkCallback endCb, int prio,
     bool usePeriodicFlushing) {

  useStagedCompletion_ = true;
  stagedCompletionStarted_ = false;
  useCompletionDetection_ = false;

  int dimensionsReceiving;
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  // received messages can be categorized into two types:
  // (1) personalized messages with this PE as the final destination
  // (2) intermediate messages to be sent along this PE's assigned dimension
  dimensionsReceiving = 2;
  finishedAssignedDim_ = false;
#else
  dimensionsReceiving = numDimensions_;
#endif

  // allocate memory on first use
  if (cntMsgSent_ == NULL) {
    cntMsgSent_ = new int*[numDimensions_];

    cntMsgReceived_ = new int[dimensionsReceiving];
    cntMsgExpected_ = new int[dimensionsReceiving];
    cntFinished_ = new int[dimensionsReceiving];

    for (int i = 0; i < numDimensions_; i++) {
      cntMsgSent_[i] = new int[individualDimensionSizes_[i]];
    }
  }

  std::fill(cntMsgReceived_, cntMsgReceived_ + dimensionsReceiving, 0);
  std::fill(cntMsgExpected_, cntMsgExpected_ + dimensionsReceiving, 0);
  std::fill(cntFinished_, cntFinished_ + dimensionsReceiving, 0);

  for (int i = 0; i < numDimensions_; i++) {
    std::fill(cntMsgSent_[i],
              cntMsgSent_[i] + individualDimensionSizes_[i], 0);
  }

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
void MeshStreamer<dtype>::
init(int numContributors, CkCallback startCb, CkCallback endCb,
     CProxy_CompletionDetector detector,
     int prio, bool usePeriodicFlushing) {

  useStagedCompletion_ = false;
  stagedCompletionStarted_ = false;
  useCompletionDetection_ = true;
  yieldCount_ = 0;
  prio_ = prio;
  userCallback_ = endCb;

  // to facilitate completion, enable flushing after all contributors
  //  have finished submitting items
  CkCallback flushCb(CkIndex_MeshStreamer<dtype>::enablePeriodicFlushing(),
                     this->thisProxy);
  CkCallback finish(CkIndex_MeshStreamer<dtype>::finish(),
		    this->thisProxy);
  detector_ = detector;
  detectorLocalObj_ = detector_.ckLocalBranch();
  initLocalClients();

  detectorLocalObj_->start_detection(numContributors, startCb, flushCb,
                                     finish , 0);

  hasSentRecently_ = false;
  if (usePeriodicFlushing) {
    enablePeriodicFlushing();
  }
}

template <class dtype>
void MeshStreamer<dtype>::
init(CkArrayID senderArrayID, CkCallback startCb, CkCallback endCb, int prio,
     bool usePeriodicFlushing) {

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
    if (destinationPe == myIndex_) {
      localDeliver(dataItem);
    }
    else if (destinationPe != TRAM_BROADCAST) {
      if (destinationPe != lastDestinationPe) {
        // do this once per sequence of items with the same destination
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
        destinationLocation = determineLocation(destinationPe,
                                                dimensionOfArrivingMsgs_);
#else
        destinationLocation = determineLocation(destinationPe, msg->dimension);
#endif
      }
      storeMessage(destinationPe, destinationLocation, &dataItem);
    }
    else /* if (destinationPe == TRAM_BROADCAST) */ {
      localBroadcast(dataItem);
      broadcast(&dataItem, msg->dimension - 1, false);
    }
    lastDestinationPe = destinationPe;
  }

#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received along route from %d %d items finalMsgCount: %d"
               " dimension: %d\n", myIndex_, env->getSrcPe(),
               msg->numDataItems, msg->finalMsgCount, msg->dimension);
#endif

  if (useStagedCompletion_) {
    markMessageReceived(msg->dimension, msg->finalMsgCount);
  }

  delete msg;
}

template <class dtype>
inline void MeshStreamer<dtype>::sendLargestBuffer() {

  int flushDimension, flushIndex, maxSize, destinationIndex, numBuffers;
  MeshStreamerMessage<dtype> ** messageBuffers;
  MeshStreamerMessage<dtype> *destinationBuffer;

  for (int i = 0; i < numDimensions_; i++) {

    messageBuffers = dataBuffers_[i];
    numBuffers = individualDimensionSizes_[i];
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
    if (i != myAssignedDim_) {
      numBuffers = 1;
    }
#endif

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

      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBuffer);
      env->shrinkUsersize((bufferSize_ - destinationBuffer->numDataItems)
                        * sizeof(dtype));
      numDataItemsBuffered_ -= destinationBuffer->numDataItems;

      destinationIndex = determineDestinationIndex(flushDimension, flushIndex);
      sendMeshStreamerMessage(destinationBuffer, flushDimension,
                              destinationIndex);

      if (useStagedCompletion_) {
        cntMsgSent_[i][flushIndex]++;
      }

      messageBuffers[flushIndex] = NULL;
    }
  }
}

template <class dtype>
inline void MeshStreamer<dtype>::flushToIntermediateDestinations() {
  for (int i = 0; i < numDimensions_; i++) {
    flushDimension(i);
  }
}

template <class dtype>
void MeshStreamer<dtype>::flushDimension(int dimension, bool sendMsgCounts) {

  MeshStreamerMessage<dtype> **messageBuffers;
  MeshStreamerMessage<dtype> *destinationBuffer;
  int destinationIndex, numBuffers;

  if (individualDimensionSizes_[dimension] == 1) {
    return;
  }

  messageBuffers = dataBuffers_[dimension];
  numBuffers = individualDimensionSizes_[dimension];

#ifdef CMK_TRAM_INTRANODE_ARBITRATION
  if (dimension != myAssignedDim_) {
    numBuffers = 1;
  }
#endif

#ifdef CMK_TRAM_VERBOSE_OUTPUT
  CkPrintf("[%d] flushDimension: %d, num buffered: %d, sendMsgCounts: %d\n",
           myIndex_, dimension, numDataItemsBuffered_, sendMsgCounts);
#endif

  for (int j = 0; j < numBuffers; j++) {
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
    if (dimension == myAssignedDim_ && j == myLocationIndex_[dimension]) {
      continue;
    }
#endif

    if(messageBuffers[j] == NULL) {
#ifdef CMK_TRAM_INTRANODE_ARBITRATION

      if (sendMsgCounts &&
          (dimension != myAssignedDim_ || j != myLocationIndex_[dimension])) {
#else
      if (sendMsgCounts && j != myLocationIndex_[dimension]) {
#endif
        messageBuffers[j] =
          new (0, 0, 8 * sizeof(int)) MeshStreamerMessage<dtype>(dimension);
        *(int *) CkPriorityPtr(messageBuffers[j]) = prio_;
        CkSetQueueing(messageBuffers[j], CK_QUEUEING_IFIFO);
      }
      else {
        continue;
      }
    }

    destinationBuffer = messageBuffers[j];

    if (destinationBuffer->numDataItems != 0) {
      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBuffer);
      env->shrinkUsersize((bufferSize_ - destinationBuffer->numDataItems)
                          * sizeof(dtype));
    }
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;

    if (useStagedCompletion_) {
      cntMsgSent_[dimension][j]++;
      if (sendMsgCounts) {
#ifdef CMK_TRAM_INTRANODE_ARBITRATION
        if (dimension != myAssignedDim_) {
          destinationBuffer->finalMsgCount = cntMsgSent_[dimension][0];
        }
        else {
          destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
        }
#else
        destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
#endif
      }
    }

    destinationIndex = determineDestinationIndex(dimension, j);
    sendMeshStreamerMessage(destinationBuffer, dimension,
                            destinationIndex);
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


template <class dtype, class ClientType>
class GroupMeshStreamer : public CBase_GroupMeshStreamer<dtype, ClientType> {
private:

  CkGroupID clientGID_;
  ClientType *clientObj_;

  void receiveAtDestination(MeshStreamerMessage<dtype> *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      const dtype& data = msg->getDataItem(i);
      clientObj_->process(data);
    }

    if (this->useStagedCompletion_) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d"
               " dimension: %d\n", this->myIndex_, env->getSrcPe(),
               msg->numDataItems, msg->finalMsgCount, msg->dimension);
#endif
      this->markMessageReceived(msg->dimension, msg->finalMsgCount);
    }
    else if (this->useCompletionDetection_){
      this->detectorLocalObj_->consume(msg->numDataItems);
    }
    QdProcess(msg->numDataItems);
    delete msg;
  }

  inline void localDeliver(const dtype& dataItem) {
    clientObj_->process(dataItem);
    if (this->useCompletionDetection_) {
      this->detectorLocalObj_->consume();
    }
    QdProcess(1);
  }

  inline void localBroadcast(const dtype& dataItem) {
    localDeliver(dataItem);
  }

  inline void initLocalClients() {
    // no action required
  }

public:

  GroupMeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
		    int *dimensionSizes,
		    CkGroupID clientGID,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0) {
    this->ctorHelper(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
               0, yieldFlag, progressPeriodInMs);
    clientGID_ = clientGID;
    clientObj_ = (ClientType *) CkLocalBranch(clientGID_);

  }

  GroupMeshStreamer(int numDimensions, int *dimensionSizes,
		    CkGroupID clientGID,
		    int bufferSize, bool yieldFlag = 0,
                    double progressPeriodInMs = -1.0) {
    this->ctorHelper(0, numDimensions, dimensionSizes, bufferSize,
               yieldFlag, progressPeriodInMs);
    clientGID_ = clientGID;
    clientObj_ = (ClientType *) CkLocalBranch(clientGID_);

  }
};

template <class dtype, class ClientType>
class LocalBroadcaster : public CkLocIterator {

public:
  CkArray *clientArrMgr_;
  const dtype *dataItem_;

  LocalBroadcaster(CkArray *clientArrMgr, const dtype *dataItem)
   : clientArrMgr_(clientArrMgr), dataItem_(dataItem) {}

  void addLocation(CkLocation& loc) {
    ClientType *clientObj =
      (ClientType *) clientArrMgr_->lookup(loc.getIndex());
    CkAssert(clientObj != NULL);
    clientObj->process(*dataItem_);
  }

};

template <class dtype, class itype, class ClientType>
class ArrayMeshStreamer : public CBase_ArrayMeshStreamer<dtype, itype,
                                                         ClientType> {

private:

  CkArrayID clientAID_;
  CkArray *clientArrayMgr_;
  CkLocMgr *clientLocMgr_;
  int numArrayElements_;
  int numLocalArrayElements_;
  std::map<itype, std::vector<ArrayDataItem<dtype, itype> > > misdeliveredItems;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
  ClientType **clientObjs_;
  int *destinationPes_;
  bool *isCachedArrayMetadata_;
#endif

  inline
  void localDeliver(const ArrayDataItem<dtype, itype>& packedDataItem) {

    itype arrayId = packedDataItem.arrayIndex;
    if (arrayId == itype(TRAM_BROADCAST)) {
      localBroadcast(packedDataItem);
      return;
    }
    ClientType *clientObj;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = (ClientType *) clientArrayMgr_->lookup(arrayId);
#endif

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
      if (this->useCompletionDetection_) {
        this->detectorLocalObj_->consume();
      }
      QdProcess(1);
    }
    else {
      // array element arrayId is no longer present locally:
      //  buffer the data item and request updated PE index
      //  to be sent to the source and this PE
      if (this->useStagedCompletion_) {
        CkAbort("Using staged completion when array locations"
                " are not guaranteed to be correct is currently"
                " not supported.");
      }
      misdeliveredItems[arrayId].push_back(packedDataItem);
      if (misdeliveredItems[arrayId].size() == 1) {
        int homePe = clientLocMgr_->homePe(arrayId);
        this->thisProxy[homePe].
          processLocationRequest(arrayId, this->myIndex_,
                                 packedDataItem.sourcePe);
      }
    }
  }

  inline
  void localBroadcast(const ArrayDataItem<dtype, itype>& packedDataItem) {

    LocalBroadcaster<dtype, ClientType>
      clientIterator(clientArrayMgr_, &packedDataItem.dataItem);
    clientLocMgr_->iterate(clientIterator);

    if (this->useCompletionDetection_) {
        this->detectorLocalObj_->consume();
    }
    QdProcess(1);
  }

  inline void initLocalClients() {

    if (this->useCompletionDetection_) {
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
      std::fill(isCachedArrayMetadata_,
                isCachedArrayMetadata_ + numArrayElements_, false);

      for (int i = 0; i < numArrayElements_; i++) {
        clientObjs_[i] =
          (ClientType*) ( clientArrayMgr_->lookup(CkArrayIndex1D(i)) );
      }
#endif
    }
  }

  inline void commonInit() {
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
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
		    int *dimensionSizes, CkArrayID clientAID,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0) {

    this->ctorHelper(maxNumDataItemsBuffered, numDimensions, dimensionSizes, 0,
                     yieldFlag, progressPeriodInMs);
    clientAID_ = clientAID;
    clientArrayMgr_ = clientAID_.ckLocalBranch();
    clientLocMgr_ = clientArrayMgr_->getLocMgr();
    commonInit();
  }

  ArrayMeshStreamer(int numDimensions, int *dimensionSizes,
		    CkArrayID clientAID, int bufferSize, bool yieldFlag = 0,
                    double progressPeriodInMs = -1.0) {

    this->ctorHelper(0, numDimensions, dimensionSizes, bufferSize, yieldFlag,
                     progressPeriodInMs);
    clientAID_ = clientAID;
    clientArrayMgr_ = clientAID_.ckLocalBranch();
    clientLocMgr_ = clientArrayMgr_->getLocMgr();
    commonInit();
  }

  ~ArrayMeshStreamer() {
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
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
    if (this->useStagedCompletion_) {
      this->markMessageReceived(msg->dimension, msg->finalMsgCount);
    }

    delete msg;
  }

  inline void broadcast(const dtype& dataItem) {
    const static bool copyIndirectly = true;

    // no data items should be submitted after all local contributors call done
    // and staged completion has begun
    CkAssert(this->stagedCompletionStarted_ == false);

    if (this->useCompletionDetection_) {
      this->detectorLocalObj_->produce(this->numMembers_);
    }
    QdCreate(this->numMembers_);

    // deliver locally
    ArrayDataItem<dtype, itype>& packedDataItem(TRAM_BROADCAST, this->myIndex_,
                                                dataItem);
    localBroadcast(packedDataItem);

    DataItemHandle tempHandle;
    tempHandle.dataItem = &dataItem;
    tempHandle.arrayIndex = TRAM_BROADCAST;

    MeshStreamer<ArrayDataItem<dtype, itype> >::
      broadcast(&tempHandle, this->numDimensions_ - 1, copyIndirectly);
  }

  inline void insertData(const dtype& dataItem, itype arrayIndex) {

    // no data items should be submitted after all local contributors call done
    // and staged completion has begun
    CkAssert(this->stagedCompletionStarted_ == false);

    if (this->useCompletionDetection_) {
      this->detectorLocalObj_->produce();
    }
    QdCreate(1);
    int destinationPe;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
    if (isCachedArrayMetadata_[arrayIndex]) {
      destinationPe =  destinationPes_[arrayIndex];
    }
    else {
      destinationPe =
        clientArrayMgr_->lastKnown(arrayIndex);
      isCachedArrayMetadata_[arrayIndex] = true;
      destinationPes_[arrayIndex] = destinationPe;
    }
#else
    destinationPe =
      clientArrayMgr_->lastKnown(arrayIndex);
#endif

    if (destinationPe == this->myIndex_) {
      ArrayDataItem<dtype, itype>
        packedDataItem(arrayIndex, this->myIndex_, dataItem);
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
      (destinationBuffer->dataItems)[numDataItems].arrayIndex =
	tempHandle->arrayIndex;
      (destinationBuffer->dataItems)[numDataItems].sourcePe = this->myIndex_;
      (destinationBuffer->dataItems)[numDataItems].dataItem =
	*(tempHandle->dataItem);
      return ++destinationBuffer->numDataItems;
    }
    else {
      // this is an item received along the route to destination
      // we can copy it from the received message
      return MeshStreamer<ArrayDataItem<dtype, itype> >::
	      copyDataItemIntoMessage(destinationBuffer, dataItemHandle);
    }
  }

  // always called on homePE for array element arrayId
  void processLocationRequest(itype arrayId, int deliveredToPe, int sourcePe) {
    int ownerPe = clientArrayMgr_->lastKnown(arrayId);
    this->thisProxy[deliveredToPe].resendMisdeliveredItems(arrayId, ownerPe);
    this->thisProxy[sourcePe].updateLocationAtSource(arrayId, sourcePe);
  }

  void resendMisdeliveredItems(itype arrayId, int destinationPe) {

    clientLocMgr_->updateLocation(arrayId, destinationPe);

    std::vector<ArrayDataItem<dtype, itype> > &bufferedItems
      = misdeliveredItems[arrayId];

    MeshLocation destinationLocation =
      this->determineLocation(destinationPe, this->numDimensions_);
    for (int i = 0; i < bufferedItems.size(); i++) {
      this->storeMessage(destinationPe, destinationLocation, &bufferedItems[i]);
    }

    bufferedItems.clear();
  }

  void updateLocationAtSource(itype arrayId, int destinationPe) {

    int prevOwner = clientArrayMgr_->lastKnown(arrayId);

    if (prevOwner != destinationPe) {
      clientLocMgr_->updateLocation(arrayId, destinationPe);

      // it is possible to also fix destinations of items buffered for arrayId,
      // but the search could be expensive; instead, with the current code
      // the items will be forwarded after being delivered to the previous owner

//    MeshLocation oldLocation = determineLocation(prevOwner, numDimensions_);

//    MeshStreamerMessage<dtype> *messageBuffer = dataBuffers_
//     [oldLocation.dimension][oldLocation.bufferIndex];

//    if (messageBuffer != NULL) {
//      // TODO: find items for arrayId, move them to buffer for destinationPe
//      // do not leave holes in messageBuffer
//    }
    }
  }

};

struct ChunkReceiveBuffer {
  int bufferNumber;
  int receivedChunks;
  char *buffer;
};

struct ChunkOutOfOrderBuffer {
  int bufferNumber;
  int receivedChunks;
  int sourcePe;
  char *buffer;

  ChunkOutOfOrderBuffer(int b, int r, int s, char *buf)
    : bufferNumber(b), receivedChunks(r), sourcePe(s), buffer(buf) {}

  bool operator==(const ChunkDataItem &chunk) {
    return ( (chunk.bufferNumber == bufferNumber) &&
             (chunk.sourcePe == sourcePe) );
  }

};

template <class dtype, class ClientType >
class GroupChunkMeshStreamer
  : public CBase_GroupChunkMeshStreamer<dtype, ClientType> {

private:
  // implementation assumes very few buffers will be received out of order
  // if this is not the case a different data structure may be preferable
  std::list<ChunkOutOfOrderBuffer> outOfOrderBuffers_;
  ChunkReceiveBuffer *lastReceived_;
  int *currentBufferNumbers_;

  CkGroupID clientGID_;
  ClientType *clientObj_;

  bool userHandlesFreeing_;
public:

  GroupChunkMeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
                         int *dimensionSizes, CkGroupID clientGID,
                         bool yieldFlag = 0, double progressPeriodInMs = -1.0,
                         bool userHandlesFreeing = false) {

    this->ctorHelper(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
                     0, yieldFlag, progressPeriodInMs);
    clientGID_ = clientGID;
    clientObj_ = (ClientType *) CkLocalBranch(clientGID_);
    userHandlesFreeing_ = userHandlesFreeing;
    commonInit();
  }

  GroupChunkMeshStreamer(int numDimensions, int *dimensionSizes,
                         CkGroupID clientGID, int bufferSize,
                         bool yieldFlag = 0, double progressPeriodInMs = -1.0,
                         bool userHandlesFreeing = false) {

    this->ctorHelper(0, numDimensions, dimensionSizes,  bufferSize, yieldFlag,
               progressPeriodInMs);
    clientGID_ = clientGID;
    clientObj_ = (ClientType *) CkLocalBranch(clientGID_);
    userHandlesFreeing_ = userHandlesFreeing;
    commonInit();
  }

  inline void commonInit() {
    lastReceived_ = new ChunkReceiveBuffer[this->numMembers_];
    currentBufferNumbers_ = new int[this->numMembers_];
    memset(lastReceived_, 0, this->numMembers_ * sizeof(ChunkReceiveBuffer));
    memset(currentBufferNumbers_, 0, this->numMembers_ * sizeof(int));
  }

  inline void insertData(dtype *dataArray, int numElements, int destinationPe,
                         void *extraData = NULL, int extraDataSize = 0) {

    char *inputData = (char *) dataArray;
    int arraySizeInBytes = numElements * sizeof(dtype);
    int totalSizeInBytes = arraySizeInBytes + extraDataSize;
    ChunkDataItem chunk;
    int offset;
    int chunkNumber = 0;
    chunk.bufferNumber = currentBufferNumbers_[destinationPe]++;
    chunk.sourcePe = this->myIndex_;
    chunk.chunkNumber = 0;
    chunk.chunkSize = CHUNK_SIZE;
    chunk.numChunks =  (int) ceil ( (float) totalSizeInBytes / CHUNK_SIZE);
    chunk.numItems = numElements;

    // loop over full chunks - handle leftovers and extra data later
    for (offset = 0; offset < arraySizeInBytes - CHUNK_SIZE;
         offset += CHUNK_SIZE) {
        memcpy(chunk.rawData, inputData + offset, CHUNK_SIZE);
        MeshStreamer<ChunkDataItem>::insertData(chunk, destinationPe);
        chunk.chunkNumber++;
    }

    // final (possibly incomplete) array chunk
    chunk.chunkSize = arraySizeInBytes - offset;
    memset(chunk.rawData, 0, CHUNK_SIZE);
    memcpy(chunk.rawData, inputData + offset, chunk.chunkSize);

    // extra data (place in last chunk if possible)
    int remainingToSend = extraDataSize;
    int tempOffset = chunk.chunkSize;
    int extraOffset = 0;
    do {
      chunk.chunkSize = std::min(tempOffset + remainingToSend, CHUNK_SIZE);
      memcpy(chunk.rawData + tempOffset, (char *) extraData + extraOffset,
             chunk.chunkSize - tempOffset);

      MeshStreamer<ChunkDataItem>::insertData(chunk, destinationPe);
      chunk.chunkNumber++;
      offset += CHUNK_SIZE;
      extraOffset += (chunk.chunkSize - tempOffset);
      remainingToSend -= (chunk.chunkSize - tempOffset);
      tempOffset = 0;
    } while (offset < totalSizeInBytes);

  }

  inline void processChunk(const ChunkDataItem& chunk) {

    ChunkReceiveBuffer &last = lastReceived_[chunk.sourcePe];

    if (last.buffer == NULL) {
      if (outOfOrderBuffers_.size() == 0) {
        // make common case fast
        last.buffer = new char[chunk.numChunks * CHUNK_SIZE];
        last.receivedChunks = 0;
      }
      else {
        // check if chunks for this buffer have been received previously
        std::list<ChunkOutOfOrderBuffer>::iterator storedBuffer =
          find(outOfOrderBuffers_.begin(), outOfOrderBuffers_.end(), chunk);
        if (storedBuffer != outOfOrderBuffers_.end()) {
          last.buffer = storedBuffer->buffer;
          last.receivedChunks = storedBuffer->receivedChunks;
          outOfOrderBuffers_.erase(storedBuffer);
        }
        else {
          last.buffer = new char[chunk.numChunks * CHUNK_SIZE];
          last.receivedChunks = 0;
        }
      }
      last.bufferNumber = chunk.bufferNumber;
    }
    else if (last.bufferNumber != chunk.bufferNumber) {
      // add last to list of out of order buffers
      ChunkOutOfOrderBuffer lastOutOfOrderBuffer(last.bufferNumber,
                                                 last.receivedChunks,
                                                 chunk.sourcePe, last.buffer);
      outOfOrderBuffers_.push_front(lastOutOfOrderBuffer);

      //search through list of out of order buffers for this chunk's buffer
      std::list<ChunkOutOfOrderBuffer >::iterator storedBuffer =
        find(outOfOrderBuffers_.begin(), outOfOrderBuffers_.end(), chunk);

      if (storedBuffer == outOfOrderBuffers_.end() ) {
        // allocate new buffer
        last.bufferNumber = chunk.bufferNumber;
        last.receivedChunks = 0;
        last.buffer = new char[chunk.numChunks * CHUNK_SIZE];
      }
      else {
        // use existing buffer
        last.bufferNumber = storedBuffer->bufferNumber;
        last.receivedChunks = storedBuffer->receivedChunks;
        last.buffer = storedBuffer->buffer;
        outOfOrderBuffers_.erase(storedBuffer);
      }
    }

    char *receiveBuffer = last.buffer;

    memcpy(receiveBuffer + chunk.chunkNumber * CHUNK_SIZE,
           chunk.rawData, chunk.chunkSize);
    if (++last.receivedChunks == chunk.numChunks) {
      clientObj_->receiveArray(
                  (dtype *) receiveBuffer, chunk.numItems, chunk.sourcePe);
      last.receivedChunks = 0;
      if (!userHandlesFreeing_) {
        delete [] last.buffer;
      }
      last.buffer = NULL;
    }

  }

  inline void localDeliver(const ChunkDataItem& chunk) {
    processChunk(chunk);
    if (this->useCompletionDetection_) {
      this->detectorLocalObj_->consume();
    }
    QdProcess(1);
  }

  void receiveAtDestination(
       MeshStreamerMessage<ChunkDataItem> *msg) {

    for (int i = 0; i < msg->numDataItems; i++) {
      const ChunkDataItem& chunk = msg->getDataItem(i);
      processChunk(chunk);
    }

    if (this->useStagedCompletion_) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n",
               this->myIndex_, env->getSrcPe(), msg->numDataItems,
               msg->finalMsgCount);
#endif
      this->markMessageReceived(msg->dimension, msg->finalMsgCount);
    }
    else if (this->useCompletionDetection_){
      this->detectorLocalObj_->consume(msg->numDataItems);
    }
    QdProcess(msg->numDataItems);
    delete msg;

  }

  inline void localBroadcast(const ChunkDataItem& dataItem) {
    localDeliver(dataItem);
  }

  inline void initLocalClients() {
    // no action required
  }

};



#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
