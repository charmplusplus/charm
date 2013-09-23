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
// allocation of up to a factor of OVERALLOCATION_FACTOR more space to
// take advantage of nonuniform filling of buffers
#define OVERALLOCATION_FACTOR 4

// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH
// #define CACHE_ARRAY_METADATA // only works for 1D array clients
// #define STREAMER_VERBOSE_OUTPUT

#define TRAM_BROADCAST (-100)

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

  int numMembers_;
  int *individualDimensionSizes_;
  int *combinedDimensionSizes_;

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

  virtual void localDeliver(const dtype& dataItem) = 0;
  virtual void localBroadcast(const dtype& dataItem) = 0;

  virtual void initLocalClients() = 0;

  void sendLargestBuffer();
  void flushToIntermediateDestinations();
  void flushDimension(int dimension, bool sendMsgCounts = false);

protected:

  int numDimensions_;
  bool useStagedCompletion_;
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
  void init(int numLocalContributors, CkCallback startCb, CkCallback endCb,
            int prio, bool usePeriodicFlushing);
  void init(int numContributors, CkCallback startCb, CkCallback endCb,
            CProxy_CompletionDetector detector,
            int prio, bool usePeriodicFlushing);
  void init(CkArrayID senderArrayID, CkCallback startCb, CkCallback endCb,
            int prio, bool usePeriodicFlushing);
  void init(CkCallback endCb, int prio);

  void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
  virtual void receiveAtDestination(MeshStreamerMessage<dtype> *msg) = 0;
  void flushIfIdle();
  void finish();

  // non entry
  inline bool isPeriodicFlushEnabled() {
    return isPeriodicFlushEnabled_;
  }
  virtual void insertData(const dtype& dataItem, int destinationPe);
  virtual void broadcast(const dtype& dataItem);

  void sendMeshStreamerMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                               int dimension, int destinationIndex);

  void registerPeriodicProgressFunction();
  // flushing begins only after enablePeriodicFlushing has been invoked
  inline void enablePeriodicFlushing(){
    isPeriodicFlushEnabled_ = true;
    registerPeriodicProgressFunction();
  }

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

  inline bool stagedCompletionStarted() {
    return (useStagedCompletion_ && dimensionToFlush_ != numDimensions_ - 1);
  }

  inline void startStagedCompletion() {

    if (individualDimensionSizes_[dimensionToFlush_] != 1) {
      flushDimension(dimensionToFlush_, true);
    }
    dimensionToFlush_--;

    checkForCompletedStages();
  }

  inline void markMessageReceived(int dimension, int finalCount) {

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

  inline void checkForCompletedStages() {

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
  CkAssert(combinedDimensionSizes_[0] * individualDimensionSizes_[0]== CkNumPes());

  // a bufferSize input of 0 indicates it should be calculated by the library
  if (bufferSize_ == 0) {
    CkAssert(maxNumDataItemsBuffered_ > 0);
    // buffers for dimensions with the
    //   same index as the sender's are not allocated/used
    bufferSize_ = OVERALLOCATION_FACTOR * maxNumDataItemsBuffered_
      / (sumAlongAllDimensions - numDimensions_ + 1);
  }
  else {
    maxNumDataItemsBuffered_ =
      bufferSize_ * (sumAlongAllDimensions - numDimensions_ + 1)
      / OVERALLOCATION_FACTOR;
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
  for (int i = 0; i < numDimensions_; i++) {
    myLocationIndex_[i] = remainder / combinedDimensionSizes_[i];
    remainder -= combinedDimensionSizes_[i] * myLocationIndex_[i];
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
inline MeshLocation MeshStreamer<dtype>::
determineLocation(int destinationPe, int dimensionReceivedAlong) {

#ifdef CACHE_LOCATIONS
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
#ifdef CACHE_LOCATIONS
      cachedLocations_[destinationPe] = destinationLocation;
      isCached_[destinationPe] = true;
#endif
      return destinationLocation;
    }
  }

  CkAbort("Error. MeshStreamer::determineLocation called with destinationPe "
          "equal to sender's PE. This is unexpected and may cause errors.\n");
  // to prevent warnings
  return destinationLocation;
}

template <class dtype>
inline int MeshStreamer<dtype>::
copyDataItemIntoMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        const void *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->addDataItem(*((const dtype *)dataItemHandle));
}

template <class dtype>
inline void MeshStreamer<dtype>::
sendMeshStreamerMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        int dimension, int destinationIndex) {

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
}

template <class dtype>
inline void MeshStreamer<dtype>::
storeMessage(int destinationPe, const MeshLocation& destinationLocation,
             const void *dataItem, bool copyIndirectly) {

  int dimension = destinationLocation.dimension;
  int bufferIndex = destinationLocation.bufferIndex;
  MeshStreamerMessage<dtype> ** messageBuffers = dataBuffers_[dimension];

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    if (dimension == 0) {
      // personalized messages do not require destination indices
      messageBuffers[bufferIndex] =
        new (0, bufferSize_, 8 * sizeof(int))
         MeshStreamerMessage<dtype>(dimension);
    }
    else {
      messageBuffers[bufferIndex] =
        new (bufferSize_, bufferSize_, 8 * sizeof(int))
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
  CkAssert(stagedCompletionStarted() == false);

  // produce and consume once per PE
  if (useCompletionDetection_) {
    detectorLocalObj_->produce(CkNumPes());
  }
  QdCreate(CkNumPes());

  // deliver locally
  localBroadcast(dataItem);

  broadcast(&dataItem, numDimensions_ - 1, copyIndirectly);
}

template <class dtype>
inline void MeshStreamer<dtype>::
broadcast(const void *dataItemHandle, int dimension, bool copyIndirectly) {

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
inline void MeshStreamer<dtype>::
insertData(const dtype& dataItem, int destinationPe) {

  // no data items should be submitted after all local contributors call done
  // and staged completion has begun
  CkAssert(stagedCompletionStarted() == false);

  if (useCompletionDetection_) {
    detectorLocalObj_->produce();
  }
  QdCreate(1);
  if (destinationPe == CkMyPe()) {
    localDeliver(dataItem);
    return;
  }

  insertData((const void *) &dataItem, destinationPe);
}

template <class dtype>
void MeshStreamer<dtype>::init(CkCallback endCb, int prio) {

  useStagedCompletion_ = false;
  useCompletionDetection_ = false;

  yieldCount_ = 0;
  userCallback_ = endCb;
  prio_ = prio;

  initLocalClients();

  hasSentRecently_ = false;
  enablePeriodicFlushing();
}

template <class dtype>
void MeshStreamer<dtype>::
init(int numLocalContributors, CkCallback startCb, CkCallback endCb, int prio,
     bool usePeriodicFlushing) {

  useStagedCompletion_ = true;
  useCompletionDetection_ = false;
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
void MeshStreamer<dtype>::
init(int numContributors, CkCallback startCb, CkCallback endCb,
     CProxy_CompletionDetector detector,
     int prio, bool usePeriodicFlushing) {

  useStagedCompletion_ = false;
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
inline void MeshStreamer<dtype>::sendLargestBuffer() {

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
      env->shrinkUsersize((bufferSize_ - destinationBuffer->numDataItems)
                        * sizeof(dtype));
      numDataItemsBuffered_ -= destinationBuffer->numDataItems;
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
          new (0, 0, 8 * sizeof(int)) MeshStreamerMessage<dtype>(dimension);
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
      env->shrinkUsersize((bufferSize_ - destinationBuffer->numDataItems)
                          * sizeof(dtype));
    }
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;

    if (useStagedCompletion_) {
      cntMsgSent_[dimension][j]++;
      if (sendMsgCounts) {
        destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
      }
    }

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

    if (MeshStreamer<dtype>::useStagedCompletion_) {
#ifdef STREAMER_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n",
               CkMyPe(), env->getSrcPe(), msg->numDataItems,
               msg->finalMsgCount);
#endif
      this->markMessageReceived(msg->dimension, msg->finalMsgCount);
    }
    else if (MeshStreamer<dtype>::useCompletionDetection_){
      this->detectorLocalObj_->consume(msg->numDataItems);
    }
    QdProcess(msg->numDataItems);
    delete msg;
  }

  inline void localDeliver(const dtype& dataItem) {
    clientObj_->process(dataItem);
    if (MeshStreamer<dtype>::useCompletionDetection_) {
      MeshStreamer<dtype>::detectorLocalObj_->consume();
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
#ifdef CACHE_ARRAY_METADATA
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
#ifdef CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = (ClientType *) clientArrayMgr_->lookup(arrayId);
#endif

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
      if (MeshStreamer<ArrayDataItem<dtype, itype> >
           ::useCompletionDetection_) {
        MeshStreamer<ArrayDataItem<dtype, itype> >
         ::detectorLocalObj_->consume();
      }
      QdProcess(1);
    }
    else {
      // array element arrayId is no longer present locally:
      //  buffer the data item and request updated PE index
      //  to be sent to the source and this PE
      if (MeshStreamer<ArrayDataItem<dtype, itype> >
          ::useStagedCompletion_) {
        CkAbort("Using staged completion when array locations"
                " are not guaranteed to be correct is currently"
                " not supported.");
      }
      misdeliveredItems[arrayId].push_back(packedDataItem);
      if (misdeliveredItems[arrayId].size() == 1) {
        int homePe = clientLocMgr_->homePe(arrayId);
        this->thisProxy[homePe].
          processLocationRequest(arrayId, CkMyPe(), packedDataItem.sourcePe);
      }
    }
  }

  inline
  void localBroadcast(const ArrayDataItem<dtype, itype>& packedDataItem) {

    LocalBroadcaster<dtype, ClientType>
      clientIterator(clientArrayMgr_, &packedDataItem.dataItem);
    clientLocMgr_->iterate(clientIterator);

    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useCompletionDetection_) {
        MeshStreamer<ArrayDataItem<dtype, itype> >
         ::detectorLocalObj_->consume();
    }
    QdProcess(1);
  }

  inline void initLocalClients() {

    if (MeshStreamer<ArrayDataItem<dtype, itype> >
         ::useCompletionDetection_) {
#ifdef CACHE_ARRAY_METADATA
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
      this->markMessageReceived(msg->dimension, msg->finalMsgCount);
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
         ::useCompletionDetection_) {
      MeshStreamer<ArrayDataItem<dtype, itype> >
        ::detectorLocalObj_->produce(CkNumPes());
    }
    QdCreate(CkNumPes());

    // deliver locally
    ArrayDataItem<dtype, itype>& packedDataItem(TRAM_BROADCAST, CkMyPe(),
                                                dataItem);
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
         ::useCompletionDetection_) {
      MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->produce();
    }
    QdCreate(1);
    int destinationPe;
#ifdef CACHE_ARRAY_METADATA
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

    if (destinationPe == CkMyPe()) {
      ArrayDataItem<dtype, itype> packedDataItem(arrayIndex, CkMyPe(), dataItem);
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
      (destinationBuffer->dataItems)[numDataItems].sourcePe = CkMyPe();
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
      determineLocation(destinationPe, MeshStreamer
                        <ArrayDataItem<dtype, itype> >::numDimensions_);
    for (int i = 0; i < bufferedItems.size(); i++) {
      storeMessage(destinationPe, destinationLocation, &bufferedItems[i]);
    }

    bufferedItems.clear();
  }

  void updateLocationAtSource(itype arrayId, int destinationPe) {

    int prevOwner = clientArrayMgr_->lastKnown(arrayId);

    if (prevOwner != destinationPe) {
      clientLocMgr_->updateLocation(arrayId, destinationPe);

//    // could also try to correct destinations of items buffered for arrayId,
//    // but it would take significant additional computation, so leaving it out;
//    // the items will get forwarded after being delivered to the previous owner
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
    lastReceived_ = new ChunkReceiveBuffer[CkNumPes()];
    currentBufferNumbers_ = new int[CkNumPes()];
    memset(lastReceived_, 0, CkNumPes() * sizeof(ChunkReceiveBuffer));
    memset(currentBufferNumbers_, 0, CkNumPes() * sizeof(int));
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
    chunk.sourcePe = CkMyPe();
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
#ifdef STREAMER_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d\n",
               CkMyPe(), env->getSrcPe(), msg->numDataItems,
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
