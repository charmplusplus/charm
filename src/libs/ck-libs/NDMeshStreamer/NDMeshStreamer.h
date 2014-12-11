#ifndef NDMESH_STREAMER_H
#define NDMESH_STREAMER_H

#include <algorithm>
#include <vector>
#include <list>
#include <map>
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"
#include "completion.h"
#include "ckarray.h"
#include "VirtualRouter.h"

// limit total number of buffered data items to
// maxNumDataItemsBuffered_ (flush when limit is reached) but allow
// allocation of up to a factor of CMK_TRAM_OVERALLOCATION_FACTOR more space to
// take advantage of nonuniform filling of buffers
#define CMK_TRAM_OVERALLOCATION_FACTOR 4

// #define CMK_TRAM_CACHE_ARRAY_METADATA // only works for 1D array clients
// #define CMK_TRAM_VERBOSE_OUTPUT

#define TRAM_BROADCAST (-100)

extern void QdCreate(int n);
extern void QdProcess(int n);

template<class dtype>
class MeshStreamerMessage : public CMessage_MeshStreamerMessage<dtype> {

public:

  int finalMsgCount;
  int msgType;
  int numDataItems;
  int *destinationPes;
  dtype *dataItems;

  MeshStreamerMessage(int t): numDataItems(0), msgType(t) {
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

template <class dtype, class RouterType>
class MeshStreamer : public CBase_MeshStreamer<dtype, RouterType> {

private:
  int bufferSize_;
  int maxNumDataItemsBuffered_;
  int numDataItemsBuffered_;

  CkCallback userCallback_;
  bool yieldFlag_;

  double progressPeriodInMs_;
  bool isPeriodicFlushEnabled_;
  bool hasSentRecently_;
  std::vector<std::vector<MeshStreamerMessage<dtype> * > > dataBuffers_;

  CProxy_CompletionDetector detector_;
  int prio_;
  int yieldCount_;

  // only used for staged completion
  std::vector<std::vector<int> > cntMsgSent_;
  std::vector<int> cntMsgReceived_;
  std::vector<int> cntMsgExpected_;
  std::vector<int> cntFinished_;

  int numLocalDone_;
  int numLocalContributors_;
  CompletionStatus myCompletionStatus_;

  virtual void localDeliver(const dtype& dataItem) = 0;
  virtual void localBroadcast(const dtype& dataItem) = 0;

  virtual void initLocalClients() = 0;

  void sendLargestBuffer();
  void flushToIntermediateDestinations();
  void flushDimension(int dimension, bool sendMsgCounts = false);

protected:

  RouterType myRouter_;
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
  void storeMessage(int destinationPe,
                    const Route& destinationCoordinates,
                    const void *dataItem, bool copyIndirectly = false);

  void ctorHelper(int maxNumDataItemsBuffered, int numDimensions,
                  int *dimensionSizes, int bufferSize,
                  bool yieldFlag, double progressPeriodInMs);

public:

  MeshStreamer() {}
  MeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
               int *dimensionSizes, int bufferSize,
               bool yieldFlag = 0, double progressPeriodInMs = -1.0);

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

  void syncInit();

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
    myCompletionStatus_.stageIndex = initialCompletionStage;
    myRouter_.updateCompletionProgress(myCompletionStatus_);
    std::vector<int> &pendingFlushes = myCompletionStatus_.dimensionsToFlush;
    for (int i = 0; i < pendingFlushes.size(); i++) {
      flushDimension(pendingFlushes[i], true);
    }
    pendingFlushes.clear();
    checkForCompletedStages();
  }

  inline void markMessageReceived(int msgType, int finalCount) {
    cntMsgReceived_[msgType]++;
    if (finalCount != -1) {
      cntFinished_[msgType]++;
      cntMsgExpected_[msgType] += finalCount;
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      CkPrintf("[%d] received msgType: %d finalCount: %d cntFinished: %d "
               "cntMsgExpected: %d cntMsgReceived: %d\n", myIndex_, msgType,
               finalCount, cntFinished_[msgType], cntMsgExpected_[msgType],
               cntMsgReceived_[msgType]);
#endif
    }
    if (stagedCompletionStarted_) {
      checkForCompletedStages();
    }
  }

  inline void checkForCompletedStages() {
    int &currentStage = myCompletionStatus_.stageIndex;
    while (cntFinished_[currentStage] == myCompletionStatus_.numContributors &&
           cntMsgExpected_[currentStage] == cntMsgReceived_[currentStage]) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      CkPrintf("[%d] stage completion finished stage %d, received contributions"
               " from %d PEs, cntMsgExpected: %d cntMsgReceived: %d\n",
               myIndex_,  myCompletionStatus_.stageIndex,
               cntFinished_[currentStage], cntMsgExpected_[currentStage],
               cntMsgReceived_[currentStage]);
#endif
      myRouter_.updateCompletionProgress(myCompletionStatus_);
      if (myCompletionStatus_.stageIndex == finalCompletionStage) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
        CkPrintf("[%d] All done. Reducing to final callback ...\n", myIndex_);
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
        std::vector<int> &pendingFlushes =
          myCompletionStatus_.dimensionsToFlush;
        for (int i = 0; i < pendingFlushes.size(); i++) {
          flushDimension(pendingFlushes[i], true);
        }
        pendingFlushes.clear();
      }
    }
  }
};

template <class dtype, class RouterType>
MeshStreamer<dtype, RouterType>::
MeshStreamer(int maxNumDataItemsBuffered, int numDimensions,
             int *dimensionSizes, int bufferSize, bool yieldFlag,
             double progressPeriodInMs) {
  ctorHelper(maxNumDataItemsBuffered, numDimensions, dimensionSizes,
             bufferSize, yieldFlag, progressPeriodInMs);
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
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

  myRouter_.initializeRouter(numDimensions_, myIndex_, dimensionSizes);
  int maxNumBuffers = myRouter_.maxNumAllocatedBuffers();

  dataBuffers_.resize(numDimensions_);
  for (int i = 0; i < numDimensions; i++) {
    dataBuffers_[i].assign(myRouter_.numBuffersPerDimension(i),
                           (MeshStreamerMessage<dtype> *) NULL);
  }

  // a bufferSize input of 0 indicates it should be calculated by the library
  if (bufferSize_ == 0) {
    CkAssert(maxNumDataItemsBuffered_ > 0);
    bufferSize_ = CMK_TRAM_OVERALLOCATION_FACTOR * maxNumDataItemsBuffered_
      / maxNumBuffers;
  }
  else {
    maxNumDataItemsBuffered_ = bufferSize_ * maxNumBuffers
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

#ifdef CMK_TRAM_VERBOSE_OUTPUT
  CkPrintf("[%d] Instance initialized. Buffer size: %d, Capacity: %d, "
           "Yield: %d, Flush period: %f, Maximum number of buffers: %d\n",
           myIndex_, bufferSize_, maxNumDataItemsBuffered_, yieldFlag_,
           progressPeriodInMs_, maxNumBuffers);
#endif

  useStagedCompletion_ = false;
  stagedCompletionStarted_ = false;
  useCompletionDetection_ = false;

  yieldCount_ = 0;
  userCallback_ = CkCallback();
  prio_ = -1;

  initLocalClients();

  hasSentRecently_ = false;

}

template <class dtype, class RouterType>
inline int MeshStreamer<dtype, RouterType>::
copyDataItemIntoMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        const void *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->addDataItem(*((const dtype *)dataItemHandle));
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
sendMeshStreamerMessage(MeshStreamerMessage<dtype> *destinationBuffer,
                        int dimension, int destinationIndex) {

  bool personalizedMessage = myRouter_.isMessagePersonalized(dimension);

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

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
storeMessage(int destinationPe, const Route& destinationRoute,
             const void *dataItem, bool copyIndirectly) {

  int dimension = destinationRoute.dimension;
  int bufferIndex = destinationRoute.dimensionIndex;
  std::vector<MeshStreamerMessage<dtype> *> &messageBuffers
    = dataBuffers_[dimension];

  bool personalizedMessage = myRouter_.isMessagePersonalized(dimension);

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    int numDestIndices = bufferSize_;
    // personalized messages do not require destination indices
    if (personalizedMessage) {
      numDestIndices = 0;
    }
    messageBuffers[bufferIndex] =
      new (numDestIndices, bufferSize_, 8 * sizeof(int))
      MeshStreamerMessage<dtype>(myRouter_.determineMsgType(dimension));

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

    sendMeshStreamerMessage(destinationBuffer, dimension,
                            destinationRoute.destinationPe);
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

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::broadcast(const dtype& dataItem) {

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

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
broadcast(const void *dataItemHandle, int dimension, bool copyIndirectly) {

  if (!myRouter_.isBroadcastSupported()) {
    CkAbort("Broadcast is not supported by this virtual routing scheme\n");
  }

  Route destinationRoute;
  destinationRoute.dimension = dimension;

  while (destinationRoute.dimension != -1) {
    for (int i = 0;
         i < myRouter_.numBuffersPerDimension(destinationRoute.dimension);
         i++) {

      if (!myRouter_.isBufferInUse(destinationRoute.dimension, i)) {
        destinationRoute.dimensionIndex = i;
        storeMessage(TRAM_BROADCAST, destinationRoute,
                     dataItemHandle, copyIndirectly);
      }
      // release control to scheduler if requested by the user,
      //   assume caller is threaded entry
      if (yieldFlag_ && ++yieldCount_ == 1024) {
        yieldCount_ = 0;
        CthYield();
      }
    }
    destinationRoute.dimension--;
  }
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
insertData(const void *dataItemHandle, int destinationPe) {

  const static bool copyIndirectly = true;

  Route destinationRoute;
  myRouter_.determineInitialRoute(destinationPe, destinationRoute);
  storeMessage(destinationPe, destinationRoute, dataItemHandle,
               copyIndirectly);
  // release control to scheduler if requested by the user,
  //   assume caller is threaded entry
  if (yieldFlag_ && ++yieldCount_ == 1024) {
    yieldCount_ = 0;
    CthYield();
  }
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
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

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::init(CkCallback startCb, int prio) {

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

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
init(int numLocalContributors, CkCallback startCb, CkCallback endCb, int prio,
     bool usePeriodicFlushing) {

  useStagedCompletion_ = true;
  stagedCompletionStarted_ = false;
  useCompletionDetection_ = false;

  int dimensionsReceiving = myRouter_.numMsgTypes();

  cntMsgSent_.resize(numDimensions_);
  for (int i = 0; i < numDimensions_; i++) {
    cntMsgSent_[i].assign(myRouter_.numBuffersPerDimension(i), 0);
  }
  cntMsgReceived_.assign(dimensionsReceiving, 0);
  cntMsgExpected_.assign(dimensionsReceiving, 0);
  cntFinished_.assign(dimensionsReceiving, 0);

  yieldCount_ = 0;
  userCallback_ = endCb;
  prio_ = prio;
  numLocalDone_ = 0;
  numLocalContributors_ = numLocalContributors;
  initLocalClients();

  hasSentRecently_ = false;
  if (usePeriodicFlushing) {
    enablePeriodicFlushing();
  }

  CkCallback syncInitCb(CkIndex_MeshStreamer<dtype, RouterType>::syncInit(),
                        this->thisProxy);
  this->contribute(syncInitCb);
  this->contribute(startCb);
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
syncInit() {

  if (numLocalContributors_ == 0) {
    startStagedCompletion();
  }

}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
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
  CkCallback flushCb(CkIndex_MeshStreamer<dtype, RouterType>::
                     enablePeriodicFlushing(), this->thisProxy);
  CkCallback finish(CkIndex_MeshStreamer<dtype, RouterType>::finish(),
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

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
init(CkArrayID senderArrayID, CkCallback startCb, CkCallback endCb, int prio,
     bool usePeriodicFlushing) {

  CkArray *senderArrayMgr = senderArrayID.ckLocalBranch();
  int numLocalElements = senderArrayMgr->getLocMgr()->numLocalElements();
  init(numLocalElements, startCb, endCb, prio, usePeriodicFlushing);
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::finish() {

  isPeriodicFlushEnabled_ = false;

  if (!userCallback_.isInvalid()) {
    this->contribute(userCallback_);
    userCallback_ = CkCallback();      // nullify the current callback
  }
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
receiveAlongRoute(MeshStreamerMessage<dtype> *msg) {

  int destinationPe, lastDestinationPe;
  Route destinationRoute;

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
	myRouter_.determineRoute(destinationPe,
				 myRouter_.dimensionReceived(msg->msgType),
				 destinationRoute);
      }
      storeMessage(destinationPe, destinationRoute, &dataItem);
    }
    else /* if (destinationPe == TRAM_BROADCAST) */ {
      localBroadcast(dataItem);
      broadcast(&dataItem, msg->msgType - 1, false);
    }
    lastDestinationPe = destinationPe;
  }

#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received along route from %d %d items finalMsgCount: %d"
               " msgType: %d\n", myIndex_, env->getSrcPe(),
               msg->numDataItems, msg->finalMsgCount, msg->msgType);
#endif

  if (useStagedCompletion_) {
    markMessageReceived(msg->msgType, msg->finalMsgCount);
  }

  delete msg;
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::sendLargestBuffer() {

  int flushDimension, flushIndex, maxSize, destinationIndex;
  MeshStreamerMessage<dtype> *destinationBuffer;

  for (int i = 0; i < numDimensions_; i++) {
    std::vector<MeshStreamerMessage<dtype> *> &messageBuffers = dataBuffers_[i];

    flushDimension = i;
    maxSize = 0;
    for (int j = 0; j < messageBuffers.size(); j++) {
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

      destinationIndex =
        myRouter_.nextPeAlongRoute(flushDimension, flushIndex);
      sendMeshStreamerMessage(destinationBuffer, flushDimension,
                              destinationIndex);

      if (useStagedCompletion_) {
        cntMsgSent_[i][flushIndex]++;
      }

      messageBuffers[flushIndex] = NULL;
    }
  }
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::flushToIntermediateDestinations() {
  for (int i = 0; i < numDimensions_; i++) {
    flushDimension(i);
  }
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
flushDimension(int dimension, bool sendMsgCounts) {

  std::vector<MeshStreamerMessage<dtype> *>
    &messageBuffers = dataBuffers_[dimension];
#ifdef CMK_TRAM_VERBOSE_OUTPUT
  CkPrintf("[%d] flushDimension: %d, num buffered: %d, sendMsgCounts: %d\n",
           myIndex_, dimension, numDataItemsBuffered_, sendMsgCounts);
#endif

  for (int j = 0; j < messageBuffers.size(); j++) {

    if (!myRouter_.isBufferInUse(dimension, j) ||
        (messageBuffers[j] == NULL && !sendMsgCounts)) {
      continue;
    }
    if(messageBuffers[j] == NULL && sendMsgCounts) {
        messageBuffers[j] = new (0, 0, 8 * sizeof(int))
          MeshStreamerMessage<dtype>(myRouter_.determineMsgType(dimension));
        *(int *) CkPriorityPtr(messageBuffers[j]) = prio_;
        CkSetQueueing(messageBuffers[j], CK_QUEUEING_IFIFO);
    }
    else {
      // if not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(messageBuffers[j]);
      env->shrinkUsersize((bufferSize_ - messageBuffers[j]->numDataItems)
                          * sizeof(dtype));
    }

    MeshStreamerMessage<dtype> *destinationBuffer = messageBuffers[j];
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;
    if (useStagedCompletion_) {
      cntMsgSent_[dimension][j]++;
      if (sendMsgCounts) {
        destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
      }
    }
    int destinationIndex = myRouter_.nextPeAlongRoute(dimension, j);
    sendMeshStreamerMessage(destinationBuffer, dimension, destinationIndex);
    messageBuffers[j] = NULL;
  }
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::flushIfIdle(){

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

template <class dtype, class RouterType>
void periodicProgressFunction(void *MeshStreamerObj, double time) {

  MeshStreamer<dtype, RouterType> *properObj =
    static_cast<MeshStreamer<dtype, RouterType>*>(MeshStreamerObj);

  if (properObj->isPeriodicFlushEnabled()) {
    properObj->flushIfIdle();
    properObj->registerPeriodicProgressFunction();
  }
}

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::registerPeriodicProgressFunction() {
  CcdCallFnAfter(periodicProgressFunction<dtype, RouterType>, (void *) this,
                 progressPeriodInMs_);
}

template <class dtype, class ClientType, class RouterType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype, ClientType> >
class GroupMeshStreamer :
  public CBase_GroupMeshStreamer<dtype, ClientType, RouterType, EntryMethod> {
private:

  CkGroupID clientGID_;
  ClientType *clientObj_;

  void receiveAtDestination(MeshStreamerMessage<dtype> *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      const dtype& data = msg->getDataItem(i);
      EntryMethod((char *) &data, clientObj_);
    }

    if (this->useStagedCompletion_) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope *env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d"
               " msgType: %d\n", this->myIndex_, env->getSrcPe(),
               msg->numDataItems, msg->finalMsgCount, msg->msgType);
#endif
      this->markMessageReceived(msg->msgType, msg->finalMsgCount);
    }
    else if (this->useCompletionDetection_){
      this->detectorLocalObj_->consume(msg->numDataItems);
    }
    QdProcess(msg->numDataItems);
    delete msg;
  }

  inline void localDeliver(const dtype& dataItem) {
    EntryMethod((char *) &dataItem, clientObj_);
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


template <class dtype, class ClientType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype,ClientType> >
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
    EntryMethod((char *) dataItem_, clientObj);
  }

};

template <class dtype, class itype, class ClientType, class RouterType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype,ClientType> >
class ArrayMeshStreamer :
  public CBase_ArrayMeshStreamer<dtype, itype, ClientType, RouterType, EntryMethod> {

private:

  CkArrayID clientAID_;
  CkArray *clientArrayMgr_;
  CkLocMgr *clientLocMgr_;
  int numArrayElements_;
  int numLocalArrayElements_;
  std::map<itype, std::vector<ArrayDataItem<dtype, itype> > > misdeliveredItems;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
  std::vector<ClientType *> clientObjs_;
  std::vector<int> destinationPes_;
  std::vector<bool> isCachedArrayMetadata_;
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
    clientObj = (ClientType *) clientArrayMgr_->lookup((CkArrayIndex)arrayId);
#endif

    if (clientObj != NULL) {
      EntryMethod((char *) &packedDataItem.dataItem, clientObj);
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

    LocalBroadcaster<dtype, ClientType, EntryMethod>
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
      numArrayElements_ = (clientArrayMgr_->getNumInitial()).data()[0];
      clientObjs_.resize(numArrayElements_);
      destinationPes_.resize(numArrayElements_);
      isCachedArrayMetadata_.assign(numArrayElements_, false);

      for (int i = 0; i < numArrayElements_; i++) {
        clientObjs_[i] =
          (ClientType*) ( clientArrayMgr_->lookup(CkArrayIndex1D(i)) );
      }
#endif
    }
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
  }

  ArrayMeshStreamer(int numDimensions, int *dimensionSizes,
                    CkArrayID clientAID, int bufferSize, bool yieldFlag = 0,
                    double progressPeriodInMs = -1.0) {

    this->ctorHelper(0, numDimensions, dimensionSizes, bufferSize, yieldFlag,
                     progressPeriodInMs);
    clientAID_ = clientAID;
    clientArrayMgr_ = clientAID_.ckLocalBranch();
    clientLocMgr_ = clientArrayMgr_->getLocMgr();
  }

  void receiveAtDestination(
       MeshStreamerMessage<ArrayDataItem<dtype, itype> > *msg) {

    for (int i = 0; i < msg->numDataItems; i++) {
      const ArrayDataItem<dtype, itype>& packedData = msg->getDataItem(i);
      localDeliver(packedData);
    }
    if (this->useStagedCompletion_) {
      this->markMessageReceived(msg->msgType, msg->finalMsgCount);
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
    ArrayDataItem<dtype, itype> packedDataItem(TRAM_BROADCAST, this->myIndex_,
                                               dataItem);
    localBroadcast(packedDataItem);

    DataItemHandle tempHandle;
    tempHandle.dataItem = &dataItem;
    tempHandle.arrayIndex = TRAM_BROADCAST;

    MeshStreamer<ArrayDataItem<dtype, itype>, RouterType>::
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
      destinationPe = clientArrayMgr_->lastKnown((CkArrayIndex)arrayIndex);
      isCachedArrayMetadata_[arrayIndex] = true;
      destinationPes_[arrayIndex] = destinationPe;
    }
#else
    destinationPe =
      clientArrayMgr_->lastKnown((CkArrayIndex)arrayIndex);
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

    MeshStreamer<ArrayDataItem<dtype, itype>, RouterType>::
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
      return MeshStreamer<ArrayDataItem<dtype, itype>, RouterType>::
        copyDataItemIntoMessage(destinationBuffer, dataItemHandle);
    }
  }

  // always called on homePE for array element arrayId
  void processLocationRequest(itype arrayId, int deliveredToPe, int sourcePe) {
    int ownerPe = clientArrayMgr_->lastKnown((CkArrayIndex)arrayId);
    this->thisProxy[deliveredToPe].resendMisdeliveredItems(arrayId, ownerPe);
    this->thisProxy[sourcePe].updateLocationAtSource(arrayId, sourcePe);
  }

  void resendMisdeliveredItems(itype arrayId, int destinationPe) {

    clientLocMgr_->updateLocation(arrayId, destinationPe);

    std::vector<ArrayDataItem<dtype, itype> > &bufferedItems
      = misdeliveredItems[arrayId];

    Route destinationRoute;
    this->myRouter_.determineInitialRoute(destinationPe, destinationRoute);
    for (int i = 0; i < bufferedItems.size(); i++) {
      this->storeMessage(destinationPe, destinationRoute, &bufferedItems[i]);
    }

    bufferedItems.clear();
  }

  void updateLocationAtSource(itype arrayId, int destinationPe) {

    int prevOwner = clientArrayMgr_->lastKnown((CkArrayIndex)arrayId);

    if (prevOwner != destinationPe) {
      clientLocMgr_->updateLocation(arrayId, destinationPe);

      // it is possible to also fix destinations of items buffered for arrayId,
      // but the search could be expensive; instead, with the current code
      // the items will be forwarded after being delivered to the previous owner

//    Route oldLocation;
//    myRouter_.determineInitialRoute(prevOwner, oldLocation);

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

template <class dtype, class ClientType, class RouterType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype,ClientType> >
class GroupChunkMeshStreamer
  : public CBase_GroupChunkMeshStreamer<dtype, ClientType, RouterType, EntryMethod> {

private:
  // implementation assumes very few buffers will be received out of order
  // if this is not the case a different data structure may be preferable
  std::list<ChunkOutOfOrderBuffer> outOfOrderBuffers_;
  std::vector<ChunkReceiveBuffer> lastReceived_;
  std::vector<int> currentBufferNumbers_;

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
    lastReceived_.resize(this->numMembers_);
    memset(&lastReceived_.front(), 0,
           this->numMembers_ * sizeof(ChunkReceiveBuffer));
    currentBufferNumbers_.assign(this->numMembers_, 0);
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
        MeshStreamer<ChunkDataItem, RouterType>::
          insertData(chunk, destinationPe);
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

      MeshStreamer<ChunkDataItem, RouterType>::insertData(chunk, destinationPe);
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
      this->markMessageReceived(msg->msgType, msg->finalMsgCount);
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

template <typename dtype, typename RouterType>
struct recursive_pup_impl<MeshStreamer<dtype, RouterType>, 1> {
  typedef MeshStreamer<dtype, RouterType> T;
  void operator()(T *obj, PUP::er &p) {
    obj->parent_pup(p);
    obj->T::pup(p);
  }
};

#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
