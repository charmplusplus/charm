#ifndef NDMESH_STREAMER_H
#define NDMESH_STREAMER_H

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>
#include <list>
#include <map>
#include <type_traits>
#include "pup.h"
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"
#include "completion.h"
#include "ckarray.h"
#include "VirtualRouter.h"
#include "pup_stl.h"
#include "debug-charm.h"

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
//below code uses templates to generate appropriate TRAM_BROADCAST array index values
template<class itype>
struct TramBroadcastInstance;

template<>
struct TramBroadcastInstance<CkArrayIndex1D>{
  static CkArrayIndex1D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex2D>{
  static CkArrayIndex2D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex3D>{
  static CkArrayIndex3D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex4D>{
  static CkArrayIndex4D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex5D>{
  static CkArrayIndex5D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex6D>{
  static CkArrayIndex6D value;
};

template<>
struct TramBroadcastInstance<CkArrayIndex>{
  static CkArrayIndex& value(int);
};

template <typename T>
struct is_PUPbytes {
  static const bool value = false;
};

template <class dtype>
struct DataItemHandle {
  CkArrayIndex arrayIndex;
  const dtype *dataItem;

  DataItemHandle(dtype* _ptr, CkArrayIndex _idx = CkArrayIndex()) : dataItem(_ptr), arrayIndex(_idx) {}
};

class MeshStreamerMessageV : public CMessage_MeshStreamerMessageV {

public:

  int finalMsgCount;
  int msgType;
  int numDataItems;
  bool fixedSize;
  int *destinationPes;
  int *sourcePes;
  char *dataItems;
  std::uint16_t *offsets;
  CkArrayIndex *destObjects;

  MeshStreamerMessageV(int t, bool isFixedSize): numDataItems(0), msgType(t), fixedSize(isFixedSize) {
    finalMsgCount = -1;
    if (!isFixedSize) {
      offsets[0] = 0;
    }
  }

  template <typename dtype>
  inline typename std::enable_if<is_PUPbytes<dtype>::value,int>::type addDataItem(dtype& dataItem, CkArrayIndex index, int sourcePe) {
    char* offset = dataItems + (numDataItems*sizeof(dtype));
    *reinterpret_cast<dtype*>(offset) = dataItem;
    destObjects[numDataItems]=index;
    sourcePes[numDataItems]=sourcePe;
    return ++numDataItems;
  }
  template <typename dtype>
  inline typename std::enable_if<!is_PUPbytes<dtype>::value,int>::type addDataItem(dtype& dataItem, CkArrayIndex index, int sourcePe) {
    size_t sz=PUP::size(dataItem);
    PUP::toMemBuf(dataItem,dataItems+offsets[numDataItems],sz);
    offsets[numDataItems+1]=offsets[numDataItems]+sz;
    destObjects[numDataItems]=index;
    sourcePes[numDataItems]=sourcePe;
    return ++numDataItems;
  }

  inline int addData(char* data, size_t sz, CkArrayIndex index, int sourcePe) {
    if (!fixedSize) {
      std::memcpy(dataItems+offsets[numDataItems],data,sz);
      offsets[numDataItems+1]=offsets[numDataItems]+sz;
    }
    else {
      char* offset = dataItems+(numDataItems*sz);
      std::memcpy(offset,data,sz);
    }
    destObjects[numDataItems]=index;
    sourcePes[numDataItems]=sourcePe;
    return ++numDataItems;
  }

  inline void markDestination(const int index, const int destinationPe) {
    destinationPes[index] = destinationPe;
  }

  template <typename dtype>
  inline typename std::enable_if<is_PUPbytes<dtype>::value,dtype>::type getDataItem(const int index) {
    char *objptr = dataItems + (numDataItems*sizeof(dtype));
    return *reinterpret_cast<dtype*>(objptr);
  }
  template <typename dtype>
  inline typename std::enable_if<!is_PUPbytes<dtype>::value,dtype>::type getDataItem(const int index) {
    dtype obj;
    size_t sz=offsets[index+1]-offsets[index];
    PUP::fromMemBuf(obj,dataItems+offsets[index],sz);
    return obj;
  }

  template <typename dtype>
  inline size_t getoffset(const std::uint16_t index) {
    if (fixedSize) {
      return sizeof(dtype)*index;
    }
    else {
      return offsets[index];
    }
  }
};

template <class dtype, class RouterType>
class MeshStreamer : public CBase_MeshStreamer<dtype, RouterType> {

private:
  int bufferSize_;
  int maxNumDataItemsBuffered_;
  int numDataItemsBuffered_;
  int maxItemsBuffered;

  CkCallback userCallback_;
  bool yieldFlag_;

  double progressPeriodInMs_;
  bool isPeriodicFlushEnabled_;
  bool hasSentRecently_;
  std::vector<std::vector<MeshStreamerMessageV * > > dataBuffers_;

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
  int tramBufferSize;
  int thresholdFractionNumerator;
  int thresholdFractionDenominator;
  int cutoffFractionNumerator;
  int cutoffFractionDenominator;

  virtual void localDeliver(const char* data, size_t size, CkArrayIndex arrayId,int sourcePe) { CkAbort("Called what should be a pure virtual base method"); }

  virtual void initLocalClients() { CkAbort("Called what should be a pure virtual base method"); }

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
              MeshStreamerMessageV *destinationBuffer,
              const DataItemHandle<dtype> *dataItemHandle, bool copyIndirectly = false);
  virtual int copyDataIntoMessage(
              MeshStreamerMessageV *destinationBuffer,
              char *dataHandle, size_t size, CkArrayIndex index);
  void createDetectors();
  void insertData(const DataItemHandle<dtype> *dataItemHandle, int destinationPe);
  void storeMessageIntermed(int destinationPe,
                    const Route& destinationCoordinates,
                    char *data, size_t size,CkArrayIndex);
  void storeMessage(int destinationPe,
                    const Route& destinationCoordinates,
                    const DataItemHandle<dtype> *dataItem, bool copyIndirectly = false);

  void ctorHelper(int maxNumDataItemsBuffered, int numDimensions,
                  int *dimensionSizes, int bufferSize,
                  bool yieldFlag, double progressPeriodInMs,
                  int mib, int tfn, int tfd,
                  int cfn, int cfd);

public:
  MeshStreamer() {}
  MeshStreamer(CkMigrateMessage *) {}

  // entry

  void receiveAlongRoute(MeshStreamerMessageV *msg);
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

  virtual void receiveAtDestination(MeshStreamerMessageV *msg) { CkAbort("Called what should be a pure virtual base method"); }

  // non entry
  void flushIfIdle();
  inline bool isPeriodicFlushEnabled() {
    return isPeriodicFlushEnabled_;
  }

  void sendMeshStreamerMessage(MeshStreamerMessageV *destinationBuffer,
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
    if (finalCount >= 0) {
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
  inline bool checkAllStagesCompleted() {
    //checks if all stages have been completed
    //if so, it resets the periodic flushing
    if (myCompletionStatus_.stageIndex == finalCompletionStage) { //has already completed all stages
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      CkPrintf("[%d] All done. Reducing to final callback ...\n", myIndex_);
#endif
      CkAssert(numDataItemsBuffered_ == 0);
      isPeriodicFlushEnabled_ = false;
      if (!userCallback_.isInvalid()) {
        this->contribute(userCallback_);
        userCallback_ = CkCallback();
      }
      return true;
    }
    else {
      return false;
    }
  }

  inline void checkForCompletedStages() {
    int &currentStage = myCompletionStatus_.stageIndex;
    if (checkAllStagesCompleted()) { //has already completed all stages
      return;
    }
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
      if (checkAllStagesCompleted()) { //has already completed all stages
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

  virtual void pup(PUP::er &p);
};

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::
ctorHelper(int maxNumDataItemsBuffered, int numDimensions,
           int *dimensionSizes, int bufferSize,
           bool yieldFlag, double progressPeriodInMs,
           int mib, int tfn, int tfd, int cfn, int cfd) {

  numDimensions_ = numDimensions;
  maxNumDataItemsBuffered_ = maxNumDataItemsBuffered;
  yieldFlag_ = yieldFlag;
  progressPeriodInMs_ = progressPeriodInMs;
  bufferSize_ = bufferSize;
  numDataItemsBuffered_ = 0;
  numMembers_ = CkNumPes();
  myIndex_ = CkMyPe();
  maxItemsBuffered = mib;
  thresholdFractionNumerator = tfn;
  thresholdFractionDenominator = tfd;
  cutoffFractionNumerator = cfn;
  cutoffFractionDenominator = cfd;

  myRouter_.initializeRouter(numDimensions_, myIndex_, dimensionSizes);
  int maxNumBuffers = myRouter_.maxNumAllocatedBuffers();

  dataBuffers_.resize(numDimensions_);
  for (int i = 0; i < numDimensions; i++) {
    dataBuffers_[i].assign(myRouter_.numBuffersPerDimension(i),
                           (MeshStreamerMessageV *) NULL);
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
copyDataItemIntoMessage(MeshStreamerMessageV *destinationBuffer,
                        const DataItemHandle<dtype> *dataItemHandle, bool copyIndirectly) {
  return destinationBuffer->template addDataItem<dtype>(const_cast<dtype&>(*(dataItemHandle->dataItem)), dataItemHandle->arrayIndex,this->myIndex_);
}

template <class dtype, class RouterType>
inline int MeshStreamer<dtype, RouterType>::
copyDataIntoMessage(MeshStreamerMessageV *destinationBuffer,
                        char *dataHandle, size_t size, CkArrayIndex index) {
  return destinationBuffer->addData(dataHandle, size, index,this->myIndex_);
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
sendMeshStreamerMessage(MeshStreamerMessageV *destinationBuffer,
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
storeMessageIntermed(int destinationPe, const Route& destinationRoute,
                 char *dataItem, size_t size,CkArrayIndex arrayId) {
  int dimension = destinationRoute.dimension;
  int bufferIndex = destinationRoute.dimensionIndex;
  std::vector<MeshStreamerMessageV *> &messageBuffers
    = dataBuffers_[dimension];

  bool personalizedMessage = myRouter_.isMessagePersonalized(dimension);

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    int numDestIndices = maxItemsBuffered;
    // personalized messages do not require destination indices
    if (personalizedMessage) {
      numDestIndices = 0;
    }
    if (!is_PUPbytes<dtype>::value) {
      messageBuffers[bufferIndex] =
        new (numDestIndices, numDestIndices, numDestIndices+1, numDestIndices, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }
    else {
      messageBuffers[bufferIndex] =
        new (numDestIndices, numDestIndices, 0, numDestIndices, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }

    *(int *) CkPriorityPtr(messageBuffers[bufferIndex]) = prio_;
    CkSetQueueing(messageBuffers[bufferIndex], CK_QUEUEING_IFIFO);
    CkAssert(messageBuffers[bufferIndex] != NULL);
  }

  MeshStreamerMessageV *destinationBuffer = messageBuffers[bufferIndex];
  int numBuffered =
    copyDataIntoMessage(destinationBuffer, dataItem, size, arrayId);
  if (!personalizedMessage) {
    destinationBuffer->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;

  // send if buffer is full
  if (numBuffered == maxItemsBuffered || destinationBuffer->template getoffset<dtype>(destinationBuffer->numDataItems)
      > (thresholdFractionNumerator*(bufferSize_/thresholdFractionDenominator))) {

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
inline void MeshStreamer<dtype, RouterType>::
storeMessage(int destinationPe, const Route& destinationRoute,
             const DataItemHandle<dtype> *dataItem, bool copyIndirectly) {
  int dimension = destinationRoute.dimension;
  int bufferIndex = destinationRoute.dimensionIndex;
  std::vector<MeshStreamerMessageV *> &messageBuffers
    = dataBuffers_[dimension];

  bool personalizedMessage = myRouter_.isMessagePersonalized(dimension);
  if (PUP::size(const_cast<dtype&>(*(dataItem->dataItem))) > (cutoffFractionNumerator*(bufferSize_/cutoffFractionDenominator))) {
    MeshStreamerMessageV* msg;
    if (!is_PUPbytes<dtype>::value) {
      msg =
        new (1, 1, 2, 1, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }
    else {
      msg =
        new (1, 1, 0, 1, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }
    *(int *) CkPriorityPtr(msg) = prio_;
    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
    copyDataItemIntoMessage(msg,dataItem,copyIndirectly);
    this->thisProxy[destinationPe].receiveAtDestination(msg);
    return;
  }

  // allocate new message if necessary
  if (messageBuffers[bufferIndex] == NULL) {
    int numDestIndices = maxItemsBuffered;
    // personalized messages do not require destination indices
    if (personalizedMessage) {
      numDestIndices = 0;
    }
    if (!is_PUPbytes<dtype>::value) {
      messageBuffers[bufferIndex] =
        new (numDestIndices, numDestIndices, numDestIndices+1, numDestIndices, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }
    else {
      messageBuffers[bufferIndex] =
        new (numDestIndices, numDestIndices, 0, numDestIndices, bufferSize_, 8 * sizeof(int))
        MeshStreamerMessageV(myRouter_.determineMsgType(dimension),is_PUPbytes<dtype>::value);
    }

    *(int *) CkPriorityPtr(messageBuffers[bufferIndex]) = prio_;
    CkSetQueueing(messageBuffers[bufferIndex], CK_QUEUEING_IFIFO);
    CkAssert(messageBuffers[bufferIndex] != NULL);
  }

  MeshStreamerMessageV *destinationBuffer = messageBuffers[bufferIndex];
  int numBuffered =
    copyDataItemIntoMessage(destinationBuffer, dataItem, copyIndirectly);
  if (!personalizedMessage) {
    destinationBuffer->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;
  if (numBuffered == maxItemsBuffered || destinationBuffer->template getoffset<dtype>(destinationBuffer->numDataItems)
      >= (cutoffFractionNumerator*(bufferSize_/cutoffFractionDenominator))) {
    // send if buffer is full
    //record number of data items sent here
    sendMeshStreamerMessage(destinationBuffer, dimension,
                            destinationRoute.destinationPe);
    if (useStagedCompletion_) {
      cntMsgSent_[dimension][bufferIndex]++;
    }
    messageBuffers[bufferIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered;
    hasSentRecently_ = true;
  }
  else if (numDataItemsBuffered_ == maxNumDataItemsBuffered_) {
    // send if total buffering capacity has been reached
    sendLargestBuffer();
    hasSentRecently_ = true;
  }
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::createDetectors() {
  // No data items should be submitted when staged completion has begun
  CkAssert(stagedCompletionStarted_ == false);

  // Increment completion detection and quiescence detection
  if (useCompletionDetection_) detectorLocalObj_->produce();
  QdCreate(1);
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::
insertData(const DataItemHandle<dtype> *dataItemHandle, int destinationPe) {
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

  detector_[CkMyPe()].start_detection(numContributors, startCb, flushCb,
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
receiveAlongRoute(MeshStreamerMessageV *msg) {

  int destinationPe, lastDestinationPe;
  Route destinationRoute;

  lastDestinationPe = -1;
  for (int i = 0; i < msg->numDataItems; i++) {
    destinationPe = msg->destinationPes[i];
    if (destinationPe == myIndex_) {
      //dtype dataItem = msg->getDataItem<dtype>(i);
      localDeliver(msg->dataItems+msg->template getoffset<dtype>(i),
                   msg->template getoffset<dtype>(i+1)-msg->template getoffset<dtype>(i),
                   msg->destObjects[i], msg->sourcePes[i]);
    }
    else if (destinationPe != TRAM_BROADCAST) {
      if (destinationPe != lastDestinationPe) {
        // do this once per sequence of items with the same destination
	myRouter_.determineRoute(destinationPe,
				 myRouter_.dimensionReceived(msg->msgType),
				 destinationRoute);
      }
      storeMessageIntermed(destinationPe, destinationRoute,
                       msg->dataItems + msg->template getoffset<dtype>(i),
                       msg->template getoffset<dtype>(i+1)-msg->template getoffset<dtype>(i),
                       msg->destObjects[i]);
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
    if (msg->finalMsgCount != -2) {
      markMessageReceived(msg->msgType, msg->finalMsgCount);
    }
#if !CMK_MULTICORE
    else if (stagedCompletionStarted_) {
      checkForCompletedStages();
    }
#endif
  }

  delete msg;
}

template <class dtype, class RouterType>
inline void MeshStreamer<dtype, RouterType>::sendLargestBuffer() {

  int flushDimension, flushIndex, maxSize, destinationIndex;
  MeshStreamerMessageV *destinationBuffer;

  for (int i = 0; i < numDimensions_; i++) {
    std::vector<MeshStreamerMessageV *> &messageBuffers = dataBuffers_[i];

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
      //env->shrinkUsersize((bufferSize_ -
      //destinationBuffer->template getoffset<dtype>(destinationBuffer->numDataItems)));
      numDataItemsBuffered_ -= destinationBuffer->numDataItems;

      destinationIndex =
        myRouter_.nextPeAlongRoute(flushDimension, flushIndex);
      
      if (destinationIndex == myIndex_) {
        destinationBuffer->finalMsgCount = -2;
      }
      
      sendMeshStreamerMessage(destinationBuffer, flushDimension,
        destinationIndex);

      if (useStagedCompletion_ && destinationIndex != myIndex_) {
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

  std::vector<MeshStreamerMessageV *>
    &messageBuffers = dataBuffers_[dimension];
#ifdef CMK_TRAM_VERBOSE_OUTPUT
  CkPrintf("[%d] flushDimension: %d, num buffered: %d, sendMsgCounts: %d\n",
           myIndex_, dimension, numDataItemsBuffered_, sendMsgCounts);
#endif

  for (int j = 0; j < messageBuffers.size(); j++) {

    if (messageBuffers[j] == NULL && !sendMsgCounts) {
      continue;
    }
    if(messageBuffers[j] == NULL && sendMsgCounts) {
        messageBuffers[j] = new (0, 0, 1, 0, 0, 8 * sizeof(int))
          MeshStreamerMessageV(myRouter_.determineMsgType(dimension), is_PUPbytes<dtype>::value);
        *(int *) CkPriorityPtr(messageBuffers[j]) = prio_;
        CkSetQueueing(messageBuffers[j], CK_QUEUEING_IFIFO);
    }
    else {
      // if not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(messageBuffers[j]);
      //const UInt s = (bufferSize_ - messageBuffers[j]->numDataItems) * sizeof(dtype);
      //const UInt s = (bufferSize_ - messageBuffers[j]->template getoffset<dtype>(messageBuffers[j]->numDataItems));
      //if (env->getUsersize() > s) {
      //  env->shrinkUsersize(s);
      //}
    }
    
    MeshStreamerMessageV *destinationBuffer = messageBuffers[j];
    int destinationIndex = myRouter_.nextPeAlongRoute(dimension, j);
    numDataItemsBuffered_ -= destinationBuffer->numDataItems;
    if (useStagedCompletion_) {
      if (destinationIndex == myIndex_) {
        destinationBuffer->finalMsgCount = -2;
      } else {
        cntMsgSent_[dimension][j]++;
        if (sendMsgCounts) {
          destinationBuffer->finalMsgCount = cntMsgSent_[dimension][j];
        }
      }
      CkAssert(!sendMsgCounts || destinationBuffer->finalMsgCount != -1);
    }
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

template <class dtype, class RouterType>
void MeshStreamer<dtype, RouterType>::pup(PUP::er &p) {
  // private members
  p|bufferSize_;
  p|maxNumDataItemsBuffered_;
  p|numDataItemsBuffered_;
  p|maxItemsBuffered;

  p|userCallback_;
  p|yieldFlag_;

  p|progressPeriodInMs_;
  p|isPeriodicFlushEnabled_;
  p|hasSentRecently_;

  p|detector_;
  p|prio_;
  p|yieldCount_;

  // only used for staged completion
  p|cntMsgSent_;
  p|cntMsgReceived_;
  p|cntMsgExpected_;
  p|cntFinished_;

  p|numLocalDone_;
  p|numLocalContributors_;
  p|myCompletionStatus_;

  // protected members
  p|myRouter_;
  p|numMembers_;
  p|myIndex_;
  p|numDimensions_;
  p|useStagedCompletion_;
  p|stagedCompletionStarted_;
  p|useCompletionDetection_;
  if (p.isUnpacking()) detectorLocalObj_ = detector_.ckLocalBranch();

  size_t outervec_size;
  std::vector<size_t> innervec_sizes;

  if (p.isPacking()) {
    outervec_size = dataBuffers_.size();
    for (int i = 0; i < outervec_size; i++) {
      innervec_sizes.push_back(dataBuffers_[i].size());
    }
  }

  p|outervec_size;
  p|innervec_sizes;

  if (p.isUnpacking()) {
    dataBuffers_.resize(outervec_size);
    for (int i = 0; i < outervec_size; i++) {
      dataBuffers_[i].resize(innervec_sizes[i]);
    }
  }

  // pup each message element
  for (int i = 0; i < outervec_size; i++) {
    for (int j = 0; j < innervec_sizes[i]; j++) {
      CkPupMessage(p, (void**) &dataBuffers_[i][j]);
    }
  }

}

template <class dtype, class ClientType, class RouterType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype, ClientType> >
class GroupMeshStreamer :
  public CBase_GroupMeshStreamer<dtype, ClientType, RouterType, EntryMethod> {
private:
  CkGroupID clientGID_;
  ClientType *clientObj_;

  void receiveAtDestination(MeshStreamerMessageV* msg) override {
    for (int i = 0; i < msg->numDataItems; i++) {
      EntryMethod(msg->dataItems + msg->getoffset<dtype>(i), clientObj_);
    }

    if (this->useStagedCompletion_) {
#ifdef CMK_TRAM_VERBOSE_OUTPUT
      envelope* env = UsrToEnv(msg);
      CkPrintf("[%d] received at dest from %d %d items finalMsgCount: %d"
               " msgType: %d\n", this->myIndex_, env->getSrcPe(),
               msg->numDataItems, msg->finalMsgCount, msg->msgType);
#endif
      this->markMessageReceived(msg->msgType, msg->finalMsgCount);
    } else if (this->useCompletionDetection_) {
      this->detectorLocalObj_->consume(msg->numDataItems);
    }
    QdProcess(msg->numDataItems);
    delete msg;
  }

  inline void localDeliver(const char* data, size_t size, CkArrayIndex arrayId,
      int sourcePe) override {
    EntryMethod(const_cast<char*>(data), clientObj_);
    if (this->useCompletionDetection_) {
      this->detectorLocalObj_->consume();
    }
    QdProcess(1);
  }

  inline void initLocalClients() override {
    // No action required
  }

public:
  GroupMeshStreamer(int numDimensions, int* dimensionSizes,
      CkGroupID clientGID, int bufferSize, bool yieldFlag,
      double progressPeriodInMs, int maxItemsBuffered,
      int _thresholdFractionNum, int _thresholdFractionDen,
      int _cutoffFractionNum, int _cutoffFractionDen) {
    this->ctorHelper(0, numDimensions, dimensionSizes, bufferSize, yieldFlag,
        progressPeriodInMs, maxItemsBuffered, _thresholdFractionNum,
        _thresholdFractionDen, _cutoffFractionNum, _cutoffFractionDen);
    clientGID_ = clientGID;
    clientObj_ = (ClientType*)CkLocalBranch(clientGID_);
  }

  GroupMeshStreamer(CkMigrateMessage*) {}

  inline void insertData(const dtype& dataItem, int destinationPe) {
    this->createDetectors();

    DataItemHandle<dtype> tempHandle(const_cast<dtype*>(&dataItem));
    MeshStreamer<dtype, RouterType>::insertData(&tempHandle, destinationPe);
  }

  void pup(PUP::er& p) override {
    p|clientGID_;
    if (p.isUnpacking()) {
      clientObj_ = (ClientType*)CkLocalBranch(clientGID_);
    }
  }
};

template <class dtype, class ClientType, class RouterType, int (*EntryMethod)(char *, void *) = defaultMeshStreamerDeliver<dtype,ClientType> >
class ArrayMeshStreamer :
  public CBase_ArrayMeshStreamer<dtype, ClientType, RouterType, EntryMethod> {

private:

  CkArrayID clientAID_;
  CkArray *clientArrayMgr_;
  CkLocMgr *clientLocMgr_;
  int numArrayElements_;
  int numLocalArrayElements_;
  std::map<CkArrayIndex, std::vector<dtype>> misdeliveredItems;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
  std::vector<ClientType *> clientObjs_;
  std::vector<int> destinationPes_;
  std::vector<bool> isCachedArrayMetadata_;
#endif
  int bufferSize;
  int thresholdFractionNum;
  int thresholdFractionDen;
  int cutoffFractionNum;
  int cutoffFractionDen;

  inline
  void localDeliver(const char* data,size_t size,CkArrayIndex arrayId, int sourcePe) override {
    ClientType *clientObj;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = (ClientType *) clientArrayMgr_->lookup((CkArrayIndex)arrayId);
#endif

    if (clientObj != NULL) {
      EntryMethod(const_cast<char*>(data), clientObj);
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
      if (!is_PUPbytes<dtype>::value) {
        dtype dataItem;
        PUP::fromMemBuf(dataItem,(void*)data,size);
        misdeliveredItems[arrayId].push_back(dataItem);
      }
      else {
        misdeliveredItems[arrayId].push_back(const_cast<dtype&>(*reinterpret_cast<const dtype*>(data)));
      }
      if (misdeliveredItems[arrayId].size() == 1) {
        int homePe = clientLocMgr_->homePe(arrayId);
        this->thisProxy[homePe].
          processLocationRequest(arrayId, this->myIndex_,
                                 sourcePe);
      }
    }
  }


  inline void initLocalClients() override {

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

  ArrayMeshStreamer(int numDimensions, int *dimensionSizes,
                    CkArrayID clientAID, int bufferSize, bool yieldFlag,
                    double progressPeriodInMs, int maxItemsBuffered,
                    int _thresholdFractionNum, int _thresholdFractionDen,
                    int _cutoffFractionNum, int _cutoffFractionDen) {

    this->ctorHelper(0, numDimensions, dimensionSizes, bufferSize, yieldFlag,
                     progressPeriodInMs, maxItemsBuffered, _thresholdFractionNum,
                     _thresholdFractionDen, _cutoffFractionNum,
                     _cutoffFractionDen);
    clientAID_ = clientAID;
    clientArrayMgr_ = clientAID_.ckLocalBranch();
    clientLocMgr_ = clientArrayMgr_->getLocMgr();
  }

  ArrayMeshStreamer(CkMigrateMessage *) {}

  void receiveAtDestination(
       MeshStreamerMessageV *msg) override {
    for (int i = 0; i < msg->numDataItems; i++) {
      //const ArrayDataItem<dtype, itype> packedData = msg->getDataItem<ArrayDataItem<dtype, itype>>(i);
      this->localDeliver(msg->dataItems+msg->template getoffset<dtype>(i),msg->template getoffset<dtype>(i+1)-msg->template getoffset<dtype>(i),msg->destObjects[i],msg->sourcePes[i]);
    }
    if (this->useStagedCompletion_) {
      this->markMessageReceived(msg->msgType, msg->finalMsgCount);
    }

    delete msg;
  }
  template <bool deliverInline = false>
  inline void insertData(const dtype& dataItem, CkArrayIndex arrayIndex) {
    this->createDetectors();

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

    if (deliverInline && destinationPe == this->myIndex_) {
      size_t sz = PUP::size(const_cast<dtype&>(dataItem));
      char* data = new char[sz];
      PUP::toMemBuf(const_cast<dtype&>(dataItem),data, sz);
      localDeliver(data,sz,arrayIndex,this->myIndex_);
      delete[] data;
      return;
    }

    // this implementation avoids copying an item before transfer into message
    DataItemHandle<dtype> tempHandle(const_cast<dtype*>(&dataItem), arrayIndex);

    MeshStreamer<dtype, RouterType>::
      insertData(&tempHandle, destinationPe);

  }

  inline int copyDataIntoMessage(

      MeshStreamerMessageV *destinationBuffer, //ArrayDataItem<dtype, itype>
      char *dataHandle, size_t size) {

      return MeshStreamer<dtype, RouterType>::
        copyDataIntoMessage(destinationBuffer, dataHandle, size);
  }

  inline int copyDataItemIntoMessage(

      MeshStreamerMessageV *destinationBuffer, //ArrayDataItem<dtype, itype>
      const DataItemHandle<dtype> *dataItemHandle, bool copyIndirectly) override {

    if (copyIndirectly == true) {
      // newly inserted items are passed through a handle to avoid copying
      int numDataItems = destinationBuffer->numDataItems;
      return destinationBuffer->template addDataItem<dtype>(const_cast<dtype&>(*(dataItemHandle->dataItem)),
          dataItemHandle->arrayIndex,this->myIndex_);
    }
    else {
      // this is an item received along the route to destination
      // we can copy it from the received message
      return MeshStreamer<dtype, RouterType>::
        copyDataItemIntoMessage(destinationBuffer, dataItemHandle);
    }
  }

  // always called on homePE for array element arrayId
  void processLocationRequest(CkArrayIndex arrayId, int deliveredToPe, int sourcePe) {
    int ownerPe = clientArrayMgr_->lastKnown((CkArrayIndex)arrayId);
    this->thisProxy[deliveredToPe].resendMisdeliveredItems(arrayId, ownerPe);
    this->thisProxy[sourcePe].updateLocationAtSource(arrayId, ownerPe);
  }

  void resendMisdeliveredItems(CkArrayIndex arrayId, int destinationPe) {

    clientLocMgr_->updateLocation(arrayId, clientLocMgr_->lookupID(arrayId),destinationPe);

    std::vector<dtype > &bufferedItems
      = misdeliveredItems[arrayId];

    Route destinationRoute;
    this->myRouter_.determineInitialRoute(destinationPe, destinationRoute);
    for (int i = 0; i < bufferedItems.size(); i++) {
      DataItemHandle<dtype> temporary(&bufferedItems[i], arrayId);
      this->storeMessage(destinationPe, destinationRoute, &temporary);
    }

    bufferedItems.clear();
  }

  void updateLocationAtSource(CkArrayIndex arrayId, int destinationPe) {

    int prevOwner = clientArrayMgr_->lastKnown((CkArrayIndex)arrayId);

    if (prevOwner != destinationPe) {
      clientLocMgr_->updateLocation(arrayId,clientLocMgr_->lookupID(arrayId), destinationPe);

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

  void pup(PUP::er &p) override {
    p|clientAID_;
    if (p.isUnpacking()) {
      clientArrayMgr_ = clientAID_.ckLocalBranch();
      clientLocMgr_ = clientArrayMgr_->getLocMgr();
    }

    p|numArrayElements_;
    p|numLocalArrayElements_;
    p|misdeliveredItems;
#ifdef CMK_TRAM_CACHE_ARRAY_METADATA
    size_t clientObjsSize;

    if (p.isPacking()) {
      clientObjsSize = clientObjs_.size();
    }
    p|clientObjsSize;

    if (p.isUnpacking()) {
      clientObjs_.resize(clientObjsSize);
    }
    for (int i = 0; i < clientObjsSize; i++) {
      p|*clientObjs_[i];
    }

    p|destinationPes_;
    p|isCachedArrayMetadata_;
#endif
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
