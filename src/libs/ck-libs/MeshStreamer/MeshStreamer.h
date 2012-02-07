#ifndef _MESH_STREAMER_H_
#define _MESH_STREAMER_H_

#include <algorithm>
#include "MeshStreamer.decl.h"
// allocate more total buffer space than the maximum buffering limit but flush upon
// reaching totalBufferCapacity_
#define BUCKET_SIZE_FACTOR 4

// #define DEBUG_STREAMER
// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH

enum MeshStreamerMessageType {PlaneMessage, ColumnMessage, PersonalizedMessage};

class MeshLocation {
 public:
  int rowIndex;
  int columnIndex;
  int planeIndex; 
  MeshStreamerMessageType msgType;
};


/*
class LocalMessage : public CMessage_LocalMessage {
public:
    int numDataItems; 
    int dataItemSize; 
    char *data;

    LocalMessage(int dataItemSizeInBytes) {
        numDataItems = 0; 
        dataItemSize = dataItemSizeInBytes; 
    }

    int addDataItem(void *dataItem) {
        CmiMemcpy(&data[numDataItems * dataItemSize], dataItem, dataItemSize);
        return ++numDataItems; 
    } 

    void *getDataItem(int index) {
        return (void *) (&data[index * dataItemSize]);  
    }

};
*/

template<class dtype>
class MeshStreamerMessage : public CMessage_MeshStreamerMessage<dtype> {
public:
    int numDataItems;
    int *destinationPes;
    dtype *data;

    MeshStreamerMessage(): numDataItems(0) {}   

    int addDataItem(const dtype &dataItem) {
        data[numDataItems] = dataItem;
        return ++numDataItems; 
    }

    void markDestination(const int index, const int destinationPe) {
	destinationPes[index] = destinationPe;
    }

    dtype &getDataItem(const int index) {
        return data[index];
    }
};

template <class dtype>
class MeshStreamerClient : public CBase_MeshStreamerClient<dtype> {
 public:
     virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg);
     virtual void process(dtype &data)=0; 
};

template <class dtype>
class MeshStreamer : public CBase_MeshStreamer<dtype> {

private:
    int bucketSize_; 
    int totalBufferCapacity_;
    int numDataItemsBuffered_;

    int numNodes_; 
    int numRows_; 
    int numColumns_; 
    int numPlanes_; 
    int planeSize_;

    CProxy_MeshStreamerClient<dtype> clientProxy_;
    MeshStreamerClient<dtype> *clientObj_;

    int myNodeIndex_;
    int myPlaneIndex_;
    int myColumnIndex_; 
    int myRowIndex_;

    CkCallback   userCallback_;
    int yieldFlag_;

    double progressPeriodInMs_; 
    bool isPeriodicFlushEnabled_; 
    double timeOfLastSend_; 

    MeshStreamerMessage<dtype> **personalizedBuffers_; 
    MeshStreamerMessage<dtype> **columnBuffers_; 
    MeshStreamerMessage<dtype> **planeBuffers_;

#ifdef CACHE_LOCATIONS
    MeshLocation *cachedLocations;
    bool *isCached; 
#endif

#ifdef SUPPORT_INCOMPLETE_MESH
    int numNodesInLastPlane_;
    int numFullRowsInLastPlane_;
    int numColumnsInLastRow_;
#endif

    void determineLocation(const int destinationPe, 
			   MeshLocation &destinationCoordinates);

    void storeMessage(MeshStreamerMessage<dtype> ** const messageBuffers, 
		      const int bucketIndex, const int destinationPe, 
		      const MeshLocation &destinationCoordinates, const dtype &dataItem);

    void flushLargestBucket(MeshStreamerMessage<dtype> ** const messageBuffers,
			    const int numBuffers, const int myIndex, 
			    const int dimensionFactor);

public:

    MeshStreamer(int totalBufferCapacity, int numRows, 
		 int numColumns, int numPlanes, 
		 const CProxy_MeshStreamerClient<dtype> &clientProxy,
                 int yieldFlag = 0, double progressPeriodInMs = -1.0);
    ~MeshStreamer();

      // entry
    void insertData(dtype &dataItem, const int destinationPe); 
    void doneInserting();
    void receiveAggregateData(MeshStreamerMessage<dtype> *msg);
    // void receivePersonalizedData(MeshStreamerMessage<dtype> *msg);

    void flushBuckets(MeshStreamerMessage<dtype> **messageBuffers, const int numBuffers);
    void flushDirect();

    bool isPeriodicFlushEnabled() {
      return isPeriodicFlushEnabled_;
    }
      // non entry
    void associateCallback(CkCallback &cb, bool automaticFinish = true) { 
      userCallback_ = cb;
      if (automaticFinish) {
        CkStartQD(CkCallback(CkIndex_MeshStreamer<dtype>::finish(NULL), this->thisProxy));
      }
    }

    void registerPeriodicProgressFunction();
    void finish(CkReductionMsg *msg);

    /*
     * Flushing begins on a PE only after enablePeriodicFlushing has been invoked.
     */
    void enablePeriodicFlushing(){
      isPeriodicFlushEnabled_ = true; 
      registerPeriodicProgressFunction();
    }
};

template <class dtype>
void MeshStreamerClient<dtype>::receiveCombinedData(MeshStreamerMessage<dtype> *msg) {
  for (int i = 0; i < msg->numDataItems; i++) {
     dtype data = ((dtype*)(msg->data))[i];
     process(data);
  }
  delete msg;
}

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(int totalBufferCapacity, int numRows, 
                           int numColumns, int numPlanes, 
                           const CProxy_MeshStreamerClient<dtype> &clientProxy,
			   int yieldFlag, double progressPeriodInMs): yieldFlag_(yieldFlag) {
  // limit total number of messages in system to totalBufferCapacity
  //   but allocate a factor BUCKET_SIZE_FACTOR more space to take
  //   advantage of nonuniform filling of buckets
  // the buffers for your own column and plane are never used
  bucketSize_ = BUCKET_SIZE_FACTOR * totalBufferCapacity / (numRows + numColumns + numPlanes - 2); 
  if (bucketSize_ <= 0) {
    bucketSize_ = 1; 
    CkPrintf("Argument totalBufferCapacity to MeshStreamer constructor "
	     "is invalid. Defaulting to a single buffer per destination.\n");
  }
  totalBufferCapacity_ = totalBufferCapacity;
  numDataItemsBuffered_ = 0; 
  numRows_ = numRows; 
  numColumns_ = numColumns;
  numPlanes_ = numPlanes; 
  numNodes_ = CkNumPes(); 
  clientProxy_ = clientProxy; 
  clientObj_ = ((MeshStreamerClient<dtype> *)CkLocalBranch(clientProxy_));
  progressPeriodInMs_ = progressPeriodInMs; 

  personalizedBuffers_ = new MeshStreamerMessage<dtype> *[numRows];
  for (int i = 0; i < numRows; i++) {
    personalizedBuffers_[i] = NULL; 
  }

  columnBuffers_ = new MeshStreamerMessage<dtype> *[numColumns];
  for (int i = 0; i < numColumns; i++) {
    columnBuffers_[i] = NULL; 
  }

  planeBuffers_ = new MeshStreamerMessage<dtype> *[numPlanes]; 
  for (int i = 0; i < numPlanes; i++) {
    planeBuffers_[i] = NULL; 
  }

  // determine plane, column, and row location of this node
  myNodeIndex_ = CkMyPe();
  planeSize_ = numRows_ * numColumns_; 
  myPlaneIndex_ = myNodeIndex_ / planeSize_; 
  int indexWithinPlane = myNodeIndex_ - myPlaneIndex_ * planeSize_;
  myRowIndex_ = indexWithinPlane / numColumns_;
  myColumnIndex_ = indexWithinPlane - myRowIndex_ * numColumns_; 

  isPeriodicFlushEnabled_ = false; 

#ifdef CACHE_LOCATIONS
  cachedLocations = new MeshLocation[numNodes_];
  isCached = new bool[numNodes_];
  std::fill(isCached, isCached + numNodes_, false);
#endif

#ifdef SUPPORT_INCOMPLETE_MESH
  numNodesInLastPlane_ = numNodes_ % planeSize_; 
  numFullRowsInLastPlane_ = numNodesInLastPlane_ / numColumns_;
  numColumnsInLastRow_ = numNodesInLastPlane_ - numFullRowsInLastPlane_ * numColumns_;  
#endif
}

template <class dtype>
MeshStreamer<dtype>::~MeshStreamer() {

  for (int i = 0; i < numRows_; i++)
      delete personalizedBuffers_[i]; 

  for (int i = 0; i < numColumns_; i++)
      delete columnBuffers_[i]; 

  for (int i = 0; i < numPlanes_; i++)
      delete planeBuffers_[i]; 

  delete[] personalizedBuffers_;
  delete[] columnBuffers_;
  delete[] planeBuffers_; 

}

template <class dtype>
void MeshStreamer<dtype>::determineLocation(const int destinationPe, 
					    MeshLocation &destinationCoordinates) { 

  int nodeIndex, indexWithinPlane; 

#ifdef CACHE_LOCATIONS
  if (isCached[destinationPe] == true) {
    destinationCoordinates = cachedLocations[destinationPe]; 
    return;
  }
#endif

  nodeIndex = destinationPe;
  destinationCoordinates.planeIndex = nodeIndex / planeSize_; 
  if (destinationCoordinates.planeIndex != myPlaneIndex_) {
    destinationCoordinates.msgType = PlaneMessage;     
  }
  else {
    indexWithinPlane = 
      nodeIndex - destinationCoordinates.planeIndex * planeSize_;
    destinationCoordinates.rowIndex = indexWithinPlane / numColumns_;
    destinationCoordinates.columnIndex = 
      indexWithinPlane - destinationCoordinates.rowIndex * numColumns_; 
    if (destinationCoordinates.columnIndex != myColumnIndex_) {
      destinationCoordinates.msgType = ColumnMessage; 
    }
    else {
      destinationCoordinates.msgType = PersonalizedMessage;
    }
  }

#ifdef CACHE_LOCATIONS
  cachedLocations[destinationPe] = destinationCoordinates;
#endif

}

template <class dtype>
void MeshStreamer<dtype>::storeMessage(MeshStreamerMessage<dtype> ** const messageBuffers, 
				       const int bucketIndex, const int destinationPe, 
				       const MeshLocation& destinationCoordinates,
				       const dtype &dataItem) {

  // allocate new message if necessary
  if (messageBuffers[bucketIndex] == NULL) {
    if (destinationCoordinates.msgType == PersonalizedMessage) {
      messageBuffers[bucketIndex] = 
        new (0, bucketSize_) MeshStreamerMessage<dtype>();
    }
    else {
      messageBuffers[bucketIndex] = 
        new (bucketSize_, bucketSize_) MeshStreamerMessage<dtype>();
    }
#ifdef DEBUG_STREAMER
    CkAssert(messageBuffers[bucketIndex] != NULL);
#endif
  }
  
  MeshStreamerMessage<dtype> *destinationBucket = messageBuffers[bucketIndex];
  
  int numBuffered = destinationBucket->addDataItem(dataItem); 
  if (destinationCoordinates.msgType != PersonalizedMessage) {
    destinationBucket->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;
  // copy data into message and send if buffer is full
  if (numBuffered == bucketSize_) {
    int destinationIndex;
    switch (destinationCoordinates.msgType) {

    case PlaneMessage:
      destinationIndex = myNodeIndex_ + 
	(destinationCoordinates.planeIndex - myPlaneIndex_) * planeSize_;  
#ifdef SUPPORT_INCOMPLETE_MESH
      if (destinationIndex >= numNodes_) {
	int numValidRows = numFullRowsInLastPlane_; 
	if (numColumnsInLastRow_ > myColumnIndex_) {
	  numValidRows++; 
	}
	destinationIndex = destinationCoordinates.planeIndex * planeSize_ + 
	  myColumnIndex_ + (myRowIndex_ % numValidRows) * numColumns_; 
      }
#endif      
      this->thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
      break;
    case ColumnMessage:
      destinationIndex = myNodeIndex_ + 
	(destinationCoordinates.columnIndex - myColumnIndex_);
#ifdef SUPPORT_INCOMPLETE_MESH
      if (destinationIndex >= numNodes_) {
	destinationIndex = destinationCoordinates.planeIndex * planeSize_ + 
	  destinationCoordinates.columnIndex + 
	  (myColumnIndex_ % numFullRowsInLastPlane_) * numColumns_; 
      }
#endif      
      this->thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
      break;
    case PersonalizedMessage:
      destinationIndex = myNodeIndex_ + 
	(destinationCoordinates.rowIndex - myRowIndex_) * numColumns_;
      clientProxy_[destinationIndex].receiveCombinedData(destinationBucket);      
      //      this->thisProxy[destinationIndex].receivePersonalizedData(destinationBucket);
      break;
    default: 
      CkError("Incorrect MeshStreamer message type\n");
      break;
    }
    messageBuffers[bucketIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered; 

    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }

  }

  if (numDataItemsBuffered_ == totalBufferCapacity_) {

    flushLargestBucket(personalizedBuffers_, numRows_, myRowIndex_, numColumns_);
    flushLargestBucket(columnBuffers_, numColumns_, myColumnIndex_, 1);
    flushLargestBucket(planeBuffers_, numPlanes_, myPlaneIndex_, planeSize_);

    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }

  }

}

template <class dtype>
void MeshStreamer<dtype>::insertData(dtype &dataItem, const int destinationPe) {
  static int count = 0;

  if (destinationPe == CkMyPe()) {
    clientObj_->process(dataItem);
    return;
  }

  MeshLocation destinationCoordinates;

  determineLocation(destinationPe, destinationCoordinates);

  // determine which array of buffers is appropriate for this message
  MeshStreamerMessage<dtype> **messageBuffers;
  int bucketIndex; 

  switch (destinationCoordinates.msgType) {
  case PlaneMessage:
    messageBuffers = planeBuffers_; 
    bucketIndex = destinationCoordinates.planeIndex; 
    break;
  case ColumnMessage:
    messageBuffers = columnBuffers_; 
    bucketIndex = destinationCoordinates.columnIndex; 
    break;
  case PersonalizedMessage:
    messageBuffers = personalizedBuffers_; 
    bucketIndex = destinationCoordinates.rowIndex; 
    break;
  default: 
    CkError("Unrecognized MeshStreamer message type\n");
    break;
  }

  storeMessage(messageBuffers, bucketIndex, destinationPe, destinationCoordinates, 
	       dataItem);

    // release control to scheduler if requested by the user, 
    //   assume caller is threaded entry
  if (yieldFlag_ && ++count == 1024) {
    count = 0; 
    CthYield();
  }
}

template <class dtype>
void MeshStreamer<dtype>::doneInserting() {
  this->contribute(CkCallback(CkIndex_MeshStreamer<dtype>::finish(NULL), this->thisProxy));
}

template <class dtype>
void MeshStreamer<dtype>::finish(CkReductionMsg *msg) {

  isPeriodicFlushEnabled_ = false; 
  flushDirect();

  if (!userCallback_.isInvalid()) {
    CkStartQD(userCallback_);
    userCallback_ = CkCallback();      // nullify the current callback
  }

  //  delete msg; 
}


template <class dtype>
void MeshStreamer<dtype>::receiveAggregateData(MeshStreamerMessage<dtype> *msg) {

  int destinationPe; 
  MeshStreamerMessageType msgType;   
  MeshLocation destinationCoordinates;

  for (int i = 0; i < msg->numDataItems; i++) {
    destinationPe = msg->destinationPes[i];
    dtype &dataItem = msg->getDataItem(i);
    determineLocation(destinationPe, destinationCoordinates);
#ifdef DEBUG_STREAMER
    CkAssert(destinationCoordinates.planeIndex == myPlaneIndex_);

    if (destinationCoordinates.msgType == PersonalizedMessage) {
      CkAssert(destinationCoordinates.columnIndex == myColumnIndex_);
    }
#endif    

    MeshStreamerMessage<dtype> **messageBuffers;
    int bucketIndex; 

    switch (destinationCoordinates.msgType) {
    case ColumnMessage:
      messageBuffers = columnBuffers_; 
      bucketIndex = destinationCoordinates.columnIndex; 
      break;
    case PersonalizedMessage:
      messageBuffers = personalizedBuffers_; 
      bucketIndex = destinationCoordinates.rowIndex; 
      break;
    default: 
      CkError("Incorrect MeshStreamer message type\n");
      break;
    }

    storeMessage(messageBuffers, bucketIndex, destinationPe, 
		 destinationCoordinates, dataItem);
    
  }

  delete msg;

}

/*
void MeshStreamer::receivePersonalizedData(MeshStreamerMessage *msg) {

  // sort data items into messages for each core on this node

  LocalMessage *localMsgs[numPesPerNode_];
  int dataSize = bucketSize_ * dataItemSize_;

  for (int i = 0; i < numPesPerNode_; i++) {
    localMsgs[i] = new (dataSize) LocalMessage(dataItemSize_);
  }

  int destinationPe;
  for (int i = 0; i < msg->numDataItems; i++) {

    destinationPe = msg->destinationPes[i]; 
    void *dataItem = msg->getDataItem(i);   
    localMsgs[destinationPe % numPesPerNode_]->addDataItem(dataItem);

  }

  for (int i = 0; i < numPesPerNode_; i++) {
    if (localMsgs[i]->numDataItems > 0) {
      clientProxy_[myNodeIndex_ * numPesPerNode_ + i].receiveCombinedData(localMsgs[i]);
    }
    else {
      delete localMsgs[i];
    }
  }

  delete msg; 

}
*/

template <class dtype>
void MeshStreamer<dtype>::flushLargestBucket(MeshStreamerMessage<dtype> ** const messageBuffers,
                                      const int numBuffers, const int myIndex, 
                                      const int dimensionFactor) {

  int flushIndex, maxSize, destinationIndex;
  MeshStreamerMessage<dtype> *destinationBucket; 
  maxSize = 0;
  for (int i = 0; i < numBuffers; i++) {
    if (messageBuffers[i] != NULL && messageBuffers[i]->numDataItems > maxSize) {
      maxSize = messageBuffers[i]->numDataItems;
      flushIndex = i;
    } 
  }
  if (maxSize > 0) {
    destinationBucket = messageBuffers[flushIndex];
    destinationIndex = myNodeIndex_ + (flushIndex - myIndex) * dimensionFactor;

    if (destinationBucket->numDataItems < bucketSize_) {
      // not sending the full buffer, shrink the message size
      envelope *env = UsrToEnv(destinationBucket);
      env->setTotalsize(env->getTotalsize() - (bucketSize_ - destinationBucket->numDataItems) * sizeof(dtype));
    }
    numDataItemsBuffered_ -= destinationBucket->numDataItems;

    if (messageBuffers == personalizedBuffers_) {
      clientProxy_[destinationIndex].receiveCombinedData(destinationBucket);
    }
    else {
      this->thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
    }
    messageBuffers[flushIndex] = NULL;
  }
}

template <class dtype>
void MeshStreamer<dtype>::flushBuckets(MeshStreamerMessage<dtype> **messageBuffers, const int numBuffers)
{

    for (int i = 0; i < numBuffers; i++) {
       if(messageBuffers[i] == NULL)
           continue;
       //flush all messages in i bucket
       numDataItemsBuffered_ -= messageBuffers[i]->numDataItems;
       if (messageBuffers == personalizedBuffers_) {
         int destinationPe = myNodeIndex_ + (i - myRowIndex_) * numColumns_; 
         clientProxy_[destinationPe].receiveCombinedData(messageBuffers[i]);
       }
       else {
         for (int j = 0; j < messageBuffers[i]->numDataItems; j++) {
           MeshStreamerMessage<dtype> *directMsg = 
             new (0, 1) MeshStreamerMessage<dtype>();
#ifdef DEBUG_STREAMER
           CkAssert(directMsg != NULL);
#endif
           int destinationPe = messageBuffers[i]->destinationPes[j]; 
           dtype dataItem = messageBuffers[i]->getDataItem(j);   
           directMsg->addDataItem(dataItem);
           clientProxy_[destinationPe].receiveCombinedData(directMsg);
         }
         delete messageBuffers[i];
       }
       messageBuffers[i] = NULL;
    }

}

template <class dtype>
void MeshStreamer<dtype>::flushDirect(){

    if (!isPeriodicFlushEnabled_ || 
	1000 * (CkWallTimer() - timeOfLastSend_) >= progressPeriodInMs_) {
      flushBuckets(planeBuffers_, numPlanes_);
      flushBuckets(columnBuffers_, numColumns_);
      flushBuckets(personalizedBuffers_, numRows_);
    }

    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }

#ifdef DEBUG_STREAMER
    //CkPrintf("[%d] numDataItemsBuffered_: %d\n", CkMyPe(), numDataItemsBuffered_);
    CkAssert(numDataItemsBuffered_ == 0); 
#endif

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
  CcdCallFnAfter(periodicProgressFunction<dtype>, (void *) this, progressPeriodInMs_); 
}


#define CK_TEMPLATES_ONLY
#include "MeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
