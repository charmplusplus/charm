#ifndef _MESH_STREAMER_H_
#define _MESH_STREAMER_H_

#include "MeshStreamer.decl.h"

// allocate more total buffer space then the maximum buffering limit but flush upon
// reaching totalBufferCapacity_
#define BUCKET_SIZE_FACTOR 4

//#define DEBUG_STREAMER 1

enum MeshStreamerMessageType {PlaneMessage, ColumnMessage, PersonalizedMessage};

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

    dtype getDataItem(const int index) {
        return data[index];
    }
};

template <class dtype>
class MeshStreamerClient : public Group {
 public:
     MeshStreamerClient();
     virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg);

};

template <class dtype>
class MeshStreamer : public Group {

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

    int myNodeIndex_;
    int myPlaneIndex_;
    int myColumnIndex_; 
    int myRowIndex_;

    CkCallback   userCallback_;

    MeshStreamerMessage<dtype> **personalizedBuffers_; 
    MeshStreamerMessage<dtype> **columnBuffers_; 
    MeshStreamerMessage<dtype> **planeBuffers_;

    void determineLocation(const int destinationPe, int &row, int &column, 
        int &plane, MeshStreamerMessageType &msgType);

    void storeMessage(MeshStreamerMessage<dtype> ** const messageBuffers, 
        const int bucketIndex, const int destinationPe, 
        const int rowIndex, const int columnIndex, 
        const int planeIndex,
        const MeshStreamerMessageType msgType, const dtype &dataItem);

    void flushLargestBucket(MeshStreamerMessage<dtype> ** const messageBuffers,
			    const int numBuffers, const int myIndex, 
			    const int dimensionFactor);
public:

    MeshStreamer(int totalBufferCapacity, int numRows, 
		 int numColumns, int numPlanes, 
		 const CProxy_MeshStreamerClient<dtype> &clientProxy);
    ~MeshStreamer();

      // entry
    void insertData(const dtype &dataItem, const int destinationPe); 
    void receiveAggregateData(MeshStreamerMessage<dtype> *msg);
    // void receivePersonalizedData(MeshStreamerMessage<dtype> *msg);

    void flushBuckets(MeshStreamerMessage<dtype> **messageBuffers, const int numBuffers);
    void flushDirect();

      // non entry
    void associateCallback(CkCallback &cb) { 
              userCallback_ = cb;
              CkStartQD(CkCallback(CkIndex_MeshStreamer<dtype>::flushDirect(), thisProxy));
         }
};

template <class dtype>
MeshStreamerClient<dtype>::MeshStreamerClient() {}

template <class dtype>
void MeshStreamerClient<dtype>::receiveCombinedData(MeshStreamerMessage<dtype> *msg) {
  CkError("Default implementation of receiveCombinedData should never be called\n");
  delete msg;
}

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(int totalBufferCapacity, int numRows, 
                           int numColumns, int numPlanes, 
                           const CProxy_MeshStreamerClient<dtype> &clientProxy) {
  // limit total number of messages in system to totalBufferCapacity
  //   but allocate a factor BUCKET_SIZE_FACTOR more space to take
  //   advantage of nonuniform filling of buckets
  // the buffers for your own column and plane are never used
  bucketSize_ = BUCKET_SIZE_FACTOR * totalBufferCapacity / (numRows + numColumns + numPlanes - 2); 
  totalBufferCapacity_ = totalBufferCapacity;
  numDataItemsBuffered_ = 0; 
  numRows_ = numRows; 
  numColumns_ = numColumns;
  numPlanes_ = numPlanes; 
  numNodes_ = CkNumPes(); 
  clientProxy_ = clientProxy; 

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
void MeshStreamer<dtype>::determineLocation(const int destinationPe, int &rowIndex, 
                                     int &columnIndex, int &planeIndex, 
                                     MeshStreamerMessageType &msgType) {
  
  int nodeIndex, indexWithinPlane; 

  nodeIndex = destinationPe;
  planeIndex = nodeIndex / planeSize_; 
  if (planeIndex != myPlaneIndex_) {
    msgType = PlaneMessage;     
  }
  else {
    indexWithinPlane = nodeIndex - planeIndex * planeSize_;
    rowIndex = indexWithinPlane / numColumns_;
    columnIndex = indexWithinPlane - rowIndex * numColumns_; 
    if (columnIndex != myColumnIndex_) {
      msgType = ColumnMessage; 
    }
    else {
      msgType = PersonalizedMessage;
    }
  }

}

template <class dtype>
void MeshStreamer<dtype>::storeMessage(MeshStreamerMessage<dtype> ** const messageBuffers, 
                                const int bucketIndex, const int destinationPe, 
                                const int rowIndex, const int columnIndex, 
                                const int planeIndex, 
                                const MeshStreamerMessageType msgType, 
                                const dtype &dataItem) {

  // allocate new message if necessary
  if (messageBuffers[bucketIndex] == NULL) {
    if (msgType == PersonalizedMessage) {
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
  if (msgType != PersonalizedMessage) {
    destinationBucket->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;
  // copy data into message and send if buffer is full
  if (numBuffered == bucketSize_) {
    int destinationIndex;
    CProxy_MeshStreamer<dtype> thisProxy(thisgroup);
    switch (msgType) {

    case PlaneMessage:
      destinationIndex = 
        myNodeIndex_ + (planeIndex - myPlaneIndex_) * planeSize_;      
      thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
      break;
    case ColumnMessage:
      destinationIndex = myNodeIndex_ + (columnIndex - myColumnIndex_);
      thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
      break;
    case PersonalizedMessage:
      destinationIndex = myNodeIndex_ + (rowIndex - myRowIndex_) * numColumns_;
      clientProxy_[destinationIndex].receiveCombinedData(destinationBucket);      
      //      thisProxy[destinationIndex].receivePersonalizedData(destinationBucket);
      break;
    default: 
      CkError("Incorrect MeshStreamer message type\n");
      break;
    }
    messageBuffers[bucketIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered; 
  }

  if (numDataItemsBuffered_ == totalBufferCapacity_) {

    flushLargestBucket(personalizedBuffers_, numRows_, myRowIndex_, numColumns_);
    flushLargestBucket(columnBuffers_, numColumns_, myColumnIndex_, 1);
    flushLargestBucket(planeBuffers_, numPlanes_, myPlaneIndex_, planeSize_);

  }

}

template <class dtype>
void MeshStreamer<dtype>::insertData(const dtype &dataItem, const int destinationPe) {

  int planeIndex, columnIndex, rowIndex; // location of destination
  int indexWithinPlane; 

  MeshStreamerMessageType msgType; 

  determineLocation(destinationPe, rowIndex, columnIndex, planeIndex, msgType);

  // determine which array of buffers is appropriate for this message
  MeshStreamerMessage<dtype> **messageBuffers;
  int bucketIndex; 

  switch (msgType) {
  case PlaneMessage:
    messageBuffers = planeBuffers_; 
    bucketIndex = planeIndex; 
    break;
  case ColumnMessage:
    messageBuffers = columnBuffers_; 
    bucketIndex = columnIndex; 
    break;
  case PersonalizedMessage:
    messageBuffers = personalizedBuffers_; 
    bucketIndex = rowIndex; 
    break;
  default: 
    CkError("Unrecognized MeshStreamer message type\n");
    break;
  }

  storeMessage(messageBuffers, bucketIndex, destinationPe, rowIndex, 
               columnIndex, planeIndex, msgType, dataItem);
}

template <class dtype>
void MeshStreamer<dtype>::receiveAggregateData(MeshStreamerMessage<dtype> *msg) {

  int rowIndex, columnIndex, planeIndex, destinationPe; 
  MeshStreamerMessageType msgType;   

  for (int i = 0; i < msg->numDataItems; i++) {
    destinationPe = msg->destinationPes[i];
    dtype dataItem = msg->getDataItem(i);
    determineLocation(destinationPe, rowIndex, columnIndex, 
                      planeIndex, msgType);
#ifdef DEBUG_STREAMER
    CkAssert(planeIndex == myPlaneIndex_);

    if (msgType == PersonalizedMessage) {
      CkAssert(columnIndex == myColumnIndex_);
    }
#endif    

    MeshStreamerMessage<dtype> **messageBuffers;
    int bucketIndex; 

    switch (msgType) {
    case ColumnMessage:
      messageBuffers = columnBuffers_; 
      bucketIndex = columnIndex; 
      break;
    case PersonalizedMessage:
      messageBuffers = personalizedBuffers_; 
      bucketIndex = rowIndex; 
      break;
    default: 
      CkError("Incorrect MeshStreamer message type\n");
      break;
    }

    storeMessage(messageBuffers, bucketIndex, destinationPe, rowIndex, 
                 columnIndex, planeIndex, msgType, dataItem);
    
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
      CProxy_MeshStreamer<dtype> thisProxy(thisgroup);
      thisProxy[destinationIndex].receiveAggregateData(destinationBucket);
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
    flushBuckets(planeBuffers_, numPlanes_);
    flushBuckets(columnBuffers_, numColumns_);
    flushBuckets(personalizedBuffers_, numRows_);

#ifdef DEBUG_STREAMER
    //CkPrintf("[%d] numDataItemsBuffered_: %d\n", CkMyPe(), numDataItemsBuffered_);
    CkAssert(numDataItemsBuffered_ == 0); 
#endif

    if (!userCallback_.isInvalid()) {
        CkStartQD(userCallback_);
        userCallback_ = CkCallback();      // nullify the current callback
    }
}

#define CK_TEMPLATES_ONLY
#include "MeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
