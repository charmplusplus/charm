#ifndef NDMESH_STREAMER_H_
#define NDMESH_STREAMER_H_

#include <algorithm>
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"

// allocate more total buffer space than the maximum buffering limit but flush 
//   upon reaching totalBufferCapacity_
#define BUFFER_SIZE_FACTOR 4

// #define DEBUG_STREAMER
// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH

struct MeshLocation {
  int dimension; 
  int bufferIndex; 
}; 

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
class MeshStreamerGroupClient : public CBase_MeshStreamerGroupClient<dtype> {
 public:
     virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg);
     virtual void process(dtype &data)=0; 
};

template <class dtype>
class MeshStreamerArrayClient : public CBase_MeshStreamerArrayClient<dtype> {
 public:
     // virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg);
  // would like to make it pure virtual but charm will try to
  // instantiate the abstract class, leading to errors
  virtual void process(dtype &data) {} //=0; 
  MeshStreamerArrayClient() {}
  MeshStreamerArrayClient(CkMigrateMessage *msg) {}
};

template <class dtype>
class MeshStreamer : public CBase_MeshStreamer<dtype> {

private:
    int bufferSize_; 
    int totalBufferCapacity_;
    int numDataItemsBuffered_;

    int numMembers_; 
    int numDimensions_;
    int *individualDimensionSizes_;
    int *combinedDimensionSizes_;

    int myIndex_;
    int *myLocationIndex_;

    CkCallback   userCallback_;
    int yieldFlag_;

    double progressPeriodInMs_; 
    bool isPeriodicFlushEnabled_; 
    double timeOfLastSend_; 


    MeshStreamerMessage<dtype> ***dataBuffers_;

#ifdef CACHE_LOCATIONS
    MeshLocation *cachedLocations;
    bool *isCached; 
#endif

    MeshLocation determineLocation(int destinationPe);

    void storeMessage(int destinationPe, 
		      const MeshLocation &destinationCoordinates, 
		      void *dataItem, bool copyIndirectly = false);

    virtual void deliverToDestination(
                 int destinationPe, 
                 MeshStreamerMessage<dtype> *destinationBuffer) = 0;

    virtual void localDeliver(dtype &dataItem) = 0; 

    void flushLargestBuffer();

protected:

    virtual int copyDataItemIntoMessage(
		MeshStreamerMessage<dtype> *destinationBuffer, 
		void *dataItemHandle, bool copyIndirectly = false);

public:

    MeshStreamer(int totalBufferCapacity, int numDimensions, 
		 int *dimensionSizes,
                 int yieldFlag = 0, double progressPeriodInMs = -1.0);
    ~MeshStreamer();

      // entry
    void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
    void flushDirect();
    void finish(CkReductionMsg *msg);

      // non entry
    bool isPeriodicFlushEnabled() {
      return isPeriodicFlushEnabled_;
    }
    virtual void insertData(dtype &dataItem, int destinationPe); 
    void insertData(void *dataItemHandle, int destinationPe);
    void doneInserting();
    void associateCallback(CkCallback &cb, bool automaticFinish = true) { 
      userCallback_ = cb;
      if (automaticFinish) {
        CkStartQD(CkCallback(CkIndex_MeshStreamer<dtype>::finish(NULL), 
			     this->thisProxy));
      }
    }
    void flushAllBuffers();
    void registerPeriodicProgressFunction();

    // flushing begins only after enablePeriodicFlushing has been invoked

    void enablePeriodicFlushing(){
      isPeriodicFlushEnabled_ = true; 
      registerPeriodicProgressFunction();
    }
};

template <class dtype>
void MeshStreamerGroupClient<dtype>::receiveCombinedData(
                                MeshStreamerMessage<dtype> *msg) {
  for (int i = 0; i < msg->numDataItems; i++) {
    dtype &data = msg->getDataItem(i);
    process(data);
  }
  delete msg;
}

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(
		     int totalBufferCapacity, int numDimensions, 
		     int *dimensionSizes, 
		     int yieldFlag, 
                     double progressPeriodInMs)
 :numDimensions_(numDimensions), 
  totalBufferCapacity_(totalBufferCapacity), 
  yieldFlag_(yieldFlag), 
  progressPeriodInMs_(progressPeriodInMs)
{
  // limit total number of messages in system to totalBufferCapacity
  //   but allocate a factor BUFFER_SIZE_FACTOR more space to take
  //   advantage of nonuniform filling of buffers

  int sumAlongAllDimensions = 0;   
  individualDimensionSizes_ = new int[numDimensions_];
  combinedDimensionSizes_ = new int[numDimensions_ + 1];
  myLocationIndex_ = new int[numDimensions_];
  memcpy(individualDimensionSizes_, dimensionSizes, 
	 numDimensions * sizeof(int)); 
  combinedDimensionSizes_[0] = 1; 
  for (int i = 0; i < numDimensions; i++) {
    sumAlongAllDimensions += individualDimensionSizes_[i];
    combinedDimensionSizes_[i + 1] = 
      combinedDimensionSizes_[i] * individualDimensionSizes_[i];
  }

  // except for personalized messages, the buffers for dimensions with the 
  //   same index as the sender's are not used
  bufferSize_ = BUFFER_SIZE_FACTOR * totalBufferCapacity 
    / (sumAlongAllDimensions - numDimensions_ + 1); 
  if (bufferSize_ <= 0) {
    bufferSize_ = 1; 
    CkPrintf("Argument totalBufferCapacity to MeshStreamer constructor "
	     "is invalid. Defaulting to a single buffer per destination.\n");
  }
  totalBufferCapacity_ = totalBufferCapacity;
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
  for (int i = numDimensions_ - 1; i >= 0; i--) {    
    myLocationIndex_[i] = remainder / combinedDimensionSizes_[i];
    remainder -= combinedDimensionSizes_[i] * myLocationIndex_[i];
  }

  isPeriodicFlushEnabled_ = false; 

#ifdef CACHE_LOCATIONS
  cachedLocations = new MeshLocation[numMembers_];
  isCached = new bool[numMembers_];
  std::fill(isCached, isCached + numMembers_, false);
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

#ifdef CACHE_LOCATIONS
  delete[] cachedLocations;
  delete[] isCached; 
#endif

}


template <class dtype>
inline
MeshLocation MeshStreamer<dtype>::determineLocation(int destinationPe) { 

#ifdef CACHE_LOCATIONS
  if (isCached[destinationPe]) {    
    return cachedLocations[destinationPe]; 
  }
#endif

  MeshLocation destinationLocation;
  int remainder = destinationPe;
  int dimensionIndex; 
  for (int i = numDimensions_ - 1; i >= 0; i--) {        
    dimensionIndex = remainder / combinedDimensionSizes_[i];
    
    if (dimensionIndex != myLocationIndex_[i]) {
      destinationLocation.dimension = i; 
      destinationLocation.bufferIndex = dimensionIndex; 
#ifdef CACHE_LOCATIONS
      cachedLocations[destinationPe] = destinationLocation;
      isCached[destinationPe] = true; 
#endif
      return destinationLocation;
    }

    remainder -= combinedDimensionSizes_[i] * dimensionIndex;
  }

  // all indices agree - message to oneself
  destinationLocation.dimension = 0; 
  destinationLocation.bufferIndex = myLocationIndex_[0];
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
        new (0, bufferSize_) MeshStreamerMessage<dtype>();
    }
    else {
      messageBuffers[bufferIndex] = 
        new (bufferSize_, bufferSize_) MeshStreamerMessage<dtype>();
    }
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
      deliverToDestination(destinationIndex, destinationBuffer);
    }
    else {
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }

    messageBuffers[bufferIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered; 

    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }

  }

  // send if total buffering capacity has been reached
  if (numDataItemsBuffered_ == totalBufferCapacity_) {
    flushLargestBuffer();
    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }
  }

}

template <class dtype>
void MeshStreamer<dtype>::insertData(void *dataItemHandle, int destinationPe) {
  static int count = 0;
  const static bool copyIndirectly = true;

  MeshLocation destinationLocation = determineLocation(destinationPe);
  storeMessage(destinationPe, destinationLocation, dataItemHandle, 
	       copyIndirectly); 

  // release control to scheduler if requested by the user, 
  //   assume caller is threaded entry
  if (yieldFlag_ && ++count == 1024) {
    count = 0; 
    CthYield();
  }

}

template <class dtype>
inline
void MeshStreamer<dtype>::insertData(dtype &dataItem, int destinationPe) {

  if (destinationPe == CkMyPe()) {
    localDeliver(dataItem);
    return;
  }

  insertData((void *) &dataItem, destinationPe);
}

template <class dtype>
void MeshStreamer<dtype>::doneInserting() {
  this->contribute(CkCallback(CkIndex_MeshStreamer<dtype>::finish(NULL), 
			      this->thisProxy));
}

template <class dtype>
void MeshStreamer<dtype>::finish(CkReductionMsg *msg) {

  isPeriodicFlushEnabled_ = false; 
  flushDirect();

  if (!userCallback_.isInvalid()) {
    CkStartQD(userCallback_);
    userCallback_ = CkCallback();      // nullify the current callback
  }

  // TODO: TEST IF THIS DELETE STILL CAUSES UNEXPLAINED CRASHES
  //  delete msg; 
}

template <class dtype>
void MeshStreamer<dtype>::receiveAlongRoute(MeshStreamerMessage<dtype> *msg) {

  int destinationPe; 
  MeshLocation destinationLocation;

  for (int i = 0; i < msg->numDataItems; i++) {
    destinationPe = msg->destinationPes[i];
    dtype &dataItem = msg->getDataItem(i);
    destinationLocation = determineLocation(destinationPe);
    if (destinationPe == CkMyPe()) {
      localDeliver(dataItem);
    }
    else {
      storeMessage(destinationPe, destinationLocation, &dataItem);   
    }
  }

  delete msg;

}

template <class dtype>
void MeshStreamer<dtype>::flushLargestBuffer() {

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
      }
      numDataItemsBuffered_ -= destinationBuffer->numDataItems;

      if (flushDimension == 0) {
        deliverToDestination(destinationIndex, destinationBuffer);
      }
      else {
	this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
      }
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
        deliverToDestination(destinationPe, messageBuffers[j]);
      }	 
      else {

	for (int k = 0; k < messageBuffers[j]->numDataItems; k++) {

	  MeshStreamerMessage<dtype> *directMsg = 
	    new (0, 1) MeshStreamerMessage<dtype>();
#ifdef DEBUG_STREAMER
	  CkAssert(directMsg != NULL);
#endif
	  int destinationPe = messageBuffers[j]->destinationPes[k]; 
	  dtype &dataItem = messageBuffers[j]->getDataItem(k);   
	  directMsg->addDataItem(dataItem);
          deliverToDestination(destinationPe,directMsg);
	}
	delete messageBuffers[j];
      }
      messageBuffers[j] = NULL;
    }
  }
}

template <class dtype>
void MeshStreamer<dtype>::flushDirect(){

    if (!isPeriodicFlushEnabled_ || 
	1000 * (CkWallTimer() - timeOfLastSend_) >= progressPeriodInMs_) {
      flushAllBuffers();
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
  CcdCallFnAfter(periodicProgressFunction<dtype>, (void *) this, 
		 progressPeriodInMs_); 
}


template <class dtype>
class GroupMeshStreamer : public MeshStreamer<dtype> {
private:

  CProxy_MeshStreamerGroupClient<dtype> clientProxy_;
  MeshStreamerGroupClient<dtype> *clientObj_;

  void deliverToDestination(int destinationPe, 
                            MeshStreamerMessage<dtype> *destinationBuffer) {
    clientProxy_[destinationPe].receiveCombinedData(destinationBuffer);
  }

  void localDeliver(dtype &dataItem) {
    clientObj_->process(dataItem);
  }

public:

  GroupMeshStreamer(int totalBufferCapacity, int numDimensions,
		    int *dimensionSizes, 
		    const CProxy_MeshStreamerGroupClient<dtype> &clientProxy,
		    int yieldFlag = 0, double progressPeriodInMs = -1.0)
   :MeshStreamer<dtype>(totalBufferCapacity, numDimensions, dimensionSizes, 
                         yieldFlag, progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientObj_ = 
      ((MeshStreamerGroupClient<dtype> *)CkLocalBranch(clientProxy_));
  }


};

template <class dtype>
class ArrayMeshStreamer : public MeshStreamer<ArrayDataItem<dtype> > {
private:

  CProxy_MeshStreamerArrayClient<dtype> clientProxy_;
  CkArray *clientArrayMgr_;
  MeshStreamerArrayClient<dtype> *clientObj_;


  void deliverToDestination(
       int destinationPe, 
       MeshStreamerMessage<ArrayDataItem<dtype> > *destinationBuffer) { 
    ( (CProxy_ArrayMeshStreamer<dtype>) 
      this->thisProxy )[destinationPe].receiveArrayData(destinationBuffer);
  }

  void localDeliver(ArrayDataItem<dtype> &packedDataItem) {
    int arrayId = packedDataItem.arrayIndex; 
    MeshStreamerArrayClient<dtype> *clientObj = 
      clientProxy_[arrayId].ckLocal();

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
    }
    else {
      // array element is no longer present locally - redeliver using proxy
      clientProxy_[arrayId].process(packedDataItem.dataItem);
    }
  }

public:

  struct DataItemHandle {
    int arrayIndex; 
    dtype *dataItem;
  };

  ArrayMeshStreamer(int totalBufferCapacity, int numDimensions,
		    int *dimensionSizes, 
		    const CProxy_MeshStreamerArrayClient<dtype> &clientProxy,
		    int yieldFlag = 0, double progressPeriodInMs = -1.0)
    :MeshStreamer<ArrayDataItem<dtype> >(totalBufferCapacity, numDimensions, 
					dimensionSizes, yieldFlag, 
					progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientArrayMgr_ = clientProxy_.ckLocalBranch();
  }

  void receiveArrayData(MeshStreamerMessage<ArrayDataItem<dtype> > *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      ArrayDataItem<dtype> &packedData = msg->getDataItem(i);
      localDeliver(packedData);
    }
    delete msg;
  }

  void insertData(dtype &dataItem, int arrayIndex) {

    int destinationPe = 
      clientArrayMgr_->lastKnown(clientProxy_[arrayIndex].ckGetIndex());
    static ArrayDataItem<dtype> packedDataItem;
    if (destinationPe == CkMyPe()) {
      // copying here is necessary - user code should not be 
      // passed back a reference to the original item
      packedDataItem.arrayIndex = arrayIndex; 
      packedDataItem.dataItem = dataItem;
      localDeliver(packedDataItem);
      return;
    }

    // this implementation avoids copying an item before transfer into message

    static DataItemHandle tempHandle; 
    tempHandle.arrayIndex = arrayIndex; 
    tempHandle.dataItem = &dataItem;

    MeshStreamer<ArrayDataItem<dtype> >::insertData(&tempHandle, destinationPe);

  }

  int copyDataItemIntoMessage(
      MeshStreamerMessage<ArrayDataItem <dtype> > *destinationBuffer, 
      void *dataItemHandle, bool copyIndirectly) {

    if (copyIndirectly == true) {
      // newly inserted items are passed through a handle to avoid copying
      int numDataItems = destinationBuffer->numDataItems;
      DataItemHandle *tempHandle = (DataItemHandle *) dataItemHandle;
      (destinationBuffer->data)[numDataItems].dataItem = 
	*(tempHandle->dataItem);
      (destinationBuffer->data)[numDataItems].arrayIndex = 
	tempHandle->arrayIndex;
      return ++destinationBuffer->numDataItems;
    }
    else {
      // this is an item received along the route to destination
      // we can copy it from the received message
      return MeshStreamer<ArrayDataItem<dtype> >::copyDataItemIntoMessage(destinationBuffer, dataItemHandle);
    }
  }

};

#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
