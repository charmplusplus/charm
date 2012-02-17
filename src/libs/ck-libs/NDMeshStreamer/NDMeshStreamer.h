#ifndef _NDMESH_STREAMER_H_
#define _NDMESH_STREAMER_H_

#include <algorithm>
#include "NDMeshStreamer.decl.h"

// allocate more total buffer space than the maximum buffering limit but flush 
//   upon reaching totalBufferCapacity_
#define BUFFER_SIZE_FACTOR 4

// #define DEBUG_STREAMER
// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH

typedef struct {
  int dimension; 
  int bufferIndex; 
} MeshLocation;

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
    int bufferSize_; 
    int totalBufferCapacity_;
    int numDataItemsBuffered_;

    int numNodes_; 
    int numDimensions_;
    int *individualDimensionSizes_;
    int *combinedDimensionSizes_;

    CProxy_MeshStreamerClient<dtype> clientProxy_;
    MeshStreamerClient<dtype> *clientObj_;

    int myNodeIndex_;
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

    /*
#ifdef SUPPORT_INCOMPLETE_MESH
    int numNodesInLastPlane_;
    int numFullRowsInLastPlane_;
    int numColumnsInLastRow_;
#endif
    */

    MeshLocation determineLocation(int destinationPe);

    void storeMessage(int destinationPe, 
		      const MeshLocation &destinationCoordinates, 
		      const dtype &dataItem);

    void flushLargestBuffer();

public:

    MeshStreamer(int totalBufferCapacity, int numDimensions,
		 int *dimensionSizes, 
		 const CProxy_MeshStreamerClient<dtype> &clientProxy,
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
    void insertData(dtype &dataItem, int destinationPe); 
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
void MeshStreamerClient<dtype>::receiveCombinedData(
                                MeshStreamerMessage<dtype> *msg) {
  for (int i = 0; i < msg->numDataItems; i++) {
     dtype data = ((dtype*)(msg->data))[i];
     process(data);
  }
  delete msg;
}

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(
		     int totalBufferCapacity, int numDimensions, 
		     int *dimensionSizes, 
                     const CProxy_MeshStreamerClient<dtype> &clientProxy,
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
  numNodes_ = CkNumPes(); 
  clientProxy_ = clientProxy; 
  clientObj_ = ((MeshStreamerClient<dtype> *)CkLocalBranch(clientProxy_));

  dataBuffers_ = new MeshStreamerMessage<dtype> **[numDimensions_]; 
  for (int i = 0; i < numDimensions; i++) {
    int numNodesAlongDimension = individualDimensionSizes_[i]; 
    dataBuffers_[i] = new MeshStreamerMessage<dtype> *[numNodesAlongDimension];

    for (int j = 0; j < numNodesAlongDimension; j++) {
      dataBuffers_[i][j] = NULL;
    }
  }

  // determine location indices for this node
  myNodeIndex_ = CkMyPe();

  int remainder = myNodeIndex_;
  for (int i = numDimensions_ - 1; i >= 0; i--) {    
    myLocationIndex_[i] = remainder / combinedDimensionSizes_[i];
    remainder -= combinedDimensionSizes_[i] * myLocationIndex_[i];
  }

  isPeriodicFlushEnabled_ = false; 

#ifdef CACHE_LOCATIONS
  cachedLocations = new MeshLocation[numNodes_];
  isCached = new bool[numNodes_];
  std::fill(isCached, isCached + numNodes_, false);
#endif

  /*
#ifdef SUPPORT_INCOMPLETE_MESH
  numNodesInLastPlane_ = numNodes_ % planeSize_; 
  numFullRowsInLastPlane_ = numNodesInLastPlane_ / numColumns_;
  numColumnsInLastRow_ = numNodesInLastPlane_ - 
    numFullRowsInLastPlane_ * numColumns_;  
#endif
  */
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
void MeshStreamer<dtype>::storeMessage(
			  int destinationPe, 
			  const MeshLocation& destinationLocation,
			  const dtype &dataItem) {

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
  
  int numBuffered = destinationBuffer->addDataItem(dataItem); 
  if (dimension != 0) {
    destinationBuffer->markDestination(numBuffered-1, destinationPe);
  }
  numDataItemsBuffered_++;

  // copy data into message and send if buffer is full
  if (numBuffered == bufferSize_) {

    int destinationIndex;

    destinationIndex = myNodeIndex_ + 
      (bufferIndex - myLocationIndex_[dimension]) * 
      combinedDimensionSizes_[dimension];

    if (dimension == 0) {
      clientProxy_[destinationIndex].receiveCombinedData(destinationBuffer);      
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

  if (numDataItemsBuffered_ == totalBufferCapacity_) {
    flushLargestBuffer();
    if (isPeriodicFlushEnabled_) {
      timeOfLastSend_ = CkWallTimer();
    }
  }

}

template <class dtype>
void MeshStreamer<dtype>::insertData(dtype &dataItem, int destinationPe) {
  static int count = 0;

  if (destinationPe == CkMyPe()) {
    clientObj_->process(dataItem);
    return;
  }

  MeshLocation destinationLocation = determineLocation(destinationPe);
  storeMessage(destinationPe, destinationLocation, dataItem); 

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
      clientObj_->process(dataItem);
    }
    else {
      storeMessage(destinationPe, destinationLocation, dataItem);   
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
      destinationIndex = myNodeIndex_ + 
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
	clientProxy_[destinationIndex].receiveCombinedData(destinationBuffer);
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
	int destinationPe = myNodeIndex_ + j - myLocationIndex_[i];
	clientProxy_[destinationPe].receiveCombinedData(messageBuffers[j]);
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
	  clientProxy_[destinationPe].receiveCombinedData(directMsg);
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
  CcdCallFnAfter(periodicProgressFunction<dtype>, (void *) this, progressPeriodInMs_); 
}


#define CK_TEMPLATES_ONLY
#include "NDMeshStreamer.def.h"
#undef CK_TEMPLATES_ONLY

#endif
