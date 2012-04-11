#ifndef NDMESH_STREAMER_H
#define NDMESH_STREAMER_H

#include <algorithm>
#include "NDMeshStreamer.decl.h"
#include "DataItemTypes.h"
#include "completion.h"
#include "ckarray.h"

// allocate more total buffer space than the maximum buffering limit but flush 
//   upon reaching totalBufferCapacity_
#define BUFFER_SIZE_FACTOR 4

// #define DEBUG_STREAMER
// #define CACHE_LOCATIONS
// #define SUPPORT_INCOMPLETE_MESH
// #define CACHE_ARRAY_METADATA // only works for 1D array clients

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
class MeshStreamerClient {
 protected:
  CompletionDetector *detectorLocalObj_;
 public:
  // would like to make it pure virtual but charm will try to
  // instantiate the abstract class, leading to errors
  virtual void process(dtype &data) {
    CkAbort("Error. MeshStreamerClient::process() is being called. "
            "This virtual function should have been defined by the user.\n");
  };     
  void setDetector(CompletionDetector *detectorLocalObj) {
    detectorLocalObj_ = detectorLocalObj;
  }
};

template <class dtype>
class MeshStreamerGroupClient : public CBase_MeshStreamerGroupClient<dtype>,
  public MeshStreamerClient<dtype> {
 public:

  virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      dtype &data = msg->getDataItem(i);
      process(data);
    }
    MeshStreamerClient<dtype>::detectorLocalObj_->consume(msg->numDataItems);
    delete msg;
  }
};

template <class dtype>
class MeshStreamerArrayClient :  public CBase_MeshStreamerArrayClient<dtype>, 
  public MeshStreamerClient<dtype>
{

public:

  // virtual void receiveCombinedData(MeshStreamerMessage<dtype> *msg);
  MeshStreamerArrayClient() {}
  MeshStreamerArrayClient(CkMigrateMessage *msg) {}
  void receiveRedeliveredItem(dtype data) {
    MeshStreamerClient<dtype>::detectorLocalObj_->consume();
    process(data);
  }

  void pup(PUP::er &p) {
    CBase_MeshStreamerArrayClient<dtype>::pup(p);
  }

};

template <class dtype>
class MeshStreamerArray2DClient : 
  public CBase_MeshStreamerArray2DClient<dtype>, 
  public MeshStreamerClient<dtype> 
{

public:
  MeshStreamerArray2DClient() {}
  MeshStreamerArray2DClient(CkMigrateMessage *msg) {}
  void receiveRedeliveredItem(dtype data) {
    MeshStreamerClient<dtype>::detectorLocalObj_->consume();
    process(data);
  }
  void pup(PUP::er &p) {
    CBase_MeshStreamerArray2DClient<dtype>::pup(p);
  }

};

template <class dtype>
class MeshStreamerArray3DClient : 
  public CBase_MeshStreamerArray3DClient<dtype>, 
  public MeshStreamerClient<dtype> 
{

public:
  MeshStreamerArray3DClient() {}
  MeshStreamerArray3DClient(CkMigrateMessage *msg) {}
  void receiveRedeliveredItem(dtype data) {
    MeshStreamerClient<dtype>::detectorLocalObj_->consume();
    process(data);
  }
  void pup(PUP::er &p) {
    CBase_MeshStreamerArray3DClient<dtype>::pup(p);
  }

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
    bool yieldFlag_;

    double progressPeriodInMs_; 
    bool isPeriodicFlushEnabled_; 
    bool hasSentRecently_;

    MeshStreamerMessage<dtype> ***dataBuffers_;

    CProxy_CompletionDetector detector_;
    int prio_;

#ifdef CACHE_LOCATIONS
    MeshLocation *cachedLocations_;
    bool *isCached_; 
#endif

    MeshLocation determineLocation(int destinationPe);

    void storeMessage(int destinationPe, 
		      const MeshLocation &destinationCoordinates, 
		      void *dataItem, bool copyIndirectly = false);

    virtual void deliverToDestination(
                 int destinationPe, 
                 MeshStreamerMessage<dtype> *destinationBuffer) = 0;

    virtual void localDeliver(dtype &dataItem) = 0; 

    virtual int numElementsInClient() = 0;

    virtual void initLocalClients() = 0;

    void flushLargestBuffer();

protected:

    CompletionDetector *detectorLocalObj_;
    virtual int copyDataItemIntoMessage(
		MeshStreamerMessage<dtype> *destinationBuffer, 
		void *dataItemHandle, bool copyIndirectly = false);

public:

    MeshStreamer(int totalBufferCapacity, int numDimensions, 
		 int *dimensionSies,
                 bool yieldFlag = 0, double progressPeriodInMs = -1.0);
    ~MeshStreamer();

      // entry
    void receiveAlongRoute(MeshStreamerMessage<dtype> *msg);
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
      detectorLocalObj_->done(numContributorsFinished);
    }

};

template <class dtype>
MeshStreamer<dtype>::MeshStreamer(
		     int totalBufferCapacity, int numDimensions, 
		     int *dimensionSizes, 
		     bool yieldFlag, 
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
  detectorLocalObj_ = NULL;

#ifdef CACHE_LOCATIONS
  cachedLocations_ = new MeshLocation[numMembers_];
  isCached_ = new bool[numMembers_];
  std::fill(isCached_, isCached_ + numMembers_, false);
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
  delete[] cachedLocations_;
  delete[] isCached_; 
#endif

}


template <class dtype>
inline
MeshLocation MeshStreamer<dtype>::determineLocation(int destinationPe) { 

#ifdef CACHE_LOCATIONS
  if (isCached_[destinationPe]) {    
    return cachedLocations_[destinationPe]; 
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
      cachedLocations_[destinationPe] = destinationLocation;
      isCached_[destinationPe] = true; 
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
        new (0, bufferSize_, sizeof(int)) MeshStreamerMessage<dtype>();
    }
    else {
      messageBuffers[bufferIndex] = 
        new (bufferSize_, bufferSize_, sizeof(int)) MeshStreamerMessage<dtype>();
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
      deliverToDestination(destinationIndex, destinationBuffer);
    }
    else {
      this->thisProxy[destinationIndex].receiveAlongRoute(destinationBuffer);
    }

    messageBuffers[bufferIndex] = NULL;
    numDataItemsBuffered_ -= numBuffered; 
    hasSentRecently_ = true; 

  }
  // send if total buffering capacity has been reached
  else if (numDataItemsBuffered_ == totalBufferCapacity_) {
    flushLargestBuffer();
    hasSentRecently_ = true; 
  }

}

template <class dtype>
inline
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

  detectorLocalObj_->produce();
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
void MeshStreamer<dtype>::associateCallback(
			  int numContributors,
			  CkCallback startCb, CkCallback endCb, 
			  CProxy_CompletionDetector detector, 
			  int prio) {
  prio_ = prio;
  userCallback_ = endCb; 
  static CkCallback finish(CkIndex_MeshStreamer<dtype>::finish(), 
			   this->thisProxy);
  detector_ = detector;      
  detectorLocalObj_ = detector_.ckLocalBranch();
  initLocalClients();

  detectorLocalObj_->start_detection(numContributors, startCb, finish , 0);
  
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
	*((int *) env->getPrioPtr()) = prio_;
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
	    new (0, 1, sizeof(int)) MeshStreamerMessage<dtype>();
	  *(int *) CkPriorityPtr(directMsg) = prio_;
	  CkSetQueueing(directMsg, CK_QUEUEING_IFIFO);

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

  void deliverToDestination(int destinationPe, 
                            MeshStreamerMessage<dtype> *destinationBuffer) {
    clientProxy_[destinationPe].receiveCombinedData(destinationBuffer);
  }

  void localDeliver(dtype &dataItem) {
    clientObj_->process(dataItem);
    MeshStreamer<dtype>::detectorLocalObj_->consume();
  }

  int numElementsInClient() {
    // client is a group - there is one element per PE
    return CkNumPes();
  }

  void initLocalClients() {
    clientObj_->setDetector(MeshStreamer<dtype>::detectorLocalObj_);
  }

public:

  GroupMeshStreamer(int totalBufferCapacity, int numDimensions,
		    int *dimensionSizes, 
		    const CProxy_MeshStreamerGroupClient<dtype> &clientProxy,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0)
   :MeshStreamer<dtype>(totalBufferCapacity, numDimensions, dimensionSizes, 
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

    ((MeshStreamerClient<dtype> *) (clientArrMgr_->lookup(loc.getIndex())))
      ->setDetector(detectorLocalObj_); 

  }

};

template <class dtype, class ctype, class itype>
class ArrayMeshStreamer : public MeshStreamer<ArrayDataItem<dtype, itype> > {
  
private:
  
  ctype clientProxy_;
  CkArray *clientArrayMgr_;
  int numArrayElements_;
#ifdef CACHE_ARRAY_METADATA
  MeshStreamerArrayClient<dtype> **clientObjs_;
  int *destinationPes_;
  bool *isCachedArrayMetadata_;
#endif

  void deliverToDestination(
       int destinationPe, 
       MeshStreamerMessage<ArrayDataItem<dtype, itype> > *destinationBuffer) {
    ( (CProxy_ArrayMeshStreamer<dtype, ctype, itype>) 
      this->thisProxy )[destinationPe].receiveArrayData(destinationBuffer);
  }

  void localDeliver(ArrayDataItem<dtype, itype> &packedDataItem) {
    itype arrayId = packedDataItem.arrayIndex; 

    MeshStreamerClient<dtype> *clientObj;
#ifdef CACHE_ARRAY_METADATA
    clientObj = clientObjs_[arrayId];
#else
    clientObj = clientProxy_[arrayId].ckLocal();
#endif

    if (clientObj != NULL) {
      clientObj->process(packedDataItem.dataItem);
      MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->consume();
    }
    else { 
      // array element is no longer present locally - redeliver using proxy
      clientProxy_[arrayId].receiveRedeliveredItem(packedDataItem.dataItem);
    }
  }

  int numElementsInClient() {
    return numArrayElements_;
  }

  void initLocalClients() {

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

  }

public:

  struct DataItemHandle {
    itype arrayIndex; 
    dtype *dataItem;
  };

  ArrayMeshStreamer(int totalBufferCapacity, int numDimensions,
		    int *dimensionSizes, const ctype &clientProxy,
		    bool yieldFlag = 0, double progressPeriodInMs = -1.0)
    :MeshStreamer<ArrayDataItem<dtype, itype> >(
      totalBufferCapacity, numDimensions, dimensionSizes, yieldFlag, 
      progressPeriodInMs) 
  {
    clientProxy_ = clientProxy; 
    clientArrayMgr_ = clientProxy_.ckLocalBranch();

    numArrayElements_ = (clientArrayMgr_->getNumInitial()).data()[0];

#ifdef CACHE_ARRAY_METADATA
    clientObjs_ = new MeshStreamerArrayClient<dtype>*[numArrayElements_];
    destinationPes_ = new int[numArrayElements_];
    isCachedArrayMetadata_ = new bool[numArrayElements_];
    std::fill(isCachedArrayMetadata_, 
	      isCachedArrayMetadata_ + numArrayElements_, false);
#endif
  }

  ~ArrayMeshStreamer() {
#ifdef CACHE_ARRAY_METADATA
    delete [] clientObjs_;
    delete [] destinationPes_;
    delete [] isCachedArrayMetadata_; 
#endif
  }

  void receiveArrayData(
       MeshStreamerMessage<ArrayDataItem<dtype, itype> > *msg) {
    for (int i = 0; i < msg->numDataItems; i++) {
      ArrayDataItem<dtype, itype> &packedData = msg->getDataItem(i);
      localDeliver(packedData);
    }
    delete msg;
  }

  void insertData(dtype &dataItem, itype arrayIndex) {

    MeshStreamer<ArrayDataItem<dtype, itype> >::detectorLocalObj_->produce();
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

  static ArrayDataItem<dtype, itype> packedDataItem;
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
      (destinationBuffer->data)[numDataItems].dataItem = 
	*(tempHandle->dataItem);
      (destinationBuffer->data)[numDataItems].arrayIndex = 
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
