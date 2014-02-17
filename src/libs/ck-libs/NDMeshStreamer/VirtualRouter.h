#ifndef VIRTUAL_ROUTER_H
#define VIRTUAL_ROUTER_H

#include <algorithm>
#include <vector>

// #define CMK_TRAM_CACHE_ROUTE

static const int routeNotFound = -1;

static const int initialCompletionStage = -2;
static const int finalCompletionStage = -1;

struct Route {
  int dimension;
  int dimensionIndex;
  int destinationPe;
};

struct CompletionStatus {
  int stageIndex;
  int numContributors;
  std::vector<int> dimensionsToFlush;
};

// use CRTP to avoid overhead of virtual functions
template <class Derived>
class VirtualRouter {

protected:

  int numDimensions_;
  int myIndex_;
  int numMembers_;
  std::vector<int> individualDimensionSizes_;
  std::vector<int> combinedDimensionSizes_;
  std::vector<int> myLocationIndex_;

  int initialRoutingDimension_;

#ifdef CMK_TRAM_CACHE_ROUTE
  std::vector <Route> cachedRoutes_;
  std::vector <bool>  isCached_;
#endif

public:

  // a temporary function to provide some necessary parameters
  void initializeRouter(int numDimensions,
                        int myIndex,
                        int *dimensionSizes) {
    numMembers_ = CkNumPes();
    numDimensions_ = numDimensions;
    myIndex_ = myIndex;
    individualDimensionSizes_.assign(dimensionSizes,
                                     dimensionSizes + numDimensions_);
    combinedDimensionSizes_.resize(numDimensions_);
    myLocationIndex_.resize(numDimensions_);

    int sumAlongAllDimensions = 0;
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

#ifdef CMK_TRAM_CACHE_ROUTE
    cachedRoutes_.resize(numMembers_);
    isCached_.resize(numMembers_);
    std::fill(isCached_, isCached_ + numMembers_, false);
#endif

    static_cast<Derived*>(this)->additionalInitialization();
  }

  // // skeleton for defining a new routing class:
  // class NewRouter: public VirtualRouter<NewRouter> {
  // public:
  //   // inline frequently accessed and/or short functions
  //   void additionalInitialization();
  //   inline int nextPeAlongRoute(int dimension, int dimensionIndex);
  //   inline void determineInitialRoute(int destinationPe,
  //                                     Route &routeToDestination);
  //   inline void determineRoute(int destinationPe,
  //                              int dimensionReceivedAlong,
  //                              Route &routeToDestination);
  //   void updateCompletionProgress(CompletionStatus &currentStatus);
  //   inline int numBuffersPerDimension(int dimension);
  //   inline int maxNumAllocatedBuffers();
  //   inline int numMsgTypes();
  //   inline bool isMessagePersonalized(int dimension);
  //   inline int dimensionReceived(int msgType);
  //   inline int determineMsgType(int dimension);
  //   inline bool isBufferInUse(int dimension, int index);
  //   inline bool isBroadcastSupported();
  // };
};

template <class Derived>
class MeshRouter: public VirtualRouter<Derived> {

private:

  inline void assignRoute(int dimension, int dimensionIndex,
                     Route &routeToDestination) {
      routeToDestination.dimension = dimension;
      routeToDestination.dimensionIndex = dimensionIndex;
      routeToDestination.destinationPe =
        nextPeAlongRoute(dimension, dimensionIndex);

#ifdef CMK_TRAM_CACHE_ROUTE
        this->cachedRoutes_[destinationPe] = routeToDestination;
        this->isCached_[destinationPe] = true;
#endif

  }

protected:

  inline int routeAlongDimension(int destinationPe, int dimension) {

    int blockIndex = destinationPe / this->combinedDimensionSizes_[dimension];

    int dimensionIndex =
      blockIndex - blockIndex / this->individualDimensionSizes_[dimension]
      * this->individualDimensionSizes_[dimension];

    return dimensionIndex;
  }

public:

  void additionalInitialization() {
    this->initialRoutingDimension_ = this->numDimensions_ - 1;
  }

  inline int nextPeAlongRoute(int dimension, int dimensionIndex) {
    int destinationPe =
      this->myIndex_ + (dimensionIndex - this->myLocationIndex_[dimension]) *
      this->combinedDimensionSizes_[dimension];

    return destinationPe;

  }

  inline void determineInitialRoute(int destinationPe,
                                    Route &routeToDestination) {
    // treat newly inserted items as if they were received along
    // a higher dimension (e.g. for a 3D mesh, received along 4th dimension)
    static_cast<Derived*>(this)->
      determineRoute(destinationPe, this->initialRoutingDimension_ + 1,
                     routeToDestination);
  }

  inline void determineRoute(int destinationPe, int dimensionReceivedAlong,
                             Route &routeToDestination) {

#ifdef CMK_TRAM_CACHE_ROUTE
    if (this->isCached_[destinationPe]) {
      return this->cachedRoutes_[destinationPe];
    }
#endif

    for (int i = dimensionReceivedAlong - 1; i >= 0; i--) {
      int dimensionIndex = routeAlongDimension(destinationPe, i);
      if (dimensionIndex != this->myLocationIndex_[i]) {
        static_cast<Derived*>(this)->
          assignRoute(i, dimensionIndex, routeToDestination);
        return;
      }
    }

    routeToDestination.dimension = routeNotFound;
  }

  void updateCompletionProgress(CompletionStatus &currentStatus) {
    if (currentStatus.stageIndex == initialCompletionStage) {
      currentStatus.stageIndex = this->numDimensions_ - 1;
    }
    else {
      currentStatus.stageIndex--;
    }

    int currentStage = currentStatus.stageIndex;
    if (currentStage == finalCompletionStage) {
      return;
    }

    currentStatus.numContributors =
        this->individualDimensionSizes_[currentStage] - 1;
    currentStatus.dimensionsToFlush.push_back(currentStage);
  }

  inline int numBuffersPerDimension(int dimension) {
    return this->individualDimensionSizes_[dimension];
  }

  inline int maxNumAllocatedBuffers() {
    int numBuffers = 0;
    for (int i = 0; i < this->numDimensions_; i++) {
      // no buffer is used for the same index as the sender's
      numBuffers += numBuffersPerDimension(i) - 1;
    }
    return numBuffers;
  }

  inline int numMsgTypes() {
    return this->numDimensions_;
  }

  inline bool isMessagePersonalized(int dimension) {
    return dimension == 0;
  }

  inline int dimensionReceived(int msgType) {
    // for MeshRouter, the type of the message is the dimension
    return msgType;
  }

  inline int determineMsgType(int dimension) {
    return dimension;
  }

  inline bool isBufferInUse(int dimension, int index) {
    return index != this->myLocationIndex_[dimension];
  }

  inline bool isBroadcastSupported() {
    return false;
  }

};

class SimpleMeshRouter: public MeshRouter<SimpleMeshRouter> {};

// The node-aware routing scheme partitions PEs into teams where each team
//  is responsible for sending messages to the peers along one dimension of the
//  topology; items that need to be sent along the non-assigned dimension are
//  first forwarded to one of the PEs in the team responsible for sending along
//  that dimension. Such forwarding messages will always be delivered intranode
//  as long as the last dimension in the topology comprises PEs within the same
//  node. The scheme improves aggregation at the cost of additional intranode
//  traffic.

class NodeAwareMeshRouter: public MeshRouter<NodeAwareMeshRouter> {

private:
  int myAssignedDim_;

  // the number of PEs per team assigned to buffer messages for each dimension
  int teamSize_;
  int dimensionOfArrivingMsgs_;
  std::vector<int> forwardingDestinations_;

  // messages can be categorized into two types:
  // (1) personalized messages sent directly to the final destination
  // (2) intermediate messages sent along the path to the destination
  enum {personalizedMsgType, forwardMsgType, msgTypeCount};

  int numSendingToMe(int msgType) {
    int senderCount = 0;
    if (msgType == forwardMsgType) {
      if (myAssignedDim_ > numDimensions_ - 1) {
        // one of the left over PEs that do not have an assigned dimension
        senderCount = 0;
      }
      else {
        // contributors forwarding messages along the last dimension
        senderCount = numDimensions_ - 1;
        // contributors sending along their assigned dimension
        if (myAssignedDim_ != numDimensions_ - 2) {
          senderCount +=
            individualDimensionSizes_[dimensionOfArrivingMsgs_] - 1;
        }
        // contributors that were not assigned a team
        int offset = myLocationIndex_[numDimensions_ - 1] +
          (numDimensions_ - myAssignedDim_) * teamSize_;

        while (offset < individualDimensionSizes_[numDimensions_ - 1]) {
          senderCount++;
          offset += teamSize_;
        }
      }
    }
    else if (msgType == personalizedMsgType) {
      senderCount =
        myAssignedDim_ == numDimensions_ - 1 ? teamSize_ - 1 : teamSize_;
    }
    else {
      CkAbort("In function NodeAwareMeshRouter::numSendingToMe(int msgType): "
              "invalid message type.\n");
    }
    return senderCount;
  }

public:

  void additionalInitialization() {

    // need at least as many PEs per node as dimensions in the virtual topology
    if (individualDimensionSizes_[numDimensions_ - 1] < numDimensions_) {
      CkAbort("Error: Last dimension in TRAM virtual topology must have size "
              "greater than or equal to number of dimensions in the topology.");
    }

    teamSize_ = individualDimensionSizes_[numDimensions_ - 1] / numDimensions_;
    myAssignedDim_ = myLocationIndex_[numDimensions_ - 1] / teamSize_;
    // if the number of PEs per node does not divide evenly by the number of
    //  dimensions, some PEs will be left with an invalid dimension assignment;
    //  this is fine - just use them to forward data locally
    if (myAssignedDim_ > numDimensions_ - 1) {
      dimensionOfArrivingMsgs_ = numDimensions_ - 1;
    }
    else {
      dimensionOfArrivingMsgs_ =
        myAssignedDim_ == numDimensions_ - 1 ? 0 : myAssignedDim_ + 1;
    }
    forwardingDestinations_.resize(numDimensions_);

    int baseIndex = myIndex_ - myLocationIndex_[numDimensions_ - 1] +
      myIndex_ % teamSize_;
    for (int i = 0; i < numDimensions_; i++) {
      forwardingDestinations_[i] = baseIndex + i * teamSize_;
    }

    // in this scheme skip routing along the last (intranode) dimension
    // until the last step
    initialRoutingDimension_ = numDimensions_ - 2;

  }

  inline void assignRoute(int dimension, int dimensionIndex,
                   Route &routeToDestination) {
      routeToDestination.dimension = dimension;
      routeToDestination.dimensionIndex =
        myAssignedDim_ == dimension ? dimensionIndex : 0;
      routeToDestination.destinationPe =
        nextPeAlongRoute(dimension, dimensionIndex);
  }

  inline int nextPeAlongRoute(int dimension, int dimensionIndex) {
    int destinationPe;

    if (dimension == myAssignedDim_) {
      destinationPe =
        MeshRouter<NodeAwareMeshRouter>::nextPeAlongRoute(dimension,
                                                          dimensionIndex);

      // adjust the destination index so that the message will arrive at
      //  a PE responsible for sending along the proper dimension
      if (dimension == 0) {
        destinationPe += (numDimensions_ - 1) * teamSize_;
      }
      else if (dimension != numDimensions_ - 1) {
        destinationPe -= teamSize_;
      }
    }
    else {
      // items to be sent along dimensions other than my assigned dimension
      // are batched into a single message per dimension, to be forwarded
      // intranode to a responsible PE
      destinationPe = forwardingDestinations_[dimension];
    }

    return destinationPe;
  }

  inline void determineRoute(int destinationPe, int dimensionReceivedAlong,
                             Route &routeToDestination) {

    MeshRouter<NodeAwareMeshRouter>::
      determineRoute(destinationPe, dimensionReceivedAlong, routeToDestination);

    if (routeToDestination.dimension == routeNotFound) {
      int dimensionIndex =
        routeAlongDimension(destinationPe, numDimensions_ - 1);
      assignRoute(numDimensions_ - 1, dimensionIndex, routeToDestination);
    }

  }

  void updateCompletionProgress(CompletionStatus &currentStatus) {
    if (currentStatus.stageIndex == initialCompletionStage) {
      currentStatus.stageIndex = forwardMsgType;
    }
    else {
      currentStatus.stageIndex--;
    }

    int currentStage = currentStatus.stageIndex;
    if (currentStage == finalCompletionStage) {
      return;
    }

    currentStatus.numContributors = numSendingToMe(currentStage);

    if (currentStatus.stageIndex == forwardMsgType) {
      for (int i = numDimensions_ - 2; i >= dimensionOfArrivingMsgs_;
           i--) {
        currentStatus.dimensionsToFlush.push_back(i);
      }
    }
    else {
      for (int i = dimensionOfArrivingMsgs_ - 1; i >= 0; i--) {
        currentStatus.dimensionsToFlush.push_back(i);
      }
      currentStatus.dimensionsToFlush.push_back(numDimensions_ - 1);
    }
  }

  inline int numBuffersPerDimension(int dimension) {
    int numBuffers =
      dimension == myAssignedDim_ ? individualDimensionSizes_[dimension] : 1;

    return numBuffers;
  }

  inline int maxNumAllocatedBuffers() {
    int numBuffers = 0;
    for (int i = 0; i < numDimensions_; i++) {
      numBuffers += numBuffersPerDimension(i);
    }
    return numBuffers;
  }

  inline int numMsgTypes() {
    return msgTypeCount;
  }

  inline bool isMessagePersonalized(int dimension) {
    return
      dimension == numDimensions_ - 1 && myAssignedDim_ == numDimensions_ - 1;
  }

  inline int dimensionReceived(int msgType) {
    CkAssert(msgType == forwardMsgType);
    return dimensionOfArrivingMsgs_;
  }

  inline int determineMsgType(int dimension) {
    return !isMessagePersonalized(dimension);
  }

  inline bool isBufferInUse(int dimension, int index) {
    return dimension != myAssignedDim_
      || index != myLocationIndex_[dimension];
  }

  inline bool isBroadcastSupported() {
    return false;
  }

};

#endif
