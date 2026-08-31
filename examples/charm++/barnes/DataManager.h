#ifndef __DATA_MANAGER_H__
#define __DATA_MANAGER_H__

#include "Particle.h"

#include "OrientedBox.h"
#include "barnes.decl.h"
#include "Node.h"
#include "Descriptor.h"
#include "ActiveBinInfo.h"

#include "Traversal_decls.h"
#include "Request.h"

#ifdef GPU_GRAVITY
#include "GpuBatch.h"
#endif

class TreePiece;

#include <map>
using namespace std;

class TreePieceCounter : public CkLocIterator {            
  public:
  int count;
  CkHashtableT<CkArrayIndex, int> registered;               
  TreePieceCounter() : count(0) { }                      
  void addLocation(CkLocation &loc) {
    registered.put(loc.getIndex()) = ++count;               
  }
  void reset() {                                            
    count = 0;
    registered.empty();                                     
  }                                                         
};

struct RequestedMomentsDescriptor {
  Node<ForceData> *node;
  int numOutstanding;

  RequestedMomentsDescriptor() : 
    node(NULL), numOutstanding(-1)
  {
  }

  RequestedMomentsDescriptor(Node<ForceData> *nd, int n) : 
    node(nd), numOutstanding(n)
  {
  }

};

struct CacheStats {
  int outstandingRequests;
  int outstandingDeliveries;

  CacheStats() : 
    outstandingRequests(0),
    outstandingDeliveries(0)
  {
  }

  void incrRequests(){ outstandingRequests++; }
  void decrRequests(int n=1){ outstandingRequests -= n; }
  void incrDeliveries(){ outstandingDeliveries++; }
  void decrDeliveries(int n=1){ outstandingDeliveries -= n; }
  bool test(){ return (outstandingRequests==0) && (outstandingDeliveries==0); }
};

class DataManager : public CBase_DataManager {
  int numRankBits;
  double prevIterationStart;
  double avgIterationRuntime;


  CkVec<Particle> myParticles;
  int myNumParticles;

  bool firstSplitterRound;

  Node<NodeDescriptor> *sortingRoot;
  int numTreePieces;

  int iteration;
  int decompIterations;
  ActiveBinInfo<NodeDescriptor> activeBins;

  TreePieceCounter localTreePieces;
  int numLocalTreePieces;
  CkVec<TreePieceDescriptor> submittedParticles;
  Node<ForceData> *root;

  Key *keyRanges;
  bool haveRanges;
  RangeMsg *rangeMsg;
  CkVec<Node<ForceData>*> myBuckets;

  // I am done constructing the tree 
  // from particles present on this PE
  bool doneTreeBuild;
  //CkVec<RequestedMomentsDescriptor> requestedMoments;
  map<Key,Node<ForceData>*> nodeTable;

  map<Key,CkVec<int> > pendingMoments;

  // I have processed the moment 
  // contributions from all other PEs, so that
  // the tree on this PE is now ready for 
  // traversal
  bool treeMomentsReady;
  CkVec<RequestMsg *> bufferedNodeRequests;
  CkVec<RequestMsg *> bufferedParticleRequests;

  Traversal<NodeDescriptor> scaffoldTrav;
  Traversal<ForceData> fillTrav;

  map<Key,Request> nodeRequestTable;
  map<Key,Request> particleRequestTable;

  int numTreePiecesDoneTraversals;
  CacheStats nodeReqs;
  CacheStats partReqs;

  Real savedEnergy;

  // The next iteration cannot start until both the universe bounding box has
  // been reduced and every tree piece has come through AtSync, because
  // decompose() sends particles to tree pieces by index and then counts the
  // ones registered on this PE. Neither condition implies the other, so both
  // are latched and whichever arrives second starts the iteration.
  bool haveUniverse;
  bool treePiecesSettled;
  BoundingBox nextUniverse;
  void startNextIteration();

  // Async LB. With the split barrier the end of an iteration and the end of
  // the balancing step it started arrive separately, so the decomposition may
  // begin -- and run all the way through its histogram rounds -- while
  // elements are still moving. senseTreePieces() is the one point that may
  // not: it snapshots the local element set and processSubmittedParticles()
  // then waits for exactly that many submissions, so an element arriving or
  // leaving across it either stalls the tree build or overruns the vector.
  //
  // migrationsSettled is that release. Under the unsplit barrier it is already
  // set by the time the decomposition gets here, so this path costs nothing.
  // Starts set: nothing is migrating at startup, and the first decomposition
  // is driven from Main before any tree piece has finished an iteration and so
  // before anything could set it.
  bool migrationsSettled;
  // Set when the decomposition reached the point above. Whichever of the two
  // is second runs the rest.
  bool atDistribute;
  // Held across the gate on the PEs that are handed their ranges.
  RangeMsg *pendingRangeMsg;
  void distributeParticles();

  // Opens and closes the load-balancing measurement window. Lives here, not in
  // the tree pieces, because a balancer is allowed to leave a PE with none and
  // the window still has to be managed there. The closing edge is duplicated
  // in TreePiece::finishIteration, which is the only place that runs before
  // the decision; both calls are idempotent.
  void updateLbInstrumentation();

  // Whether a balancing step was still running when this iteration's tree
  // pieces reported, and when the decomposition first reached the gate without
  // it having finished. Together they say what the split actually bought: the
  // decomposition either covered the step outright or stalled, and by how
  // long.
  bool stepThisIteration;
  double decompStalledAt;

  // The body of finishIteration, after the accelerations are in host memory.
  void finishIterationTail();

#ifdef GPU_GRAVITY
  GpuParticleStore gpuParticles;
#endif

#ifdef STATISTICS
  CmiUInt8 numInteractions[3];
#endif

  void kickDriftKick(OrientedBox<Real> &box, Real &energy);

  void hashParticleCoordinates(const OrientedBox<Real> &universe);
  void initHistogramParticles();
  void sendHistogram();
  
  void senseTreePieces();
  void buildTree();

  void printTree();
  void flushParticles();

  void processSubmittedParticles();
  void makeMoments();
  void flushMomentRequests();
  void respondToMomentsRequest(Node<ForceData> *,CkVec<int>&);
  Node<ForceData> *lookupNode(Key k);

  void updateLeafMoments(Node<ForceData> *node, MomentsExchangeStruct &data);
  void passMomentsUpward(Node<ForceData> *node);
  void treeReady();

  void startTraversal();
  void flushBufferedRemoteDataRequests();

  void freeCachedData();
  void freeTree();
  void finishIteration();

  void findMinVByA(DtReductionStruct &);

  void markNaNBuckets();

  public:
  DataManager();

  void loadParticles(const CkCallback &cb);

  void decompose(const BoundingBox &universe);
  void receiveHistogram(CkReductionMsg *msg);
  void receiveSplitters(SplitterMsg *msg);
  void sendParticles(RangeMsg *msg);
  void sendParticlesToTreePiece(Node<NodeDescriptor> *nd, int tp);

  void receiveMoments(MomentsMsg *msg);
  
  // called by tree pieces
  void submitParticles(CkVec<ParticleMsg *> *vec, int numParticles, TreePiece *tp, Key smallestKey, Key largestKey); 
  void requestMoments(Key k, int replyTo);
  void advance(CkReductionMsg *);
#ifdef STATISTICS
  void traversalsDone(CmiUInt8 pnInter, CmiUInt8 ppInter, CmiUInt8 openCrit);
#else
  void traversalsDone();
#endif

  // called by tree piece that is making a request
  void requestNode(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *callbackTraversal);
  void requestParticles(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *callbackTraversal);

  // called by tree piece that is forwarding a remote request
  void requestNode(RequestMsg *msg);
  void requestParticles(RequestMsg *msg);
  
  void recvParticles(ParticleReplyMsg *msg);
  void recvNode(NodeReplyMsg *msg);

  void recvUnivBoundingBox(CkReductionMsg *msg);
  void treePiecesReady(CkReductionMsg *msg);
  void treePiecesMigrated(CkReductionMsg *msg);

#ifdef GPU_GRAVITY
  // Take a stream and the device arrays. Called from an entry method, never
  // from the constructor: HAPI has not picked this PE's device at that point.
  void ensureDevice();
  int particleOffset(Particle *p){ return (int)(p - myParticles.getVec()); }
  const float4 *devicePositions() const { return gpuParticles.positions(); }
  float4 *deviceAccel() const { return gpuParticles.accel(); }
  cudaEvent_t uploadEvent() const { return gpuParticles.uploadEvent(); }
  void forcesReady();
#endif

  void quiescence();

  void addBucketNodeInteractions(Key k, CmiUInt8 pn);
  void addBucketPartInteractions(Key k, CmiUInt8 pp);
};

#endif
