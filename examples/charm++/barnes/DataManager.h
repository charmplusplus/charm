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
#include "Profile.h"

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
  // The indices themselves, so the PE can say which tree pieces it holds
  // without walking the hashtable. Published every iteration so that every
  // sender knows where to address its block.
  CkVec<int> indices;
  TreePieceCounter() : count(0) { }                      
  void addLocation(CkLocation &loc) {
    registered.put(loc.getIndex()) = ++count;               
    indices.push_back(loc.getIndex().data()[0]);
  }
  void reset() {                                            
    count = 0;
    registered.empty();                                     
    indices.length() = 0;
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
  // Whether this iteration's tree came from the device, in which case its
  // moments are already in place.
  bool treeMirrored;
  // Whether this iteration's particles were assembled on the device. Not the
  // same question as whether a device is attached: the first exchange runs
  // before anything is, so it takes the host route and the device array is
  // still empty when ensureDevice() first succeeds.
  bool assembledOnDevice;
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

  // Stage 0 instrumentation. See Profile.h and GPU_RESIDENT_PLAN.md section 4.
  PhaseProfile prof;

  // The body of finishIteration, after the accelerations are in host memory.
  void finishIterationTail();

  // Correctness harness. Writes this PE's accelerations on the last iteration
  // when BARNES_ACCEL_DUMP names a prefix; compare_accel.py folds the per-PE
  // files together. See GPU_RESIDENT_PLAN.md section 4, Stage 0.
  void dumpAccelerations();

  // decompose() prints this and decomposeTail() needs it after the device
  // round trip, so it is held rather than passed.
  BoundingBox decomposeUniverse;

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
  // Fill the pending bins' descriptors, and their nodes' particle ranges, from
  // the device. Replaces walking a host copy of the particles to answer the
  // same three questions per bin.
  bool fillBinCounts();
  
  void senseTreePieces();

  // Stage 5a. tpToPe[t] is the PE holding tree piece t, refreshed every
  // iteration by a max reduction over each PE's own element set. A sender
  // needs it to group its sorting-tree leaves by destination PE.
  // The sorting tree's leaves, one per tree piece, recorded by the preorder
  // walk and consumed once the map says where each tree piece lives.
  struct LeafRef {
    Node<NodeDescriptor> *node;
    int tp;
    LeafRef() : node(NULL), tp(-1) {}
    LeafRef(Node<NodeDescriptor> *n, int t) : node(n), tp(t) {}
  };
  CkVec<LeafRef> leafList;

  CkVec<int> tpToPe;
  CkVec<ParticleBlockMsg *> recvdBlocks;
  int numBlocksRecvd;
  // How many PEs will actually send this one a block. Without it the exchange
  // has to send to every PE so the receiver can count to CkNumPes(), which is
  // P^2 messages -- 25600 at 160 ranks, nearly all empty, because SFC ordering
  // and a block map send a PE's particles to only a handful of neighbours.
  int expectedBlocks;
  bool haveExpected;
  // Per source: how many particles its block says are coming, and whether the
  // device payload has landed. Assembly waits for both halves.
  CkVec<int> srcParts;
  CkVec<char> srcPayloadIn;
  int payloadsRecvd;
  int outstandingExchangeSends;
  void publishTreePieceMap();
  void freeSortingTree();
  void maybeAssemble();
  void sendParticleBlocks();
  void assembleReceivedBlocks();
  void buildTree();
  // Mirror the device tree instead of rebuilding from the particles. The two
  // have identical structure -- the device build applies the same refine
  // predicate against the same key ranges -- so this copies structure,
  // ownership and moments across and leaves nothing for the host to read.
  bool buildTreeFromDevice();
  // Stage 6. Compares the flat device tree against the host tree built over
  // the same particles; reports under BARNES_TREE_CHECK. It is the only check
  // on the device build available while the traversal is still host code.
  void checkDeviceTree();
  // Push the host's completed moments for boundary nodes into the device tree.
  // The device build only sees particles resident here, so those nodes are
  // short the mass that lives on other PEs until this runs.
  void patchDeviceBoundaryMoments();

  void printTree();
  void flushParticles();

  void processSubmittedParticles();
  void makeMoments();
  // The ownership frontier: the nodes at which the owner range narrows to a
  // single tree piece. Determined by keyRanges alone, so every PE enumerates
  // the same list in the same order, which is what lets one reduction stand in
  // for the whole moment exchange.
  void collectFrontier(Node<ForceData> *n, CkVec<Node<ForceData>*> &out);
  void contributeFrontierMoments();
  void fillBoundaryMoments(Node<ForceData> *n);
  CkVec<Node<ForceData>*> frontier;

  // Stage 1 (LET). Each PE's own domain box, gathered from every PE by riding
  // along on the frontier reduction. A sender walks its tree against these to
  // decide what a destination could possibly need.
  CkVec<OrientedBox<Real> > peBoxes;
  OrientedBox<Real> myDomain;
  // What a push to `dest` would contain, without building it. The predicate is
  // the walk's own: a cell the destination would accept is sent as a
  // multipole, one it would open is descended into, and an opened leaf sends
  // its particles.
  void letSize(Node<ForceData> *n, const OrientedBox<Real> &dest,
               int &nodes, int &parts);
  void reportLetSizes();

  // Stage 1 (LET). Walk the local tree against a destination's domain and
  // collect what it could need; ship one message per destination; splice what
  // arrives. Runs alongside the request/reply path rather than replacing it,
  // so the request counters measure whether the push was actually sufficient.
  void collectLet(Node<ForceData> *n, const OrientedBox<Real> &dest,
                  CkVec<Key> &keys, CkVec<Real> &mom, CkVec<int> &npart,
                  CkVec<ExternalParticle> &parts);
  void sendLets();
  Node<ForceData> *descendToKey(Key k);
  int letsExpected, letsRecvd;
  bool letsDone;
  // A push can arrive before this PE's own frontier callback has run, and the
  // splice needs the frontier types to know what it may grow into. Hold them
  // until it has.
  bool frontierReady;
  CkVec<LetMsg *> pendingLets;
  void spliceLet(LetMsg *msg);
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
  void recordLeaf(Node<NodeDescriptor> *nd, int tp);
  void beginDistribute();
  void recvTreePieceMap(CkReductionMsg *msg);
  void recvSenderCounts(CkReductionMsg *msg);
  void exchangeSendDone();
  void recvFrontierMoments(CkReductionMsg *msg);
  void recvLet(LetMsg *msg);
  void receiveParticleBlock(ParticleBlockMsg *msg);
  // Stage 5b. The block message carries the bookkeeping; the particles come
  // straight from the sender's device memory into ours.
  void recvParticleDevice(int fromPe, int nbytes, char *buf);
  void recvParticleDevice(int fromPe, int &nbytes, char *&buf,
                          CkDeviceBufferPost *devicePost);

  void receiveMoments(MomentsMsg *msg);
  
  // called by tree pieces
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

  // The rest of advance(), once the integrator's reduction is in host memory.
  // Public because it is an entry method under GPU_GRAVITY; on the CPU path
  // advance() simply calls it.
  void advanceTail();
  // The rest of decompose(), once the keys are generated and the particles are
  // back. Same arrangement.
  void decomposeTail();

  void recvUnivBoundingBox(CkReductionMsg *msg);
  void treePiecesReady(CkReductionMsg *msg);
  void treePiecesMigrated(CkReductionMsg *msg);

#ifdef GPU_GRAVITY
  // Take a stream and the device arrays. Called from an entry method, never
  // from the constructor: HAPI has not picked this PE's device at that point.
  void ensureDevice();
  int particleOffset(Particle *p){ return (int)(p - myParticles.getVec()); }
  const float4 *devicePositions() const { return gpuParticles.positions(); }
  // The flat device tree the local walk traverses; NULL until it is built.
  const DeviceNode *deviceNodes() const { return gpuParticles.deviceNodes(); }
  cudaEvent_t treeEvent() const { return gpuParticles.treeEvent(); }
  float4 *deviceAccel() const { return gpuParticles.accel(); }
  cudaEvent_t uploadEvent() const { return gpuParticles.uploadEvent(); }
  void forcesReady();
#endif

  void quiescence();

  void addBucketNodeInteractions(Key k, CmiUInt8 pn);
  void addBucketPartInteractions(Key k, CmiUInt8 pp);
};

#endif
