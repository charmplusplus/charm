#include "TreePiece.h"
#include "Messages.h"
#include "DataManager.h"
#include "Parameters.h"

#include <fstream>

extern CProxy_Main mainProxy;
extern CProxy_DataManager dataManagerProxy;
extern Parameters globalParams;

#ifdef GPU_GRAVITY
#include "StreamPool.h"
extern CProxy_StreamPool streamPool;
#endif

// Everything a tree piece owns is rederived every iteration: particles arrive
// from the decomposition, buckets are handed out by the local DataManager in
// prepare(), and the traversal states are reset in startTraversal(). So this
// is all a migrated tree piece needs to come up with, and it is why pup()
// carries only the iteration counter.
void TreePiece::initIterationState(){
  myNumParticles = 0;
  numDecompMsgsRecvd = 0;
  decompMsgsRecvd.length() = 0;
  // Reset per iteration, not just once. These bound the key range this tree
  // piece submitted, and startTraversal() splits the PE's buckets by them; a
  // range accumulated across iterations would describe a tree piece that no
  // longer exists, and after a migration it would not even describe this PE.
  smallestKey = ~Key(0);
  largestKey = Key(0);
  numTraversalsDone = 0;
  myNumBuckets = 0;
  myBuckets = NULL;
  root = NULL;
}

#ifdef GPU_GRAVITY
void TreePiece::initGpuState(){
  outstandingKernels = 0;
  traversalsComplete = false;
}
#endif

TreePiece::TreePiece() :
  localTraversalState(),
  remoteTraversalState(),
  localStateID(0),
  remoteStateID(1),
  localTraversalWorker(),
  remoteTraversalWorker(),
  totalNumTraversals(2),
  iteration(0)
{
  initIterationState();
#ifdef GPU_GRAVITY
  initGpuState();
#endif
  usesAtSync = true;
  myDM = dataManagerProxy.ckLocalBranch();
}

// The stock version of this constructor was empty, which left myDM dangling
// and every counter uninitialised. Nothing migrated before, so it never ran.
TreePiece::TreePiece(CkMigrateMessage *m) :
  CBase_TreePiece(m),
  localTraversalState(),
  remoteTraversalState(),
  localStateID(0),
  remoteStateID(1),
  localTraversalWorker(),
  remoteTraversalWorker(),
  totalNumTraversals(2),
  iteration(0)
{
  initIterationState();
#ifdef GPU_GRAVITY
  initGpuState();
#endif
  usesAtSync = true;
  myDM = dataManagerProxy.ckLocalBranch();
}

void TreePiece::receiveParticles(ParticleMsg *msg){
  int msgNumParticles = msg->numParticles;
  decompMsgsRecvd.push_back(msg);
  myNumParticles += msgNumParticles;
  numDecompMsgsRecvd++;
  if(smallestKey > msg->part[0].key) smallestKey = msg->part[0].key;
  if(largestKey < msg->part[msgNumParticles-1].key) largestKey = msg->part[msgNumParticles-1].key;

  if(numDecompMsgsRecvd == CkNumPes()){
    submitParticles();
    numDecompMsgsRecvd = 0;
  }
}

void TreePiece::receiveParticles(){
  numDecompMsgsRecvd++;
  if(numDecompMsgsRecvd == CkNumPes()){
    submitParticles();
    numDecompMsgsRecvd = 0;
  }
}

void TreePiece::submitParticles(){
  myDM->submitParticles(&decompMsgsRecvd,myNumParticles,this,smallestKey,largestKey);
}

void TreePiece::prepare(Node<ForceData> *_root, Node<ForceData> **buckets, int bucketStart, int bucketEnd){
  root = _root;
  myBuckets = buckets+bucketStart;
  myNumBuckets = bucketEnd-bucketStart;
}

void TreePiece::startTraversal(){
  numTraversalsDone = 0;
  trav.setDataManager(myDM);
#ifdef GPU_GRAVITY
  traversalsComplete = false;
#endif

  if(myNumBuckets == 0){
    localGravityDone();
    remoteGravityDone();
    return;
  }

  remoteTraversalState.reset(this,myNumBuckets,myBuckets);
  remoteTraversalWorker.reset(this,&remoteTraversalState,*myBuckets);
  RescheduleMsg *msg = new (NUM_PRIORITY_BITS) RescheduleMsg;
  *(int *)CkPriorityPtr(msg) = REMOTE_GRAVITY_PRIORITY;
  CkSetQueueing(msg, CK_QUEUEING_IFIFO);
  thisProxy[thisIndex].doRemoteGravity(msg);

  localTraversalState.reset(this,myNumBuckets,myBuckets);
  localTraversalWorker.reset(this,&localTraversalState,*myBuckets);

  msg = new (NUM_PRIORITY_BITS) RescheduleMsg;
  *(int *)CkPriorityPtr(msg) = LOCAL_GRAVITY_PRIORITY;
  CkSetQueueing(msg, CK_QUEUEING_IFIFO);
  thisProxy[thisIndex].doLocalGravity(msg);
}

void TreePiece::doLocalGravity(RescheduleMsg *msg){
  int i;
  for(i = 0; i < globalParams.yieldPeriod &&
                 localTraversalState.current < myNumBuckets;
                 i++){
    trav.topDownTraversal(root,&localTraversalWorker,&localTraversalState);
    localTraversalState.current++;
    localTraversalState.currentBucketPtr++;
    // Guarded. currentBucketPtr walks one past this tree piece's slice of the
    // PE's bucket vector on the last bucket, and for the last tree piece on a
    // PE that is one past the vector itself. The stock code read the pointer
    // there and never used it; setContext now looks the bucket up.
    if(localTraversalState.current < myNumBuckets)
      localTraversalWorker.setContext(*localTraversalState.currentBucketPtr);
  }

#ifdef GPU_GRAVITY
  // Bound the interaction list. This is an entry method of this chare, so a
  // launch from here is attributed to it.
  if(batch.wantsFlush()) flushGpu();
#endif

  if(localTraversalState.decrPending(i)){
    localGravityDone();
    delete msg;
  }
  else if(localTraversalState.current < myNumBuckets) {
    thisProxy[thisIndex].doLocalGravity(msg);
  }
}

void TreePiece::doRemoteGravity(RescheduleMsg *msg){
  int i;
  for(i = 0; i < globalParams.yieldPeriod &&
                 remoteTraversalState.current < myNumBuckets;
                 i++){
    trav.topDownTraversal(root,&remoteTraversalWorker,&remoteTraversalState);
    remoteTraversalState.current++;
    remoteTraversalState.currentBucketPtr++;
    if(remoteTraversalState.current < myNumBuckets)
      remoteTraversalWorker.setContext(*remoteTraversalState.currentBucketPtr);
  }

#ifdef GPU_GRAVITY
  if(batch.wantsFlush()) flushGpu();
#endif

  if(remoteTraversalState.decrPending(i)){
    remoteGravityDone();
    delete msg;
  }
  else if (remoteTraversalState.current < myNumBuckets) {
    thisProxy[thisIndex].doRemoteGravity(msg);
  }
}

void TreePiece::localGravityDone(){
  traversalDone();
}

void TreePiece::remoteGravityDone(){
  traversalDone();
}

void TreePiece::traversalDone(){
  numTraversalsDone++;

  if(numTraversalsDone != totalNumTraversals) return;

#ifdef GPU_GRAVITY
  // Not a direct call. A remote traversal can finish inside
  // DataManager::recvParticles or recvNode, where the running object is the
  // group and not this chare; a kernel launched there would be charged to the
  // group, which no balancer can move. Bouncing through an entry method of
  // this chare puts the final launch back under its correlation id.
  //
  // traversalsComplete is deliberately not set here. Between this send and its
  // delivery an earlier flush's HAPI callback can arrive, and if that callback
  // found the flag already set and no launch outstanding it would declare the
  // tree piece done while the tail of the interaction list was still sitting
  // in the batch.
  thisProxy[thisIndex].finishGpuWork();
#else
  reportTraversalsDone();
  finishIteration();
#endif
}

void TreePiece::reportTraversalsDone(){
#ifdef STATISTICS
  CmiUInt8 pn = localTraversalState.numInteractions[0]+remoteTraversalState.numInteractions[0];
  CmiUInt8 pp = localTraversalState.numInteractions[1]+remoteTraversalState.numInteractions[1];
  CmiUInt8 oc = localTraversalState.numInteractions[2]+remoteTraversalState.numInteractions[2];
  dataManagerProxy[CkMyPe()].traversalsDone(pn,pp,oc);
#else
  dataManagerProxy[CkMyPe()].traversalsDone();
#endif
}

#ifdef GPU_GRAVITY
int TreePiece::particleOffset(Particle *p) const {
  return myDM->particleOffset(p);
}

void TreePiece::ensureDevice(){
  if(batch.attached()) return;
  batch.attach(streamPool.ckLocalBranch()->acquire(), globalParams.gpuFlushLimit);
}

void TreePiece::flushGpu(){
  if(batch.empty()) return;
  ensureDevice();
  myDM->ensureDevice();

  CkCallback cb(CkIndex_TreePiece::gpuWorkDone(), thisProxy[thisIndex]);
  if(batch.flush(myDM->devicePositions(), myDM->deviceAccel(),
                 myDM->uploadEvent(), globalParams.epssq, cb)){
    outstandingKernels++;
  }
}

void TreePiece::finishGpuWork(){
  traversalsComplete = true;
  flushGpu();
  maybeReportDone();
}

void TreePiece::gpuWorkDone(){
  outstandingKernels--;
  maybeReportDone();
}

void TreePiece::maybeReportDone(){
  if(!traversalsComplete || outstandingKernels > 0) return;
  traversalsComplete = false;
  // Reporting only from here is what lets the DataManager read the
  // accumulator back with no event bookkeeping of its own: by the time every
  // local tree piece has reported, every kernel any of them launched has run.
  reportTraversalsDone();
  finishIteration();
}
#endif

bool TreePiece::isLbIteration() const {
  if(globalParams.lbPeriod <= 0) return false;
  if(iteration < globalParams.firstLbIteration) return false;
  if(iteration >= globalParams.iterations) return false;
  return ((iteration - globalParams.firstLbIteration) % globalParams.lbPeriod) == 0;
}

void TreePiece::finishIteration(){
  localTraversalState.finishedIteration();
  remoteTraversalState.finishedIteration();

  initIterationState();

  iteration++;

  checkTraversals();

  // Every iteration ends at this barrier, whether or not the balancer runs, so
  // that the DataManager has one condition to wait on before it starts the
  // next decomposition. It has to wait: decompose() sends particles to tree
  // pieces by index and then counts the ones registered locally, and both
  // would be wrong if an element were still in flight.
  if(isLbIteration()) AtSync();
  else lbBarrierDone();
}

void TreePiece::ResumeFromSync(){
  // This may be a different PE from the one the constructor ran on.
  myDM = dataManagerProxy.ckLocalBranch();
  lbBarrierDone();
}

void TreePiece::lbBarrierDone(){
  CkCallback cb(CkIndex_DataManager::treePiecesReady(NULL), dataManagerProxy);
  contribute(0,0,CkReduction::nop,cb);
}

void TreePiece::checkTraversals(){
}

void TreePiece::quiescence(){
  CkPrintf("QUIESCENCE tree piece %d proc %d submitted %d numBuckets %d trav_done %d outstanding local %d remote %d\n",
                thisIndex,
                CkMyPe(),
                myNumParticles,
                myNumBuckets,
                numTraversalsDone,
                localTraversalState.pending,
                remoteTraversalState.pending);

  CkCallback cb(CkIndex_Main::quiescenceExit(),mainProxy);
  contribute(0,0,CkReduction::sum_int,cb);
}

void TreePiece::requestMoments(Key k, int replyTo){
  myDM->requestMoments(k,replyTo);
}

void TreePiece::requestParticles(RequestMsg *msg){
  // forward the request to the DM, since TPs don't
  // really own particles or nodes
  myDM->requestParticles(msg);
}

void TreePiece::requestNode(RequestMsg *msg){
  myDM->requestNode(msg);
}

int TreePiece::getIteration() {
  return iteration;
}

void TreePiece::pup(PUP::er &p){
  p | iteration;
  // Nothing else travels. AtSync is reached from finishIteration(), after the
  // interaction list has been consumed and every per-iteration counter has
  // been reset, so the destination reconstructs the rest in
  // initIterationState(). The device buffers are released by the destructor on
  // the old PE and taken again on first use on the new one.
}

#include "Traversal_defs.h"
