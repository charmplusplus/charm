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
  lbState = LB_IDLE;
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
  lbState = LB_IDLE;
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

// thread_local: in this build PEs are threads in a process, so a file-scope
// double would aggregate the whole process and hide the per-PE distribution.
thread_local double _tpWalkLocal = 0.0;
thread_local double _tpWalkRemote = 0.0;

void TreePiece::doLocalGravity(RescheduleMsg *msg){
  const double _t0 = CmiWallTimer();
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

  _tpWalkLocal += CmiWallTimer() - _t0;

  if(localTraversalState.decrPending(i)){
    localGravityDone();
    delete msg;
  }
  else if(localTraversalState.current < myNumBuckets) {
    thisProxy[thisIndex].doLocalGravity(msg);
  }
}

void TreePiece::doRemoteGravity(RescheduleMsg *msg){
  const double _t0 = CmiWallTimer();
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

  _tpWalkRemote += CmiWallTimer() - _t0;

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
  return isBalancingIteration(globalParams, iteration);
}

void TreePiece::finishIteration(){
  localTraversalState.finishedIteration();
  remoteTraversalState.finishedIteration();

  initIterationState();

  iteration++;

  checkTraversals();

  // Every iteration ends at this barrier, whether or not the balancer runs, so
  // that the DataManager has one condition to wait on before it starts the
  // next decomposition. It has to wait somewhere: decompose() sends particles
  // to tree pieces by index and then counts the ones registered locally, and
  // both would be wrong if an element were still in flight. Where it waits is
  // what the two paths below differ in.
  if(iteration >= globalParams.iterations){
    static thread_local bool _walkPrinted = false;
    if(!_walkPrinted && getenv("BARNES_WALK_REPORT") != NULL){
      _walkPrinted = true;
      CkPrintf("[WALK] pe %2d local %.4f s  remote %.4f s  total %.4f s\n",
               CkMyPe(), _tpWalkLocal, _tpWalkRemote,
               _tpWalkLocal + _tpWalkRemote);
    }
  }

  if(!isLbIteration()){
    lbBarrierDone(0);
    return;
  }

  if(!globalParams.asyncLb){
    // The element stops here and the DataManager hears nothing until the whole
    // step -- strategy and migrations -- is over.
    lbState = LB_SYNC;
    AtSync();
    return;
  }

  // Close the measurement window. This has to happen here rather than in the
  // DataManager's own end-of-iteration hook: the last tree piece's
  // reportTraversalsDone() only *starts* the reduction that reaches advance(),
  // so advance() lands after this point, and the strategy would read a window
  // that stayed open across its own decision.
  if(globalParams.lbWindow > 0) LBTurnInstrumentOff();

  // Flushes this element's GPU counters and feeds MetaBalancer's sample
  // stream. A no-op when MetaBalancer is off, so it is safe to call on this
  // application's own -lbperiod cadence either way.
  AtSyncSample();
  // Set before the call, not after: an element stopped at the tentative count
  // is handed back through ResumeFromSync, which then has to know that this
  // iteration was never reported.
  lbState = LB_BLOCKED;
  if(AtSyncStart() == CkMigratable::AtSyncStatus::Blocked) return;
  startLbOverlap();
}

// What the split buys here. A tree piece has no work of its own between
// iterations -- the DataManager drives everything -- so the work that overlaps
// the step is the next decomposition: the kick-drift-kick, the universe
// bounding box reduction and every histogram round run while the strategy runs
// and elements move. Only senseTreePieces() needs the elements to be still,
// and lbMigrationDone() is what releases it.
void TreePiece::startLbOverlap(){
  lbState = LB_OVERLAP;
  lbBarrierDone(1);
  // Resumes inline when the step is already over, in which case ResumeFromSync
  // runs from inside this call and finds LB_OVERLAP, which is correct.
  AtSyncWait();
}

void TreePiece::ResumeFromSync(){
  // This may be a different PE from the one the constructor ran on.
  myDM = dataManagerProxy.ckLocalBranch();

  const int was = lbState;
  lbState = LB_IDLE;
  switch(was){
    case LB_OVERLAP:
      // The iteration was reported when the step started. This is the step
      // ending, which is the other half.
      lbMigrationDone();
      break;
    case LB_BLOCKED:
      // Released from the tentative count -- either joined to the step or let
      // go to keep iterating. Either way the iteration is still unreported.
      startLbOverlap();
      break;
    default:
      // Unsplit AtSync: one event, so one report, and nothing is moving by the
      // time it is made.
      lbBarrierDone(0);
      break;
  }
}

void TreePiece::lbBarrierDone(int stepInFlight){
  // max_int rather than nop: the reduction has to carry whether anything is
  // still moving. The DataManager would otherwise need its own copy of the
  // balancing cadence to work that out, and two counters that have to agree is
  // a worse thing to own than one extra int.
  CkCallback cb(CkIndex_DataManager::treePiecesReady(NULL), dataManagerProxy);
  contribute(sizeof(int),&stepInFlight,CkReduction::max_int,cb);
}

void TreePiece::lbMigrationDone(){
  CkCallback cb(CkIndex_DataManager::treePiecesMigrated(NULL), dataManagerProxy);
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
  // Travels because an element parked in AtSyncWait() can be moved by the step
  // it is waiting on, and the destination is where its ResumeFromSync runs.
  p | lbState;
  // Nothing else travels. AtSync is reached from finishIteration(), after the
  // interaction list has been consumed and every per-iteration counter has
  // been reset, so the destination reconstructs the rest in
  // initIterationState(). The device buffers are released by the destructor on
  // the old PE and taken again on first use on the new one.
}

#include "Traversal_defs.h"
