#include "DataManager.h"
#include "Reduction.h"
#include "defines.h"
#include "Messages.h"
#include "Parameters.h"

#include "Worker.h"
#include "TreePiece.h"

#include "Request.h"

#ifdef GPU_GRAVITY
#include "StreamPool.h"
extern CProxy_StreamPool streamPool;
#endif

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
extern CProxy_TreePiece treePieceProxy;
extern CProxy_Main mainProxy;
extern Parameters globalParams;

void copyMomentsToNode(Node<ForceData> *node, const MomentsExchangeStruct &mes){
  CkAssert(node->getKey() == mes.key);

  node->data.moments = mes.moments;
  node->data.box = mes.box;
  NodeType type = mes.type;
  node->setType(Node<ForceData>::makeRemote(type));
}

DataManager::DataManager() : 
  numTreePieces(1),
  firstSplitterRound(false),
  decompIterations(0),
  iteration(0),
  haveRanges(false),
  keyRanges(NULL),
  rangeMsg(NULL),
  numLocalTreePieces(-1),
  doneTreeBuild(false),
  treeMomentsReady(false),
  numTreePiecesDoneTraversals(0),
  prevIterationStart(0.0),
  haveUniverse(false),
  treePiecesSettled(false),
  migrationsSettled(true),
  atDistribute(false),
  pendingRangeMsg(NULL),
  stepThisIteration(false),
  decompStalledAt(0.0),
  numBlocksRecvd(0),
  expectedBlocks(0),
  haveExpected(false),
  payloadsRecvd(0),
  outstandingExchangeSends(0),
  letsExpected(0),
  letsRecvd(0),
  letsDone(false),
  frontierReady(false)
{
#ifdef STATISTICS
  numInteractions[0] = 0;
  numInteractions[1] = 0;
  numInteractions[2] = 0;
#endif
  avgIterationRuntime = 0.0;
  savedEnergy = 0.0;
}

void DataManager::loadParticles(const CkCallback &cb){
  numRankBits = LOG_BRANCH_FACTOR;

  const char *fname = globalParams.filename.c_str();
  const long long npart = (long long)globalParams.numParticles;

  std::ifstream partFile;
  partFile.open(fname, ios::in | ios::binary);
  CkAssert(partFile.is_open());

  // 64-bit throughout. The byte offset is the particle index times 32, so an
  // int overflows at 67.1M particles and the seek below goes negative -- which
  // partFile.fail() does not reliably catch, so the run would read the wrong
  // slice rather than stop. Anything worth running on more than a node or two
  // is past that limit.
  const long long myid = CkMyPe();
  const long long npes = CkNumPes();

  long long avgParticlesPerPE = npart/npes;
  const long long rem = npart-npes*avgParticlesPerPE;
  long long firstParticle;
  if(myid < rem){
    avgParticlesPerPE++;
    firstParticle = myid*avgParticlesPerPE;
  }
  else{
    firstParticle = myid*avgParticlesPerPE+rem;
  }
  myNumParticles = (int)avgParticlesPerPE;

  const std::streamoff offset =
      (std::streamoff)firstParticle * SIZE_PER_PARTICLE + PREAMBLE_SIZE;

  myParticles.reserve(myNumParticles);
  myParticles.length() = myNumParticles;

  partFile.clear();
  partFile.seekg(offset,ios::beg);
  if(partFile.fail()){
    std::ostringstream oss;
    oss << "couldn't seek to position " << offset << " on PE " << CkMyPe() << " position " << partFile.tellg() << endl;
    CkAbort("%s", oss.str().c_str());
  }
  BoundingBox myBox;

  // In blocks. One 32-byte read per particle is a syscall per particle, and at
  // scale that is hundreds of ranks doing millions of tiny reads against one
  // file on a shared filesystem.
  const int BATCH = 4096;
  Real *buf = new Real[(size_t)BATCH * REALS_PER_PARTICLE];
  int numParticlesDone = 0;

  while(numParticlesDone < myNumParticles){
    const int n = (myNumParticles - numParticlesDone < BATCH)
                      ? (myNumParticles - numParticlesDone) : BATCH;
    const std::streamsize want = (std::streamsize)n * SIZE_PER_PARTICLE;
    partFile.read((char *)buf, want);
    if(partFile.gcount() != want){
      std::ostringstream oss;
      oss << "short read on PE " << CkMyPe() << ": wanted " << want
          << " got " << partFile.gcount() << endl;
      CkAbort("%s", oss.str().c_str());
    }

    for(int i = 0; i < n; i++){
      const Real *tmp = buf + (size_t)i * REALS_PER_PARTICLE;
      Particle &q = myParticles[numParticlesDone + i];
      q.position.x = tmp[0];
      q.position.y = tmp[1];
      q.position.z = tmp[2];
      q.velocity.x = tmp[3];
      q.velocity.y = tmp[4];
      q.velocity.z = tmp[5];
      q.mass = tmp[6];

      q.acceleration.x = 0.0;
      q.acceleration.y = 0.0;
      q.acceleration.z = 0.0;
      q.potential = 0.0;
      myBox.grow(q.position);
    }
    numParticlesDone += n;
  }
  delete[] buf;

  CkAssert(numParticlesDone == myNumParticles);
  myBox.numParticles = myNumParticles;

  partFile.close();

  contribute(sizeof(BoundingBox),&myBox,BoundingBoxGrowReductionType,cb);
}

void DataManager::hashParticleCoordinates(const OrientedBox<Real> &universe){
  Key prepend;
  prepend = 1L;
  prepend <<= (TREE_KEY_BITS-1);

  Real xsz = universe.greater_corner.x-universe.lesser_corner.x;
  Real ysz = universe.greater_corner.y-universe.lesser_corner.y;
  Real zsz = universe.greater_corner.z-universe.lesser_corner.z;

  for(unsigned int i = 0; i < myNumParticles; i++){
    Particle *p = &(myParticles[i]);
    Key xint = ((Key) (((p->position.x-universe.lesser_corner.x)*(BOXES_PER_DIM*1.0))/xsz)); 
    Key yint = ((Key) (((p->position.y-universe.lesser_corner.y)*(BOXES_PER_DIM*1.0))/ysz)); 
    Key zint = ((Key) (((p->position.z-universe.lesser_corner.z)*(BOXES_PER_DIM*1.0))/zsz)); 

    Key mask = Key(0x1);
    Key k = Key(0x0);
    int shiftBy = 0;
    for(int j = 0; j < BITS_PER_DIM; j++){
      k |= ((zint & mask) <<  shiftBy);
      k |= ((yint & mask) << (shiftBy+1));
      k |= ((xint & mask) << (shiftBy+2));
      mask <<= 1;
      // minus 1 because mask itself has shifted
      // left by one position
      shiftBy += (NDIMS-1);
    }
    k |= prepend; 
    myParticles[i].key = k;
  }
}

void DataManager::decompose(const BoundingBox &universe){
  prof.begin(PhaseProfile::KEYGEN);
  decomposeUniverse = universe;

#ifdef GPU_GRAVITY
  // The particles have been on the device since processSubmittedParticles and
  // the integrator left them there, so the keys are generated where they are.
  // Not on the first decomposition: nothing has been uploaded yet, because the
  // upload happens in processSubmittedParticles further down this same
  // pipeline.
  if(gpuParticles.attached()){
    const OrientedBox<Real> &b = universe.box;
    CkCallback cb(CkIndex_DataManager::decomposeTail(), CkMyPe(), thisgroup);
    gpuParticles.hashKeys(b.lesser_corner.x, b.lesser_corner.y, b.lesser_corner.z,
                          b.greater_corner.x - b.lesser_corner.x,
                          b.greater_corner.y - b.lesser_corner.y,
                          b.greater_corner.z - b.lesser_corner.z,
                          cb);
    return;
  }
#endif

  hashParticleCoordinates(universe.box);
  decomposeTail();
}

void DataManager::decomposeTail(){
  const BoundingBox &universe = decomposeUniverse;

#ifdef GPU_GRAVITY
  // The drifted positions, the new velocities and the keys, back into the host
  // particles. This is the O(N) transfer Stages 3 and 5 remove; until the sort
  // and the exchange are device code the host cannot do without it.
  if(gpuParticles.attached())
    gpuParticles.applyIntegrated(myParticles.getVec(), myNumParticles);
#endif
  prof.end(PhaseProfile::KEYGEN);

  prof.begin(PhaseProfile::SORT_PRE);
#ifdef GPU_GRAVITY
  // Stage 3: hashKeys() sorted on the device and applyIntegrated wrote the
  // particles back in key order, so there is nothing to do here. The timer
  // stays so the phase table keeps its shape across the two paths -- it should
  // read zero on the device path, and that is the point.
  if(!gpuParticles.attached())
#endif
  myParticles.quickSort();
  prof.end(PhaseProfile::SORT_PRE);

  if(CkMyPe()==0){
    float memMB = (1.0*CmiMemoryUsage())/(1<<20);
    ostringstream oss; 
#ifdef STATISTICS
    CkPrintf("(%d) prev time %g s\n", CkMyPe(), CmiWallTimer()-prevIterationStart);
    CkPrintf("(%d) start iteration %d\n", CkMyPe(), iteration);
    CkPrintf("(%d) mem %.2f MB\n", CkMyPe(), memMB);
    CkPrintf("(%d) univ %f %f %f %f %f %f energy %f\n", 
              CkMyPe(),
              universe.box.lesser_corner.x,
              universe.box.lesser_corner.y,
              universe.box.lesser_corner.z,
              universe.box.greater_corner.x,
              universe.box.greater_corner.y,
              universe.box.greater_corner.z,
              universe.energy);
#endif

    avgIterationRuntime += (CmiWallTimer()-prevIterationStart);
    prevIterationStart = CkWallTimer();
  }

  numTreePieces = 1;
  initHistogramParticles();
  // Opens here and closes when the gate in distributeParticles() lets go, so
  // the span covers every round trip of the histogram and not just the
  // counting. That distinction is the whole point of section 3.3 of the plan.
  prof.begin(PhaseProfile::HIST);
  sendHistogram();
}

void DataManager::initHistogramParticles(){
  int rootDepth = 0;
  
  sortingRoot = new Node<NodeDescriptor>(Key(1),
                         rootDepth,
                         myParticles.getVec(),
                         myNumParticles);
  activeBins.addNewNode(sortingRoot);

  // don't access myParticles through ckvec after this
  // anyway. these must be reset before this DM starts
  // to receive submitted particles from TPs placed on it
  myNumParticles = 0;
  myParticles.length() = 0;
}

void DataManager::sendHistogram(){

  CkCallback cb(CkIndex_DataManager::receiveHistogram(NULL),0,this->thisgroup);
  contribute(sizeof(NodeDescriptor)*activeBins.getNumCounts(),activeBins.getCounts(),NodeDescriptorReductionType,cb);
  activeBins.reset();
}

// executed on PE 0
void DataManager::receiveHistogram(CkReductionMsg *msg){

  int numRecvdBins = msg->getSize()/sizeof(NodeDescriptor);
  NodeDescriptor *descriptors = (NodeDescriptor *)msg->getData();

  // XXX remove this and make a refine function for ActiveBinInfo
  CkVec<int> binsToRefine;

  binsToRefine.reserve(2*numRecvdBins);
  binsToRefine.length() = 0;

  int particlesHistogrammed = 0;

  CkVec<pair<Node<NodeDescriptor>*,bool> > *active = activeBins.getActive();
  CkAssert(numRecvdBins == active->length());

  const Real target = (Real)(DECOMP_TOLERANCE*globalParams.ppc);

  for(int i = 0; i < numRecvdBins; i++){
    if(descriptors[i].numParticles > target){
      // Refine this bin. One level is the original behaviour and still the
      // default; with -decomplevels above one, a bin far over the target is
      // taken down several levels in this round instead of one per round.
      //
      // The whole cost of the histogram is the round trips -- the counting
      // itself is a binary search per bin, because the particles are already
      // sorted -- so the only thing worth optimising here is how few rounds it
      // takes to converge.
      int levels = 1;
      if(globalParams.decompLevels > 1 && target > 0.0){
        const Real ratio = descriptors[i].numParticles / target;
        while(levels < globalParams.decompLevels &&
              (Real)(1 << levels) < ratio) levels++;
        // Splitting a clustered bin k levels deep can leave most of the
        // children empty, and every child costs a tree piece whether or not it
        // holds anything. Back off until the budget covers it rather than
        // aborting: one level always fits if anything does, and the next round
        // will pick the bin up again.
        while(levels > 1 &&
              numTreePieces + ((1 << levels) - 1) > globalParams.numTreePieces){
          levels--;
        }
      }

      binsToRefine.push_back(i);
      binsToRefine.push_back(levels);
      numTreePieces += (1 << levels) - 1;
      if(numTreePieces > globalParams.numTreePieces){
        CkPrintf("have %d treepieces need %d\n",globalParams.numTreePieces,numTreePieces);
        CkAbort("Need more tree pieces!\n");
      }
    }
    else{
      Node<NodeDescriptor> *nd = (*active)[i].first;
      nd->data = descriptors[i];
    }

    particlesHistogrammed += descriptors[i].numParticles;
  }

  // Two ints per bin: the index and the number of levels.
  int numInts = binsToRefine.length();
  int numBinsToRefine = numInts / 2;

  if(numBinsToRefine > 0){
    SplitterMsg *m = new (numInts,0) SplitterMsg;
    memcpy(m->splitBins,binsToRefine.getVec(),sizeof(int)*numInts);
    m->nSplitBins = numBinsToRefine; 
    thisProxy.receiveSplitters(m);
    decompIterations++;
  }
  else{
    // create tree pieces and send proxy
    #ifdef STATISTICS
    CkPrintf("[0] decomp done after %d iterations used treepieces %d\n", decompIterations, numTreePieces);
    // -p is a budget and placement is derived from it, so a budget far above
    // what the decomposition actually uses costs real time -- measured at
    // 0.134 s/iter with -p sized to the used count against 0.146 with the
    // block cyclic default and 0.506 with a plain block map over the same
    // oversized budget.
    static bool warnedBudget = false;
    if(!warnedBudget && numTreePieces * 2 < globalParams.numTreePieces){
      warnedBudget = true;
      CkPrintf("[0] note: -p=%d is %.1fx the %d tree pieces actually used; "
               "sizing it near that and passing -blockmap=1 keeps neighbouring "
               "key ranges on one PE and cuts remote traffic\n",
               globalParams.numTreePieces,
               (double)globalParams.numTreePieces/(double)numTreePieces,
               numTreePieces);
    }
    #endif
    decompIterations = 0;

    // Everything past here needs the local tree pieces to hold still, so it
    // lives in distributeParticles() behind the migration gate. Broadcast,
    // because every PE now has to reach the tree-piece map reduction before
    // anything can be sent -- the key ranges used to be that signal, and they
    // are not available until after the reduction.
    thisProxy.beginDistribute();
  }

  delete msg;
}

// Walk the sorting tree and record its leaves. Nothing is sent here: the
// leaves have to be grouped by destination PE first, and that needs the map.
// On PE 0 this is also where the key ranges are filled, because PE 0's sorting
// tree is the only one carrying the globally reduced NodeDescriptors.
void DataManager::flushParticles(){
  leafList.length() = 0;
  ParticleFlushWorker pfw(this);
  scaffoldTrav.preorderTraversal(sortingRoot,&pfw);
}

void DataManager::recordLeaf(Node<NodeDescriptor> *nd, int tp){
  CkAssert(nd->getNumChildren() == 0);
  leafList.push_back(LeafRef(nd,tp));

  // only PE 0 has the correct ranges
  if(CkMyPe() == 0){
    if(nd->data.numParticles > 0){
      CkAssert(nd->data.smallestKey <= nd->data.largestKey);
    } else {
      CkAssert(nd->data.smallestKey == nd->data.largestKey);
    }
    keyRanges[(tp<<1)] = nd->data.smallestKey;
    keyRanges[(tp<<1)+1] = nd->data.largestKey;
  }
}

void DataManager::freeSortingTree(){
  if(sortingRoot == NULL) return;
  FreeTreeWorker<NodeDescriptor> freeWorker;
  scaffoldTrav.postorderTraversal(sortingRoot,&freeWorker);
  delete sortingRoot;
  sortingRoot = NULL;
  leafList.length() = 0;
}

// Every PE says which tree pieces it holds; the max reduction folds those into
// one array that every PE then has. A sender needs it to address its blocks.
// Refreshed every iteration because the balancer moves elements between them.
void DataManager::publishTreePieceMap(){
  const int n = globalParams.numTreePieces;
  CkVec<int> mine;
  mine.resize(n);
  for(int i = 0; i < n; i++) mine[i] = -1;
  for(int i = 0; i < localTreePieces.indices.length(); i++){
    const int idx = localTreePieces.indices[i];
    if(idx >= 0 && idx < n) mine[idx] = CkMyPe();
  }
  CkCallback cb(CkIndex_DataManager::recvTreePieceMap(NULL),thisProxy);
  contribute(sizeof(int)*n, mine.getVec(), CkReduction::max_int, cb);
}

void DataManager::recvTreePieceMap(CkReductionMsg *msg){
  const int n = msg->getSize()/sizeof(int);
  tpToPe.resize(n);
  memcpy(tpToPe.getVec(), msg->getData(), sizeof(int)*n);
  delete msg;

  sendParticleBlocks();

  if(CkMyPe() == 0){
    // The ranges are filled now, so the other PEs can be told.
    int numKeys = numTreePieces*2;
    RangeMsg *rmsg = new (numKeys) RangeMsg;
    rmsg->numTreePieces = numTreePieces;
    memcpy(rmsg->keys,keyRanges,sizeof(Key)*numKeys);
    thisProxy.sendParticles(rmsg);
  }
  maybeAssemble();
}

// One message per destination PE, holding every local leaf that belongs to a
// tree piece living there. Empty blocks are still sent: the receiver counts to
// CkNumPes() to know the exchange is complete.
void DataManager::sendParticleBlocks(){
  const int npes = CkNumPes();

  CkVec<int> nTps, nParts;
  nTps.resize(npes); nParts.resize(npes);
  for(int q = 0; q < npes; q++){ nTps[q] = 0; nParts[q] = 0; }

  for(int i = 0; i < leafList.length(); i++){
    const int tp = leafList[i].tp;
    const int q = (tp < tpToPe.length()) ? tpToPe[tp] : -1;
    if(q < 0) continue;   // no element registered anywhere for this index
    nTps[q]++;
    nParts[q] += leafList[i].node->getNumParticles();
  }

  CkVec<ParticleBlockMsg *> out;
  out.resize(npes);
  CkVec<int> tpFill, partFill;
  tpFill.resize(npes); partFill.resize(npes);
  CkVec<CkVec<int> > sendOffs, sendCnts;
  sendOffs.resize(npes); sendCnts.resize(npes);
  for(int q = 0; q < npes; q++){ sendOffs[q].length()=0; sendCnts[q].length()=0; }
  for(int q = 0; q < npes; q++){
    const int t = nTps[q] > 0 ? nTps[q] : 1;
    const int pcount = nParts[q] > 0 ? nParts[q] : 1;
    out[q] = new (t, t, 2*t, pcount, 0) ParticleBlockMsg;
    out[q]->numTps = nTps[q];
    out[q]->numParticles = nParts[q];
    out[q]->fromPe = CkMyPe();
    tpFill[q] = 0; partFill[q] = 0;
  }

  for(int i = 0; i < leafList.length(); i++){
    const int tp = leafList[i].tp;
    const int q = (tp < tpToPe.length()) ? tpToPe[tp] : -1;
    if(q < 0) continue;
    Node<NodeDescriptor> *nd = leafList[i].node;
    const int np = nd->getNumParticles();

    ParticleBlockMsg *m = out[q];
    const int k = tpFill[q]++;
    m->tpIndex[k] = tp;
    m->tpCount[k] = np;
    if(np > 0){
      Particle *src = nd->getParticles();
      m->tpKeys[2*k]   = src[0].key;
      m->tpKeys[2*k+1] = src[np-1].key;
#ifdef GPU_GRAVITY
      if(globalParams.deviceExchange && gpuParticles.attached()){
        // Record the device range instead of copying through the host.
        sendOffs[q].push_back(particleOffset(src));
        sendCnts[q].push_back(np);
      }
      else
#endif
      memcpy(m->parts + partFill[q], src, sizeof(Particle)*np);
      partFill[q] += np;
    }
    else{
      m->tpKeys[2*k]   = ~Key(0);
      m->tpKeys[2*k+1] = Key(0);
    }
  }

  // Tell every PE how many senders it should expect, then send only to the
  // ones that actually have something. One reduction of P ints replaces P^2
  // empty messages.
  CkVec<int> willSend;
  willSend.resize(npes);
  for(int q = 0; q < npes; q++) willSend[q] = (nTps[q] > 0) ? 1 : 0;
  CkCallback cb(CkIndex_DataManager::recvSenderCounts(NULL),thisProxy);
  contribute(sizeof(int)*npes, willSend.getVec(), CkReduction::sum_int, cb);

  for(int q = 0; q < npes; q++){
    if(nTps[q] <= 0){ delete out[q]; continue; }
    thisProxy[q].receiveParticleBlock(out[q]);
#ifdef GPU_GRAVITY
    if(globalParams.deviceExchange && gpuParticles.attached()){
      char *buf = gpuParticles.stageSend(q, sendOffs[q].getVec(),
                                         sendCnts[q].getVec(),
                                         sendOffs[q].length(), nParts[q]);
      const int nbytes = (int)GpuParticleStore::stageBytes(nParts[q]);
      thisProxy[q].recvParticleDevice(CkMyPe(), nbytes,
          CkDeviceBuffer(buf, CkCallback(CkIndex_DataManager::exchangeSendDone(),
                                         CkMyPe(), thisgroup),
                         gpuParticles.deviceStream()));
      outstandingExchangeSends++;
    }
#endif
  }

  // The leaves point into myParticles and have been copied out, so the sorting
  // tree can go.
  freeSortingTree();
}

void DataManager::recvSenderCounts(CkReductionMsg *msg){
  const int *counts = (const int *)msg->getData();
  expectedBlocks = counts[CkMyPe()];
  haveExpected = true;
  delete msg;
  maybeAssemble();
}

// The staging region a push lands in. Sized from the count the block message
// carries, which is why the two are matched by sender.
void DataManager::recvParticleDevice(int fromPe, int &nbytes, char *&buf,
                                     CkDeviceBufferPost *devicePost){
  buf = gpuParticles.recvSlot(fromPe,
          nbytes / (int)(sizeof(float4)*2 + sizeof(unsigned long long)));
  devicePost[0].hapi_stream = gpuParticles.deviceStream();
}

void DataManager::recvParticleDevice(int fromPe, int nbytes, char *buf){
  while(srcPayloadIn.length() <= fromPe) srcPayloadIn.push_back(0);
  srcPayloadIn[fromPe] = 1;
  payloadsRecvd++;
  maybeAssemble();
}

// The staging region is read asynchronously by the transport, so it may not be
// rebuilt until every send out of it has drained.
void DataManager::exchangeSendDone(){
  outstandingExchangeSends--;
}

void DataManager::receiveParticleBlock(ParticleBlockMsg *msg){
  recvdBlocks.push_back(msg);
  numBlocksRecvd++;
  maybeAssemble();
}

void DataManager::maybeAssemble(){
  if(!haveRanges) return;
  if(!haveExpected) return;
  if(numBlocksRecvd != expectedBlocks) return;
#ifdef GPU_GRAVITY
  // Both halves: the bookkeeping and the particles it describes.
  if(globalParams.deviceExchange && gpuParticles.attached()){
    int need = 0;
    for(int b = 0; b < recvdBlocks.length(); b++)
      if(recvdBlocks[b]->numParticles > 0) need++;
    if(payloadsRecvd < need) return;
  }
#endif
  assembleReceivedBlocks();
}

// Build this PE's particle array and its TreePieceDescriptors straight from
// the received blocks. The tree pieces play no part: senseTreePieces() already
// said which elements are here, and the blocks say how many particles each of
// them got and over what key range.
void DataManager::assembleReceivedBlocks(){
  const int n = globalParams.numTreePieces;

  // Fold the per-tree-piece contributions from every sender.
  CkVec<int> count; count.resize(n);
  CkVec<Key> smallest, largest;
  smallest.resize(n); largest.resize(n);
  for(int i = 0; i < n; i++){
    count[i] = 0;
    smallest[i] = ~Key(0);
    largest[i] = Key(0);
  }
  for(int b = 0; b < recvdBlocks.length(); b++){
    ParticleBlockMsg *m = recvdBlocks[b];
    for(int k = 0; k < m->numTps; k++){
      const int tp = m->tpIndex[k];
      if(m->tpCount[k] == 0) continue;
      count[tp] += m->tpCount[k];
      if(smallest[tp] > m->tpKeys[2*k])   smallest[tp] = m->tpKeys[2*k];
      if(largest[tp]  < m->tpKeys[2*k+1]) largest[tp]  = m->tpKeys[2*k+1];
    }
  }

  // One descriptor per local element, in index order.
  submittedParticles.length() = 0;
  myNumParticles = 0;
  CkVec<int> local = localTreePieces.indices;
  local.quickSort();
  for(int i = 0; i < local.length(); i++){
    const int tp = local[i];
    TreePiece *owner = treePieceProxy[tp].ckLocal();
    TreePieceDescriptor d(NULL, count[tp], owner, tp, smallest[tp], largest[tp]);
    submittedParticles.push_back(d);
    myNumParticles += count[tp];
  }

  // Where each tree piece's particles start in myParticles.
  CkVec<int> offset; offset.resize(n);
  for(int i = 0; i < n; i++) offset[i] = -1;
  int running = 0;
  for(int i = 0; i < submittedParticles.length(); i++){
    offset[submittedParticles[i].index] = running;
    running += submittedParticles[i].numParticles;
  }

  myParticles.resize(myNumParticles);
  CkVec<Particle> staged;
  for(int b = 0; b < recvdBlocks.length(); b++){
    ParticleBlockMsg *m = recvdBlocks[b];
    const Particle *src = m->parts;
#ifdef GPU_GRAVITY
    // Pull this sender's block back from the device, in the order its
    // bookkeeping lists the tree pieces.
    if(globalParams.deviceExchange && gpuParticles.attached() &&
       m->numParticles > 0){
      staged.resize(m->numParticles);
      gpuParticles.unstageRecv(m->fromPe, m->numParticles, staged.getVec());
      src = staged.getVec();
    }
#endif
    int p = 0;
    for(int k = 0; k < m->numTps; k++){
      const int tp = m->tpIndex[k];
      const int np = m->tpCount[k];
      if(np == 0) continue;
      CkAssert(offset[tp] >= 0);
      memcpy(myParticles.getVec()+offset[tp], src + p, sizeof(Particle)*np);
      offset[tp] += np;
      p += np;
    }
    delete m;
  }
  recvdBlocks.length() = 0;
  numBlocksRecvd = 0;
  haveExpected = false;
  expectedBlocks = 0;
  payloadsRecvd = 0;
  for(int i = 0; i < srcPayloadIn.length(); i++) srcPayloadIn[i] = 0;

  processSubmittedParticles();
}

void DataManager::receiveSplitters(SplitterMsg *msg){

  // process bins to refine. splitBins is (index, levels) pairs.
  activeBins.processRefineLevels(msg->splitBins,msg->nSplitBins);

  // We traverse the final tree to flush particles to 
  // appropriate tree pieces

  // here, we will know of bins that have not
  // been refined or deleted in the present 
  // iteration; we can send particles to these
  // CkVec<Node<NodeDescriptor>*> &unrefined = activeBins.getUnrefined();

  sendHistogram();
  delete msg;
}


void DataManager::sendParticles(RangeMsg *msg){

  if(CkMyPe() != 0){
    // The ranges, which only PE 0 can compute. The blocks may already be here
    // or may still be coming; whichever completes second starts the assembly.
    numTreePieces = msg->numTreePieces;
    keyRanges = msg->keys;
    rangeMsg = msg;
    haveRanges = true;
    maybeAssemble();
  }
  else{
    CkAssert(numTreePieces == msg->numTreePieces);
    CkAssert(haveRanges);
    delete msg;
  }

  // there are tree piece on this PE
  // these will eventually receive their respective
  // particles and submit them to the DM. Also, the
  // DM will receive the rangeKeys from the decomposition
  // leader (i.e. DM on PE 0) When all particles and 
  // rangeKeys have been received, the DM begins tree
  // construction
}

void DataManager::senseTreePieces(){
  localTreePieces.reset();
  CkLocMgr *mgr = treePieceProxy.ckLocMgr();
  mgr->iterate(localTreePieces);
  numLocalTreePieces = localTreePieces.count;
}


void DataManager::processSubmittedParticles(){
  // myParticles and submittedParticles were filled by assembleReceivedBlocks.
  // The per-tree-piece concatenation that used to live here went with the
  // per-tree-piece messages.
  submittedParticles.quickSort();

  prof.handoff(PhaseProfile::DISTRIB, PhaseProfile::SORT_POST);
  myParticles.quickSort();
  prof.end(PhaseProfile::SORT_POST);

  prof.begin(PhaseProfile::BUILD);
  buildTree();
  prof.end(PhaseProfile::BUILD);
  // add dummy tree piece whose index is larger than
  // that of all others. this is required to mark the
  // boundary of nodes/particles owned by this PE.
  submittedParticles.push_back(TreePieceDescriptor(globalParams.numTreePieces));

#ifdef GPU_GRAVITY
  // myParticles is final now -- resized, filled, sorted, and pointed into by
  // every bucket in the tree just built -- so this is the earliest point at
  // which it can go to the device, and the traversals that read it back are
  // still several messages away.
  ensureDevice();
  prof.begin(PhaseProfile::UPLOAD);
  gpuParticles.upload(myParticles.getVec(), myNumParticles);
  prof.end(PhaseProfile::UPLOAD);

  // Stage 6: the same tree, on the device. The host tree built above is still
  // what the traversal walks; this one is for the device traversal to come,
  // and is checked against the host's moments under BARNES_TREE_CHECK.
  gpuParticles.buildDeviceTree(keyRanges, numTreePieces,
                               (int)((Real)globalParams.ppb*BUCKET_TOLERANCE));
#endif

  // makeMoments also sends out requests for moments
  // of remote nodes. The span closes in treeReady(), so it covers the cross-PE
  // exchange rather than only the local postorder pass.
  prof.begin(PhaseProfile::MOMENTS);
  makeMoments();

  doneTreeBuild = true;

  // One reduction instead of the request/reply chain up the tree.
  contributeFrontierMoments();
}

static void collectHostNodes(Node<ForceData> *n,
                             map<Key, Node<ForceData>*> &out){
  if(n == NULL) return;
  out[n->getKey()] = n;
  for(int i = 0; i < n->getNumChildren(); i++)
    collectHostNodes(n->getChild(i), out);
}

// Compare the flat device tree against the host tree over the same particles.
//
// Nodes are matched by SFC key, which both builds assign identically, so a
// mismatch in *shape* shows up as a key present on one side and not the other.
// The moments are then compared where the keys agree. This is the only handle
// on the device build until a device traversal exists to disagree with the
// host one, so it reports counts as well as errors: a tree with the right
// moments and the wrong number of nodes is still wrong.
void DataManager::patchDeviceBoundaryMoments(){
#ifdef GPU_GRAVITY
  if(!globalParams.deviceWalk) return;
  if(!gpuParticles.attached() || myNumParticles == 0) return;
  if(root == NULL) return;

  // No readback: the scatter kernel finds each node by descending the key, so
  // the host never has to learn where the device put them.
  map<Key, Node<ForceData>*> hostNodes;
  collectHostNodes(root, hostNodes);

  CkVec<GpuMomentPatch> patches;
  for(map<Key,Node<ForceData>*>::iterator it = hostNodes.begin();
      it != hostNodes.end(); ++it){
    Node<ForceData> *h = it->second;
    const NodeType t = h->getType();
    // Only the nodes whose moments the device could not have computed: the
    // ones that own particles on other PEs.
    if(t != Boundary) continue;

    const MultipoleMoments &m = h->data.moments;
    GpuMomentPatch p;
    p.key = (unsigned long long)it->first;
    p.cmMass = make_float4(m.cm.x, m.cm.y, m.cm.z, m.totalMass);
    p.rsq = m.rsq;
    p.qxx = m.qxx; p.qxy = m.qxy; p.qxz = m.qxz;
    p.qyy = m.qyy; p.qyz = m.qyz;
    patches.push_back(p);
  }

  if(patches.length() > 0)
    gpuParticles.patchMoments(patches.getVec(), patches.length());

  if(getenv("BARNES_WALK_DEBUG") != NULL){
    CkPrintf("[WALKDBG] pe %d patched %d boundary moments\n",
             CkMyPe(), patches.length());
  }
#endif
}

void DataManager::checkDeviceTree(){
#ifdef GPU_GRAVITY
  if(getenv("BARNES_TREE_CHECK") == NULL) return;
  if(!gpuParticles.attached() || myNumParticles == 0) return;

  const int cap = 1 << 20;
  DeviceNode *nodes = new DeviceNode[cap];
  const int n = gpuParticles.readTree(nodes, cap);
  const int m = (n < cap) ? n : cap;

  // The host side, keyed the same way. Walked from the root rather than read
  // out of nodeTable: that map only holds the nodes the remote-request path
  // needs to look up, not the whole tree.
  map<Key, Node<ForceData>*> hostByKey;
  collectHostNodes(root, hostByKey);

  int matched = 0, missing = 0, leafMismatch = 0, compared = 0;
  double maxCmErr = 0.0, maxMassErr = 0.0, maxRsqErr = 0.0;
  for(int i = 0; i < m; i++){
    const DeviceNode &d = nodes[i];
    map<Key,Node<ForceData>*>::iterator it = hostByKey.find((Key)d.key);
    if(it == hostByKey.end()){ missing++; continue; }
    Node<ForceData> *h = it->second;
    matched++;

    if((h->getNumChildren() == 0) != (d.firstChild < 0)) leafMismatch++;

    // A Boundary node's host moments include contributions from other PEs,
    // which the device tree cannot have: it only knows the particles resident
    // here. Only fully local subtrees are comparable.
    const NodeType ht = h->getType();
    if(ht != Internal && ht != Bucket && ht != EmptyBucket) continue;

    const MultipoleMoments &hm = h->data.moments;
    const double mass = hm.totalMass;
    if(mass <= 0.0) continue;
    compared++;
    const double dm = fabs((double)d.cmMass.w - mass)/mass;
    if(dm > maxMassErr) maxMassErr = dm;

    const double scale = sqrt((double)hm.rsq) > 0.0 ? sqrt((double)hm.rsq) : 1.0;
    const double dx = (double)d.cmMass.x - hm.cm.x;
    const double dy = (double)d.cmMass.y - hm.cm.y;
    const double dz = (double)d.cmMass.z - hm.cm.z;
    const double dc = sqrt(dx*dx+dy*dy+dz*dz)/scale;
    if(dc > maxCmErr) maxCmErr = dc;

    if(hm.rsq > 0.0){
      const double dr = fabs((double)d.rsq - hm.rsq)/hm.rsq;
      if(dr > maxRsqErr) maxRsqErr = dr;
    }
  }

  CkPrintf("[TREECHECK] pe %d iter %d: device nodes %d, host nodes %zu, "
           "matched %d, device-only %d, leaf-shape mismatches %d\n",
           CkMyPe(), iteration, n, hostByKey.size(), matched, missing,
           leafMismatch);
  CkPrintf("[TREECHECK] pe %d compared %d nodes with mass: max rel err "
           "mass %.3e  cm %.3e (of the opening radius)  rsq %.3e\n",
           CkMyPe(), compared, maxMassErr, maxCmErr, maxRsqErr);

  delete[] nodes;
#endif
}

void DataManager::buildTree(){

  int rootDepth = 0;
  root = new Node<ForceData>(Key(1),rootDepth,myParticles.getVec(),myNumParticles);
  root->setOwners(0,numTreePieces-1);
  nodeTable[Key(1)] = root;
  if(myNumParticles == 0){
    return;
  }

  OwnershipActiveBinInfo<ForceData> abi(keyRanges);
  abi.addNewNode(root);
  int numFatNodes = 1;

  int limit = ((Real)globalParams.ppb*BUCKET_TOLERANCE);

  CkVec<int> refines;

  while(numFatNodes > 0){
    abi.reset();
    refines.length() = 0;

    CkVec<std::pair<Node<ForceData>*,bool> > *active = abi.getActive();
    // discard node when:
    // 1. ownerEnd of node is < curTP
    // 2. if not 1, check for numparticles in node 
    // if ownerStart of node is > curTP, curTP = curTP->next
    // when a node its split, 
    for(int i = 0; i < active->length(); i++){
      Node<ForceData> *node = (*active)[i].first;
      if((node->getOwnerEnd() > node->getOwnerStart()) || 
         (node->getNumParticles() > limit)){
        refines.push_back(i);
      }
    }

    abi.processRefine(refines.getVec(), refines.length());
    numFatNodes = abi.getNumCounts();
  }

}

// mass, cm(3), rsq, quadrupole(5), box lesser(3), box greater(3), type.
//
// The type travels because copyMomentsToNode used to set it: the receiver
// needs to know whether the owner holds a Bucket or an interior node, since
// that is what decides whether its walk asks for particles or for a subtree.
#define FRONTIER_W 17

void DataManager::makeMoments(){
  if(root == NULL) return;

  MomentsWorker mw(submittedParticles,
                   nodeTable,
                   myBuckets
                   );
  fillTrav.postorderTraversal(root,&mw);
}

// Every node at which the owner range narrows to one tree piece. The
// refinement that produced these ranges came from keyRanges, which is global,
// so this preorder walk yields the same list on every PE -- that common
// ordering is the whole trick.
void DataManager::collectFrontier(Node<ForceData> *n, CkVec<Node<ForceData>*> &out){
  if(n == NULL) return;
  if(n->getOwnerStart() == n->getOwnerEnd() || n->getNumChildren() == 0){
    out.push_back(n);
    return;
  }
  for(int i = 0; i < n->getNumChildren(); i++) collectFrontier(n->getChild(i), out);
}

// One reduction in place of the whole moment exchange.
//
// A frontier node belongs to exactly one tree piece, so exactly one PE holds
// its particles and can compute its moments; everyone else contributes zeros
// and a plain sum is the answer. That is why nothing has to be reduced in
// additive form here -- there is only ever one contributor per node.
//
// What this replaces was a message per remote node and then a chain of
// childMomentsReady notifications up the tree, so the tree was not ready until
// a round trip per level had completed. That chain is the part that scales
// badly: its length is the tree depth and every link is a network hop.
void DataManager::contributeFrontierMoments(){
  frontier.length() = 0;
  collectFrontier(root, frontier);

  const int n = frontier.length();
  // The bounding box travels too. getMomentsFromChildren derives rsq from the
  // box, so a boundary node whose remote children have no box would get a
  // meaningless opening radius and the walk would open the wrong cells.
  // Six more slots per PE at the end: this PE's own domain box, which the LET
  // walk needs for every other PE. It rides along rather than costing a second
  // collective.
  const int boxBase = n*FRONTIER_W + 1;
  const int total = boxBase + 6*CkNumPes();
  CkVec<Real> mine;
  mine.resize(total);
  for(int i = 0; i < total; i++) mine[i] = 0.0;
  // A tripwire on the assumption this whole scheme rests on: that every PE
  // enumerates the same frontier in the same order. If the lengths ever
  // diverge the sum below is meaningless, so carry the count and check it.
  mine[n*FRONTIER_W] = (Real)n;

  for(int i = 0; i < n; i++){
    Node<ForceData> *f = frontier[i];
    // Remote means someone else owns it; leave zeros and take theirs.
    if(f->getType() == Remote || f->getType() == RemoteBucket ||
       f->getType() == RemoteEmptyBucket) continue;
    const MultipoleMoments &m = f->data.moments;
    Real *o = mine.getVec() + i*FRONTIER_W;
    o[16] = (Real)(int)f->getType();

    // An empty node's box is the reset one, HUGE_VAL against -HUGE_VAL, and
    // summing that would poison the result. Its moments are zero anyway, so
    // only the box is withheld.
    if(f->getNumParticles() <= 0) continue;

    o[0] = m.totalMass;
    o[1] = m.cm.x; o[2] = m.cm.y; o[3] = m.cm.z;
    o[4] = m.rsq;
    o[5] = m.qxx; o[6] = m.qxy; o[7] = m.qxz; o[8] = m.qyy; o[9] = m.qyz;
    o[10] = f->data.box.lesser_corner.x;
    o[11] = f->data.box.lesser_corner.y;
    o[12] = f->data.box.lesser_corner.z;
    o[13] = f->data.box.greater_corner.x;
    o[14] = f->data.box.greater_corner.y;
    o[15] = f->data.box.greater_corner.z;
  }

  if(getenv("BARNES_FRONTIER_DEBUG") != NULL){
    fprintf(stderr, "[FRONTIER] pe %d contributing frontier=%d reals=%d\n",
            CkMyPe(), n, n*FRONTIER_W + 1);
    fflush(stderr);
  }
  // This PE's domain: the union of the frontier nodes it owns.
  myDomain.reset();
  for(int i = 0; i < n; i++){
    Node<ForceData> *f = frontier[i];
    if(f->getType() == Remote || f->getType() == RemoteBucket ||
       f->getType() == RemoteEmptyBucket) continue;
    if(f->getNumParticles() <= 0) continue;
    myDomain.grow(f->data.box);
  }
  if(myDomain.initialized()){
    Real *b = mine.getVec() + boxBase + 6*CkMyPe();
    b[0] = myDomain.lesser_corner.x;
    b[1] = myDomain.lesser_corner.y;
    b[2] = myDomain.lesser_corner.z;
    b[3] = myDomain.greater_corner.x;
    b[4] = myDomain.greater_corner.y;
    b[5] = myDomain.greater_corner.z;
  }

  CkCallback cb(CkIndex_DataManager::recvFrontierMoments(NULL),thisProxy);
  contribute(sizeof(Real)*total, mine.getVec(), CkReduction::sum_float, cb);
}

// Boundary nodes -- everything above the frontier -- from their children, now
// that the frontier is complete. Their moments were never touched by
// MomentsWorker, so they are still zero and the accumulation is safe.
void DataManager::fillBoundaryMoments(Node<ForceData> *n){
  if(n == NULL) return;
  // Only Boundary nodes are outstanding. MomentsWorker already computed every
  // Internal node's moments from its own subtree, and every Bucket's from its
  // particles; recursing into those and calling getMomentsFromChildren again
  // would accumulate on top of what is already there and double them.
  if(n->getType() != Boundary) return;
  for(int i = 0; i < n->getNumChildren(); i++) fillBoundaryMoments(n->getChild(i));
  n->getMomentsFromChildren();
  // passMomentsUpward did both. The owner span is what requestNode picks a
  // target from, so dropping it sends requests to the wrong tree pieces.
  n->getOwnershipFromChildren();
}

// What a push to `dest` would carry, counted rather than built.
//
// The predicate is the destination's own walk: a cell it would accept becomes
// one multipole, a cell it would open is descended into, and a leaf it opens
// contributes its particles. Measuring against the destination's whole domain
// rather than its individual buckets is deliberately conservative -- a larger
// box opens more, so the result is a superset of what any one of its buckets
// could ask for, which is what makes a push sufficient.
void DataManager::letSize(Node<ForceData> *n, const OrientedBox<Real> &dest,
                          int &nodes, int &parts){
  if(n == NULL) return;
  const NodeType t = n->getType();
  // Only what this PE actually owns can be pushed.
  if(t == Remote || t == RemoteBucket || t == RemoteEmptyBucket) return;
  if(t == EmptyBucket) return;

  // openCriterionBucket, with the destination's domain in place of a bucket.
  const MultipoleMoments &m = n->data.moments;
  if(m.totalMass <= 0.0) return;
  Real dx = dest.lesser_corner.x - m.cm.x;
  Real ex = m.cm.x - dest.greater_corner.x;
  if(dx < ex) dx = ex;  if(dx < 0) dx = 0;
  Real dy = dest.lesser_corner.y - m.cm.y;
  Real ey = m.cm.y - dest.greater_corner.y;
  if(dy < ey) dy = ey;  if(dy < 0) dy = 0;
  Real dz = dest.lesser_corner.z - m.cm.z;
  Real ez = m.cm.z - dest.greater_corner.z;
  if(dz < ez) dz = ez;  if(dz < 0) dz = 0;
  const bool open = (globalParams.tolsq*(dx*dx+dy*dy+dz*dz) < m.rsq);

  if(!open){ nodes++; return; }          // accepted: one multipole travels
  if(n->getNumChildren() == 0){          // opened leaf: its particles travel
    nodes++;
    parts += n->getNumParticles();
    return;
  }
  nodes++;
  for(int i = 0; i < n->getNumChildren(); i++)
    letSize(n->getChild(i), dest, nodes, parts);
}

void DataManager::reportLetSizes(){
  if(getenv("BARNES_LET_SIZE") == NULL) return;
  for(int q = 0; q < CkNumPes(); q++){
    if(q == CkMyPe() || !peBoxes[q].initialized()) continue;
    int nodes = 0, parts = 0;
    letSize(root, peBoxes[q], nodes, parts);
    CkPrintf("[LET] pe %d -> pe %d: %d cells, %d particles (%.1f KB)\n",
             CkMyPe(), q, nodes, parts,
             (nodes*(double)sizeof(Node<ForceData>) +
              parts*(double)sizeof(ExternalParticle))/1024.0);
  }
}

// mass, cm(3), rsq, quadrupole(5), box(6), type.
#define LET_W 17

// Collect what `dest` could need from this PE's tree.
//
// Same shape as letSize: accept -> one multipole travels; open -> descend;
// opened leaf -> its particles travel. Emitting a node whether it is accepted
// or opened is what lets the receiver rebuild the structure, and marking an
// accepted cell RemoteBucket-or-Remote by whether particles came with it is
// what tells its walk which of the two it is looking at.
void DataManager::collectLet(Node<ForceData> *n, const OrientedBox<Real> &dest,
                             CkVec<Key> &keys, CkVec<Real> &mom,
                             CkVec<int> &npart,
                             CkVec<ExternalParticle> &parts){
  if(n == NULL) return;
  const NodeType t = n->getType();
  if(t == Remote || t == RemoteBucket || t == RemoteEmptyBucket) return;
  if(t == EmptyBucket) return;

  const MultipoleMoments &m = n->data.moments;
  if(m.totalMass <= 0.0) return;

  Real dx = dest.lesser_corner.x - m.cm.x;
  Real ex = m.cm.x - dest.greater_corner.x;
  if(dx < ex) dx = ex;  if(dx < 0) dx = 0;
  Real dy = dest.lesser_corner.y - m.cm.y;
  Real ey = m.cm.y - dest.greater_corner.y;
  if(dy < ey) dy = ey;  if(dy < 0) dy = 0;
  Real dz = dest.lesser_corner.z - m.cm.z;
  Real ez = m.cm.z - dest.greater_corner.z;
  if(dz < ez) dz = ez;  if(dz < 0) dz = 0;
  const bool open = (globalParams.tolsq*(dx*dx+dy*dy+dz*dz) < m.rsq);

  const bool leaf = (n->getNumChildren() == 0);
  if(open && !leaf){
    // Interior: the destination will descend, so send the node as a marker and
    // recurse. It carries moments too, in case a shallower bucket accepts it.
    keys.push_back(n->getKey());
    Real o[LET_W];
    o[0]=m.totalMass; o[1]=m.cm.x; o[2]=m.cm.y; o[3]=m.cm.z; o[4]=m.rsq;
    o[5]=m.qxx; o[6]=m.qxy; o[7]=m.qxz; o[8]=m.qyy; o[9]=m.qyz;
    o[10]=n->data.box.lesser_corner.x; o[11]=n->data.box.lesser_corner.y;
    o[12]=n->data.box.lesser_corner.z; o[13]=n->data.box.greater_corner.x;
    o[14]=n->data.box.greater_corner.y; o[15]=n->data.box.greater_corner.z;
    o[16]=(Real)(int)Internal;
    for(int i = 0; i < LET_W; i++) mom.push_back(o[i]);
    npart.push_back(0);
    for(int i = 0; i < n->getNumChildren(); i++)
      collectLet(n->getChild(i), dest, keys, mom, npart, parts);
    return;
  }

  // Accepted, or an opened leaf. Either way one entry; particles only when the
  // destination would open it.
  keys.push_back(n->getKey());
  Real o[LET_W];
  o[0]=m.totalMass; o[1]=m.cm.x; o[2]=m.cm.y; o[3]=m.cm.z; o[4]=m.rsq;
  o[5]=m.qxx; o[6]=m.qxy; o[7]=m.qxz; o[8]=m.qyy; o[9]=m.qyz;
  o[10]=n->data.box.lesser_corner.x; o[11]=n->data.box.lesser_corner.y;
  o[12]=n->data.box.lesser_corner.z; o[13]=n->data.box.greater_corner.x;
  o[14]=n->data.box.greater_corner.y; o[15]=n->data.box.greater_corner.z;

  if(open && leaf && n->getNumParticles() > 0){
    o[16] = (Real)(int)Bucket;
    for(int i = 0; i < LET_W; i++) mom.push_back(o[i]);
    npart.push_back(n->getNumParticles());
    Particle *pp = n->getParticles();
    for(int i = 0; i < n->getNumParticles(); i++){
      ExternalParticle e;
      e.position = pp[i].position;
      e.mass = pp[i].mass;
      parts.push_back(e);
    }
  }
  else{
    o[16] = (Real)(int)Internal;   // multipole only
    for(int i = 0; i < LET_W; i++) mom.push_back(o[i]);
    npart.push_back(0);
  }
}

void DataManager::sendLets(){
  letsExpected = 0;
  for(int q = 0; q < CkNumPes(); q++)
    if(q != CkMyPe() && peBoxes[q].initialized()) letsExpected++;

  for(int q = 0; q < CkNumPes(); q++){
    if(q == CkMyPe() || !peBoxes[q].initialized()) continue;
    CkVec<Key> keys; CkVec<Real> mom; CkVec<int> npart;
    CkVec<ExternalParticle> parts;
    // From the frontier down, not from the root. Above the frontier the tree
    // is shared: those nodes are Boundary on every PE and their moments were
    // completed by the frontier reduction. Emitting them here would ship one
    // PE's partial view of a node the destination already has complete, and
    // the splice would overwrite the good copy with it.
    for(int i = 0; i < frontier.length(); i++){
      Node<ForceData> *f = frontier[i];
      const NodeType ft = f->getType();
      if(ft == Remote || ft == RemoteBucket || ft == RemoteEmptyBucket) continue;
      collectLet(f, peBoxes[q], keys, mom, npart, parts);
    }

    const int nn = keys.length();
    const int np = parts.length();
    LetMsg *m = new (nn, nn*LET_W, nn, (np > 0 ? np : 1), 0) LetMsg;
    m->numNodes = nn;
    m->numParts = np;
    m->fromPe = CkMyPe();
    if(nn > 0){
      memcpy(m->keys, keys.getVec(), sizeof(Key)*nn);
      memcpy(m->mom, mom.getVec(), sizeof(Real)*nn*LET_W);
      memcpy(m->npart, npart.getVec(), sizeof(int)*nn);
    }
    if(np > 0) memcpy(m->parts, parts.getVec(), sizeof(ExternalParticle)*np);
    thisProxy[q].recvLet(m);
  }
  if(letsExpected == 0){ letsDone = true; treeReady(); }
}

// Find the node with this key, creating it if the local tree does not go that
// deep. The key is the path: bit 63 is the leading one and the bits below it,
// most significant first, are the child choices.
Node<ForceData> *DataManager::descendToKey(Key k){
  int depth = 0;
  Key t = k;
  while(t > Key(1)){ t >>= 1; depth++; }

  Node<ForceData> *cur = root;
  for(int level = depth - 1; level >= 0 && cur != NULL; level--){
    if(cur->getNumChildren() == 0){
      // Only ever grow below something this PE does not own; refining a local
      // subtree would cut its buckets loose from myParticles.
      const NodeType ct = cur->getType();
      if(ct != Remote && ct != RemoteBucket && ct != RemoteEmptyBucket) return NULL;
      cur->refine();
      for(int i = 0; i < cur->getNumChildren(); i++){
        cur->getChild(i)->setType(Remote);
        cur->getChild(i)->setCached();
      }
    }
    const int child = (int)((k >> level) & Key(1));
    cur = cur->getChild(child);
  }
  return cur;
}

void DataManager::recvLet(LetMsg *msg){
  // The reduction callback and a push from a faster PE are two queued
  // messages with no order between them. Splicing before the frontier types
  // are set would find untyped nodes, refuse to grow into them, and silently
  // drop the payload.
  if(!frontierReady){
    pendingLets.push_back(msg);
    return;
  }
  spliceLet(msg);
}

void DataManager::spliceLet(LetMsg *msg){
  int po = 0;
  int droppedNull = 0, droppedOwned = 0, droppedParts = 0, placed = 0;
  for(int i = 0; i < msg->numNodes; i++){
    Node<ForceData> *n = descendToKey(msg->keys[i]);
    const int np = msg->npart[i];
    if(n == NULL){ po += np; droppedNull++; droppedParts += np; continue; }

    // Never write over a node this PE owns or shares. Anything above the
    // frontier is already complete here, and anything below one of our own
    // frontier nodes is ours.
    const NodeType nt = n->getType();
    if(nt != Remote && nt != RemoteBucket && nt != RemoteEmptyBucket){
      po += np;
      droppedOwned++; droppedParts += np;
      continue;
    }
    placed++;

    const Real *o = msg->mom + (size_t)i*LET_W;
    MultipoleMoments &m = n->data.moments;
    m.totalMass = o[0];
    m.cm = Vector3D<Real>(o[1], o[2], o[3]);
    m.rsq = o[4];
    m.qxx = o[5]; m.qxy = o[6]; m.qxz = o[7]; m.qyy = o[8]; m.qyz = o[9];
    n->data.box.lesser_corner  = Vector3D<Real>(o[10], o[11], o[12]);
    n->data.box.greater_corner = Vector3D<Real>(o[13], o[14], o[15]);

    if(np > 0){
      ExternalParticle *dst = new ExternalParticle[np];
      memcpy(dst, msg->parts + po, sizeof(ExternalParticle)*np);
      n->setParticles((Particle *)dst, np);
      n->setType(RemoteBucket);
    }
    else{
      n->setType(Node<ForceData>::makeRemote((NodeType)(int)o[16]));
    }
    n->setCached();
    po += np;
  }
  if(getenv("BARNES_LET_DEBUG") != NULL){
    CkPrintf("[LETDBG] pe %d <- pe %d: %d nodes (%d placed, %d no-path, "
             "%d owned), %d particles, %d particles dropped\n",
             CkMyPe(), msg->fromPe, msg->numNodes, placed, droppedNull,
             droppedOwned, msg->numParts, droppedParts);
  }
  delete msg;

  if(++letsRecvd == letsExpected){
    letsDone = true;
    treeReady();
  }
}

void DataManager::recvFrontierMoments(CkReductionMsg *msg){
  const Real *all = (const Real *)msg->getData();
  const int n = frontier.length();

  // Every PE contributed its own count, so the sum must be n per PE. Check the
  // message size first: if the frontiers diverged the contributions had
  // different lengths and indexing by our own n would read past the end.
  const int got = msg->getSize()/(int)sizeof(Real);
  const int want = n*FRONTIER_W + 1 + 6*CkNumPes();
  if(got != want){
    CkPrintf("[FRONTIER] pe %d: reduced %d reals, expected %d (frontier %d) -- "
             "the PEs did not enumerate the same frontier\n",
             CkMyPe(), got, want, n);
    CkAbort("frontier mismatch");
  }
  const Real counted = all[n*FRONTIER_W];
  if(counted != (Real)(n*CkNumPes())){
    CkPrintf("[FRONTIER] pe %d: counts summed to %g, expected %g\n",
             CkMyPe(), (double)counted, (double)(n*CkNumPes()));
    CkAbort("frontier count mismatch");
  }

  for(int i = 0; i < n; i++){
    Node<ForceData> *f = frontier[i];
    if(f->getType() != Remote && f->getType() != RemoteBucket &&
       f->getType() != RemoteEmptyBucket) continue;
    const Real *o = all + i*FRONTIER_W;
    MultipoleMoments &m = f->data.moments;
    m.totalMass = o[0];
    m.cm = Vector3D<Real>(o[1], o[2], o[3]);
    m.rsq = o[4];
    m.qxx = o[5]; m.qxy = o[6]; m.qxz = o[7]; m.qyy = o[8]; m.qyz = o[9];
    if(m.totalMass > 0.0){
      f->data.box.lesser_corner  = Vector3D<Real>(o[10], o[11], o[12]);
      f->data.box.greater_corner = Vector3D<Real>(o[13], o[14], o[15]);
    }
    else{
      // Nothing there. The reset box makes grow() a no-op in the parent.
      f->data.box.reset();
    }
    // What copyMomentsToNode did: the owner's type, seen from here.
    f->setType(Node<ForceData>::makeRemote((NodeType)(int)o[16]));
  }
  delete msg;

  const bool dbg = getenv("BARNES_FRONTIER_DEBUG") != NULL;
  if(dbg){ fprintf(stderr,"[FRONTIER] pe %d filled remotes\n",CkMyPe()); fflush(stderr); }
  // Unpack every PE's domain box before anything uses it.
  peBoxes.resize(CkNumPes());
  for(int q = 0; q < CkNumPes(); q++){
    const Real *b = all + (n*FRONTIER_W + 1) + 6*q;
    if(b[0] == 0.0 && b[3] == 0.0 && b[1] == 0.0 && b[4] == 0.0){
      peBoxes[q].reset();   // that PE holds nothing
    }
    else{
      peBoxes[q].lesser_corner  = Vector3D<Real>(b[0], b[1], b[2]);
      peBoxes[q].greater_corner = Vector3D<Real>(b[3], b[4], b[5]);
    }
  }

  fillBoundaryMoments(root);
  if(dbg){ fprintf(stderr,"[FRONTIER] pe %d boundary done, root mass %g\n",
                   CkMyPe(), root?(double)root->data.moments.totalMass:-1.0); fflush(stderr); }
  reportLetSizes();
  if(globalParams.useLet){
    letsRecvd = 0;
    letsDone = false;
    frontierReady = true;
    // Anything that arrived early can be spliced now.
    CkVec<LetMsg *> held = pendingLets;
    pendingLets.length() = 0;
    sendLets();
    for(int i = 0; i < held.length(); i++) spliceLet(held[i]);
  }
  else treeReady();
  if(dbg){ fprintf(stderr,"[FRONTIER] pe %d treeReady returned\n",CkMyPe()); fflush(stderr); }
}

Node<ForceData> *DataManager::lookupNode(Key k){
  map<Key,Node<ForceData>*>::iterator it;
  it = nodeTable.find(k);
  if(it == nodeTable.end()) return NULL;
  else return it->second;
}

void DataManager::requestMoments(Key k, int replyTo){
  pendingMoments[k].push_back(replyTo);
  TB_DEBUG("(%d) received requestMoments %lu from pe %d doneTreeBuild %d\n", 
          CkMyPe(), k, replyTo, doneTreeBuild);

  if(doneTreeBuild){    
    Node<ForceData> *node = lookupNode(k);
    if(node == NULL){
      CkPrintf("(%d) recvd request from %d for moments %lu\n", CkMyPe(), replyTo, k);
      CkAbort("bad request\n");
    }
    bool ready = node->allChildrenMomentsReady();
    TB_DEBUG("(%d) node %lu ready %d\n", CkMyPe(), k, ready);

    if(ready){
      map<Key,CkVec<int> >::iterator it = pendingMoments.find(k);
      CkVec<int> &requestors = it->second;
      respondToMomentsRequest(node,requestors);
      pendingMoments.erase(it);
    }
  }
}

void DataManager::flushMomentRequests(){
  CkAssert(doneTreeBuild);
  map<Key,CkVec<int> >::iterator it;
  for(it = pendingMoments.begin(); it != pendingMoments.end();){
    Key k = it->first;
    Node<ForceData> *node = lookupNode(k);
    CkAssert(node != NULL);
    if(node->allChildrenMomentsReady()){
      CkVec<int> &requestors = it->second;
      respondToMomentsRequest(node,requestors);
      map<Key,CkVec<int> >::iterator kill = it;
      ++it;
      pendingMoments.erase(kill);
    }
    else{
      ++it;
    }
  }
}

void DataManager::respondToMomentsRequest(Node<ForceData> *node, CkVec<int> &replyTo){
  for(int i = 0; i < replyTo.length(); i++){
    MomentsMsg *m = new (NUM_PRIORITY_BITS) MomentsMsg(node);
    *(int *)CkPriorityPtr(m) = RECV_MOMENTS_PRIORITY;
    CkSetQueueing(m,CK_QUEUEING_IFIFO);
    TB_DEBUG("(%d) responding to %d with node %lu\n", CkMyPe(), replyTo[i], node->getKey());
    thisProxy[replyTo[i]].receiveMoments(m);
  }
  replyTo.length() = 0;
}

void DataManager::receiveMoments(MomentsMsg *msg){
  Node<ForceData> *node = lookupNode(msg->data.key);
  CkAssert(node != NULL);

  // update moments of leaf and pass these on 
  // to parent recursively; if there are requests
  // for these nodes, respond to them
  updateLeafMoments(node,msg->data);
   
  delete msg;
}

void DataManager::updateLeafMoments(Node<ForceData> *node, MomentsExchangeStruct &data){
  copyMomentsToNode(node,data);
  TB_DEBUG("(%d) updateLeafMoments %lu\n", CkMyPe(), node->getKey());
  passMomentsUpward(node);
}

void DataManager::passMomentsUpward(Node<ForceData> *node){
  TB_DEBUG("(%d) passUp %lu\n", CkMyPe(), node->getKey());
  map<Key,CkVec<int> >::iterator it = pendingMoments.find(node->getKey());
  if(it != pendingMoments.end()){
    CkVec<int> &requestors = it->second;
    respondToMomentsRequest(node,requestors);
    pendingMoments.erase(it);
  }

  Node<ForceData> *parent = node->getParent();
  if(parent == NULL){
    CkAssert(node->getKey() == Key(1));
    treeReady();
  }else{
    parent->childMomentsReady();
    TB_DEBUG("[%d] parent %lu children ready %d\n", CkMyPe(), parent->getKey(), parent->getNumChildrenMomentsReady());
    if(parent->allChildrenMomentsReady()){
      parent->getMomentsFromChildren();
      parent->getOwnershipFromChildren();
      passMomentsUpward(parent);
    }
  }
}

// doneTreeBuild: built local tree and sent out requests for remote 

void DataManager::treeReady(){
  prof.handoff(PhaseProfile::MOMENTS, PhaseProfile::TRAV);
#ifdef GPU_GRAVITY
  if(getenv("BARNES_WALK_DEBUG") != NULL && gpuParticles.attached()){
    DeviceNode rt[4];
    const int n = gpuParticles.readTree(rt, 4);
    CkPrintf("[WALKDBG] pe %d device tree: %d nodes; root mass=%g rsq=%g "
             "cm=(%g %g %g) type=%d firstChild=%d partCount=%d | "
             "host root mass=%g rsq=%g\n",
             CkMyPe(), n, rt[0].cmMass.w, rt[0].rsq,
             rt[0].cmMass.x, rt[0].cmMass.y, rt[0].cmMass.z,
             rt[0].type, rt[0].firstChild, rt[0].partCount,
             root ? root->data.moments.totalMass : -1.0,
             root ? root->data.moments.rsq : -1.0);
  }
#endif
  // Both here rather than next to the build: the host moments are only
  // complete once passMomentsUpward has reached the root.
  patchDeviceBoundaryMoments();
  checkDeviceTree();
  treeMomentsReady = true;
  flushBufferedRemoteDataRequests();
  startTraversal();
}

void DataManager::flushBufferedRemoteDataRequests(){
  CkAssert(treeMomentsReady);
  for(int i = 0; i < bufferedNodeRequests.length(); i++){
    RequestMsg *msg = bufferedNodeRequests[i];
    requestNode(msg);
  }
  for(int i = 0; i < bufferedParticleRequests.length(); i++){
    RequestMsg *msg = bufferedParticleRequests[i];
    requestParticles(msg);
  }
  bufferedNodeRequests.length() = 0;
  bufferedParticleRequests.length() = 0;
}

bool CompareNodePtrToKey(void *a, Key k){
  Node<ForceData> *node = *((Node<ForceData>**)a);
  return (Node<ForceData>::getParticleLevelKey(node) >= k);
}

void DataManager::startTraversal(){
  Node<ForceData> **bucketPtrs = myBuckets.getVec();
  submittedParticles[0].bucketStartIdx = 0;
  int start = 0;
  int end = myBuckets.length();

  if(end > 0){
    for(int i = 0; i < numLocalTreePieces-1; i++){
      TreePieceDescriptor &descr = submittedParticles[i];
      int bucketIdx = binary_search_ge<Node<ForceData>*>(descr.largestKey,bucketPtrs,start,end,CompareNodePtrToKey);
      descr.bucketEndIdx = bucketIdx;
      descr.owner->prepare(root,myBuckets.getVec(),descr.bucketStartIdx,descr.bucketEndIdx);
      int tpIndex = descr.owner->getIndex();
      treePieceProxy[tpIndex].startTraversal();
      submittedParticles[i+1].bucketStartIdx = bucketIdx;
      start = bucketIdx;
    }
    TreePieceDescriptor &descr = submittedParticles[numLocalTreePieces-1];
    descr.bucketEndIdx = myBuckets.length();
    descr.owner->prepare(root,myBuckets.getVec(),descr.bucketStartIdx,descr.bucketEndIdx);
    int tpIndex = descr.owner->getIndex();
    treePieceProxy[tpIndex].startTraversal();
  }
  else if(numLocalTreePieces > 0){
    for(int i = 0; i < numLocalTreePieces; i++){
      TreePieceDescriptor &descr = submittedParticles[i];
      descr.owner->prepare(root,myBuckets.getVec(),0,0);
      int tpIndex = descr.owner->getIndex();
      treePieceProxy[tpIndex].startTraversal();
    }
  }
  else{
    finishIteration();
  }
}

void DataManager::requestParticles(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *traversal){
  Key key = leaf->getKey();
  Request &request = particleRequestTable[key];
  if(!request.sent){
    partReqs.incrRequests();
    prof.partReqSent++;
    request.sentAt = CmiWallTimer();

    if(leaf->isCached()) request.parentCached = true;
    else request.parentCached = false;

    RequestMsg *reqMsg = new (NUM_PRIORITY_BITS) RequestMsg(key,CkMyPe());
    *(int *)CkPriorityPtr(reqMsg) = REQUEST_PARTICLES_PRIORITY;
    CkSetQueueing(reqMsg,CK_QUEUEING_IFIFO);

    CkAssert(leaf->getOwnerStart() == leaf->getOwnerEnd());
    int owner = leaf->getOwnerStart();
    treePieceProxy[owner].requestParticles(reqMsg);
    request.sent = true;
    
    request.parent = leaf;
    RRDEBUG("(%d) REQUEST particles %lu from tp %d\n", 
            CkMyPe(), key, owner);
  }
  request.requestors.push_back(Requestor(worker,state,traversal,worker->getContext()));
  partReqs.incrDeliveries();
}

void DataManager::requestParticles(RequestMsg *msg){
  if(!treeMomentsReady){
    bufferedParticleRequests.push_back(msg);
    return;
  }
  prof.partReqServed++;

  RRDEBUG("(%d) REPLY particles key %lu to %d\n", 
          CkMyPe(), msg->key, msg->replyTo);

  map<Key,Node<ForceData>*>::iterator it = nodeTable.find(msg->key);
  CkAssert(it != nodeTable.end());
  Node<ForceData> *bucket = it->second;
  CkAssert(bucket->getType() == Bucket);

  Particle *data = bucket->getParticles();
  int np = bucket->getNumParticles();

  ParticleReplyMsg *pmsg = new (np,NUM_PRIORITY_BITS) ParticleReplyMsg;
  *(int *)CkPriorityPtr(pmsg) = RECV_PARTICLES_PRIORITY;
  CkSetQueueing(pmsg,CK_QUEUEING_IFIFO);

  pmsg->key = msg->key;
  pmsg->np = np;
  for(int i = 0; i < np; i++){
    pmsg->data[i] = data[i];
  }

  thisProxy[msg->replyTo].recvParticles(pmsg);
  delete msg;
}

void DataManager::requestNode(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *traversal){
  Key key = leaf->getKey();
  Request &request = nodeRequestTable[key];
  if(!request.sent){
    nodeReqs.incrRequests();
    prof.nodeReqSent++;
    request.sentAt = CmiWallTimer();

    if(leaf->isCached()) request.parentCached = true;
    else request.parentCached = false;

    RequestMsg *reqMsg = new (NUM_PRIORITY_BITS) RequestMsg(key,CkMyPe());
    *(int *)CkPriorityPtr(reqMsg) = REQUEST_NODE_PRIORITY;
    CkSetQueueing(reqMsg,CK_QUEUEING_IFIFO);

    int numOwners = leaf->getOwnerEnd()-leaf->getOwnerStart()+1;
    int requestOwner = leaf->getOwnerStart()+(rand()%numOwners);
    RRDEBUG("(%d) REQUEST node %lu from tp %d\n", 
            CkMyPe(), key, requestOwner);
    treePieceProxy[requestOwner].requestNode(reqMsg);
    request.sent = true;
    request.parent = leaf;
  }
  request.requestors.push_back(Requestor(worker,state,traversal,worker->getContext()));
  nodeReqs.incrDeliveries();
}

void DataManager::requestNode(RequestMsg *msg){
  if(!treeMomentsReady){
    bufferedNodeRequests.push_back(msg);
    return;
  }
  prof.nodeReqServed++;

  RRDEBUG("(%d) REPLY node %lu to %d\n", 
          CkMyPe(), msg->key, msg->replyTo);


  map<Key,Node<ForceData>*>::iterator it = nodeTable.find(msg->key);
  CkAssert(it != nodeTable.end());
  Node<ForceData> *node = it->second;

  if(node->getNumChildren() == 0){
    CkPrintf("[%d] children of leaf node %lu type %s requested!\n", CkMyPe(), node->getKey(), NodeTypeString[node->getType()].c_str());
    CkAbort("Leaf children request\n");
  }

  TreeSizeWorker tsz(node->getDepth()+globalParams.cacheLineSize);
  fillTrav.topDownTraversal_local(node,&tsz);

  int nn = tsz.getNumNodes();

  NodeReplyMsg *nmsg = new (nn,NUM_PRIORITY_BITS) NodeReplyMsg;
  *(int *)CkPriorityPtr(nmsg) = RECV_NODE_PRIORITY;
  CkSetQueueing(nmsg,CK_QUEUEING_IFIFO);

  nmsg->key = msg->key;
  nmsg->nn = nn;

  Node<ForceData> *emptyBuf = nmsg->data;
  node->serialize(NULL,emptyBuf,globalParams.cacheLineSize);
  CkAssert(emptyBuf == nmsg->data+nn);

  thisProxy[msg->replyTo].recvNode(nmsg);

  delete msg;
}

void DataManager::recvParticles(ParticleReplyMsg *msg){
  map<Key,Request>::iterator it = particleRequestTable.find(msg->key);
  CkAssert(it != particleRequestTable.end());

  RRDEBUG("(%d) RECVD particle REPLY for key %lu\n", 
          CkMyPe(), msg->key);

  Request &req = it->second;
  CkAssert(req.requestors.length() > 0);
  CkAssert(req.sent);
  CkAssert(req.msg == NULL);

  req.msg = msg;
  req.data = msg->data;
  
  // attach particles to bucket in tree 
  Node<ForceData> *leaf = req.parent;
  CkAssert(leaf != NULL);
  CkAssert(leaf->getType() == RemoteBucket);
  leaf->setParticles((Particle *)msg->data,msg->np);

  partReqs.decrRequests();
  partReqs.decrDeliveries(req.requestors.length());
  prof.partReplies++;
  prof.partReqLatency += CmiWallTimer() - req.sentAt;
  req.deliverParticles(msg->np);
}

void DataManager::recvNode(NodeReplyMsg *msg){
  map<Key,Request>::iterator it = nodeRequestTable.find(msg->key);
  CkAssert(it != nodeRequestTable.end());

  RRDEBUG("(%d) RECVD node REPLY for key %lu\n", 
          CkMyPe(), msg->key);

  Request &req = it->second;
  CkAssert(req.requestors.length() > 0);
  CkAssert(req.sent);
  CkAssert(req.msg == NULL);

  req.msg = msg;
  req.data = msg->data;
  
  // attach recvd subtree to appropriate point in local tree
  Node<ForceData> *node = req.parent;
  CkAssert(node != NULL);
  
  node->deserialize(msg->data, msg->nn);

  ostringstream oss;
  oss << "(" << CkMyPe() << ") key check: " << msg->key << endl;
  TreeChecker checker(oss);
  fillTrav.topDownTraversal_local(node,&checker);

  nodeReqs.decrRequests();
  nodeReqs.decrDeliveries(req.requestors.length());
  prof.nodeReplies++;
  prof.nodeReqLatency += CmiWallTimer() - req.sentAt;
  RRDEBUG("(%d) DELIVERING key %lu\n", 
          CkMyPe(), msg->key);

  req.deliverNode();
  RRDEBUG("(%d) DELIVERED key %lu\n", 
          CkMyPe(), msg->key);
}

#ifdef STATISTICS
void DataManager::traversalsDone(CmiUInt8 pnInter, CmiUInt8 ppInter, CmiUInt8 openCrit)
#else
void DataManager::traversalsDone()
#endif
{
  numTreePiecesDoneTraversals++;
#ifdef STATISTICS
  numInteractions[0] += pnInter;
  numInteractions[1] += ppInter;
  numInteractions[2] += openCrit;
#endif
  if(numTreePiecesDoneTraversals == numLocalTreePieces){
    finishIteration();
  }
}

#ifdef GPU_GRAVITY
void DataManager::ensureDevice(){
  if(gpuParticles.attached()) return;
  gpuParticles.attach(streamPool.ckLocalBranch()->acquire());
}

// Read the accelerations back. No event bookkeeping across the tree pieces'
// streams is needed: a tree piece calls traversalsDone only from its own HAPI
// callback, so all of them having reported means all of their kernels have
// run.
void DataManager::forcesReady(){
  finishIterationTail();
}
#endif

void DataManager::finishIteration(){
  prof.handoff(PhaseProfile::TRAV, PhaseProfile::FINISH);
#ifdef GPU_GRAVITY
  // A PE can hold no tree pieces at all -- the balancer is allowed to empty
  // one -- in which case nothing was ever uploaded and there is no stream to
  // read back on.
  if(!gpuParticles.attached()){
    finishIterationTail();
    return;
  }
  // The accelerations stay on the device: the integrator reads them there and
  // the only thing the host needs from them on this path is the NaN flag.
  // Everything below moves to forcesReady.
  CkCallback cb(CkIndex_DataManager::forcesReady(), CkMyPe(), thisgroup);
  gpuParticles.nanCheck(cb);
#else
  finishIterationTail();
#endif
}

// Written at the end of the last iteration, while the accelerations still
// mean something: the integrator zeroes them. Keyed by SFC key rather than by
// index, because the two builds do not have to agree about which PE holds a
// particle -- only about the force on it.
void DataManager::dumpAccelerations(){
  const char *prefix = getenv("BARNES_ACCEL_DUMP");
  if(prefix == NULL) return;
  if(iteration != globalParams.iterations - 1) return;

#ifdef GPU_GRAVITY
  // They are on the device on this path, and nothing else this iteration
  // wants them on the host. The run is one reduction from finishing, so the
  // blocking readback costs nothing that matters.
  if(gpuParticles.attached()){
    gpuParticles.downloadAccelSync();
    gpuParticles.applyAccel(myParticles.getVec(), myNumParticles);
  }
#endif

  std::ostringstream name;
  name << prefix << "." << CkMyPe();
  std::ofstream out(name.str().c_str());
  out.precision(9);
  out << std::scientific;
  for(int i = 0; i < myNumParticles; i++){
    const Particle &p = myParticles[i];
    out << p.key << " "
        << p.acceleration.x << " " << p.acceleration.y << " "
        << p.acceleration.z << " " << p.potential << "\n";
  }
  out.close();
}

void DataManager::finishIterationTail(){
  // can't advance particles here, because other PEs
  // might not have finished their traversals yet,
  // and therefore might need my particles

  CkAssert(nodeReqs.test());
  CkAssert(partReqs.test());

  InteractionChecker ic;
  fillTrav.postorderTraversal(root,&ic);

  dumpAccelerations();

  DtReductionStruct dtred;
#ifdef GPU_GRAVITY
  if(gpuParticles.attached()){
    // Reduced on the device by nanCheck(); findMinVByA was a whole O(N) host
    // pass whose only product was this flag.
    dtred.haveNaN = (gpuParticles.reduction().haveNaN != 0);
    dtred.vbya = -1.0;
  }
  else
#endif
  findMinVByA(dtred);

#ifdef STATISTICS
  dtred.pnInteractions = numInteractions[0];
  dtred.ppInteractions = numInteractions[1];
  dtred.openCrit = numInteractions[2];
  numInteractions[0] = 0;
  numInteractions[1] = 0;
  numInteractions[2] = 0;
#endif

  CkCallback cb(CkIndex_DataManager::advance(NULL),thisProxy);
  contribute(sizeof(DtReductionStruct),&dtred,DtReductionType,cb);
  prof.end(PhaseProfile::FINISH);
}

void DataManager::advance(CkReductionMsg *msg){
  prof.begin(PhaseProfile::ADVANCE);

  DtReductionStruct *dtred = (DtReductionStruct *)(msg->getData());
  if(dtred->haveNaN){
    CkPrintf("(%d) iteration %d NaN accel detected! Exit...\n", CkMyPe(), iteration);
#ifdef GPU_GRAVITY
    // markNaNBuckets reads the accelerations that produced the NaN, and on
    // this path they are still on the device. The run is stopping, so this
    // blocks rather than threading another entry method through.
    if(gpuParticles.attached()){
      gpuParticles.downloadAccelSync();
      gpuParticles.applyAccel(myParticles.getVec(), myNumParticles);
    }
#endif
    markNaNBuckets();
    printTree();
    CkCallback exitCb(CkCallback::ckExit);
    contribute(0,0,CkReduction::sum_int,exitCb);
    delete msg;
    return;
  }

  if(CkMyPe() == 0){
#ifdef STATISTICS
    CkPrintf("[STATS] node inter %lu part inter %lu open crit %lu dt %f\n", dtred->pnInteractions, dtred->ppInteractions, dtred->openCrit, globalParams.dtime);
#endif
  }
  delete msg;

#ifdef GPU_GRAVITY
  if(gpuParticles.attached()){
    // kick, drift, kick on the device, with the bounding box, the potential
    // and the two kinetic sums reduced in the same pass. advanceTail() picks
    // up when those forty bytes are in host memory.
    const Real dt_k1 = (iteration == 0) ? globalParams.dtime : globalParams.dthf;
    CkCallback cb(CkIndex_DataManager::advanceTail(), CkMyPe(), thisgroup);
    gpuParticles.integrate(dt_k1, globalParams.dtime, globalParams.dthf, cb);
    return;
  }
#endif
  advanceTail();
}

void DataManager::advanceTail(){
  BoundingBox myBox;

#ifdef GPU_GRAVITY
  if(gpuParticles.attached()){
    const GpuKdkReduction &r = gpuParticles.reduction();
    // The device sums the three energy terms separately; the branch that says
    // how to seed the total is the host's, exactly as in kickDriftKick.
    Real energy = (iteration == 0) ? (Real)(r.preKinetic/2.0) : savedEnergy;
    energy += r.potential;
    myBox.energy = energy;
    savedEnergy = (Real)(r.postKinetic/2.0);
    // An empty PE reduces to an inverted box. It is never merged -- the
    // reducer skips a contribution with no particles -- but leave it reset
    // rather than propagate infinities.
    if(myNumParticles > 0){
      myBox.box.lesser_corner  = Vector3D<Real>(r.minx, r.miny, r.minz);
      myBox.box.greater_corner = Vector3D<Real>(r.maxx, r.maxy, r.maxz);
    }
  }
  else
#endif
  kickDriftKick(myBox.box,myBox.energy);

  Real pad = 0.001;
  myBox.expand(pad);
  myBox.numParticles = myNumParticles;

  CkAssert(pendingMoments.empty());
  // safe to reset here, since all tree pieces 
  // must have finished iteration
  freeCachedData();

  submittedParticles.length() = 0;
  haveRanges = false;
  myBuckets.length() = 0;
  doneTreeBuild = false;
  treeMomentsReady = false;
  numTreePiecesDoneTraversals = 0;

  firstSplitterRound = true;
  frontierReady = false;
  freeTree();
  nodeTable.clear();

  CkAssert(activeBins.getNumCounts() == 0);

  if(CkMyPe() == 0) delete[] keyRanges;
  else delete rangeMsg;

  iteration++;
  updateLbInstrumentation();
  prof.iterations++;
  prof.end(PhaseProfile::ADVANCE);
  CkCallback cb;
  if(iteration == globalParams.iterations){
    prof.report();
    cb = CkCallback(CkIndex_Main::niceExit(),mainProxy);
    if (thisIndex == 0) 
      CkPrintf("(%d) finished all %d iterations with avg time %f\n", CkMyPe(), iteration, avgIterationRuntime/globalParams.iterations);
    contribute(0,0,CkReduction::sum_int,cb);
  }
  else{
    cb = CkCallback(CkIndex_DataManager::recvUnivBoundingBox(NULL),thisProxy);
    contribute(sizeof(BoundingBox),&myBox,BoundingBoxGrowReductionType,cb);
  }
}

void DataManager::recvUnivBoundingBox(CkReductionMsg *msg){
  nextUniverse = *((BoundingBox *)msg->getData());
  haveUniverse = true;
  delete msg;
  startNextIteration();
}

// Broadcast from the tree pieces' end-of-iteration reduction. Under the
// unsplit barrier they contribute to it from ResumeFromSync on a balancing
// iteration and directly otherwise; under the split barrier a balancing
// iteration contributes as soon as the step has been joined, which is what
// lets the decomposition below start while the step runs.
void DataManager::treePiecesReady(CkReductionMsg *msg){
  // Max over the elements: 1 if any of them is still owed a migration.
  const int stepInFlight = *((int *)msg->getData());
  stepThisIteration = (stepInFlight != 0);
  treePiecesSettled = true;
  // Nothing is moving, so the local element set is stable already and the
  // decomposition runs straight through. This is every iteration of an
  // unsplit-barrier run, and every non-balancing iteration of a split one.
  if(!stepInFlight) migrationsSettled = true;
  delete msg;
  startNextIteration();
}

// Broadcast from the tree pieces' second reduction, contributed when
// AtSyncWait() releases them. Every element everywhere has come out of the
// step by the time this completes, so no tree piece is in flight to or from
// any PE.
void DataManager::treePiecesMigrated(CkReductionMsg *msg){
  migrationsSettled = true;
  delete msg;
  distributeParticles();
}

// The strategy should read a short, recent window: opened after the previous
// step's migrations settled and closed at the decision, rather than the whole
// period between steps, most of which is stale by the time it is read. The
// same switch gates CUPTI tracing, which is by far the more expensive half, so
// this is also what keeps an instrumented run affordable -- the tracing that
// no strategy will ever read is simply not done.
void DataManager::updateLbInstrumentation(){
  if(globalParams.lbWindow <= 0) return;
  // Level-triggered, not edge-triggered: whether the window is open is a
  // function of the iteration alone, so there is no switch state that can end
  // up out of step with the schedule. Open when one of the next lbWindow
  // iterations is a balancing iteration; closed at the balancing iteration
  // itself, since lbWindow is clamped below the period.
  bool inWindow = false;
  for(int k = 1; k <= globalParams.lbWindow && !inWindow; k++){
    inWindow = isBalancingIteration(globalParams, iteration + k);
  }
  if(inWindow) LBTurnInstrumentOn();
  else LBTurnInstrumentOff();
}

// The tail of the decomposition: hand each tree piece its particles and start
// the tree build. Runs once the decomposition has reached this point and the
// balancer has finished moving elements, in whichever order those happen.
void DataManager::distributeParticles(){
  if(!atDistribute || !migrationsSettled){
    // The decomposition is here and the elements are not still. Time the gap:
    // it is the part of the step the decomposition failed to cover.
    if(atDistribute && decompStalledAt == 0.0) decompStalledAt = CkWallTimer();
    return;
  }
  if(CkMyPe() == 0 && stepThisIteration){
    if(decompStalledAt != 0.0){
      CkPrintf("[LBOVERLAP] iteration %d: decomposition waited %f s for the step\n",
               iteration, CkWallTimer()-decompStalledAt);
    }
    else{
      CkPrintf("[LBOVERLAP] iteration %d: step finished before the decomposition "
               "needed it -- fully overlapped\n", iteration);
    }
  }
  prof.handoff(PhaseProfile::HIST, PhaseProfile::DISTRIB);

  decompStalledAt = 0.0;
  stepThisIteration = false;
  atDistribute = false;
  migrationsSettled = false;

  senseTreePieces();
  if(CkMyPe() == 0){
    keyRanges = new Key[numTreePieces*2];
  }
  // Records the leaves and, on PE 0, fills the key ranges. Nothing goes on the
  // wire until the map arrives.
  flushParticles();
  if(CkMyPe() == 0) haveRanges = true;

  publishTreePieceMap();
}

void DataManager::beginDistribute(){
  atDistribute = true;
  distributeParticles();
}

void DataManager::startNextIteration(){
  if(!haveUniverse || !treePiecesSettled) return;
  haveUniverse = false;
  treePiecesSettled = false;
  decompose(nextUniverse);
}

void DataManager::freeCachedData(){
  map<Key,Request>::iterator it;

  for(it = particleRequestTable.begin(); it != particleRequestTable.end(); it++){
    Request &request = it->second;
    CkAssert(request.sent);
    CkAssert(request.data != NULL);
    CkAssert(request.requestors.length() == 0);
    CkAssert(request.msg != NULL);
    
    // no need to set the particles of a cached
    // bucket to NULL: we will delete the bucket
    // anyway
    if(!request.parentCached){
      request.parent->setParticles(NULL,0);
    }

    delete (ParticleReplyMsg *)(request.msg);
  }

  for(it = nodeRequestTable.begin(); it != nodeRequestTable.end(); it++){
    Request &request = it->second;
    CkAssert(request.sent);
    CkAssert(request.data != NULL);
    CkAssert(request.requestors.length() == 0);
    CkAssert(request.msg != NULL);

    // tell the root of the nodes in this 
    // fetched entry that its children don't
    // exist anymore
    // cached parents may be deleted before
    // their children, so we don't set their
    // children
    if(!request.parentCached){
      request.parent->setChildren(NULL,0);
    }

    delete (NodeReplyMsg *)(request.msg);
  }

  nodeRequestTable.clear();
  particleRequestTable.clear();
}

void DataManager::quiescence(){
  CkPrintf("QUIESCENCE dm %d pieces done %d (%d) nodereq %d partreq %d\n",
              CkMyPe(),
              numTreePiecesDoneTraversals,
              numLocalTreePieces,
              nodeReqs.test(),
              partReqs.test()
              );
  
  CkCallback cb(CkIndex_Main::quiescenceExit(),mainProxy);
  contribute(0,0,CkReduction::sum_int,cb);
}

void DataManager::freeTree(){
  if(root != NULL){
    FreeTreeWorker<ForceData> freeWorker;
    fillTrav.postorderTraversal(root,&freeWorker); 
    delete root;
    root = NULL;
  }
}

void DataManager::printTree(){
  ostringstream oss;
  oss << "pe" << CkMyPe() << "." << iteration << ".dot";

  ofstream ofs(oss.str().c_str());

  ofs << "digraph PE" << CkMyPe() << "_" << iteration << " {" << endl;
  ofs << "node [style=\"filled\"]" << endl;
  if(root != NULL){
    Node<ForceData> &rootRef = *root;
    ofs << rootRef;
  }
  ofs << "}" << endl;
  ofs.close();
}

void DataManager::addBucketNodeInteractions(Key k, CmiUInt8 pn){
#ifdef CHECK_NUM_INTERACTIONS
  Node<ForceData> *node = nodeTable[k];
  node->addNodeInteractions(pn);
#endif
}

void DataManager::addBucketPartInteractions(Key k, CmiUInt8 pp){
#ifdef CHECK_NUM_INTERACTIONS
  Node<ForceData> *node = nodeTable[k];
  node->addPartInteractions(pp);
#endif
}

void DataManager::kickDriftKick(OrientedBox<Real> &box, Real &energy){
  Vector3D<Real> dv;

  Particle *pstart = myParticles.getVec();
  Particle *pend = myParticles.getVec()+myNumParticles;
  Real dt_k1, dt_k2;
  if(iteration == 0){
    dt_k1 = globalParams.dtime;
    // won't have KE stored by a previous 
    // iteration for this one
    for(Particle *p = pstart; p != pend; p++){
      energy += p->mass*(p->velocity.lengthSquared()); 
    }
    energy /= 2.0;
  }
  else{
    dt_k1 = globalParams.dthf;
    energy = savedEnergy;
    savedEnergy = 0.0;
  }

  dt_k2 = globalParams.dthf;

  for(Particle *p = pstart; p != pend; p++){
    energy += p->mass*p->potential;
    // kick
    p->velocity += dt_k1*p->acceleration;
    savedEnergy += p->mass*(p->velocity.lengthSquared()); 
    // drift
    p->position += globalParams.dtime*p->velocity;
    // kick
    p->velocity += dt_k2*p->acceleration;
    
    box.grow(p->position);

    p->acceleration = Vector3D<Real>(0.0);
    p->potential = 0.0;

  }
  savedEnergy /= 2.0;
}

void DataManager::findMinVByA(DtReductionStruct &dtred){
  if(myNumParticles == 0) {
    dtred.haveNaN = false;
    dtred.vbya = -1.0; 
    return;
  }
  
  dtred.haveNaN = false;

  for(int i = 0; i < myNumParticles; i++){
    Real v = myParticles[i].velocity.length();
    Real a = myParticles[i].acceleration.length();
    CkAssert(!isnan(v));
    if(isnan(a)) dtred.haveNaN = true;
   }

  dtred.vbya = -1.0;
}

void DataManager::markNaNBuckets(){
  for(int i = 0; i < myBuckets.length(); i++){
    Node<ForceData> *bucket = myBuckets[i];
    Particle *part = bucket->getParticles();
    int numParticles = bucket->getNumParticles();
    for(int j = 0; j < numParticles; j++){
      if(isnan(part[j].acceleration.length())){
        bucket->setType(Invalid);
        break;
      }
    }
  }
}



#include "Traversal_defs.h"

