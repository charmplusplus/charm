#include "defines.h"

#ifdef GPU_GRAVITY

#include "GpuBatch.h"
#include "Particle.h"

#include <cstring>

namespace {

// The tag handed to CUPTI for a gravity launch.
//
// gpuStableWorkBucket hashes the tag together with the launch geometry, so the
// tag has to name a *class* of launches rather than this launch's exact size:
// a raw interaction count would give every launch its own identity and the
// estimator would never collect a second sample for any of them. Half-octave
// buckets keep the spread inside a class under 1.5x and leave about 120
// classes over the whole representable range.
uint64_t gravityWorkClass(unsigned long long pairs){
  if (pairs == 0) return 0;
  const int e = 63 - __builtin_clzll(pairs);  // floor(log2(pairs))
  const int half = (e >= 1) ? (int)((pairs >> (e - 1)) & 1ULL) : 0;
  return (uint64_t)(2 * e + half) + 1;
}

}  // namespace

void GpuTraversalBatch::growSources(){
  const int newCap = capSrcs ? 2 * capSrcs : (64 * 1024);
  GpuSource *buf = NULL;
  hapiCheck(hapiMallocHost((void **)&buf, sizeof(GpuSource) * newCap));
  if (hSrcs != NULL){
    memcpy(buf, hSrcs, sizeof(GpuSource) * nSrcs);
    hapiCheck(hapiFreeHost(hSrcs));
  }
  hSrcs = buf;
  capSrcs = newCap;
}

void GpuTraversalBatch::growDescs(){
  const int newCap = capDescs ? 2 * capDescs : 1024;
  GpuBucketDesc *buf = NULL;
  hapiCheck(hapiMallocHost((void **)&buf, sizeof(GpuBucketDesc) * newCap));
  if (hDescs != NULL){
    memcpy(buf, hDescs, sizeof(GpuBucketDesc) * nDescs);
    hapiCheck(hapiFreeHost(hDescs));
  }
  hDescs = buf;
  capDescs = newCap;
}

// The device buffers are freed while an earlier launch may still be reading
// them, so growth -- which happens a handful of times over a whole run, since
// each step doubles -- drains the stream first.
void GpuTraversalBatch::growDevice(){
  if (nSrcs > devCapSrcs){
    if (dSrcs != NULL){
      hapiCheck(cudaStreamSynchronize(stream));
      hapiCheck(hapiFree(dSrcs));
    }
    devCapSrcs = nSrcs + nSrcs / 2 + 1024;
    hapiCheck(hapiMalloc((void **)&dSrcs, sizeof(GpuSource) * devCapSrcs));
  }
  if (nDescs > devCapDescs){
    if (dDescs != NULL){
      hapiCheck(cudaStreamSynchronize(stream));
      hapiCheck(hapiFree(dDescs));
    }
    devCapDescs = nDescs + nDescs / 2 + 256;
    hapiCheck(hapiMalloc((void **)&dDescs, sizeof(GpuBucketDesc) * devCapDescs));
  }
}

bool GpuTraversalBatch::flush(const float4 *dPartPos, float4 *dAccel,
                              cudaEvent_t uploadDone, float epssq,
                              const CkCallback &cb){
  closeRun();
  if (nDescs == 0){
    nSrcs = 0;
    return false;
  }

  growDevice();

  // The particle array this kernel reads is uploaded by the DataManager on its
  // own stream. Ordering against that upload here, rather than synchronizing
  // the device, is what lets the tree pieces of a PE overlap with each other.
  hapiCheck(cudaStreamWaitEvent(stream, uploadDone, 0));

  hapiCheck(cudaMemcpyAsync(dSrcs, hSrcs, sizeof(GpuSource) * nSrcs,
                            cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaMemcpyAsync(dDescs, hDescs, sizeof(GpuBucketDesc) * nDescs,
                            cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaEventRecord(h2dDone, stream));

  unsigned long long pairs = 0;
  for (int i = 0; i < nDescs; i++)
    pairs += (unsigned long long)hDescs[i].srcCount * (unsigned long long)hDescs[i].partCount;

  {
    // The grid is one block per run, which says almost nothing about how much
    // work the launch carries: run lengths vary by an order of magnitude with
    // the local particle density, and the number of runs varies with how often
    // the two traversals interleaved. Without a tag the scaling model files
    // every gravity launch under one identity and learns an average of the
    // whole distribution.
    hapiCuptiKernelTagScope workTag(gravityWorkClass(pairs));
    invokeGravity(dSrcs, dDescs, nDescs, dPartPos, dAccel, epssq, stream);
  }

  // By const reference, not through the void* overload: that one dereferences
  // the pointer and copies the callback without taking ownership, so handing
  // it a `new CkCallback` leaks one per launch.
  hapiAddCallback(stream, cb);

  // Refilling the staging buffers has to wait for the upload to drain them.
  // This is the only place the PE blocks on the device. A tree piece normally
  // flushes once per iteration on a list of a few megabytes, so the wait is
  // tens of microseconds against an iteration of tens of milliseconds, and it
  // is what buys one pinned buffer per tree piece instead of a double-buffered
  // pair -- which would double the pinned footprint of every PE.
  hapiCheck(cudaEventSynchronize(h2dDone));
  nSrcs = 0;
  nDescs = 0;
  return true;
}

// Not hapiCheck'd: this runs from a destructor, including at teardown after
// the CUDA context may already be gone, and aborting there would turn a clean
// exit into a crash.
void GpuTraversalBatch::release(){
  if (stream != NULL) cudaStreamSynchronize(stream);
  if (hSrcs != NULL){ hapiFreeHost(hSrcs); hSrcs = NULL; }
  if (hDescs != NULL){ hapiFreeHost(hDescs); hDescs = NULL; }
  if (dSrcs != NULL){ hapiFree(dSrcs); dSrcs = NULL; }
  if (dDescs != NULL){ hapiFree(dDescs); dDescs = NULL; }
  if (h2dDone != NULL){ cudaEventDestroy(h2dDone); h2dDone = NULL; }
  nSrcs = capSrcs = nDescs = capDescs = 0;
  devCapSrcs = devCapDescs = 0;
  curTarget = NULL;
  stream = NULL;
}

void GpuParticleStore::ensure(int n){
  // The reduction scratch does not scale with n, so it is allocated once and
  // kept across resizes.
  if (dRed == NULL){
    hapiCheck(hapiMallocHost((void **)&hRed, sizeof(GpuKdkReduction)));
    hapiCheck(hapiMalloc((void **)&dRed, sizeof(GpuKdkReduction)));
    hapiCheck(hapiMalloc((void **)&dPartials,
                         sizeof(GpuKdkReduction) * REDUCE_BLOCKS));
    std::memset(hRed, 0, sizeof(GpuKdkReduction));
  }

  if (n <= cap) return;
  if (hPos != NULL){
    // Nothing can still be reading these: the previous iteration's readback
    // has already been applied, and everything the tree pieces launched was
    // ordered after the upload on their own streams and has completed -- a
    // tree piece reports its traversal done only from its HAPI callback.
    hapiCheck(hapiFreeHost(hPos));
    hapiCheck(hapiFreeHost(hVel));
    hapiCheck(hapiFreeHost(hAccel));
    hapiCheck(hapiFreeHost(hKey));
    hapiCheck(hapiFree(dPos));
    hapiCheck(hapiFree(dVel));
    hapiCheck(hapiFree(dAccel));
    hapiCheck(hapiFree(dKey));
    hapiCheck(hapiFree(dPosAlt));
    hapiCheck(hapiFree(dVelAlt));
    hapiCheck(hapiFree(dKeyAlt));
    hapiCheck(hapiFree(dIdx));
    hapiCheck(hapiFree(dIdxAlt));
    if (dSortTemp != NULL) hapiCheck(hapiFree(dSortTemp));
  }
  cap = n + n / 4 + 1024;
  hapiCheck(hapiMallocHost((void **)&hPos, sizeof(float4) * cap));
  hapiCheck(hapiMallocHost((void **)&hVel, sizeof(float4) * cap));
  hapiCheck(hapiMallocHost((void **)&hAccel, sizeof(float4) * cap));
  hapiCheck(hapiMallocHost((void **)&hKey, sizeof(unsigned long long) * cap));
  hapiCheck(hapiMalloc((void **)&dPos, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dVel, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dAccel, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dKey, sizeof(unsigned long long) * cap));
  hapiCheck(hapiMalloc((void **)&dPosAlt, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dVelAlt, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dKeyAlt, sizeof(unsigned long long) * cap));
  hapiCheck(hapiMalloc((void **)&dIdx, sizeof(int) * cap));
  hapiCheck(hapiMalloc((void **)&dIdxAlt, sizeof(int) * cap));

  // CUB sizes its scratch from the item count, so it is queried at the
  // capacity rather than at n and reused until the next resize.
  sortTempBytes = gpuSortTempBytes(cap);
  dSortTemp = NULL;
  if (sortTempBytes > 0)
    hapiCheck(hapiMalloc((void **)&dSortTemp, sortTempBytes));
}

// The build allocates children two at a time and stops at bucket size, so the
// node count is bounded by a small multiple of the bucket count. Sizing from
// the capacity keeps it stable across a resize.
void GpuParticleStore::ensureTree(int n){
  const int want = 4 * (n / 8 + 1) + 1024;
  if (dtree.nodes != NULL && dtree.capacity >= want) return;
  if (dtree.nodes != NULL){
    hapiCheck(hapiFree(dtree.nodes));
    hapiCheck(hapiFree(dtree.active));
    hapiCheck(hapiFree(dtree.activeNext));
  }
  else{
    hapiCheck(hapiMalloc((void **)&dtree.nodeCount, sizeof(int)));
    hapiCheck(hapiMalloc((void **)&dtree.activeCount, sizeof(int)));
    hapiCheck(hapiMalloc((void **)&dtree.activeNextCount, sizeof(int)));
    hapiCheck(hapiMalloc((void **)&dtree.levelStart,
                         sizeof(int) * (DTREE_MAX_LEVELS + 2)));
  }
  dtree.capacity = want;
  hapiCheck(hapiMalloc((void **)&dtree.nodes, sizeof(DeviceNode) * want));
  hapiCheck(hapiMalloc((void **)&dtree.active, sizeof(int) * want));
  hapiCheck(hapiMalloc((void **)&dtree.activeNext, sizeof(int) * want));
}

void GpuParticleStore::buildDeviceTree(const Key *owners, int numTreePieces,
                                       int ppbLimit){
  if (nParts <= 0 || owners == NULL) return;
  ensureTree(nParts);

  const int nOwners = 2 * numTreePieces;
  if (nOwners > ownerCap){
    if (dOwners != NULL) hapiCheck(hapiFree(dOwners));
    ownerCap = nOwners + nOwners / 4 + 64;
    hapiCheck(hapiMalloc((void **)&dOwners,
                         sizeof(unsigned long long) * ownerCap));
  }
  hapiCheck(cudaMemcpyAsync(dOwners, owners,
                            sizeof(unsigned long long) * nOwners,
                            cudaMemcpyHostToDevice, stream));

  invokeBuildTree(dKey, nParts, dOwners, numTreePieces, ppbLimit, dtree, stream);
  invokeTreeMoments(dPos, dtree, stream);
  hapiCheck(cudaEventRecord(treeBuilt, stream));
}

void GpuParticleStore::patchMoments(const GpuMomentPatch *patches, int n){
  if (n <= 0 || dtree.nodes == NULL) return;
  if (n > patchCap){
    if (hPatch != NULL){ hapiCheck(hapiFreeHost(hPatch)); hapiCheck(hapiFree(dPatch)); }
    patchCap = n + n/4 + 64;
    hapiCheck(hapiMallocHost((void **)&hPatch, sizeof(GpuMomentPatch)*patchCap));
    hapiCheck(hapiMalloc((void **)&dPatch, sizeof(GpuMomentPatch)*patchCap));
  }
  std::memcpy(hPatch, patches, sizeof(GpuMomentPatch)*n);
  hapiCheck(cudaMemcpyAsync(dPatch, hPatch, sizeof(GpuMomentPatch)*n,
                            cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaMemsetAsync(dtree.activeNextCount, 0, sizeof(int), stream));
  invokeScatterMoments(dtree.nodes, dPatch, n, dtree.activeNextCount, stream);
  int missed = 0;
  hapiCheck(cudaMemcpyAsync(&missed, dtree.activeNextCount, sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));
  if(missed > 0)
    CkPrintf("[PATCH] pe %d: %d of %d boundary moments had no node in the "
             "device tree\n", CkMyPe(), missed, n);
  // The walks wait on this, so it has to be re-recorded after the patch.
  hapiCheck(cudaEventRecord(treeBuilt, stream));
}

int GpuParticleStore::treeNodeCount(){
  if (dtree.nodeCount == NULL) return 0;
  int n = 0;
  hapiCheck(cudaMemcpyAsync(&n, dtree.nodeCount, sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));
  return n;
}

int GpuParticleStore::readTree(DeviceNode *out, int maxNodes){
  const int n = treeNodeCount();
  const int m = (n < maxNodes) ? n : maxNodes;
  if (m > 0){
    hapiCheck(cudaMemcpyAsync(out, dtree.nodes, sizeof(DeviceNode) * m,
                              cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaStreamSynchronize(stream));
  }
  return n;
}

void GpuParticleStore::sortByKey(){
  if (nParts <= 0) return;
  invokeSortByKey(dKey, dKeyAlt, dPos, dPosAlt, dVel, dVelAlt,
                  dIdx, dIdxAlt, dSortTemp, sortTempBytes, nParts, stream);
  // The sorted arrays become the live ones. dAccel needs no permutation: the
  // integrator zeroed it, and upload() memsets it again next iteration.
  float4 *tp = dPos; dPos = dPosAlt; dPosAlt = tp;
  float4 *tv = dVel; dVel = dVelAlt; dVelAlt = tv;
  unsigned long long *tk = dKey; dKey = dKeyAlt; dKeyAlt = tk;
}

void GpuParticleStore::upload(const Particle *parts, int n){
  nParts = n;
  // Before the early return: the reduction scratch does not depend on n, and
  // an empty PE still runs the integrator and reads its result back.
  ensure(n);
  if (n == 0){
    // Still record the event: a tree piece with no buckets of its own can sit
    // on a PE whose DataManager holds nothing, and its (empty) flush path
    // still waits on this handle.
    hapiCheck(cudaEventRecord(uploaded, stream));
    return;
  }

  for (int i = 0; i < n; i++){
    const Particle &p = parts[i];
    hPos[i] = make_float4(p.position.x, p.position.y, p.position.z, p.mass);
    hVel[i] = make_float4(p.velocity.x, p.velocity.y, p.velocity.z, 0.f);
    hKey[i] = (unsigned long long)p.key;
  }

  hapiCheck(cudaMemcpyAsync(dPos, hPos, sizeof(float4) * n,
                            cudaMemcpyHostToDevice, stream));
  // The integrator needs the velocities on the device; the exchange still
  // routes them through the host, so they are re-uploaded every iteration.
  hapiCheck(cudaMemcpyAsync(dVel, hVel, sizeof(float4) * n,
                            cudaMemcpyHostToDevice, stream));
  // The keys too. hashKeys() regenerates them on the device each iteration,
  // but it only runs from the second one onward -- the first decompose happens
  // before anything has been uploaded -- and the tree build splits on them.
  hapiCheck(cudaMemcpyAsync(dKey, hKey, sizeof(unsigned long long) * n,
                            cudaMemcpyHostToDevice, stream));
  // A memset rather than a kernel on purpose. processSubmittedParticles runs
  // inside whichever entry method delivered the last batch of particles, which
  // may well belong to a tree piece; a clearing kernel launched there would be
  // charged to that object as GPU load it did not ask for. CUPTI reports a
  // memset as a memset, not as a kernel, so it stays out of the model.
  hapiCheck(cudaMemsetAsync(dAccel, 0, sizeof(float4) * n, stream));
  hapiCheck(cudaEventRecord(uploaded, stream));
}

void GpuParticleStore::nanCheck(const CkCallback &cb){
  invokeNanCheck(dAccel, nParts, dPartials, dRed, stream);
  hapiCheck(cudaMemcpyAsync(hRed, dRed, sizeof(GpuKdkReduction),
                            cudaMemcpyDeviceToHost, stream));
  hapiAddCallback(stream, cb);
}

void GpuParticleStore::integrate(float dt_k1, float dtime, float dt_k2,
                                 const CkCallback &cb){
  invokeKickDriftKick(dPos, dVel, dAccel, nParts, dt_k1, dtime, dt_k2,
                      dPartials, dRed, stream);
  hapiCheck(cudaMemcpyAsync(hRed, dRed, sizeof(GpuKdkReduction),
                            cudaMemcpyDeviceToHost, stream));
  hapiAddCallback(stream, cb);
}

void GpuParticleStore::hashKeys(float lx, float ly, float lz,
                                float xsz, float ysz, float zsz, bool readback,
                                const CkCallback &cb){
  invokeHashKeys(dPos, dKey, nParts, lx, ly, lz, xsz, ysz, zsz,
                 BITS_PER_DIM, stream);
  // Stage 3: the particles go back to the host already in key order, so
  // decomposeTail has no sort left to do.
  sortByKey();
  if (nParts > 0 && readback){
    // The O(N) transfer Stage 5 exists to remove. The host needs the
    // particles because the all-to-all is still host code.
    hapiCheck(cudaMemcpyAsync(hPos, dPos, sizeof(float4) * nParts,
                              cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaMemcpyAsync(hVel, dVel, sizeof(float4) * nParts,
                              cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaMemcpyAsync(hKey, dKey, sizeof(unsigned long long) * nParts,
                              cudaMemcpyDeviceToHost, stream));
  }
  hapiAddCallback(stream, cb);
}

void GpuParticleStore::applyIntegrated(Particle *parts, int n) const {
  for (int i = 0; i < n; i++){
    const float4 p = hPos[i];
    const float4 v = hVel[i];
    Particle &q = parts[i];
    q.position = Vector3D<Real>(p.x, p.y, p.z);
    q.velocity = Vector3D<Real>(v.x, v.y, v.z);
    q.key = (Key)hKey[i];
    // The device already zeroed its own accumulator; keep the host copy in
    // step so a GPU=0 comparison of the two builds sees the same state.
    q.acceleration = Vector3D<Real>(0.0);
    q.potential = 0.0;
  }
}

// The abort path is about to stop the run, so it blocks rather than threading
// another entry method through for a case that happens once.
void GpuParticleStore::downloadAccelSync(){
  if (nParts == 0) return;
  hapiCheck(cudaMemcpyAsync(hAccel, dAccel, sizeof(float4) * nParts,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));
}

void GpuParticleStore::downloadAccel(const CkCallback &cb){
  if (nParts > 0)
    hapiCheck(cudaMemcpyAsync(hAccel, dAccel, sizeof(float4) * nParts,
                              cudaMemcpyDeviceToHost, stream));
  hapiAddCallback(stream, cb);
}

void GpuParticleStore::applyAccel(Particle *parts, int n) const {
  for (int i = 0; i < n; i++){
    const float4 a = hAccel[i];
    parts[i].acceleration = Vector3D<Real>(a.x, a.y, a.z);
    parts[i].potential = a.w;
  }
}

// --- Stage 5b: device-resident particle exchange ---------------------------

static void ensureRegion(CkVec<char *> &bufs, CkVec<int> &caps, int idx,
                         size_t want){
  while(bufs.length() <= idx){ bufs.push_back(NULL); caps.push_back(0); }
  if((size_t)caps[idx] >= want) return;
  if(bufs[idx] != NULL) hapiCheck(hapiFree(bufs[idx]));
  caps[idx] = (int)(want + want/4 + 4096);
  hapiCheck(hapiMalloc((void **)&bufs[idx], caps[idx]));
}

char *GpuParticleStore::stageSend(int dest, const int *offs, const int *cnts,
                                  int nranges, int total){
  if(total <= 0) return NULL;
  ensureRegion(dSend, sendCap, dest, stageBytes(total));
  char *base = dSend[dest];

  // [pos][vel][key], each block contiguous, so the receiver splits it with
  // three copies and no per-particle work on the device. The ranges come from
  // the sorting tree's leaves and are already contiguous per tree piece.
  float4 *dp = (float4 *)base;
  float4 *dv = (float4 *)(base + (size_t)total*sizeof(float4));
  unsigned long long *dk =
      (unsigned long long *)(base + (size_t)total*sizeof(float4)*2);

  int at = 0;
  for(int r = 0; r < nranges; r++){
    const int o = offs[r], c = cnts[r];
    if(c <= 0) continue;
    hapiCheck(cudaMemcpyAsync(dp + at, dPos + o, sizeof(float4)*c,
                              cudaMemcpyDeviceToDevice, stream));
    hapiCheck(cudaMemcpyAsync(dv + at, dVel + o, sizeof(float4)*c,
                              cudaMemcpyDeviceToDevice, stream));
    hapiCheck(cudaMemcpyAsync(dk + at, dKey + o,
                              sizeof(unsigned long long)*c,
                              cudaMemcpyDeviceToDevice, stream));
    at += c;
  }
  // The send reads this region asynchronously, so it must be settled first.
  hapiCheck(cudaStreamSynchronize(stream));
  return base;
}

char *GpuParticleStore::recvSlot(int src, int total){
  ensureRegion(dRecv, recvCap, src, stageBytes(total > 0 ? total : 1));
  return dRecv[src];
}

void GpuParticleStore::unstageRecv(int src, int total, Particle *out){
  if(total <= 0) return;
  const size_t bytes = stageBytes(total);
  if((int)bytes > hStageCap){
    if(hStage != NULL) hapiCheck(hapiFreeHost(hStage));
    hStageCap = (int)(bytes + bytes/4 + 4096);
    hapiCheck(hapiMallocHost((void **)&hStage, hStageCap));
  }
  hapiCheck(cudaMemcpyAsync(hStage, dRecv[src], bytes,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));

  const float4 *p = (const float4 *)hStage;
  const float4 *v = (const float4 *)(hStage + (size_t)total*sizeof(float4));
  const unsigned long long *k =
      (const unsigned long long *)(hStage + (size_t)total*sizeof(float4)*2);
  for(int i = 0; i < total; i++){
    out[i].position = Vector3D<Real>(p[i].x, p[i].y, p[i].z);
    out[i].mass = p[i].w;
    out[i].velocity = Vector3D<Real>(v[i].x, v[i].y, v[i].z);
    out[i].key = (Key)k[i];
    out[i].acceleration = Vector3D<Real>(0.0);
    out[i].potential = 0.0;
  }
}

void GpuParticleStore::beginAssemble(int total){
  nParts = total;
  ensure(total);
}

// One range of one sender's block into its place in the destination array.
// Contiguous on both sides, so three device-to-device copies and no kernel.
void GpuParticleStore::assembleRange(int srcPe, int srcTotal, int srcOff,
                                     int dstOff, int cnt){
  if(cnt <= 0) return;
  if(srcPe < 0 || srcPe >= dRecv.length() || dRecv[srcPe] == NULL ||
     srcOff + cnt > srcTotal || dstOff + cnt > nParts ||
     (size_t)recvCap[srcPe] < stageBytes(srcTotal)){
    CkPrintf("[ASSEMBLE] pe %d BAD src=%d srcTotal=%d srcOff=%d cnt=%d "
             "dstOff=%d nParts=%d regions=%d cap=%d\n",
             CkMyPe(), srcPe, srcTotal, srcOff, cnt, dstOff, nParts,
             dRecv.length(),
             (srcPe >= 0 && srcPe < recvCap.length()) ? recvCap[srcPe] : -1);
    return;
  }
  char *base = dRecv[srcPe];
  const float4 *sp = (const float4 *)base + srcOff;
  const float4 *sv = (const float4 *)(base + (size_t)srcTotal*sizeof(float4)) + srcOff;
  const unsigned long long *sk =
      (const unsigned long long *)(base + (size_t)srcTotal*sizeof(float4)*2) + srcOff;

  hapiCheck(cudaMemcpyAsync(dPos + dstOff, sp, sizeof(float4)*cnt,
                            cudaMemcpyDeviceToDevice, stream));
  hapiCheck(cudaMemcpyAsync(dVel + dstOff, sv, sizeof(float4)*cnt,
                            cudaMemcpyDeviceToDevice, stream));
  hapiCheck(cudaMemcpyAsync(dKey + dstOff, sk,
                            sizeof(unsigned long long)*cnt,
                            cudaMemcpyDeviceToDevice, stream));
}

void GpuParticleStore::endAssemble(){
  if(nParts <= 0){
    hapiCheck(cudaEventRecord(uploaded, stream));
    return;
  }
  // The blocks arrive in tree piece order, which is key order between pieces
  // but not within the concatenation, so the array still has to be sorted --
  // on the device now, where it already lives.
  sortByKey();
  hapiCheck(cudaMemsetAsync(dAccel, 0, sizeof(float4) * nParts, stream));
  hapiCheck(cudaEventRecord(uploaded, stream));
}

void GpuParticleStore::readbackParticles(Particle *out, int n){
  if(n <= 0) return;
  hapiCheck(cudaMemcpyAsync(hPos, dPos, sizeof(float4)*n,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaMemcpyAsync(hVel, dVel, sizeof(float4)*n,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaMemcpyAsync(hKey, dKey, sizeof(unsigned long long)*n,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));
  for(int i = 0; i < n; i++){
    out[i].position = Vector3D<Real>(hPos[i].x, hPos[i].y, hPos[i].z);
    out[i].mass = hPos[i].w;
    out[i].velocity = Vector3D<Real>(hVel[i].x, hVel[i].y, hVel[i].z);
    out[i].key = (Key)hKey[i];
    out[i].acceleration = Vector3D<Real>(0.0);
    out[i].potential = 0.0;
  }
}

void GpuParticleStore::binCounts(const Key *keys, const int *depths, int nbins,
                                 int *start, int *count, Key *first, Key *last){
  if(nbins <= 0) return;
  if(nbins > binCap){
    if(hBinKey != NULL){
      hapiCheck(hapiFreeHost(hBinKey));   hapiCheck(hapiFree(dBinKey));
      hapiCheck(hapiFreeHost(hBinDepth)); hapiCheck(hapiFree(dBinDepth));
      hapiCheck(hapiFreeHost(hBinStart)); hapiCheck(hapiFree(dBinStart));
      hapiCheck(hapiFreeHost(hBinCount)); hapiCheck(hapiFree(dBinCount));
      hapiCheck(hapiFreeHost(hBinFirst)); hapiCheck(hapiFree(dBinFirst));
      hapiCheck(hapiFreeHost(hBinLast));  hapiCheck(hapiFree(dBinLast));
    }
    binCap = nbins + nbins/4 + 256;
    hapiCheck(hapiMallocHost((void **)&hBinKey,   sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinKey,       sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMallocHost((void **)&hBinFirst, sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinFirst,     sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMallocHost((void **)&hBinLast,  sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinLast,      sizeof(unsigned long long)*binCap));
    hapiCheck(hapiMallocHost((void **)&hBinDepth, sizeof(int)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinDepth,     sizeof(int)*binCap));
    hapiCheck(hapiMallocHost((void **)&hBinStart, sizeof(int)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinStart,     sizeof(int)*binCap));
    hapiCheck(hapiMallocHost((void **)&hBinCount, sizeof(int)*binCap));
    hapiCheck(hapiMalloc((void **)&dBinCount,     sizeof(int)*binCap));
  }

  for(int i = 0; i < nbins; i++){
    hBinKey[i] = (unsigned long long)keys[i];
    hBinDepth[i] = depths[i];
  }
  hapiCheck(cudaMemcpyAsync(dBinKey, hBinKey, sizeof(unsigned long long)*nbins,
                            cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaMemcpyAsync(dBinDepth, hBinDepth, sizeof(int)*nbins,
                            cudaMemcpyHostToDevice, stream));

  invokeBinCount(dKey, nParts, dBinKey, dBinDepth, nbins,
                 dBinStart, dBinCount, dBinFirst, dBinLast, stream);

  hapiCheck(cudaMemcpyAsync(hBinStart, dBinStart, sizeof(int)*nbins,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaMemcpyAsync(hBinCount, dBinCount, sizeof(int)*nbins,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaMemcpyAsync(hBinFirst, dBinFirst, sizeof(unsigned long long)*nbins,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaMemcpyAsync(hBinLast, dBinLast, sizeof(unsigned long long)*nbins,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));

  for(int i = 0; i < nbins; i++){
    start[i] = hBinStart[i];
    count[i] = hBinCount[i];
    first[i] = (Key)hBinFirst[i];
    last[i]  = (Key)hBinLast[i];
  }
}

void GpuParticleStore::gatherExternal(const int *offs, const int *cnts,
                                      int nranges, int total, void *out){
  if(total <= 0) return;
  if(total > gatherCap){
    if(dGather != NULL){ hapiCheck(hapiFree(dGather)); hapiCheck(hapiFreeHost(hGather)); }
    gatherCap = total + total/4 + 1024;
    hapiCheck(hapiMalloc((void **)&dGather, sizeof(float4)*gatherCap));
    hapiCheck(hapiMallocHost((void **)&hGather, sizeof(float4)*gatherCap));
  }
  int at = 0;
  for(int r = 0; r < nranges; r++){
    if(cnts[r] <= 0) continue;
    hapiCheck(cudaMemcpyAsync(dGather + at, dPos + offs[r],
                              sizeof(float4)*cnts[r],
                              cudaMemcpyDeviceToDevice, stream));
    at += cnts[r];
  }
  hapiCheck(cudaMemcpyAsync(hGather, dGather, sizeof(float4)*total,
                            cudaMemcpyDeviceToHost, stream));
  hapiCheck(cudaStreamSynchronize(stream));
  std::memcpy(out, hGather, sizeof(float4)*total);
}

void GpuParticleStore::release(){
  if (stream != NULL) cudaStreamSynchronize(stream);
  if (hPos != NULL){ hapiFreeHost(hPos); hPos = NULL; }
  if (hVel != NULL){ hapiFreeHost(hVel); hVel = NULL; }
  if (hAccel != NULL){ hapiFreeHost(hAccel); hAccel = NULL; }
  if (hKey != NULL){ hapiFreeHost(hKey); hKey = NULL; }
  if (hRed != NULL){ hapiFreeHost(hRed); hRed = NULL; }
  if (dPos != NULL){ hapiFree(dPos); dPos = NULL; }
  if (dVel != NULL){ hapiFree(dVel); dVel = NULL; }
  if (dAccel != NULL){ hapiFree(dAccel); dAccel = NULL; }
  if (dKey != NULL){ hapiFree(dKey); dKey = NULL; }
  if (dPosAlt != NULL){ hapiFree(dPosAlt); dPosAlt = NULL; }
  if (dVelAlt != NULL){ hapiFree(dVelAlt); dVelAlt = NULL; }
  if (dKeyAlt != NULL){ hapiFree(dKeyAlt); dKeyAlt = NULL; }
  if (dIdx != NULL){ hapiFree(dIdx); dIdx = NULL; }
  if (dIdxAlt != NULL){ hapiFree(dIdxAlt); dIdxAlt = NULL; }
  if (dSortTemp != NULL){ hapiFree(dSortTemp); dSortTemp = NULL; }
  if (dOwners != NULL){ hapiFree(dOwners); dOwners = NULL; }
  if (hPatch != NULL){ hapiFreeHost(hPatch); hPatch = NULL; }
  if (dPatch != NULL){ hapiFree(dPatch); dPatch = NULL; }
  patchCap = 0;
  for(int i = 0; i < dSend.length(); i++) if(dSend[i]) hapiFree(dSend[i]);
  for(int i = 0; i < dRecv.length(); i++) if(dRecv[i]) hapiFree(dRecv[i]);
  dSend.length() = 0; dRecv.length() = 0;
  sendCap.length() = 0; recvCap.length() = 0;
  if (hStage != NULL){ hapiFreeHost(hStage); hStage = NULL; }
  hStageCap = 0;
  if (dGather != NULL){ hapiFree(dGather); dGather = NULL; }
  if (hGather != NULL){ hapiFreeHost(hGather); hGather = NULL; }
  gatherCap = 0;
  ownerCap = 0;
  if (dtree.nodes != NULL){
    hapiFree(dtree.nodes);
    hapiFree(dtree.active);
    hapiFree(dtree.activeNext);
    hapiFree(dtree.nodeCount);
    hapiFree(dtree.activeCount);
    hapiFree(dtree.activeNextCount);
    hapiFree(dtree.levelStart);
    clearTreeHandles();
  }
  sortTempBytes = 0;
  if (dPartials != NULL){ hapiFree(dPartials); dPartials = NULL; }
  if (dRed != NULL){ hapiFree(dRed); dRed = NULL; }
  if (uploaded != NULL){ cudaEventDestroy(uploaded); uploaded = NULL; }
  if (treeBuilt != NULL){ cudaEventDestroy(treeBuilt); treeBuilt = NULL; }
  nParts = 0;
  cap = 0;
  stream = NULL;
}

#endif // GPU_GRAVITY
