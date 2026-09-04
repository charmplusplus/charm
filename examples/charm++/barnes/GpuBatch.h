#ifndef __GPU_BATCH_H__
#define __GPU_BATCH_H__

#ifdef GPU_GRAVITY

#include "charm++.h"

#include "barnes_cuda.h"
#include "common.h"
#include "defaults.h"
#include "hapi.h"
#include "MultipoleMoments.h"
#include "Vector3D.h"

// One tree piece's interaction list, and the device resources that evaluate it.
//
// The traversal appends sources here instead of calling grav(). A source is
// tagged with the *target* bucket it acts on rather than being written into a
// per-bucket list, because the two traversals of a tree piece interleave --
// doLocalGravity and doRemoteGravity yield to each other every yieldPeriod
// buckets, and a remote traversal resumes whenever a deferred reply lands --
// so the sequence of targets is not monotone. Comparing against the last
// target turns that sequence into contiguous runs, and each run becomes one
// CUDA block.
//
// Everything is allocated on first use, never from a constructor: tree pieces
// are created during startup, before HAPI has picked this PE's device.
class GpuTraversalBatch {
public:
  GpuTraversalBatch()
      : stream(NULL), h2dDone(NULL),
        hSrcs(NULL), nSrcs(0), capSrcs(0),
        hDescs(NULL), nDescs(0), capDescs(0),
        dSrcs(NULL), devCapSrcs(0), dDescs(NULL), devCapDescs(0),
        curTarget(NULL), curStart(0), curPartStart(0), curPartCount(0),
        // Not zero: wantsFlush is consulted from the first traversal, which
        // can run before attach() has had a chance to read the command line.
        flushLimit(DEFAULT_GPU_FLUSH_LIMIT)
  {
  }

  ~GpuTraversalBatch() { release(); }

  // Bind the stream this batch launches on. Called from an entry method.
  void attach(cudaStream_t s, int limit){
    stream = s;
    flushLimit = limit;
    if (h2dDone == NULL)
      hapiCheck(cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming));
  }

  bool attached() const { return stream != NULL; }
  cudaStream_t getStream() const { return stream; }

  // The hot path: one call per node-bucket or particle-bucket interaction.
  // Two forms, because a particle source has no quadrupole and a cell does.
  inline GpuSource &openSource(const void *target, int partStart, int partCount){
    if (target != curTarget){
      closeRun();
      curTarget = target;
      curStart = nSrcs;
      curPartStart = partStart;
      curPartCount = partCount;
    }
    if (nSrcs == capSrcs) growSources();
    return hSrcs[nSrcs++];
  }

  // A particle: a point mass, quadrupole zero.
  inline void addSource(const void *target, int partStart, int partCount,
                        Real mass, const Vector3D<Real> &pos){
    GpuSource &s = openSource(target, partStart, partCount);
    s.x = pos.x;
    s.y = pos.y;
    s.z = pos.z;
    s.mass = mass;
#ifndef MONOPOLE_ONLY
    s.qxx = s.qxy = s.qxz = s.qyy = s.qyz = 0.f;
#endif
  }

  // A cell: the monopole and the reduced quadrupole about its centre of mass.
  inline void addSource(const void *target, int partStart, int partCount,
                        const MultipoleMoments &m){
    GpuSource &s = openSource(target, partStart, partCount);
    s.x = m.cm.x;
    s.y = m.cm.y;
    s.z = m.cm.z;
    s.mass = m.totalMass;
#ifndef MONOPOLE_ONLY
    s.qxx = m.qxx;
    s.qxy = m.qxy;
    s.qxz = m.qxz;
    s.qyy = m.qyy;
    s.qyz = m.qyz;
#endif
  }

  // True once the list is large enough that the tree piece should launch. The
  // caller decides *when* to act on this, because a launch has to happen
  // inside a TreePiece entry method for CUPTI to attribute it to the tree
  // piece -- and sources are also appended from DataManager entry methods,
  // when a deferred node or particle reply is delivered.
  bool wantsFlush() const { return nSrcs >= flushLimit; }
  bool empty() const { return nSrcs == 0 && nDescs == 0; }

  // Close the open run, ship the list, and launch. Returns true if a kernel
  // was launched, in which case `cb` will fire when it completes.
  bool flush(const float4 *dPartPos, float4 *dAccel, cudaEvent_t uploadDone,
             float epssq, const CkCallback &cb);

  void release();

private:
  void closeRun(){
    if (curTarget == NULL) return;
    const int n = nSrcs - curStart;
    if (n > 0 && curPartCount > 0){
      if (nDescs == capDescs) growDescs();
      GpuBucketDesc &d = hDescs[nDescs++];
      d.srcStart = curStart;
      d.srcCount = n;
      d.partStart = curPartStart;
      d.partCount = curPartCount;
    }
    curTarget = NULL;
  }

  void growSources();
  void growDescs();
  void growDevice();

  cudaStream_t stream;
  cudaEvent_t h2dDone;

  // Pinned, so the upload is a real async DMA. The traversal writes straight
  // into it; there is no second host copy.
  GpuSource *hSrcs;
  int nSrcs, capSrcs;
  GpuBucketDesc *hDescs;
  int nDescs, capDescs;

  GpuSource *dSrcs;
  int devCapSrcs;
  GpuBucketDesc *dDescs;
  int devCapDescs;

  const void *curTarget;
  int curStart;
  int curPartStart, curPartCount;

  int flushLimit;
};

struct Particle;

// The PE's particles on the device, and the accumulator the kernels write.
//
// Owned by the DataManager rather than by a tree piece because that is where
// the particles are: tree pieces hand their particles to the local DataManager
// every iteration, which sorts them into one array and builds one tree over
// it, and a bucket is a contiguous range of that array. A tree piece owns
// buckets, not particles.
//
// This is also why migration needs nothing device-side. The array is rebuilt
// from scratch every iteration out of whatever particles the local tree pieces
// submitted, so a tree piece that moved simply causes a different set of
// particles to be uploaded on each of the two PEs.
//
// Stage 2 added the velocity and the key, and with them the integrator, the
// key generation and the per-PE reductions -- see barnes_cuda.h. The particles
// still come back once per iteration because the sort and the exchange are
// still host code; what went away is three O(N) host passes over them.
class GpuParticleStore {
public:
  GpuParticleStore()
      : stream(NULL), uploaded(NULL), treeBuilt(NULL),
        hPos(NULL), hVel(NULL), hAccel(NULL), hKey(NULL), hRed(NULL),
        dPos(NULL), dVel(NULL), dAccel(NULL), dKey(NULL),
        dPosAlt(NULL), dVelAlt(NULL), dKeyAlt(NULL),
        dIdx(NULL), dIdxAlt(NULL), dSortTemp(NULL), sortTempBytes(0),
        dPartials(NULL), dRed(NULL),
        dOwners(NULL), ownerCap(0),
        hPatch(NULL), dPatch(NULL), patchCap(0),
        hStage(NULL), hStageCap(0),
        dGather(NULL), hGather(NULL), gatherCap(0),
        dRemote(NULL), remoteCap(0), remoteUsed(0),
        hLet(NULL), dLet(NULL), letCap(0),
        hBinKey(NULL), dBinKey(NULL), hBinFirst(NULL), dBinFirst(NULL),
        hBinLast(NULL), dBinLast(NULL), hBinDepth(NULL), dBinDepth(NULL),
        hBinStart(NULL), dBinStart(NULL), hBinCount(NULL), dBinCount(NULL),
        binCap(0),
        nParts(0), cap(0)
  {
    clearTreeHandles();
  }

  ~GpuParticleStore() { release(); }

  // dtree holds raw device pointers; zeroing them lets ensureTree tell a first
  // call from a resize, and release() from a double free.
  void clearTreeHandles(){
    dtree.nodes = NULL; dtree.nodeCount = NULL;
    dtree.active = NULL; dtree.activeNext = NULL;
    dtree.activeCount = NULL; dtree.activeNextCount = NULL;
    dtree.levelStart = NULL; dtree.capacity = 0;
  }

  void attach(cudaStream_t s){
    stream = s;
    if (uploaded == NULL)
      hapiCheck(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
    if (treeBuilt == NULL)
      hapiCheck(cudaEventCreateWithFlags(&treeBuilt, cudaEventDisableTiming));
  }

  bool attached() const { return stream != NULL; }

  // Ship this iteration's particles and clear the accumulator.
  void upload(const Particle *parts, int n);

  // Ordering handle for the tree pieces: their kernels read dPos.
  cudaEvent_t uploadEvent() const { return uploaded; }
  const float4 *positions() const { return dPos; }
  float4 *accel() const { return dAccel; }
  int count() const { return nParts; }

  // --- Stage 2 ------------------------------------------------------------
  // Each of these launches on the stream and then reads back a fixed forty
  // bytes; `cb` fires when that is in host memory. They are separate calls
  // rather than one because a Charm reduction sits between each pair -- see
  // the sync-point budget in GPU_RESIDENT_PLAN.md section 3.2.

  // haveNaN over the accelerations. Runs before integrate(), which zeroes
  // them.
  void nanCheck(const CkCallback &cb);

  // kick, drift, kick. Leaves the drifted positions and the new velocities on
  // the device and the accumulator zeroed.
  void integrate(float dt_k1, float dtime, float dt_k2, const CkCallback &cb);

  // SFC keys from the drifted positions, the sort into key order, and then
  // the particles back to the host: the exchange is still host code. Stage 5
  // is what removes the transfer.
  void hashKeys(float lx, float ly, float lz,
                float xsz, float ysz, float zsz, bool readback,
                const CkCallback &cb);

  // Valid once the callback from any of the three above has fired.
  const GpuKdkReduction &reduction() const { return *hRed; }

  // Write the drifted positions, the new velocities and the keys back into the
  // host particles. Pairs with hashKeys().
  void applyIntegrated(Particle *parts, int n) const;

  // --- Stage 6 ------------------------------------------------------------
  // Build the flat device tree over the particles already resident here, and
  // its moments. `owners` is the host's keyRanges. Launches on this store's
  // stream; nothing comes back until checkTree asks.
  void buildDeviceTree(const Key *owners, int numTreePieces, int ppbLimit);

  // Recorded after the build and its moments. The tree pieces walk on their
  // own streams, so they have to wait on this and not on uploadEvent -- the
  // upload is recorded before the build even starts.
  cudaEvent_t treeEvent() const { return treeBuilt; }

  // Push completed moments for the boundary nodes down into the device tree
  // and re-record treeBuilt, so a walk waiting on it sees them.
  void patchMoments(const GpuMomentPatch *patches, int n);

  // --- Stage 5b -----------------------------------------------------------
  // Gather the particles bound for one destination into a contiguous device
  // staging region, laid out as [pos][vel][key]. The ranges come from the
  // sorting tree's leaves and are already contiguous per tree piece, so this
  // is a handful of device-to-device copies and no packing kernel.
  char *stageSend(int dest, const int *offs, const int *cnts, int nranges,
                  int total);
  // Where an incoming push should land.
  char *recvSlot(int src, int total);
  // Pull one arrived block back to the host as Particles.
  void unstageRecv(int src, int total, struct Particle *out);
  static size_t stageBytes(int n){ return (size_t)n*(sizeof(float4)*2 + sizeof(unsigned long long)); }

  // Assemble this iteration's particles on the device, straight out of the
  // staging regions the pushes landed in. Replaces the host concatenation, the
  // host sort and the re-upload that followed it: the particles never leave.
  void beginAssemble(int total);
  void assembleRange(int srcPe, int srcTotal, int srcOff, int dstOff, int cnt);
  void endAssemble();
  // Bring the assembled, sorted particles back for whatever still runs on the
  // host -- the histogram's sorting tree and the host tree build.
  void readbackParticles(struct Particle *out, int n);

  // Extent of each key-prefix bin over the sorted device keys. Blocking; the
  // decomposition needs the answer before it can decide the next round.
  void binCounts(const Key *keys, const int *depths, int nbins,
                 int *start, int *count, Key *first, Key *last);

  // Positions and masses for a set of particle ranges, as ExternalParticle.
  // dPos is already (x, y, z, mass), which is exactly that layout, so this is
  // a gather and one copy back -- collectLet never has to touch the host
  // array, which is the last thing keeping it alive.
  void gatherExternal(const int *offs, const int *cnts, int nranges,
                      int total, void *out);

  // --- pushed trees, device resident ---------------------------------------
  // Gather one destination's payload and leave it on the device for the
  // transport to read; the send must not be rebuilt until its callback fires.
  float4 *stageLetSend(int dest, const int *offs, const int *cnts,
                       int nranges, int total);
  // Where an incoming payload should land, and where it ends up afterwards:
  // everything received is concatenated into one block, because a spliced node
  // indexes that block and not a per-sender one.
  float4 *letRecvSlot(int src, int n);
  int appendRemote(int src, int n);      // returns the offset it landed at
  void resetRemote(){ remoteUsed = 0; }
  const float4 *deviceRemote() const { return dRemote; }
  // Splice the pushed cells into the tree and re-record treeBuilt.
  int insertLet(const struct GpuLetNode *nodes, int n);
  cudaStream_t deviceStream() const { return stream; }


  // Read the tree back for the self-check. Blocking, and only used under
  // BARNES_TREE_CHECK.
  int  readTree(DeviceNode *out, int maxNodes);
  int  treeNodeCount();

  const DeviceNode *deviceNodes() const { return dtree.nodes; }

  // The abort path only: markNaNBuckets prints the accelerations that produced
  // a NaN, so they have to come back before the run stops.
  void downloadAccel(const CkCallback &cb);
  void downloadAccelSync();
  void applyAccel(Particle *parts, int n) const;

  void release();

private:
  void ensure(int n);
  // Sorts in place from the caller's point of view: the result lands in the
  // alternate buffers and the pointers are swapped.
  void sortByKey();

  cudaStream_t stream;
  cudaEvent_t uploaded;
  cudaEvent_t treeBuilt;
  float4 *hPos;    // pinned staging, (x, y, z, mass); used both directions
  float4 *hVel;    // pinned staging, (vx, vy, vz, _); used both directions
  float4 *hAccel;  // pinned staging, (ax, ay, az, potential); abort path only
  unsigned long long *hKey;
  GpuKdkReduction *hRed;
  float4 *dPos;
  float4 *dVel;
  float4 *dAccel;
  unsigned long long *dKey;
  // Stage 3. Radix sort is not in place, so position, velocity and key each
  // need a second buffer to land in; the pointers are swapped afterwards.
  float4 *dPosAlt;
  float4 *dVelAlt;
  unsigned long long *dKeyAlt;
  int *dIdx, *dIdxAlt;
  void *dSortTemp;
  size_t sortTempBytes;
  // Bin query scratch, sized to the largest round seen.
  unsigned long long *hBinKey, *dBinKey, *hBinFirst, *dBinFirst, *hBinLast, *dBinLast;
  int *hBinDepth, *dBinDepth, *hBinStart, *dBinStart, *hBinCount, *dBinCount;
  int binCap;

  GpuMomentPatch *hPatch;
  GpuMomentPatch *dPatch;
  int patchCap;
  // One send and one receive region per PE, grown on demand.
  CkVec<char *> dSend, dRecv;
  CkVec<int> sendCap, recvCap;
  char *hStage;
  int hStageCap;
  float4 *dGather, *hGather;
  int gatherCap;
  CkVec<float4 *> dLetSend, dLetRecv;
  CkVec<int> letSendCap, letRecvCap;
  float4 *dRemote;
  int remoteCap, remoteUsed;
  struct GpuLetNode *hLet, *dLet;
  int letCap;
  GpuKdkReduction *dPartials;  // REDUCE_BLOCKS entries
  GpuKdkReduction *dRed;

  // Stage 6. The scratch struct holds device pointers; the store owns them.
  DeviceTreeScratch dtree;
  unsigned long long *dOwners;
  int ownerCap;
  void ensureTree(int n);
  void ensureTreeCapacity(int want);
  int nParts;
  int cap;
};

#endif // GPU_GRAVITY
#endif // __GPU_BATCH_H__
