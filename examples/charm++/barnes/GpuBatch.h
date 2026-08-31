#ifndef __GPU_BATCH_H__
#define __GPU_BATCH_H__

#ifdef GPU_GRAVITY

#include "charm++.h"

#include "barnes_cuda.h"
#include "common.h"
#include "defaults.h"
#include "hapi.h"
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
  inline void addSource(const void *target, int partStart, int partCount,
                        Real mass, const Vector3D<Real> &pos){
    if (target != curTarget){
      closeRun();
      curTarget = target;
      curStart = nSrcs;
      curPartStart = partStart;
      curPartCount = partCount;
    }
    if (nSrcs == capSrcs) growSources();
    GpuSource &s = hSrcs[nSrcs++];
    s.x = pos.x;
    s.y = pos.y;
    s.z = pos.z;
    s.mass = mass;
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
class GpuParticleStore {
public:
  GpuParticleStore()
      : stream(NULL), uploaded(NULL), hPos(NULL), hAccel(NULL),
        dPos(NULL), dAccel(NULL), nParts(0), cap(0)
  {
  }

  ~GpuParticleStore() { release(); }

  void attach(cudaStream_t s){
    stream = s;
    if (uploaded == NULL)
      hapiCheck(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
  }

  bool attached() const { return stream != NULL; }

  // Ship this iteration's particles and clear the accumulator.
  void upload(const Particle *parts, int n);

  // Ordering handle for the tree pieces: their kernels read dPos.
  cudaEvent_t uploadEvent() const { return uploaded; }
  const float4 *positions() const { return dPos; }
  float4 *accel() const { return dAccel; }
  int count() const { return nParts; }

  // Start the readback. `cb` fires once the accelerations are in host memory.
  void download(const CkCallback &cb);
  // Copy them into the particles, replacing whatever was there.
  void apply(Particle *parts, int n) const;

  void release();

private:
  void ensure(int n);

  cudaStream_t stream;
  cudaEvent_t uploaded;
  float4 *hPos;    // pinned staging, (x, y, z, mass)
  float4 *hAccel;  // pinned staging, (ax, ay, az, potential)
  float4 *dPos;
  float4 *dAccel;
  int nParts;
  int cap;
};

#endif // GPU_GRAVITY
#endif // __GPU_BATCH_H__
