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
  if (n <= cap) return;
  if (hPos != NULL){
    // Nothing can still be reading these: the previous iteration's readback
    // has already been applied, and everything the tree pieces launched was
    // ordered after the upload on their own streams and has completed -- a
    // tree piece reports its traversal done only from its HAPI callback.
    hapiCheck(hapiFreeHost(hPos));
    hapiCheck(hapiFreeHost(hAccel));
    hapiCheck(hapiFree(dPos));
    hapiCheck(hapiFree(dAccel));
  }
  cap = n + n / 4 + 1024;
  hapiCheck(hapiMallocHost((void **)&hPos, sizeof(float4) * cap));
  hapiCheck(hapiMallocHost((void **)&hAccel, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dPos, sizeof(float4) * cap));
  hapiCheck(hapiMalloc((void **)&dAccel, sizeof(float4) * cap));
}

void GpuParticleStore::upload(const Particle *parts, int n){
  nParts = n;
  if (n == 0){
    // Still record the event: a tree piece with no buckets of its own can sit
    // on a PE whose DataManager holds nothing, and its (empty) flush path
    // still waits on this handle.
    hapiCheck(cudaEventRecord(uploaded, stream));
    return;
  }
  ensure(n);

  for (int i = 0; i < n; i++){
    const Particle &p = parts[i];
    hPos[i] = make_float4(p.position.x, p.position.y, p.position.z, p.mass);
  }

  hapiCheck(cudaMemcpyAsync(dPos, hPos, sizeof(float4) * n,
                            cudaMemcpyHostToDevice, stream));
  // A memset rather than a kernel on purpose. processSubmittedParticles runs
  // inside whichever entry method delivered the last batch of particles, which
  // may well belong to a tree piece; a clearing kernel launched there would be
  // charged to that object as GPU load it did not ask for. CUPTI reports a
  // memset as a memset, not as a kernel, so it stays out of the model.
  hapiCheck(cudaMemsetAsync(dAccel, 0, sizeof(float4) * n, stream));
  hapiCheck(cudaEventRecord(uploaded, stream));
}

void GpuParticleStore::download(const CkCallback &cb){
  if (nParts > 0)
    hapiCheck(cudaMemcpyAsync(hAccel, dAccel, sizeof(float4) * nParts,
                              cudaMemcpyDeviceToHost, stream));
  hapiAddCallback(stream, cb);
}

void GpuParticleStore::apply(Particle *parts, int n) const {
  for (int i = 0; i < n; i++){
    const float4 a = hAccel[i];
    parts[i].acceleration = Vector3D<Real>(a.x, a.y, a.z);
    parts[i].potential = a.w;
  }
}

void GpuParticleStore::release(){
  if (stream != NULL) cudaStreamSynchronize(stream);
  if (hPos != NULL){ hapiFreeHost(hPos); hPos = NULL; }
  if (hAccel != NULL){ hapiFreeHost(hAccel); hAccel = NULL; }
  if (dPos != NULL){ hapiFree(dPos); dPos = NULL; }
  if (dAccel != NULL){ hapiFree(dAccel); dAccel = NULL; }
  if (uploaded != NULL){ cudaEventDestroy(uploaded); uploaded = NULL; }
  nParts = 0;
  cap = 0;
  stream = NULL;
}

#endif // GPU_GRAVITY
