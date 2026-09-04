#ifndef __BARNES_CUDA_H__
#define __BARNES_CUDA_H__

// Host-visible interface to the gravity kernel, and the place where the
// device-residency contract of the GPU port is written down.
//
// WHAT LIVES ON THE DEVICE
//
// Per PE (owned by the DataManager, rebuilt every iteration):
//   d_partPos   float4 (x, y, z, mass) for every particle this PE holds,
//               in the same order as DataManager::myParticles. A tree node's
//               particleStart is a pointer into that CkVec, so a bucket is a
//               contiguous [offset, count) range of this array.
//   d_accel     float4 (ax, ay, az, potential) accumulator, zeroed at upload
//               and consumed on the device by the integrator.
//   d_vel       float4 (vx, vy, vz, _), so the integrator can run here.
//   d_key       the SFC key, generated here from d_partPos.
//
// Stage 2 of GPU_RESIDENT_PLAN.md added the last two, along with the
// integrator, the key generation and the per-PE reductions. The particles
// still come back to the host once per iteration, because the sort and the
// all-to-all exchange are still host code -- that is what Stages 3 and 5 are
// for, and until they land this is scaffolding rather than a saving.
//
// Per tree piece (owned by the TreePiece, grown on demand):
//   d_srcs      the interaction list -- see GpuSource below.
//   d_buckets   one descriptor per contiguous run of sources belonging to one
//               target bucket.
//
// Nothing device-side survives a migration: a TreePiece's buffers hold only
// the current iteration's interaction list, AtSync is reached after that list
// has been consumed, and the destructor releases them on the old PE.
//
// WHY ONE SOURCE TYPE SERVES BOTH INTERACTION KINDS
//
// The CPU code has two force routines, nodeBucketForce and partBucketForce,
// but both call grav() with a mass and a position: a multipole is used only
// through moments.totalMass and moments.cm, and an ExternalParticle only
// through mass and position. They are numerically the same interaction, so the
// traversal appends both into one list and the kernel has one inner loop.

#include <cuda_runtime.h>

// A source: a point mass, plus the reduced quadrupole about it when the source
// is a cell. A particle source leaves the quadrupole zero, which makes the
// quadrupole terms vanish and costs only the arithmetic.
//
// Aligned like a float4 so that staging the list into shared memory is a
// sequence of 128-bit loads and stores rather than 32-bit ones -- this is the
// kernel's only global read that scales with the interaction count, and it is
// three times the size it was before the quadrupole. -DMONOPOLE_ONLY takes it
// back to 16 bytes, which is how the two are compared.
struct alignas(16) GpuSource {
  float x, y, z, mass;
#ifndef MONOPOLE_ONLY
  float qxx, qxy, qxz, qyy, qyz;
  // Explicit, so the 48-byte size is stated rather than inferred from the
  // alignment. Three float4 loads per source.
  float pad0, pad1, pad2;
#endif
};

// One contiguous run of sources acting on one target bucket.
//
// A bucket can appear more than once in a batch: the local and the remote
// traversal both act on it, and a remote traversal resumes when a deferred
// node or particle reply arrives. Each run becomes its own CUDA block, and the
// blocks that share a bucket accumulate through atomicAdd.
struct GpuBucketDesc {
  int srcStart;    // offset into the source array
  int srcCount;    // number of sources in this run
  int partStart;   // offset into the PE's particle array
  int partCount;   // number of particles in the target bucket
};

// Threads per block. Also the shared-memory tile width for the source list.
#define GRAV_BLOCK_SIZE 128

// accel[i] += sum over the sources of every run targeting particle i.
// One block per descriptor. numDescs blocks are launched on `stream`.
void invokeGravity(const GpuSource* d_srcs, const GpuBucketDesc* d_descs,
                   int numDescs, const float4* d_partPos, float4* d_accel,
                   float epssq, cudaStream_t stream);

// The accumulator is cleared with cudaMemsetAsync rather than a kernel -- see
// the note in GpuParticleStore::upload.

// ---------------------------------------------------------------------------
// Stage 2: integration, key generation, and the per-PE reductions.
// ---------------------------------------------------------------------------

// Everything DataManager::kickDriftKick and findMinVByA used to accumulate in
// two O(N) host passes, reduced on the device instead. Nine floats and a flag,
// whatever N is.
//
// The three kinetic/potential sums are kept separate rather than combined here
// because kickDriftKick seeds the energy differently on iteration 0 (from the
// current velocities) than afterwards (from the previous iteration's saved
// value). The host applies that branch; the device just sums.
struct GpuKdkReduction {
  float minx, miny, minz;   // bounding box of the drifted positions
  float maxx, maxy, maxz;
  float preKinetic;         // sum m*|v|^2 before the first kick
  float potential;          // sum m*phi
  float postKinetic;        // sum m*|v|^2 after the first kick
  int   haveNaN;            // any isnan(|a|)
};

// Blocks used by the reductions. Each block writes one partial and a second
// single-block kernel folds them, so the buffer is a fixed size rather than a
// function of N.
#define REDUCE_BLOCKS 256
#define REDUCE_BLOCK_SIZE 256

// haveNaN only. Runs before the integrator, because the integrator zeroes the
// accelerations and the abort path wants to print them.
void invokeNanCheck(const float4* d_accel, int n, GpuKdkReduction* d_partials,
                    GpuKdkReduction* d_out, cudaStream_t stream);

// kick, drift, kick, zero the accumulator, and reduce the box and the two
// kinetic sums plus the potential. dt_k1 is dtime on iteration 0 and dthf
// afterwards; dt_k2 is always dthf.
void invokeKickDriftKick(float4* d_pos, float4* d_vel, float4* d_accel, int n,
                         float dt_k1, float dtime, float dt_k2,
                         GpuKdkReduction* d_partials, GpuKdkReduction* d_out,
                         cudaStream_t stream);

// The SFC key of every particle, from hashParticleCoordinates. The lower
// corner and the extent of the universe are passed separately, and the kernel
// divides exactly where the host divided: folding the extent into a
// precomputed reciprocal would move a particle across a box boundary now and
// then, and the two builds would decompose differently for no reason.
//
// bitsPerDim comes from the caller rather than from defines.h because this
// header is the only one nvcc sees -- see the note in the Makefile.
void invokeHashKeys(const float4* d_pos, unsigned long long* d_key, int n,
                    float lx, float ly, float lz,
                    float xsz, float ysz, float zsz,
                    int bitsPerDim, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 3: the sort.
// ---------------------------------------------------------------------------

// Scratch DeviceRadixSort wants for n items. Queried once per resize; CUB
// computes it without touching the pointers.
size_t gpuSortTempBytes(int n);

// Sort the particles into key order. The keys are sorted against an index and
// position and velocity are gathered through it -- the acceleration and the
// potential need no permutation because the integrator has just zeroed them,
// and they are the only other fields a Particle carries.
//
// The *Out arrays receive the result and the caller swaps; d_idx and d_idxOut
// are scratch. Radix sort is stable, so particles sharing a key keep the order
// they were uploaded in. The host quickSort this replaces made no such
// promise, which is the one way the two builds can now disagree about the
// decomposition -- and only for particles that landed in the same box of the
// 21-bits-per-dimension grid.
void invokeSortByKey(const unsigned long long* d_keyIn,
                     unsigned long long* d_keyOut,
                     const float4* d_posIn, float4* d_posOut,
                     const float4* d_velIn, float4* d_velOut,
                     int* d_idx, int* d_idxOut,
                     void* d_temp, size_t tempBytes,
                     int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 6: the flat device tree.
//
// A mirror of the host's Node<ForceData> tree, built level by level straight
// from the sorted key array. It exists so that Stage 7 can walk it on the
// device; while the traversal is still host code the host tree is also built,
// and BARNES_TREE_CHECK compares the two.
//
// The build is level-synchronous. Every level's kernel appends its children
// with one atomicAdd, so a level's nodes are contiguous in the array and the
// bottom-up moment pass is a loop over level ranges. Nothing is read back
// between levels: the kernels take the active count by pointer and a fixed
// grid strides over it, so a level with no work is an empty launch rather
// than a synchronisation.
// ---------------------------------------------------------------------------

// Mirrors NodeType in Node.h; only the values the device walk distinguishes.
#define DNODE_INTERNAL     1
#define DNODE_BUCKET       2
#define DNODE_EMPTYBUCKET  3
#define DNODE_BOUNDARY     4

struct DeviceNode {
  float4 cmMass;                  // centre of mass, total mass
  float  rsq;                     // squared opening radius about the cm
  float  qxx, qxy, qxz, qyy, qyz; // reduced quadrupole, as MultipoleMoments
  float3 boxMin, boxMax;          // bounding box; the opening criterion's target side
  int    firstChild;              // index of child 0, or -1 for a leaf
  int    partStart, partCount;    // contiguous range of the sorted particles
  int    type;
  int    depth;
  int    ownerStart, ownerEnd;    // tree-piece range, as the host's owner span
  unsigned long long key;
};

// The most levels the build will descend. TREE_KEY_BITS/LOG_BRANCH_FACTOR is
// the hard limit; this is the practical one, and the build asserts it was not
// hit by leaving nodes active.
#define DTREE_MAX_LEVELS 40

// Scratch the build needs, kept together so the caller allocates once.
struct DeviceTreeScratch {
  DeviceNode* nodes;
  int*  nodeCount;      // one int, the high-water mark
  int*  active;         // node indices still to be refined
  int*  activeNext;
  int*  activeCount;    // one int
  int*  activeNextCount;
  int*  levelStart;     // DTREE_MAX_LEVELS+2 entries; levelStart[L] is the
                        // index of the first node of level L
  int   capacity;       // nodes
};

// Build the tree over n sorted particles. `owners` is the key-range array the
// host calls keyRanges, 2*numTreePieces entries. `ppbLimit` is
// globalParams.ppb*BUCKET_TOLERANCE, the same refine threshold buildTree uses.
// Returns through d_scratch; the caller reads nodeCount when it needs the size.
void invokeBuildTree(const unsigned long long* d_key, int n,
                     const unsigned long long* d_owners, int numTreePieces,
                     int ppbLimit, DeviceTreeScratch scratch,
                     cudaStream_t stream);

// Moments for every node, leaves first and then one level at a time upward.
void invokeTreeMoments(const float4* d_posm, DeviceTreeScratch scratch,
                       cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 7: the device traversal.
//
// Replaces the host walk for the LOCAL half of the traversal. The remote half
// stays on the host because the data it needs is not resident here -- that is
// what Stage 1 (push-based LET) is for.
//
// The walk mirrors Traversal::topDownTraversal exactly, including which nodes
// each half keeps, so it can be compared against the host walk one force at a
// time. work() returning 1 means *descend*; the stack holds nodes already
// accepted for descent; a leaf that is opened falls through to direct
// particle-particle summation, as processLeaf does.
//
// The opening criterion depends only on the cell and the target bucket's box,
// so it is uniform across a block. That is what lets one block walk on behalf
// of all of a bucket's particles with no divergence in the walk itself: every
// thread carries one target particle and they all take the same branch.
// ---------------------------------------------------------------------------

// A target bucket, as the host knows it. The device walk needs no mapping back
// to a device node: the box drives the criterion and the particle range names
// the targets.
struct GpuTargetBucket {
  float3 boxMin, boxMax;
  int partStart, partCount;
};

// accel[] is accumulated into, exactly as the gravity kernel does, so the
// device walk and any remaining host work can both contribute.
void invokeLocalWalk(const DeviceNode* d_nodes, const float4* d_posm,
                     const GpuTargetBucket* d_buckets, int numBuckets,
                     float4* d_accel, float epssq, float tolsq,
                     cudaStream_t stream);

// A completed moment for one device node, pushed down from the host.
//
// The device tree is built from the particles resident on this PE, so a node
// the host calls Boundary carries only local mass here. The host's copy of
// that node has the complete moments -- the cross-PE exchange filled them in
// -- and the device walk needs them, because accepting such a cell as a
// multipole is exactly the case where local-only is wrong.
// Addressed by SFC key, not by node index. A key is the path from the root --
// every bit below the leading one is a child choice -- so the device can walk
// down to the node itself. That is what lets the host push these without first
// reading the whole tree back to learn where each node landed.
struct GpuMomentPatch {
  float4 cmMass;
  float rsq, qxx, qxy, qxz, qyy, qyz;
  unsigned long long key;
};

void invokeScatterMoments(DeviceNode* d_nodes, const GpuMomentPatch* d_patch,
                          int n, cudaStream_t stream);

#endif // __BARNES_CUDA_H__
