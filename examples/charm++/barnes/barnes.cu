// The gravity kernel for the GPU port.
//
// This is a transcription of grav() in gravity.h, with one deliberate
// numerical difference. Two things make the totals differ from the CPU's:
//
//  - Summation order. A thread accumulates its particle's whole acceleration
//    in registers and commits once, where the CPU adds each interaction into
//    the particle in traversal order.
//  - The reciprocal square root. grav() computes sqrt in double and then
//    divides twice; this kernel takes rsqrtf once and multiplies. rsqrtf is
//    accurate to about 2 ulp, which is several orders of magnitude below the
//    force error the multipole approximation already carries at the default
//    theta of 0.5 -- but it is not bit-for-bit, so the CPU build is no longer
//    an exact oracle for this path.
//
// Building with -DGRAV_EXACT_RSQRT restores the sqrt-and-divide form for when
// the comparison against the CPU build needs to be tight.
//
// MAPPING
//
// One block per descriptor, one thread per particle of the target bucket, and
// the source list staged through shared memory in GRAV_BLOCK_SIZE tiles so
// every thread of the block reads each source once from L1 instead of once
// from L2 per thread.
//
// This wants buckets of about GRAV_BLOCK_SIZE particles. The CPU default of
// -b=10 leaves 118 of 128 threads idle; see the note on -b in the README.
// Buckets larger than the block are handled by striding, so correctness does
// not depend on the choice.

#include "barnes_cuda.h"

__global__ void gravKernel(const GpuSource* __restrict__ srcs,
                           const GpuBucketDesc* __restrict__ descs,
                           const float4* __restrict__ partPos,
                           float4* __restrict__ accel, float epssq) {
  const GpuBucketDesc d = descs[blockIdx.x];
  __shared__ GpuSource tile[GRAV_BLOCK_SIZE];

  for (int base = 0; base < d.partCount; base += GRAV_BLOCK_SIZE) {
    const int i = base + threadIdx.x;
    const bool active = (i < d.partCount);

    float px = 0.f, py = 0.f, pz = 0.f;
    if (active) {
      const float4 p = partPos[d.partStart + i];
      px = p.x; py = p.y; pz = p.z;
    }

    float ax = 0.f, ay = 0.f, az = 0.f, phi = 0.f;

    for (int s = 0; s < d.srcCount; s += GRAV_BLOCK_SIZE) {
      const int t = s + threadIdx.x;
      // Every thread of the block reaches both barriers: the loop bound is
      // uniform because srcCount comes from the descriptor, not from `active`.
      __syncthreads();
      if (t < d.srcCount) tile[threadIdx.x] = srcs[d.srcStart + t];
      __syncthreads();

      const int n = min(GRAV_BLOCK_SIZE, d.srcCount - s);
      if (active) {
        for (int k = 0; k < n; k++) {
          const GpuSource src = tile[k];
          // grav(): dr = source position - particle position.
          const float dx = src.x - px;
          const float dy = src.y - py;
          const float dz = src.z - pz;
          const float drsq = dx * dx + dy * dy + dz * dz + epssq;
#ifdef GRAV_EXACT_RSQRT
          const float rinv = 1.f / sqrtf(drsq);
#else
          // One special-function op replaces a square root and two divides.
          const float rinv = rsqrtf(drsq);
#endif
          const float rinv2 = rinv * rinv;
          const float mrinv = src.mass * rinv;   // M/r
#ifdef MONOPOLE_ONLY
          phi -= mrinv;
          const float coef = mrinv * rinv2;      // M/r^3
          ax += coef * dx;
          ay += coef * dy;
          az += coef * dz;
#else
          // gravQuad() in gravity.h, term for term.
          const float qzz = -(src.qxx + src.qyy);
          const float qx = src.qxx * dx + src.qxy * dy + src.qxz * dz;
          const float qy = src.qxy * dx + src.qyy * dy + src.qyz * dz;
          const float qz = src.qxz * dx + src.qyz * dy + qzz      * dz;
          const float drQdr = dx * qx + dy * qy + dz * qz;

          const float rinv5 = rinv2 * rinv2 * rinv;
          const float rinv7 = rinv5 * rinv2;

          phi -= (mrinv + 0.5f * drQdr * rinv5);

          const float coef = mrinv * rinv2 + 2.5f * drQdr * rinv7;
          ax += coef * dx - rinv5 * qx;
          ay += coef * dy - rinv5 * qy;
          az += coef * dz - rinv5 * qz;
#endif
        }
      }
    }

    if (active) {
      // Atomic because several blocks of this launch can target the same
      // bucket -- see the note on GpuBucketDesc -- and because successive
      // launches on this stream keep adding to the same accumulator.
      float4* a = accel + (d.partStart + i);
      atomicAdd(&a->x, ax);
      atomicAdd(&a->y, ay);
      atomicAdd(&a->z, az);
      atomicAdd(&a->w, phi);
    }
  }
}

void invokeGravity(const GpuSource* d_srcs, const GpuBucketDesc* d_descs,
                   int numDescs, const float4* d_partPos, float4* d_accel,
                   float epssq, cudaStream_t stream) {
  if (numDescs <= 0) return;
  gravKernel<<<numDescs, GRAV_BLOCK_SIZE, 0, stream>>>(
      d_srcs, d_descs, d_partPos, d_accel, epssq);
}

// ---------------------------------------------------------------------------
// Stage 2: integration, key generation, and the per-PE reductions.
//
// These replace three O(N) host passes -- findMinVByA, kickDriftKick, and
// hashParticleCoordinates -- and the accumulators they carried. The reductions
// are two-stage: every block folds its own threads into one partial, and a
// single-block kernel folds the partials, so the scratch buffer is a fixed
// REDUCE_BLOCKS entries rather than a function of N.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void redInit(GpuKdkReduction &r){
  r.minx = r.miny = r.minz = INFINITY;
  r.maxx = r.maxy = r.maxz = -INFINITY;
  r.preKinetic = r.potential = r.postKinetic = 0.f;
  r.haveNaN = 0;
}

__device__ __forceinline__ void redCombine(GpuKdkReduction &a,
                                           const GpuKdkReduction &b){
  a.minx = fminf(a.minx, b.minx);
  a.miny = fminf(a.miny, b.miny);
  a.minz = fminf(a.minz, b.minz);
  a.maxx = fmaxf(a.maxx, b.maxx);
  a.maxy = fmaxf(a.maxy, b.maxy);
  a.maxz = fmaxf(a.maxz, b.maxz);
  a.preKinetic  += b.preKinetic;
  a.potential   += b.potential;
  a.postKinetic += b.postKinetic;
  a.haveNaN     |= b.haveNaN;
}

// Fold one block's threads down to sh[0]. Every thread must reach this.
__device__ __forceinline__ void redBlock(GpuKdkReduction *sh,
                                         const GpuKdkReduction &mine){
  sh[threadIdx.x] = mine;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1){
    if (threadIdx.x < s) redCombine(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
}

__global__ void reduceFinalKernel(const GpuKdkReduction* __restrict__ partials,
                                  int count, GpuKdkReduction* out){
  __shared__ GpuKdkReduction sh[REDUCE_BLOCK_SIZE];
  GpuKdkReduction r;
  redInit(r);
  for (int i = threadIdx.x; i < count; i += blockDim.x) redCombine(r, partials[i]);
  redBlock(sh, r);
  if (threadIdx.x == 0) *out = sh[0];
}

// findMinVByA. The host test is isnan on the length of the acceleration, so
// that is what this computes rather than testing the components: an infinite
// component makes an infinite length, not a NaN, and the two must agree.
__global__ void nanCheckKernel(const float4* __restrict__ accel, int n,
                               GpuKdkReduction* partials){
  __shared__ GpuKdkReduction sh[REDUCE_BLOCK_SIZE];
  GpuKdkReduction r;
  redInit(r);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x){
    const float4 a = accel[i];
    const float len = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    if (isnan(len)) r.haveNaN = 1;
  }
  redBlock(sh, r);
  if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

// kickDriftKick, with the accumulators it carried folded in. The three
// kinetic/potential sums stay separate; the host applies the iteration-0
// branch that decides how the energy is seeded.
__global__ void kdkKernel(float4* pos, float4* vel, float4* accel, int n,
                          float dt_k1, float dtime, float dt_k2,
                          GpuKdkReduction* partials){
  __shared__ GpuKdkReduction sh[REDUCE_BLOCK_SIZE];
  GpuKdkReduction r;
  redInit(r);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x){
    float4 p = pos[i];
    float4 v = vel[i];
    const float4 a = accel[i];
    const float m = p.w;

    r.preKinetic += m * (v.x * v.x + v.y * v.y + v.z * v.z);
    r.potential  += m * a.w;

    // kick
    v.x += dt_k1 * a.x;
    v.y += dt_k1 * a.y;
    v.z += dt_k1 * a.z;

    r.postKinetic += m * (v.x * v.x + v.y * v.y + v.z * v.z);

    // drift
    p.x += dtime * v.x;
    p.y += dtime * v.y;
    p.z += dtime * v.z;

    // kick
    v.x += dt_k2 * a.x;
    v.y += dt_k2 * a.y;
    v.z += dt_k2 * a.z;

    r.minx = fminf(r.minx, p.x); r.maxx = fmaxf(r.maxx, p.x);
    r.miny = fminf(r.miny, p.y); r.maxy = fmaxf(r.maxy, p.y);
    r.minz = fminf(r.minz, p.z); r.maxz = fmaxf(r.maxz, p.z);

    pos[i] = p;
    vel[i] = v;
    // The next iteration's traversals accumulate into this.
    accel[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  redBlock(sh, r);
  if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

// hashParticleCoordinates. Same arithmetic, same order, so the keys match the
// host's bit for bit.
__global__ void hashKeysKernel(const float4* __restrict__ pos,
                               unsigned long long* __restrict__ key, int n,
                               float lx, float ly, float lz,
                               float xsz, float ysz, float zsz,
                               int bitsPerDim){
  const float boxesPerDim = (float)(1u << bitsPerDim);
  const unsigned long long prepend = 1ULL << 63;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x){
    const float4 p = pos[i];
    const unsigned long long xint =
        (unsigned long long)(((p.x - lx) * boxesPerDim) / xsz);
    const unsigned long long yint =
        (unsigned long long)(((p.y - ly) * boxesPerDim) / ysz);
    const unsigned long long zint =
        (unsigned long long)(((p.z - lz) * boxesPerDim) / zsz);

    unsigned long long mask = 1ULL;
    unsigned long long k = 0ULL;
    int shiftBy = 0;
    for (int j = 0; j < bitsPerDim; j++){
      k |= ((zint & mask) <<  shiftBy);
      k |= ((yint & mask) << (shiftBy + 1));
      k |= ((xint & mask) << (shiftBy + 2));
      mask <<= 1;
      // minus one because mask itself has shifted left by one position
      shiftBy += 2;
    }
    key[i] = k | prepend;
  }
}

static inline int reduceGrid(int n){
  int b = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
  if (b < 1) b = 1;
  if (b > REDUCE_BLOCKS) b = REDUCE_BLOCKS;
  return b;
}

void invokeNanCheck(const float4* d_accel, int n, GpuKdkReduction* d_partials,
                    GpuKdkReduction* d_out, cudaStream_t stream){
  const int blocks = reduceGrid(n);
  nanCheckKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(d_accel, n, d_partials);
  reduceFinalKernel<<<1, REDUCE_BLOCK_SIZE, 0, stream>>>(d_partials, blocks, d_out);
}

void invokeKickDriftKick(float4* d_pos, float4* d_vel, float4* d_accel, int n,
                         float dt_k1, float dtime, float dt_k2,
                         GpuKdkReduction* d_partials, GpuKdkReduction* d_out,
                         cudaStream_t stream){
  const int blocks = reduceGrid(n);
  kdkKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
      d_pos, d_vel, d_accel, n, dt_k1, dtime, dt_k2, d_partials);
  reduceFinalKernel<<<1, REDUCE_BLOCK_SIZE, 0, stream>>>(d_partials, blocks, d_out);
}

void invokeHashKeys(const float4* d_pos, unsigned long long* d_key, int n,
                    float lx, float ly, float lz,
                    float xsz, float ysz, float zsz,
                    int bitsPerDim, cudaStream_t stream){
  if (n <= 0) return;
  const int blocks = reduceGrid(n);
  hashKeysKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
      d_pos, d_key, n, lx, ly, lz, xsz, ysz, zsz, bitsPerDim);
}

// ---------------------------------------------------------------------------
// Stage 3: the sort. Replaces the quickSort in DataManager::decompose.
//
// Only that one. The other quickSort, in processSubmittedParticles, runs on
// particles that have just arrived from other PEs into *host* memory, and the
// host tree build reads them straight afterwards; sorting those here would be
// an H2D and a D2H to save an O(N log N) host pass, which is a loss until
// Stage 6 builds the tree on the device too.
// ---------------------------------------------------------------------------

#include <cub/cub.cuh>

__global__ void iotaKernel(int* __restrict__ idx, int n){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x){
    idx[i] = i;
  }
}

__global__ void gatherKernel(const int* __restrict__ idx,
                             const float4* __restrict__ posIn,
                             float4* __restrict__ posOut,
                             const float4* __restrict__ velIn,
                             float4* __restrict__ velOut, int n){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x){
    const int src = idx[i];
    posOut[i] = posIn[src];
    velOut[i] = velIn[src];
  }
}

size_t gpuSortTempBytes(int n){
  if (n <= 0) return 0;
  size_t bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      (void*)NULL, bytes,
      (const unsigned long long*)NULL, (unsigned long long*)NULL,
      (const int*)NULL, (int*)NULL, n);
  return bytes;
}

void invokeSortByKey(const unsigned long long* d_keyIn,
                     unsigned long long* d_keyOut,
                     const float4* d_posIn, float4* d_posOut,
                     const float4* d_velIn, float4* d_velOut,
                     int* d_idx, int* d_idxOut,
                     void* d_temp, size_t tempBytes,
                     int n, cudaStream_t stream){
  if (n <= 0) return;
  const int blocks = reduceGrid(n);

  iotaKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(d_idx, n);

  // The full key width. Bit 63 is set on every key -- hashKeysKernel ors in
  // the prepend -- so a 63-bit sort would order these identically and save a
  // pass, but that is an invariant of another kernel and not worth a silent
  // failure if it ever moves.
  size_t bytes = tempBytes;
  cub::DeviceRadixSort::SortPairs(d_temp, bytes, d_keyIn, d_keyOut,
                                  d_idx, d_idxOut, n, 0, 64, stream);

  gatherKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
      d_idxOut, d_posIn, d_posOut, d_velIn, d_velOut, n);
}

// ---------------------------------------------------------------------------
// Stage 6: the flat device tree.
// ---------------------------------------------------------------------------

// lower_bound over the sorted key array, matching binary_search_ge in util.h.
__device__ __forceinline__ int lowerBound(const unsigned long long* __restrict__ k,
                                          int lo, int hi, unsigned long long want){
  while (lo < hi){
    const int mid = lo + ((hi - lo) >> 1);
    if (k[mid] >= want) hi = mid; else lo = mid + 1;
  }
  return lo;
}

// The key of a node's second child, shifted where findSplitters tests it:
// childKey = (key<<1)|1, at bit (64 - depth - 2).
__device__ __forceinline__ unsigned long long splitKey(unsigned long long key, int depth){
  return ((key << 1) | 1ULL) << (64 - depth - 2);
}

__global__ void treeInitKernel(DeviceTreeScratch sc, int n, int numTreePieces){
  DeviceNode &r = sc.nodes[0];
  r.key = 1ULL;
  r.depth = 0;
  r.partStart = 0;
  r.partCount = n;
  r.firstChild = -1;
  r.type = DNODE_INTERNAL;
  r.ownerStart = 0;
  r.ownerEnd = numTreePieces - 1;
  *sc.nodeCount = 1;
  sc.active[0] = 0;
  *sc.activeCount = 1;
  *sc.activeNextCount = 0;
  // levelStart[L] is the index of the first node of level L. The root is
  // level 0 and occupies [0,1); the nodes a level's kernel creates belong to
  // the level below it, which is why the mark kernel writes level+2.
  for (int i = 0; i <= DTREE_MAX_LEVELS + 1; i++) sc.levelStart[i] = 1;
  sc.levelStart[0] = 0;
  sc.levelStart[1] = 1;
}

// One level. Every active node splits in two; each child becomes a leaf or
// joins the next level. The refine predicate is buildTree's: keep refining
// while the node straddles more than one tree piece, or holds more particles
// than a bucket should.
__global__ void treeLevelKernel(const unsigned long long* __restrict__ key,
                                const unsigned long long* __restrict__ owners,
                                int ppbLimit, DeviceTreeScratch sc){
  const int nActive = *sc.activeCount;
  for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < nActive;
       a += gridDim.x * blockDim.x){
    const int ni = sc.active[a];
    const DeviceNode nd = sc.nodes[ni];

    const int lo = nd.partStart;
    const int hi = nd.partStart + nd.partCount;
    const unsigned long long sk = splitKey(nd.key, nd.depth);
    const int split = lowerBound(key, lo, hi, sk);

    const int base = atomicAdd(sc.nodeCount, 2);
    sc.nodes[ni].firstChild = base;
    sc.nodes[ni].type = DNODE_INTERNAL;

    // Ownership as OwnershipActiveBinInfo::refine computes it. Each tree piece
    // contributes a low and a high key to `owners`, hence the doubling.
    const int oIdx  = lowerBound(owners, 2*nd.ownerStart, 2*nd.ownerEnd + 2, sk);
    const int tpIdx = oIdx >> 1;

    for (int c = 0; c < 2; c++){
      DeviceNode &ch = sc.nodes[base + c];
      ch.key = (nd.key << 1) | (unsigned long long)c;
      ch.depth = nd.depth + 1;
      ch.partStart = (c == 0) ? lo : split;
      ch.partCount = (c == 0) ? (split - lo) : (hi - split);
      ch.firstChild = -1;

      if (c == 0){
        ch.ownerStart = nd.ownerStart;
        ch.ownerEnd   = ((oIdx & 1) == 0) ? (tpIdx - 1) : tpIdx;
      }
      else{
        ch.ownerStart = tpIdx;
        ch.ownerEnd   = nd.ownerEnd;
      }

      if (ch.ownerEnd < ch.ownerStart){
        // Nothing owns this range; the host marks it EmptyBucket and stops.
        ch.ownerStart = ch.ownerEnd = -1;
        ch.type = DNODE_EMPTYBUCKET;
        continue;
      }

      if ((ch.ownerEnd > ch.ownerStart) || (ch.partCount > ppbLimit)){
        ch.type = DNODE_INTERNAL;
        sc.activeNext[atomicAdd(sc.activeNextCount, 1)] = base + c;
      }
      else{
        ch.type = (ch.partCount > 0) ? DNODE_BUCKET : DNODE_EMPTYBUCKET;
      }
    }
  }
}

// Record where the level that just finished ends. The active lists are swapped
// on the host -- the scratch struct is passed by value, so a swap made here
// would be thrown away with the kernel's copy of it.
__global__ void treeLevelMarkKernel(DeviceTreeScratch sc, int level){
  // This level's kernel created the nodes of level+1, so the first node of
  // level+2 is wherever the count now stands. Writing level+1 here would put
  // a node in the same range as its own children, and the bottom-up moment
  // pass would then have a parent reading children being written by the same
  // launch.
  sc.levelStart[level + 2] = *sc.nodeCount;
  // Clear the list just consumed. The host then swaps, so this counter becomes
  // the one the next level appends into and it has to start at zero.
  *sc.activeCount = 0;
}

// CUB 13 dropped cub::Min / cub::Max; these are the two reductions the moment
// pass needs and they are cheaper to spell out than to chase the replacement.
struct FMinOp { __device__ float operator()(const float &a, const float &b) const { return fminf(a,b); } };
struct FMaxOp { __device__ float operator()(const float &a, const float &b) const { return fmaxf(a,b); } };

// Node::getMomentsFromParticles, one block per leaf.
__global__ void leafMomentsKernel(const float4* __restrict__ posm,
                                  DeviceTreeScratch sc){
  const int n = *sc.nodeCount;
  typedef cub::BlockReduce<float, GRAV_BLOCK_SIZE> BR;
  __shared__ typename BR::TempStorage ts;
  __shared__ float sm, scx, scy, scz, sb[6], sr, sq[5];

  for (int ni = blockIdx.x; ni < n; ni += gridDim.x){
    DeviceNode &d = sc.nodes[ni];
    // Block-uniform, so every thread takes the same branch and the barriers
    // below stay collective.
    if (d.firstChild >= 0) continue;

    if (threadIdx.x == 0){
      d.cmMass = make_float4(0.f, 0.f, 0.f, 0.f);
      d.rsq = 0.f;
      d.qxx = d.qxy = d.qxz = d.qyy = d.qyz = 0.f;
      d.boxMin = make_float3( INFINITY,  INFINITY,  INFINITY);
      d.boxMax = make_float3(-INFINITY, -INFINITY, -INFINITY);
    }
    __syncthreads();
    if (d.partCount <= 0) continue;

    const int p0 = d.partStart, p1 = d.partStart + d.partCount;
    float m = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;
    float bx0 =  INFINITY, by0 =  INFINITY, bz0 =  INFINITY;
    float bx1 = -INFINITY, by1 = -INFINITY, bz1 = -INFINITY;
    for (int i = p0 + threadIdx.x; i < p1; i += blockDim.x){
      const float4 p = posm[i];
      m += p.w; cx += p.w*p.x; cy += p.w*p.y; cz += p.w*p.z;
      bx0 = fminf(bx0, p.x); bx1 = fmaxf(bx1, p.x);
      by0 = fminf(by0, p.y); by1 = fmaxf(by1, p.y);
      bz0 = fminf(bz0, p.z); bz1 = fmaxf(bz1, p.z);
    }
    float t;
    t = BR(ts).Sum(m);  if (threadIdx.x == 0) sm  = t; __syncthreads();
    t = BR(ts).Sum(cx); if (threadIdx.x == 0) scx = t; __syncthreads();
    t = BR(ts).Sum(cy); if (threadIdx.x == 0) scy = t; __syncthreads();
    t = BR(ts).Sum(cz); if (threadIdx.x == 0) scz = t; __syncthreads();
    t = BR(ts).Reduce(bx0, FMinOp()); if (threadIdx.x==0) sb[0]=t; __syncthreads();
    t = BR(ts).Reduce(by0, FMinOp()); if (threadIdx.x==0) sb[1]=t; __syncthreads();
    t = BR(ts).Reduce(bz0, FMinOp()); if (threadIdx.x==0) sb[2]=t; __syncthreads();
    t = BR(ts).Reduce(bx1, FMaxOp()); if (threadIdx.x==0) sb[3]=t; __syncthreads();
    t = BR(ts).Reduce(by1, FMaxOp()); if (threadIdx.x==0) sb[4]=t; __syncthreads();
    t = BR(ts).Reduce(bz1, FMaxOp()); if (threadIdx.x==0) sb[5]=t; __syncthreads();

    const float mass = sm;
    const float cmx = (mass > 0.f) ? scx/mass : 0.f;
    const float cmy = (mass > 0.f) ? scy/mass : 0.f;
    const float cmz = (mass > 0.f) ? scz/mass : 0.f;

    float r2 = 0.f, qxx = 0.f, qxy = 0.f, qxz = 0.f, qyy = 0.f, qyz = 0.f;
    for (int i = p0 + threadIdx.x; i < p1; i += blockDim.x){
      const float4 p = posm[i];
      const float sx = p.x - cmx, sy = p.y - cmy, sz = p.z - cmz;
      const float d2 = sx*sx + sy*sy + sz*sz;
      r2 = fmaxf(r2, d2);
      qxx += p.w*(3.f*sx*sx - d2);
      qxy += p.w*(3.f*sx*sy);
      qxz += p.w*(3.f*sx*sz);
      qyy += p.w*(3.f*sy*sy - d2);
      qyz += p.w*(3.f*sy*sz);
    }
    t = BR(ts).Reduce(r2, FMaxOp()); if (threadIdx.x==0) sr = t; __syncthreads();
    t = BR(ts).Sum(qxx); if (threadIdx.x==0) sq[0]=t; __syncthreads();
    t = BR(ts).Sum(qxy); if (threadIdx.x==0) sq[1]=t; __syncthreads();
    t = BR(ts).Sum(qxz); if (threadIdx.x==0) sq[2]=t; __syncthreads();
    t = BR(ts).Sum(qyy); if (threadIdx.x==0) sq[3]=t; __syncthreads();
    t = BR(ts).Sum(qyz); if (threadIdx.x==0) sq[4]=t; __syncthreads();

    if (threadIdx.x == 0){
      d.cmMass = make_float4(cmx, cmy, cmz, mass);
      d.rsq = sr;
      d.qxx = sq[0]; d.qxy = sq[1]; d.qxz = sq[2]; d.qyy = sq[3]; d.qyz = sq[4];
      d.boxMin = make_float3(sb[0], sb[1], sb[2]);
      d.boxMax = make_float3(sb[3], sb[4], sb[5]);
    }
    __syncthreads();
  }
}

// Node::getMomentsFromChildren for one level's nodes. The quadrupole shift is
// the parallel-axis one: a child brings its own reduced quadrupole plus its
// monopole taken about the offset between the two centres of mass.
__global__ void internalMomentsKernel(DeviceTreeScratch sc, int level){
  const int lo = sc.levelStart[level];
  const int hi = sc.levelStart[level + 1];
  for (int ni = lo + blockIdx.x*blockDim.x + threadIdx.x; ni < hi;
       ni += gridDim.x*blockDim.x){
    DeviceNode &d = sc.nodes[ni];
    if (d.firstChild < 0) continue;

    float m = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;
    float3 b0 = make_float3( INFINITY,  INFINITY,  INFINITY);
    float3 b1 = make_float3(-INFINITY, -INFINITY, -INFINITY);
    for (int c = 0; c < 2; c++){
      const DeviceNode &ch = sc.nodes[d.firstChild + c];
      m  += ch.cmMass.w;
      cx += ch.cmMass.w*ch.cmMass.x;
      cy += ch.cmMass.w*ch.cmMass.y;
      cz += ch.cmMass.w*ch.cmMass.z;
      if (ch.partCount > 0){
        b0.x = fminf(b0.x, ch.boxMin.x); b1.x = fmaxf(b1.x, ch.boxMax.x);
        b0.y = fminf(b0.y, ch.boxMin.y); b1.y = fmaxf(b1.y, ch.boxMax.y);
        b0.z = fminf(b0.z, ch.boxMin.z); b1.z = fmaxf(b1.z, ch.boxMax.z);
      }
    }
    const float cmx = (m > 0.f) ? cx/m : 0.f;
    const float cmy = (m > 0.f) ? cy/m : 0.f;
    const float cmz = (m > 0.f) ? cz/m : 0.f;

    float qxx = 0.f, qxy = 0.f, qxz = 0.f, qyy = 0.f, qyz = 0.f;
    for (int c = 0; c < 2; c++){
      const DeviceNode &ch = sc.nodes[d.firstChild + c];
      const float tx = ch.cmMass.x - cmx;
      const float ty = ch.cmMass.y - cmy;
      const float tz = ch.cmMass.z - cmz;
      const float t2 = tx*tx + ty*ty + tz*tz;
      const float mc = ch.cmMass.w;
      qxx += ch.qxx + mc*(3.f*tx*tx - t2);
      qxy += ch.qxy + mc*(3.f*tx*ty);
      qxz += ch.qxz + mc*(3.f*tx*tz);
      qyy += ch.qyy + mc*(3.f*ty*ty - t2);
      qyz += ch.qyz + mc*(3.f*ty*tz);
    }

    // getMomentsFromChildren takes the opening radius from the box corner
    // furthest from the centre of mass, not from the children's radii.
    const float dx = fmaxf(cmx - b0.x, b1.x - cmx);
    const float dy = fmaxf(cmy - b0.y, b1.y - cmy);
    const float dz = fmaxf(cmz - b0.z, b1.z - cmz);

    d.cmMass = make_float4(cmx, cmy, cmz, m);
    d.rsq = dx*dx + dy*dy + dz*dz;
    d.qxx = qxx; d.qxy = qxy; d.qxz = qxz; d.qyy = qyy; d.qyz = qyz;
    d.boxMin = b0; d.boxMax = b1;
  }
}

void invokeBuildTree(const unsigned long long* d_key, int n,
                     const unsigned long long* d_owners, int numTreePieces,
                     int ppbLimit, DeviceTreeScratch sc,
                     cudaStream_t stream){
  treeInitKernel<<<1, 1, 0, stream>>>(sc, n, numTreePieces);
  for (int level = 0; level < DTREE_MAX_LEVELS; level++){
    // A fixed grid strides over whatever is active, so a level with nothing
    // left is an empty launch rather than a readback.
    treeLevelKernel<<<REDUCE_BLOCKS, REDUCE_BLOCK_SIZE, 0, stream>>>(
        d_key, d_owners, ppbLimit, sc);
    treeLevelMarkKernel<<<1, 1, 0, stream>>>(sc, level);
    // The scratch goes to the kernels by value, so the flip has to happen here.
    int *t;
    t = sc.active;      sc.active = sc.activeNext;           sc.activeNext = t;
    t = sc.activeCount; sc.activeCount = sc.activeNextCount; sc.activeNextCount = t;
  }
}

void invokeTreeMoments(const float4* d_posm, DeviceTreeScratch sc,
                       cudaStream_t stream){
  leafMomentsKernel<<<REDUCE_BLOCKS, GRAV_BLOCK_SIZE, 0, stream>>>(d_posm, sc);
  for (int level = DTREE_MAX_LEVELS; level >= 0; level--){
    internalMomentsKernel<<<REDUCE_BLOCKS, REDUCE_BLOCK_SIZE, 0, stream>>>(sc, level);
  }
}

// ---------------------------------------------------------------------------
// Stage 7: the device traversal (local half).
// ---------------------------------------------------------------------------

// Deep enough for a binary tree over any realistic particle count: the walk
// pushes at most one sibling per level.
#define WALK_STACK 96

// openCriterionBucket in gravity.h, in its default (bounding box) form: the
// minimum distance from the cell's centre of mass to the target bucket's box.
__device__ __forceinline__ bool openCell(const DeviceNode &nd,
                                         const float3 &bmin, const float3 &bmax,
                                         float tolsq){
  float dx = fmaxf(fmaxf(bmin.x - nd.cmMass.x, nd.cmMass.x - bmax.x), 0.f);
  float dy = fmaxf(fmaxf(bmin.y - nd.cmMass.y, nd.cmMass.y - bmax.y), 0.f);
  float dz = fmaxf(fmaxf(bmin.z - nd.cmMass.z, nd.cmMass.z - bmax.z), 0.f);
  return (tolsq * (dx*dx + dy*dy + dz*dz) < nd.rsq);
}

// One interaction, cell or particle, in the form gravQuad uses. A particle
// source passes zero quadrupole components and the terms vanish.
__device__ __forceinline__ void accumulate(float sx, float sy, float sz, float sm,
                                           float qxx, float qxy, float qxz,
                                           float qyy, float qyz,
                                           float px, float py, float pz,
                                           float epssq,
                                           float &ax, float &ay, float &az,
                                           float &phi){
  const float dx = sx - px, dy = sy - py, dz = sz - pz;
  const float drsq = dx*dx + dy*dy + dz*dz + epssq;
#ifdef GRAV_EXACT_RSQRT
  const float rinv = 1.f / sqrtf(drsq);
#else
  const float rinv = rsqrtf(drsq);
#endif
  const float rinv2 = rinv * rinv;
  const float mrinv = sm * rinv;
#ifdef MONOPOLE_ONLY
  phi -= mrinv;
  const float coef = mrinv * rinv2;
  ax += coef*dx; ay += coef*dy; az += coef*dz;
#else
  const float qzz = -(qxx + qyy);
  const float qx = qxx*dx + qxy*dy + qxz*dz;
  const float qy = qxy*dx + qyy*dy + qyz*dz;
  const float qz = qxz*dx + qyz*dy + qzz*dz;
  const float drQdr = dx*qx + dy*qy + dz*qz;
  const float rinv5 = rinv2 * rinv2 * rinv;
  const float rinv7 = rinv5 * rinv2;
  phi -= (mrinv + 0.5f * drQdr * rinv5);
  const float coef = mrinv * rinv2 + 2.5f * drQdr * rinv7;
  ax += coef*dx - rinv5*qx;
  ay += coef*dy - rinv5*qy;
  az += coef*dz - rinv5*qz;
#endif
}

__global__ void localWalkKernel(const DeviceNode* __restrict__ nodes,
                                const float4* __restrict__ posm,
                                const GpuTargetBucket* __restrict__ buckets,
                                int numBuckets, float4* __restrict__ accel,
                                float epssq, float tolsq){
  __shared__ int stack[WALK_STACK];
  __shared__ int sp;
  __shared__ int cur;
  __shared__ float4 srcTile[GRAV_BLOCK_SIZE];

  for (int b = blockIdx.x; b < numBuckets; b += gridDim.x){
    const GpuTargetBucket tb = buckets[b];

    // One target particle per thread, striding if the bucket is larger than
    // the block.
    for (int base = 0; base < tb.partCount; base += blockDim.x){
      const int i = base + threadIdx.x;
      const bool active = (i < tb.partCount);
      float px = 0.f, py = 0.f, pz = 0.f;
      if (active){
        const float4 p = posm[tb.partStart + i];
        px = p.x; py = p.y; pz = p.z;
      }
      float ax = 0.f, ay = 0.f, az = 0.f, phi = 0.f;

      // work(root): keep and, if it is not opened, take its multipole and
      // stop. Everything below is block-uniform, so the barriers stay
      // collective even though threads branch on `active`.
      if (threadIdx.x == 0) sp = 0;
      __syncthreads();
      {
        const DeviceNode rt = nodes[0];
        const bool keep = (rt.type == DNODE_INTERNAL || rt.type == DNODE_BUCKET ||
                           rt.type == DNODE_BOUNDARY);
        if (keep){
          if (openCell(rt, tb.boxMin, tb.boxMax, tolsq)){
            if (threadIdx.x == 0) stack[sp++] = 0;
          }
          else if (active){
            accumulate(rt.cmMass.x, rt.cmMass.y, rt.cmMass.z, rt.cmMass.w,
                       rt.qxx, rt.qxy, rt.qxz, rt.qyy, rt.qyz,
                       px, py, pz, epssq, ax, ay, az, phi);
          }
        }
      }
      __syncthreads();

      while (sp > 0){
        if (threadIdx.x == 0) cur = stack[--sp];
        __syncthreads();
        const DeviceNode nd = nodes[cur];

        if (nd.firstChild < 0){
          // processLeaf: an opened bucket is summed particle by particle.
          // Particle-particle dominates -- 519M against 8M node interactions
          // in a 500K run -- so the sources are staged through shared memory
          // in tiles, the way the gravity kernel does it. Every thread then
          // reads each source once from shared instead of once from global.
          if (nd.type == DNODE_BUCKET){
            const int s0 = nd.partStart, n = nd.partCount;
            for (int t = 0; t < n; t += GRAV_BLOCK_SIZE){
              // Uniform bounds, so the barriers stay collective.
              __syncthreads();
              const int g = t + threadIdx.x;
              if (g < n) srcTile[threadIdx.x] = posm[s0 + g];
              __syncthreads();
              const int m = min(GRAV_BLOCK_SIZE, n - t);
              if (active){
                for (int k = 0; k < m; k++){
                  const float4 sp4 = srcTile[k];
                  accumulate(sp4.x, sp4.y, sp4.z, sp4.w,
                             0.f, 0.f, 0.f, 0.f, 0.f,
                             px, py, pz, epssq, ax, ay, az, phi);
                }
              }
            }
          }
          __syncthreads();
          continue;
        }

        for (int c = 0; c < 2; c++){
          const DeviceNode ch = nodes[nd.firstChild + c];
          const bool keep = (ch.type == DNODE_INTERNAL || ch.type == DNODE_BUCKET ||
                             ch.type == DNODE_BOUNDARY);
          if (!keep) continue;
          if (openCell(ch, tb.boxMin, tb.boxMax, tolsq)){
            if (threadIdx.x == 0) stack[sp++] = nd.firstChild + c;
          }
          else if (active){
            accumulate(ch.cmMass.x, ch.cmMass.y, ch.cmMass.z, ch.cmMass.w,
                       ch.qxx, ch.qxy, ch.qxz, ch.qyy, ch.qyz,
                       px, py, pz, epssq, ax, ay, az, phi);
          }
        }
        __syncthreads();
      }

      if (active){
        float4 *a = accel + (tb.partStart + i);
        atomicAdd(&a->x, ax);
        atomicAdd(&a->y, ay);
        atomicAdd(&a->z, az);
        atomicAdd(&a->w, phi);
      }
      __syncthreads();
    }
  }
}

void invokeLocalWalk(const DeviceNode* d_nodes, const float4* d_posm,
                     const GpuTargetBucket* d_buckets, int numBuckets,
                     float4* d_accel, float epssq, float tolsq,
                     cudaStream_t stream){
  if (numBuckets <= 0) return;
  const int blocks = (numBuckets < 1024) ? numBuckets : 1024;
  localWalkKernel<<<blocks, GRAV_BLOCK_SIZE, 0, stream>>>(
      d_nodes, d_posm, d_buckets, numBuckets, d_accel, epssq, tolsq);
}

// Descend from the root following the key's bits. Bit 63 is the leading one
// every key carries; the bits below it, most significant first, are the child
// choices, and the node's own depth says how many of them to consume.
__device__ __forceinline__ int findByKey(const DeviceNode* nodes,
                                         unsigned long long key){
  // Depth is the position of the leading set bit below bit 63.
  int depth = 0;
  unsigned long long k = key;
  while (k > 1ULL){ k >>= 1; depth++; }

  int cur = 0;
  for (int level = depth - 1; level >= 0; level--){
    const int child = (int)((key >> level) & 1ULL);
    const int fc = nodes[cur].firstChild;
    if (fc < 0) return -1;          // host tree is deeper here than the device
    cur = fc + child;
  }
  return cur;
}

__global__ void scatterMomentsKernel(DeviceNode* __restrict__ nodes,
                                     const GpuMomentPatch* __restrict__ patch,
                                     int n){
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
       i += gridDim.x*blockDim.x){
    const GpuMomentPatch p = patch[i];
    const int ni = findByKey(nodes, p.key);
    if (ni < 0) continue;
    DeviceNode &d = nodes[ni];
    d.cmMass = p.cmMass;
    d.rsq = p.rsq;
    d.qxx = p.qxx; d.qxy = p.qxy; d.qxz = p.qxz;
    d.qyy = p.qyy; d.qyz = p.qyz;
  }
}

void invokeScatterMoments(DeviceNode* d_nodes, const GpuMomentPatch* d_patch,
                          int n, cudaStream_t stream){
  if (n <= 0) return;
  const int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
  scatterMomentsKernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(d_nodes, d_patch, n);
}
