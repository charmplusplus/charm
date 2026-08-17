#include "hapi.h"
#include "pic2d.h"

#define TILE_SIZE 16
#define BLOCK_1D 256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Counter-based RNG: particle initial state is a pure function of the global
// particle index, so every patch can generate the full distribution and keep
// only the particles that fall inside its bounds, with no RNG state to store.
__device__ inline unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Uniform in (0,1] (never 0, safe for logf)
__device__ inline float u01(unsigned int r) {
  return ((float)(r & 0xFFFFFFu) + 1.0f) / 16777217.0f;
}

__global__ void initParticlesKernel(Particle* parts, int* np, long n_total,
    long n_bunch, int dist_type, float bunch_cx, float bunch_cy, float sigma,
    float drift, float thermal, float Lx, float Ly, float px0, float py0,
    float px1, float py1, int capacity, int* err) {
  long stride = (long)blockDim.x * gridDim.x;
  for (long g = (long)blockDim.x * blockIdx.x + threadIdx.x; g < n_total;
       g += stride) {
    unsigned long long r1 = splitmix64((unsigned long long)g);
    unsigned long long r2 = splitmix64(r1 ^ 0xA5A5A5A55A5A5A5AULL);
    float u1 = u01((unsigned int)r1);
    float u2 = u01((unsigned int)(r1 >> 32));
    float u3 = u01((unsigned int)r2);
    float u4 = u01((unsigned int)(r2 >> 32));

    Particle p;
    // Maxwellian thermal velocity (Box-Muller)
    float vr = thermal * sqrtf(-2.0f * logf(u3));
    float va = 2.0f * (float)M_PI * u4;
    p.vx = vr * cosf(va);
    p.vy = vr * sinf(va);

    if (dist_type == 1 && g < n_bunch) {
      // Gaussian bunch around (bunch_cx, bunch_cy) with drift in +x
      float rr = sigma * sqrtf(-2.0f * logf(u1));
      float ra = 2.0f * (float)M_PI * u2;
      p.x = bunch_cx + rr * cosf(ra);
      p.y = bunch_cy + rr * sinf(ra);
      p.vx += drift;
    } else {
      p.x = u1 * Lx;
      p.y = u2 * Ly;
      // Two-stream: uniform positions, counter-streaming beams in +-x
      if (dist_type == 2) p.vx += (g & 1) ? drift : -drift;
    }
    // Periodic wrap (Gaussian tails may extend past the domain). The wrap of
    // a slightly negative coordinate can round to exactly Lx, which would
    // belong to no patch; fold it back to 0. Note: patch ownership relies on
    // every device computing bit-identical positions for the same index,
    // which holds for homogeneous GPUs running the same binary.
    p.x = fmodf(p.x, Lx); if (p.x < 0.f) p.x += Lx;
    p.y = fmodf(p.y, Ly); if (p.y < 0.f) p.y += Ly;
    if (p.x >= Lx) p.x = 0.f;
    if (p.y >= Ly) p.y = 0.f;

    if (p.x >= px0 && p.x < px1 && p.y >= py0 && p.y < py1) {
      int k = atomicAdd(np, 1);
      if (k < capacity) parts[k] = p; else *err = 1;
    }
  }
}

// CIC (bilinear) charge deposition. Interior cell (i,j), i=1..W, j=1..H, is
// centered at global (px0 + i - 0.5, py0 + j - 0.5). Edge particles deposit
// into the ghost ring (depth 1 is guaranteed for particles inside the patch).
__global__ void depositKernel(const Particle* parts, int n, RealType* rho,
    float weight, float px0, float py0, int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n) return;
  Particle p = parts[t];
  float u = p.x - px0 + 0.5f;
  float v = p.y - py0 + 0.5f;
  int i0 = (int)floorf(u);
  int j0 = (int)floorf(v);
  float wx = u - i0;
  float wy = v - j0;
  atomicAdd(&rho[IDX(i0,   j0  )], weight * (1.f - wx) * (1.f - wy));
  atomicAdd(&rho[IDX(i0+1, j0  )], weight * wx * (1.f - wy));
  atomicAdd(&rho[IDX(i0,   j0+1)], weight * (1.f - wx) * wy);
  atomicAdd(&rho[IDX(i0+1, j0+1)], weight * wx * wy);
}

// Slab layout shared by the 8-way charge and E exchanges:
// [0,H) left, [H,2H) right, [2H,2H+W) top, [2H+W,2H+2W) bottom,
// then the TL, TR, BL, BR corners (1 element each).
__global__ void packChargeGhostsKernel(const RealType* rho, RealType* slab,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  if (t < H)                slab[t] = rho[IDX(0, 1 + t)];
  else if (t < 2*H)         slab[t] = rho[IDX(W+1, 1 + (t - H))];
  else if (t < 2*H + W)     slab[t] = rho[IDX(1 + (t - 2*H), 0)];
  else if (t < 2*H + 2*W)   slab[t] = rho[IDX(1 + (t - 2*H - W), H+1)];
  else if (t == 2*H + 2*W)     slab[t] = rho[IDX(0,   0)];
  else if (t == 2*H + 2*W + 1) slab[t] = rho[IDX(W+1, 0)];
  else if (t == 2*H + 2*W + 2) slab[t] = rho[IDX(0,   H+1)];
  else if (t == 2*H + 2*W + 3) slab[t] = rho[IDX(W+1, H+1)];
}

// Adds a ghost-charge strip received from direction `dir` into my interior
// boundary cells (the sender's ghost ring overlays them). Launches for
// different directions overlap at the corners but are serialized on the
// comm stream, so no atomics are needed.
__global__ void accumChargeGhostKernel(RealType* rho, const RealType* buf,
    int dir, int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  switch (dir) {
    case LEFT:   if (t < H) rho[IDX(1, 1 + t)] += buf[t]; break;
    case RIGHT:  if (t < H) rho[IDX(W, 1 + t)] += buf[t]; break;
    case TOP:    if (t < W) rho[IDX(1 + t, 1)] += buf[t]; break;
    case BOTTOM: if (t < W) rho[IDX(1 + t, H)] += buf[t]; break;
    case TL: if (t == 0) rho[IDX(1, 1)] += buf[0]; break;
    case TR: if (t == 0) rho[IDX(W, 1)] += buf[0]; break;
    case BL: if (t == 0) rho[IDX(1, H)] += buf[0]; break;
    case BR: if (t == 0) rho[IDX(W, H)] += buf[0]; break;
  }
}

// Phi send slab layout (4-way): [0,H) left interior column, [H,2H) right
// interior column, [2H,2H+W) top interior row, [2H+W,2H+2W) bottom interior
// row. The contiguous rows are copied with memcpy on the host side; this
// kernel only packs the strided columns.
__global__ void packPhiLRKernel(const RealType* phi, RealType* slab,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  if (t < H)        slab[t] = phi[IDX(1, 1 + t)];
  else if (t < 2*H) slab[t] = phi[IDX(W, 1 + (t - H))];
}

// Writes a received phi strip into the ghost column on side `dir`
__global__ void unpackPhiLRKernel(RealType* phi, const RealType* buf, int dir,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  if (t >= H) return;
  if (dir == LEFT) phi[IDX(0, 1 + t)] = buf[t];
  else             phi[IDX(W+1, 1 + t)] = buf[t];
}

// One weighted-Jacobi step of grad^2(phi) = -(rho - rho_bar) with dx = dy = 1
__global__ void jacobiPhiKernel(const RealType* phi, RealType* phi_new,
    const RealType* rho, float rho_bar, int block_width, int block_height) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;
  if (i <= block_width && j <= block_height) {
    phi_new[IDX(i,j)] = 0.25f * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)] +
        phi[IDX(i,j-1)] + phi[IDX(i,j+1)] + (rho[IDX(i,j)] - rho_bar));
  }
}

// E = -grad(phi), central differences, interior cells only; the ghost ring
// is filled by the E exchange
__global__ void efieldKernel(const RealType* phi, float2* ef, int block_width,
    int block_height) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;
  if (i <= block_width && j <= block_height) {
    float2 e;
    e.x = -0.5f * (phi[IDX(i+1,j)] - phi[IDX(i-1,j)]);
    e.y = -0.5f * (phi[IDX(i,j+1)] - phi[IDX(i,j-1)]);
    ef[IDX(i,j)] = e;
  }
}

// Packs my interior boundary ring (it overlays the neighbors' ghost rings)
// into 8 float2 strips, same slab layout as the charge exchange
__global__ void packEGhostsKernel(const float2* ef, float2* slab,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  if (t < H)                slab[t] = ef[IDX(1, 1 + t)];
  else if (t < 2*H)         slab[t] = ef[IDX(W, 1 + (t - H))];
  else if (t < 2*H + W)     slab[t] = ef[IDX(1 + (t - 2*H), 1)];
  else if (t < 2*H + 2*W)   slab[t] = ef[IDX(1 + (t - 2*H - W), H)];
  else if (t == 2*H + 2*W)     slab[t] = ef[IDX(1, 1)];
  else if (t == 2*H + 2*W + 1) slab[t] = ef[IDX(W, 1)];
  else if (t == 2*H + 2*W + 2) slab[t] = ef[IDX(1, H)];
  else if (t == 2*H + 2*W + 3) slab[t] = ef[IDX(W, H)];
}

// Writes a received E strip into the ghost ring on side `dir`
__global__ void unpackEGhostKernel(float2* ef, const float2* buf, int dir,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int W = block_width, H = block_height;
  switch (dir) {
    case LEFT:   if (t < H) ef[IDX(0, 1 + t)] = buf[t]; break;
    case RIGHT:  if (t < H) ef[IDX(W+1, 1 + t)] = buf[t]; break;
    case TOP:    if (t < W) ef[IDX(1 + t, 0)] = buf[t]; break;
    case BOTTOM: if (t < W) ef[IDX(1 + t, H+1)] = buf[t]; break;
    case TL: if (t == 0) ef[IDX(0,   0)]   = buf[0]; break;
    case TR: if (t == 0) ef[IDX(W+1, 0)]   = buf[0]; break;
    case BL: if (t == 0) ef[IDX(0,   H+1)] = buf[0]; break;
    case BR: if (t == 0) ef[IDX(W+1, H+1)] = buf[0]; break;
  }
}

// Gather E at the particle (same CIC stencil as deposit), leapfrog push,
// then classify: stayers are compacted into `stay`, leavers into the
// per-direction send slab. Classification happens against the unwrapped
// position so patches at the domain edge route correctly; the position is
// wrapped afterwards.
__global__ void pushAndMarkKernel(const Particle* parts, int n,
    const float2* ef, Particle* stay, Particle* sendslab, int exch_cap,
    int* counts, float dt, float qm, float px0, float py0, float Lx, float Ly,
    int block_width, int block_height) {
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n) return;
  int W = block_width, H = block_height;
  Particle p = parts[t];

  float u = p.x - px0 + 0.5f;
  float v = p.y - py0 + 0.5f;
  int i0 = (int)floorf(u);
  int j0 = (int)floorf(v);
  float wx = u - i0;
  float wy = v - j0;
  float2 e00 = ef[IDX(i0,   j0  )];
  float2 e10 = ef[IDX(i0+1, j0  )];
  float2 e01 = ef[IDX(i0,   j0+1)];
  float2 e11 = ef[IDX(i0+1, j0+1)];
  float ex = (1.f-wx)*(1.f-wy)*e00.x + wx*(1.f-wy)*e10.x +
             (1.f-wx)*wy*e01.x + wx*wy*e11.x;
  float ey = (1.f-wx)*(1.f-wy)*e00.y + wx*(1.f-wy)*e10.y +
             (1.f-wx)*wy*e01.y + wx*wy*e11.y;

  p.vx += qm * ex * dt;
  p.vy += qm * ey * dt;
  p.x += p.vx * dt;
  p.y += p.vy * dt;

  float lx = p.x - px0;
  float ly = p.y - py0;
  int sx = 0, sy = 0;
  if (lx < 0.f)             sx = (lx < -(float)W)  ? -2 : -1;
  else if (lx >= (float)W)  sx = (lx >= 2.f*W) ? 2 : 1;
  if (ly < 0.f)             sy = (ly < -(float)H)  ? -2 : -1;
  else if (ly >= (float)H)  sy = (ly >= 2.f*H) ? 2 : 1;
  if (sx < -1 || sx > 1 || sy < -1 || sy > 1) {
    counts[ERR_COUNTER] = 1;
    return;
  }

  if (p.x < 0.f) p.x += Lx; else if (p.x >= Lx) p.x -= Lx;
  if (p.y < 0.f) p.y += Ly; else if (p.y >= Ly) p.y -= Ly;

  const int dir_table[3][3] = {{TL, TOP, TR}, {LEFT, STAY, RIGHT}, {BL, BOTTOM, BR}};
  int dir = dir_table[sy + 1][sx + 1];
  if (dir == STAY) {
    stay[atomicAdd(&counts[STAY], 1)] = p;
  } else {
    int k = atomicAdd(&counts[dir], 1);
    // Overflow is detected on the host from counts[dir] > exch_cap
    if (k < exch_cap) sendslab[(size_t)dir * exch_cap + k] = p;
  }
}

static inline int nblocks(int n) { return (n + BLOCK_1D - 1) / BLOCK_1D; }

void invokeInitParticlesKernel(Particle* d_parts, int* d_np, long n_total,
    long n_bunch, int dist_type, float bunch_cx, float bunch_cy, float sigma,
    float drift, float thermal, float Lx, float Ly, float px0, float py0,
    float px1, float py1, int capacity, int* d_err, cudaStream_t stream) {
  dim3 block_dim(BLOCK_1D);
  dim3 grid_dim(4096);
  initParticlesKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_parts, d_np, n_total, n_bunch, dist_type, bunch_cx, bunch_cy, sigma,
      drift, thermal, Lx, Ly, px0, py0, px1, py1, capacity, d_err);
  hapiCheck(cudaPeekAtLastError());
}

void invokeDepositKernel(const Particle* d_parts, int n, RealType* d_rho,
    float weight, float px0, float py0, int block_width, int block_height,
    cudaStream_t stream) {
  if (n == 0) return;
  depositKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_parts, n, d_rho, weight, px0, py0, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokePackChargeGhostsKernel(const RealType* d_rho, RealType* d_slab,
    int block_width, int block_height, cudaStream_t stream) {
  int n = 2 * (block_width + block_height) + 4;
  packChargeGhostsKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_rho, d_slab, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeAccumChargeGhostKernel(RealType* d_rho, const RealType* d_buf,
    int dir, int block_width, int block_height, cudaStream_t stream) {
  int n = (dir == LEFT || dir == RIGHT) ? block_height :
          (dir == TOP || dir == BOTTOM) ? block_width : 1;
  accumChargeGhostKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_rho, d_buf, dir, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokePackPhiLRKernel(const RealType* d_phi, RealType* d_slab,
    int block_width, int block_height, cudaStream_t stream) {
  packPhiLRKernel<<<nblocks(2 * block_height), BLOCK_1D, 0, stream>>>(
      d_phi, d_slab, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackPhiLRKernel(RealType* d_phi, const RealType* d_buf, int dir,
    int block_width, int block_height, cudaStream_t stream) {
  unpackPhiLRKernel<<<nblocks(block_height), BLOCK_1D, 0, stream>>>(
      d_phi, d_buf, dir, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiPhiKernel(const RealType* d_phi, RealType* d_phi_new,
    const RealType* d_rho, float rho_bar, int block_width, int block_height,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_width + block_dim.x - 1) / block_dim.x,
      (block_height + block_dim.y - 1) / block_dim.y);
  jacobiPhiKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_phi, d_phi_new, d_rho, rho_bar, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeEFieldKernel(const RealType* d_phi, float2* d_efield,
    int block_width, int block_height, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_width + block_dim.x - 1) / block_dim.x,
      (block_height + block_dim.y - 1) / block_dim.y);
  efieldKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_phi, d_efield, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokePackEGhostsKernel(const float2* d_efield, float2* d_slab,
    int block_width, int block_height, cudaStream_t stream) {
  int n = 2 * (block_width + block_height) + 4;
  packEGhostsKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_efield, d_slab, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackEGhostKernel(float2* d_efield, const float2* d_buf, int dir,
    int block_width, int block_height, cudaStream_t stream) {
  int n = (dir == LEFT || dir == RIGHT) ? block_height :
          (dir == TOP || dir == BOTTOM) ? block_width : 1;
  unpackEGhostKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_efield, d_buf, dir, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokePushAndMarkKernel(const Particle* d_parts, int n,
    const float2* d_efield, Particle* d_stay, Particle* d_sendslab,
    int exch_cap, int* d_counts, float dt, float qm, float px0, float py0,
    float Lx, float Ly, int block_width, int block_height,
    cudaStream_t stream) {
  if (n == 0) return;
  pushAndMarkKernel<<<nblocks(n), BLOCK_1D, 0, stream>>>(
      d_parts, n, d_efield, d_stay, d_sendslab, exch_cap, d_counts, dt, qm,
      px0, py0, Lx, Ly, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}
