// Kokkos variant of jacobi3d.cu. The kernels are the kernels of jacobi3d.cu as functors, launched on the chare's stream
// through the Kokkos execution-space instance that wraps it (hapi_kokkos.h). The launch shapes are the ones jacobi3d.cu
// uses: 8x8x8 tiles for the 3-D kernels, 16x16 tiles for the faces, policy index 0 on threadIdx.x, the same lower
// bounds, so an A/B against jacobi3d.cu measures the launch path and the functor form and nothing else.
#include "hapi.h"
#include "jacobi3d.h"
#include "hapi_kokkos.h"

#define TILE_SIZE_3D 8
#define TILE_SIZE_2D 16
#define DIVIDEBY7 0.142857

using hapikokkos::Space;
using hapikokkos::space;
using Range3 = Kokkos::MDRangePolicy<Space, Kokkos::Rank<3>, Kokkos::IndexType<int>>;
using Range2 = Kokkos::MDRangePolicy<Space, Kokkos::Rank<2>, Kokkos::IndexType<int>>;
using Range1 = Kokkos::RangePolicy<Space, Kokkos::IndexType<int>>;

template <class P, class F>
static inline void site(std::once_flag& once, const char* label, const P& p, const F& f) {
  hapikokkos::launch(once, [&] { Kokkos::parallel_for(label, p, f); });
}

// IDX() reads block_width and block_height from the enclosing scope, hence the member names.

struct InitKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
    temperature[IDX(i,j,k)] = 0;
  }
};

struct GhostInitKernel {
  DataType* ghost;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { ghost[i] = 0; }
};

// Face kernels: the two policy indices are (j,k), (i,k) or (i,j) exactly as the kernels name them.
struct LeftBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { temperature[IDX(0,1+j,1+k)] = 1; }
};
struct RightBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { temperature[IDX(block_width+1,1+j,1+k)] = 1; }
};
struct TopBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { temperature[IDX(1+i,0,1+k)] = 1; }
};
struct BottomBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { temperature[IDX(1+i,block_height+1,1+k)] = 1; }
};
struct FrontBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { temperature[IDX(1+i,1+j,0)] = 1; }
};
struct BackBoundaryKernel {
  DataType* temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { temperature[IDX(1+i,1+j,block_depth+1)] = 1; }
};

struct JacobiKernel {
  DataType* temperature; DataType* new_temperature; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
#ifdef TEST_CORRECTNESS
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 10000;
#else
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) * DIVIDEBY7;
#endif
  }
};

struct LeftPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { ghost[block_height*k+j] = temperature[IDX(1,1+j,1+k)]; }
};
struct RightPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { ghost[block_height*k+j] = temperature[IDX(block_width,1+j,1+k)]; }
};
struct TopPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { ghost[block_width*k+i] = temperature[IDX(1+i,1,1+k)]; }
};
struct BottomPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { ghost[block_width*k+i] = temperature[IDX(1+i,block_height,1+k)]; }
};
struct FrontPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { ghost[block_width*j+i] = temperature[IDX(1+i,1+j,1)]; }
};
struct BackPackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { ghost[block_width*j+i] = temperature[IDX(1+i,1+j,block_depth)]; }
};

struct LeftUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { temperature[IDX(0,1+j,1+k)] = ghost[block_height*k+j]; }
};
struct RightUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int j, int k) const { temperature[IDX(block_width+1,1+j,1+k)] = ghost[block_height*k+j]; }
};
struct TopUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { temperature[IDX(1+i,0,1+k)] = ghost[block_width*k+i]; }
};
struct BottomUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int k) const { temperature[IDX(1+i,block_height+1,1+k)] = ghost[block_width*k+i]; }
};
struct FrontUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { temperature[IDX(1+i,1+j,0)] = ghost[block_width*j+i]; }
};
struct BackUnpackingKernel {
  DataType* temperature; DataType* ghost; int block_width, block_height, block_depth;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { temperature[IDX(1+i,1+j,block_depth+1)] = ghost[block_width*j+i]; }
};

// The face policies: (height x depth) for left/right, (width x depth) for top/bottom, (width x height) for front/back.
static inline Range2 faceJK(hapiStream_t s, int block_height, int block_depth) {
  return Range2(space(s), {0, 0}, {block_height, block_depth}, {TILE_SIZE_2D, TILE_SIZE_2D});
}
static inline Range2 faceIK(hapiStream_t s, int block_width, int block_depth) {
  return Range2(space(s), {0, 0}, {block_width, block_depth}, {TILE_SIZE_2D, TILE_SIZE_2D});
}
static inline Range2 faceIJ(hapiStream_t s, int block_width, int block_height) {
  return Range2(space(s), {0, 0}, {block_width, block_height}, {TILE_SIZE_2D, TILE_SIZE_2D});
}

void invokeInitKernel(DataType* d_temperature, int block_width, int block_height,
    int block_depth, hapiStream_t stream) {
  static std::once_flag once;
  site(once, "jacobi3d::init",
       Range3(space(stream), {0, 0, 0}, {block_width+2, block_height+2, block_depth+2},
              {TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D}),
       InitKernel{d_temperature, block_width, block_height, block_depth});
  hapiCheck(hapiPeekAtLastError());
}

void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, hapiStream_t stream) {
  static std::once_flag once;
  for (int i = 0; i < ghosts.size(); i++) {
    site(once, "jacobi3d::ghostInit", Range1(space(stream), 0, ghost_counts[i]), GhostInitKernel{ghosts[i]});
    hapiCheck(hapiPeekAtLastError());
  }
}

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], hapiStream_t stream) {
  static std::once_flag oL, oR, oT, oB, oF, oK;
  if (bounds[LEFT])
    site(oL, "jacobi3d::leftBoundary", faceJK(stream, block_height, block_depth),
         LeftBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  if (bounds[RIGHT])
    site(oR, "jacobi3d::rightBoundary", faceJK(stream, block_height, block_depth),
         RightBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  if (bounds[TOP])
    site(oT, "jacobi3d::topBoundary", faceIK(stream, block_width, block_depth),
         TopBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  if (bounds[BOTTOM])
    site(oB, "jacobi3d::bottomBoundary", faceIK(stream, block_width, block_depth),
         BottomBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  if (bounds[FRONT])
    site(oF, "jacobi3d::frontBoundary", faceIJ(stream, block_width, block_height),
         FrontBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  if (bounds[BACK])
    site(oK, "jacobi3d::backBoundary", faceIJ(stream, block_width, block_height),
         BackBoundaryKernel{d_temperature, block_width, block_height, block_depth});
  hapiCheck(hapiPeekAtLastError());
}

void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, hapiStream_t stream) {
  static std::once_flag once;
  // Interior points [1, block_width] x [1, block_height] x [1, block_depth]: tiles start at 1, as the kernel's +1 did.
  site(once, "jacobi3d::jacobi",
       Range3(space(stream), {1, 1, 1}, {block_width+1, block_height+1, block_depth+1},
              {TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D}),
       JacobiKernel{d_temperature, d_new_temperature, block_width, block_height, block_depth});
  hapiCheck(hapiPeekAtLastError());
}

void invokePackingKernels(DataType* d_temperature, DataType* d_ghosts[],
    bool bounds[], int block_width, int block_height, int block_depth,
    hapiStream_t stream) {
  static std::once_flag oL, oR, oT, oB, oF, oK;
  if (!bounds[LEFT])
    site(oL, "jacobi3d::leftPacking", faceJK(stream, block_height, block_depth),
         LeftPackingKernel{d_temperature, d_ghosts[LEFT], block_width, block_height, block_depth});
  if (!bounds[RIGHT])
    site(oR, "jacobi3d::rightPacking", faceJK(stream, block_height, block_depth),
         RightPackingKernel{d_temperature, d_ghosts[RIGHT], block_width, block_height, block_depth});
  if (!bounds[TOP])
    site(oT, "jacobi3d::topPacking", faceIK(stream, block_width, block_depth),
         TopPackingKernel{d_temperature, d_ghosts[TOP], block_width, block_height, block_depth});
  if (!bounds[BOTTOM])
    site(oB, "jacobi3d::bottomPacking", faceIK(stream, block_width, block_depth),
         BottomPackingKernel{d_temperature, d_ghosts[BOTTOM], block_width, block_height, block_depth});
  if (!bounds[FRONT])
    site(oF, "jacobi3d::frontPacking", faceIJ(stream, block_width, block_height),
         FrontPackingKernel{d_temperature, d_ghosts[FRONT], block_width, block_height, block_depth});
  if (!bounds[BACK])
    site(oK, "jacobi3d::backPacking", faceIJ(stream, block_width, block_height),
         BackPackingKernel{d_temperature, d_ghosts[BACK], block_width, block_height, block_depth});
  hapiCheck(hapiPeekAtLastError());
}

void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost, int dir,
    int block_width, int block_height, int block_depth, hapiStream_t stream) {
  static std::once_flag oL, oR, oT, oB, oF, oK;
  if (dir == LEFT)
    site(oL, "jacobi3d::leftUnpacking", faceJK(stream, block_height, block_depth),
         LeftUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  else if (dir == RIGHT)
    site(oR, "jacobi3d::rightUnpacking", faceJK(stream, block_height, block_depth),
         RightUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  else if (dir == TOP)
    site(oT, "jacobi3d::topUnpacking", faceIK(stream, block_width, block_depth),
         TopUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  else if (dir == BOTTOM)
    site(oB, "jacobi3d::bottomUnpacking", faceIK(stream, block_width, block_depth),
         BottomUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  else if (dir == FRONT)
    site(oF, "jacobi3d::frontUnpacking", faceIJ(stream, block_width, block_height),
         FrontUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  else if (dir == BACK)
    site(oK, "jacobi3d::backUnpacking", faceIJ(stream, block_width, block_height),
         BackUnpackingKernel{d_temperature, d_ghost, block_width, block_height, block_depth});
  hapiCheck(hapiPeekAtLastError());
}
