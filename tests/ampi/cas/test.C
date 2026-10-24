// MPI_Compare_and_swap, and the result paths of MPI_Fetch_and_op and MPI_Get.
//
// Each rank owns a window and operates on the window of the NEXT rank, so
// with +vp 2 on one PE the target is local (ampi::winLocalCompareAndSwap) and
// on two PEs it is remote (ampi::winRemoteCompareAndSwap). MPI-3 semantics:
// the target value is compared with the compare buffer; if they are equal the
// origin value is written to the target; in every case the result buffer
// receives the ORIGINAL target value.
#include <stdio.h>
#include <string.h>
#include "mpi.h"

// Per-rank state lives on the stack: AMPI ranks on one PE share globals.
struct Ctx {
  int rank, size, target, source;
  int errors;
};

#define check(ok, type, what, got, want) checkImpl(c, ok, type, what, got, want)
static void checkImpl(Ctx& c, bool ok, const char* type, const char* what, double got, double want)
{
  if (!ok) {
    printf("[%d] FAIL %s %s: got %.17g, expected %.17g\n", c.rank, type, what, got, want);
    c.errors++;
  }
}

// Window of N elements of T with disp_unit sizeof(T); element i of rank r
// starts as r*100+i.
template <typename T>
static void testType(Ctx& c, MPI_Datatype dt, const char* name)
{
  const int rank = c.rank, target = c.target, source = c.source;
  const int N = 8;
  T wbuf[N];
  for (int i = 0; i < N; i++) wbuf[i] = (T)(rank * 100 + i);
  MPI_Win win;
  MPI_Win_create(wbuf, sizeof(wbuf), sizeof(T), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  T orig, comp, res;

  // Fence epoch. disp 2: matching compare, the swap happens.
  // disp 3: compare differs from the target only above the low byte, no swap.
  MPI_Win_fence(0, win);
  orig = (T)(1000 + rank); comp = (T)(target * 100 + 2); res = (T)-1;
  MPI_Compare_and_swap(&orig, &comp, &res, dt, target, 2, win);
  MPI_Win_fence(0, win);
  check(res == (T)(target * 100 + 2), name, "fence match: result", res, target * 100 + 2);

  MPI_Win_fence(0, win);
  orig = (T)(2000 + rank); comp = (T)(target * 100 + 3 + 256); res = (T)-1;
  MPI_Compare_and_swap(&orig, &comp, &res, dt, target, 3, win);
  MPI_Win_fence(0, win);
  check(res == (T)(target * 100 + 3), name, "fence no-match: result", res, target * 100 + 3);
  check(orig == (T)(2000 + rank), name, "fence no-match: origin unchanged", orig, 2000 + rank);
  check(comp == (T)(target * 100 + 3 + 256), name, "fence no-match: compare unchanged", comp,
        target * 100 + 3 + 256);

  check(wbuf[2] == (T)(1000 + source), name, "fence match: target swapped", wbuf[2], 1000 + source);
  check(wbuf[3] == (T)(rank * 100 + 3), name, "fence no-match: target unchanged", wbuf[3], rank * 100 + 3);

  // Lock epoch. disp 5: match. disp 6: no match. disp 2 again: compare with
  // the value swapped in above, so the swap happens again.
  T res5 = (T)-1, res6 = (T)-1, res2 = (T)-1;
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
  orig = (T)(3000 + rank); comp = (T)(target * 100 + 5);
  MPI_Compare_and_swap(&orig, &comp, &res5, dt, target, 5, win);
  T orig6 = (T)(4000 + rank), comp6 = (T)(-7);
  MPI_Compare_and_swap(&orig6, &comp6, &res6, dt, target, 6, win);
  T orig2 = (T)(5000 + rank), comp2 = (T)(1000 + rank);
  MPI_Compare_and_swap(&orig2, &comp2, &res2, dt, target, 2, win);
  MPI_Win_unlock(target, win);
  check(res5 == (T)(target * 100 + 5), name, "lock match: result", res5, target * 100 + 5);
  check(res6 == (T)(target * 100 + 6), name, "lock no-match: result", res6, target * 100 + 6);
  check(res2 == (T)(1000 + rank), name, "lock second swap: result", res2, 1000 + rank);
  MPI_Barrier(MPI_COMM_WORLD);
  check(wbuf[5] == (T)(3000 + source), name, "lock match: target swapped", wbuf[5], 3000 + source);
  check(wbuf[6] == (T)(rank * 100 + 6), name, "lock no-match: target unchanged", wbuf[6], rank * 100 + 6);
  check(wbuf[2] == (T)(5000 + source), name, "lock second swap: target", wbuf[2], 5000 + source);

  // Fetch_and_op SUM as a cross-check of the fetch path. disp 0.
  MPI_Win_fence(0, win);
  orig = (T)5; res = (T)-1;
  MPI_Fetch_and_op(&orig, &res, dt, target, 0, MPI_SUM, win);
  MPI_Win_fence(0, win);
  check(res == (T)(target * 100), name, "fetch_and_op: result", res, target * 100);
  check(wbuf[0] == (T)(rank * 100 + 5), name, "fetch_and_op: target", wbuf[0], rank * 100 + 5);

  // Elements no operation addressed must be untouched.
  const int untouched[] = {1, 4, 7};
  for (int k = 0; k < 3; k++) {
    int i = untouched[k];
    check(wbuf[i] == (T)(rank * 100 + i), name, "untouched element", wbuf[i], rank * 100 + i);
  }

  MPI_Win_free(&win);
}

// Byte-addressed window (disp_unit 1) holding longs at byte displacements 8
// and 16: the target address must be baseAddr + disp_unit*targdisp.
static void testByteWindow(Ctx& c)
{
  const int rank = c.rank, target = c.target, source = c.source;
  long wbuf[4];
  for (int i = 0; i < 4; i++) wbuf[i] = rank * 100 + i;
  MPI_Win win;
  MPI_Win_create(wbuf, sizeof(wbuf), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  long orig = 7000 + rank, comp = target * 100 + 1, res = -1;
  MPI_Win_fence(0, win);
  MPI_Compare_and_swap(&orig, &comp, &res, MPI_LONG, target, 1 * sizeof(long), win);
  MPI_Win_fence(0, win);
  check(res == target * 100 + 1, "disp_unit 1", "cas: result", res, target * 100 + 1);
  check(wbuf[1] == 7000 + source, "disp_unit 1", "cas: target swapped", wbuf[1], 7000 + source);

  long one = 1;
  res = -1;
  MPI_Win_fence(0, win);
  MPI_Fetch_and_op(&one, &res, MPI_LONG, target, 2 * sizeof(long), MPI_SUM, win);
  MPI_Win_fence(0, win);
  check(res == target * 100 + 2, "disp_unit 1", "fetch_and_op: result", res, target * 100 + 2);
  check(wbuf[2] == rank * 100 + 3, "disp_unit 1", "fetch_and_op: target", wbuf[2], rank * 100 + 3);
  check(wbuf[0] == rank * 100 + 0, "disp_unit 1", "untouched [0]", wbuf[0], rank * 100);
  check(wbuf[3] == rank * 100 + 3, "disp_unit 1", "untouched [3]", wbuf[3], rank * 100 + 3);

  MPI_Win_free(&win);
}

// MPI_Get into a non-contiguous origin type (every other int) from a
// contiguous target.
static void testGetNoncontigOrigin(Ctx& c)
{
  const int rank = c.rank, target = c.target;
  int wbuf[4];
  for (int i = 0; i < 4; i++) wbuf[i] = rank * 100 + i;
  MPI_Win win;
  MPI_Win_create(wbuf, sizeof(wbuf), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  MPI_Datatype vec;
  MPI_Type_vector(2, 1, 2, MPI_INT, &vec);
  MPI_Type_commit(&vec);

  int gbuf[4] = {-1, -1, -1, -1};
  MPI_Win_fence(0, win);
  MPI_Get(gbuf, 1, vec, target, 1, 2, MPI_INT, win);
  MPI_Win_fence(0, win);
  const char* n = "get vector origin";
  check(gbuf[0] == target * 100 + 1, n, "gbuf[0]", gbuf[0], target * 100 + 1);
  check(gbuf[1] == -1, n, "gbuf[1] (gap)", gbuf[1], -1);
  check(gbuf[2] == target * 100 + 2, n, "gbuf[2]", gbuf[2], target * 100 + 2);
  check(gbuf[3] == -1, n, "gbuf[3] (gap)", gbuf[3], -1);
  for (int i = 0; i < 4; i++)
    check(wbuf[i] == rank * 100 + i, n, "target unchanged", wbuf[i], rank * 100 + i);

  MPI_Type_free(&vec);
  MPI_Win_free(&win);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  Ctx c;
  c.errors = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &c.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &c.size);
  c.target = (c.rank + 1) % c.size;
  c.source = (c.rank + c.size - 1) % c.size;
  const int rank = c.rank, size = c.size;

  testType<int>(c, MPI_INT, "int");
  testType<long>(c, MPI_LONG, "long");
  testType<double>(c, MPI_DOUBLE, "double");
  testByteWindow(c);
  testGetNoncontigOrigin(c);

  int total = 0;
  MPI_Allreduce(&c.errors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0) printf("cas test passed on %d ranks\n", size);
    else printf("cas test FAILED: %d errors\n", total);
  }
  // AMPI's MPI_Finalize never returns (it ends in AMPI_Exit(0)), so a
  // failure must be reported through the exit status of MPI_Abort.
  if (total != 0) MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
  return 0;
}
