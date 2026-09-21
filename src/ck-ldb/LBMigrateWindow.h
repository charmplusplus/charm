#ifndef LB_MIGRATE_WINDOW_H
#define LB_MIGRATE_WINDOW_H

// The migration window: one rule, read by the runtime that enforces it per PE
// (cklocation.C, ckWindowLimits) and by the balancer that reserves it per
// device (LBMemoryContract.h, DiffusionHelper.C: L_g). The two must be one
// number, or the plan lands more than the pool kept room for.
//
//   window per PE  = min(CHARM_LB_MIGRATE_WINDOW_MB (256 MB), reach / (4 * PEs))
//   L_g per device = window per PE * PEs of the device  <= reach / 4
//
// The quarter cap keeps a small heap plannable: leanmd at N>1 has a 2 GB heap
// and 8 PEs per device, and 8 x 256 MB reserved all of it (job 22217872: the
// verifier refused 4487 of 4520 moves, the balancer moved nothing).

#include <cstddef>
#include <cstdlib>

static inline size_t lbMigrateWindowBytes() {
  static const size_t bytes = []() {
    const char* eb = getenv("CHARM_LB_MIGRATE_WINDOW_MB");
    return eb ? (size_t)atol(eb) << 20 : ((size_t)256 << 20);
  }();
  return bytes;
}

// L_g: what a pooled device with `reach` bytes and `npes` PEs reserves for
// landings in flight.
static inline size_t lbLandingReserveBytes(size_t reach, size_t npes) {
  const size_t want = lbMigrateWindowBytes() * npes;
  const size_t cap = reach / 4;
  return want < cap ? want : cap;
}

// The per-PE byte window that adds up to L_g.
static inline size_t lbLandingWindowBytes(size_t reach, size_t npes) {
  return npes ? lbLandingReserveBytes(reach, npes) / npes : lbMigrateWindowBytes();
}

#endif
