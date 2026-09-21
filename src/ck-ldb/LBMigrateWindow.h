#ifndef LB_MIGRATE_WINDOW_H
#define LB_MIGRATE_WINDOW_H

// The migration window: one rule, read by the runtime that enforces it per PE
// (cklocation.C, ckWindowLimits) and by the balancer that reserves it per
// device (LBMemoryContract.h, DiffusionHelper.C: L_g). The two must be one
// number, or the plan lands more than the pool kept room for.
//
//   window per PE  = min(CHARM_LB_MIGRATE_WINDOW_MB (256 MB), size / (4 * PEs))
//   L_g per device = window per PE * PEs of the device  <= size / 4
//
// The quarter cap keeps a small heap plannable: leanmd at N>1 has a 2 GB heap
// and 8 PEs per device, and 8 x 256 MB reserved all of it (job 22217872: the
// verifier refused 4487 of 4520 moves, the balancer moved nothing).
//
// `size` is the pool's CAPACITY on the device -- what it has mapped, free or
// not -- and so is the base of the plan's (1-eps) margin (lbPlannableBytes).
// Both used to be taken from the bytes free at the LB step, so the reserve
// shrank with the room: every step was allowed ~70% of whatever the last one
// had left, and a device the balancer kept feeding converged on full. sph2d
// strong, N=1: the GPU that collects the idle patches went 20 GB free -> 2 GB
// -> 600 MB -> under 80 MB over four steps, where an 11 MB block no longer
// fits by shape; gated landings then waited out CHARM_LB_GATE_TIMEOUT (120 s)
// and ungated ones grew the pool (jobs 22244226, 22245278). Measured against
// capacity the reserve is the same at every step, so the device comes to rest
// with it free. A capacity of 0 means the producer did not report one (an old
// dump, a hand-built test): the free bytes stand in, which is the old rule.

#include <cstddef>
#include <cstdlib>

static inline size_t lbMigrateWindowBytes() {
  static const size_t bytes = []() {
    const char* eb = getenv("CHARM_LB_MIGRATE_WINDOW_MB");
    return eb ? (size_t)atol(eb) << 20 : ((size_t)256 << 20);
  }();
  return bytes;
}

// L_g: what a pooled device of `size` bytes and `npes` PEs reserves for
// landings in flight.
static inline size_t lbLandingReserveBytes(size_t size, size_t npes) {
  const size_t want = lbMigrateWindowBytes() * npes;
  const size_t cap = size / 4;
  return want < cap ? want : cap;
}

// The per-PE byte window that adds up to L_g.
static inline size_t lbLandingWindowBytes(size_t size, size_t npes) {
  return npes ? lbLandingReserveBytes(size, npes) / npes : lbMigrateWindowBytes();
}

// What a plan may still place on a pooled device: its free bytes less the
// reserve -- the (1-headroom) margin and L_g, both measured against the pool's
// capacity so neither shrinks as the device fills. The caller takes sigma_max
// off where the rule it implements holds one pack back (H_g).
static inline size_t lbPlanReserveBytes(size_t free, size_t capacity, size_t npes,
                                        double headroom) {
  const size_t size = capacity > free ? capacity : free;
  return (size_t)((1.0 - headroom) * (double)size) + lbLandingReserveBytes(size, npes);
}

static inline size_t lbPlannableBytes(size_t free, size_t capacity, size_t npes,
                                      double headroom) {
  const size_t hold = lbPlanReserveBytes(free, capacity, npes, headroom);
  return free > hold ? free - hold : 0;
}

#endif
