#ifndef LEANMD_JOIN_TRACE_H
#define LEANMD_JOIN_TRACE_H
#include <cstdlib>
// Sparse full-population snapshots; no clock subtraction across processes.
static inline bool joinTrace(int step) {
  static const bool enabled = std::getenv("LEANMD_JOIN_TRACE") != nullptr;
  return enabled && (step == 45 || step == 65 || step == 85 || step == 95);
}
#endif
