/* Per-entry-method hooks for two timeline tools (cknvtx.C):
 *   +nvtx            NVTX range "Chare::entry" around every entry-method delivery (payload = array index)
 *                    and an "idle" range while the scheduler has nothing to run, domain "charm", for nsys.
 *   +gputrace <dir>  the runtime's own timeline: the same entry and idle ranges written per PE to
 *                    <dir>/gputrace.<host>.<pid>.pe<N> on the CUPTI clock, next to the process's CUPTI
 *                    kernel/memcpy records (hapiTraceStart); gputrace_merge.py builds one Perfetto trace of
 *                    all GPUs. Use this for LB runs: nsys would take over the runtime's CUPTI subscription.
 *   +nvtxnoidle      either tool without the idle ranges.
 * Off by default; the hot path is one relaxed atomic load per delivery. */
#ifndef _CKNVTX_H_
#define _CKNVTX_H_
#if CMK_NVTX
#include <atomic>
extern std::atomic<bool> _nvtxOn;            // +nvtx or +gputrace given; set once at init, before any entry method runs
void _nvtxInit(char **argv);                 // rank 0 of each process, with the other command-line options
void _nvtxInitPE();                          // every PE, after registration: the idle hooks
void _nvtxStartTrace();                      // every PE, after hapiInit: rank 0 opens the +gputrace sink
void _nvtxExit();                            // every PE at exit: close its +gputrace file
void _nvtxBeginEntry(int epIdx, const void *msg, void *obj);
void _nvtxEndEntry();
// An application message's flight, for the +gputrace timeline: sent at t_send (the sender's
// hapiTraceTimestamp()), envelope on this PE at t_env, payload usable at t_land; kind and tier are
// the application's own small integers (sph3d: kind 0 halo / 1 migration, tier 0 same process /
// 1 same node / 2 other node). A no-op unless +gputrace is on.
void _nvtxTraceMsg(unsigned long long t_send, unsigned long long t_env, unsigned long long t_land, int kind, int tier);
#define CK_GPUTRACE_MSG(ts, te, tl, kind, tier) do { if (_nvtxOn.load(std::memory_order_relaxed)) _nvtxTraceMsg((ts), (te), (tl), (kind), (tier)); } while (0)
#define CK_NVTX_BEGIN_ENTRY(ep, msg, obj) do { if (_nvtxOn.load(std::memory_order_relaxed)) _nvtxBeginEntry((ep), (msg), (obj)); } while (0)
#define CK_NVTX_END_ENTRY()               do { if (_nvtxOn.load(std::memory_order_relaxed)) _nvtxEndEntry(); } while (0)
#else
static inline void _nvtxInit(char **) {}
static inline void _nvtxInitPE() {}
static inline void _nvtxStartTrace() {}
static inline void _nvtxExit() {}
#define CK_NVTX_BEGIN_ENTRY(ep, msg, obj) do { } while (0)
#define CK_NVTX_END_ENTRY()               do { } while (0)
#define CK_GPUTRACE_MSG(ts, te, tl, kind, tier) do { } while (0)
#endif
#endif
