/**
 * \addtogroup CkLdb
 */
/*@{*/

/** Scheduler telemetry.
 *
 * Reports what an elastic scheduler needs in order to decide how wide a job
 * should be: not a single "efficiency" number, but the P-invariant quantities
 * from which the speedup curve at any width can be derived.
 *
 * The snapshot is taken on PE 0 at each load-balancing step, from statistics
 * the load balancer already collects, and served over CCS on request. There is
 * no extra reduction and no extra instrumentation.
 */

#ifndef CK_TELEMETRY_H
#define CK_TELEMETRY_H

#include "BaseLB.h"
#include <stddef.h>
#include <stdint.h>

/** Wire format version. Bump on any layout change. */
#define CK_TELEMETRY_VERSION 4

/** The snapshot is valid: a window has closed and its numbers are usable. */
#define CK_TELEMETRY_VALID (1u << 0)
/** This window straddled a rescale. Migration, cold caches and (on the first
 *  rescale) isomalloc re-sync land in it, so its timings describe neither the
 *  old width nor the new one. Report it rather than let a reader average it
 *  in. */
#define CK_TELEMETRY_POST_RESCALE (1u << 1)
/** objWaitSeconds is meaningful -- the build was configured with
 *  CMK_LB_WAIT_TIME. Without it the field is zero, which is indistinguishable
 *  from a job that never waits, so a reader must check this before using it. */
#define CK_TELEMETRY_HAS_WAIT (1u << 3)
/** gpuTime is meaningful. Without it a low CPU efficiency says nothing about
 *  whether the job is saturated, because the busy resource is not being
 *  measured. */
#define CK_TELEMETRY_HAS_GPU (1u << 2)
/** The sample array holds predicted makespans at other widths. Without this a
 *  reader has only the current width and must fit a curve through history. */
#define CK_TELEMETRY_HAS_SIM (1u << 4)
/** The predictions include communication, because the job was already
 *  collecting an object communication graph. Without it they model only
 *  granularity and imbalance, and so read optimistic at large widths. */
#define CK_TELEMETRY_SIM_COMM (1u << 5)

/** How many predicted widths a snapshot can carry. */
#define CK_TELEMETRY_MAX_SAMPLES 12

/** A predicted point on the job's speedup curve. */
struct CkSchedSpeedupSample
{
  uint32_t numPes;   /**< the width predicted for */
  uint32_t reserved; /**< keeps predWall 8-byte aligned; zero */
  double predWall;   /**< predicted seconds for a window of this window's work */
};

/** A telemetry snapshot, as sent over CCS.
 *
 * Plain old data with no padding surprises: every field is 4 or 8 bytes and
 * the 4-byte fields come in pairs. Integers are in the sender's native byte
 * order, matching how Charm++ CCS handlers read their arguments elsewhere --
 * note this differs from the CCS *header*, which is big-endian.
 */
struct CkSchedTelemetry
{
  uint32_t version;    /**< CK_TELEMETRY_VERSION */
  uint32_t flags;      /**< CK_TELEMETRY_* */
  uint32_t generation; /**< rescale membership epoch this window belongs to */
  uint32_t windowSeq;  /**< monotone across the run; 0 before the first window */
  uint32_t numPes;     /**< the width this window was measured at */
  uint32_t numObjs;    /**< chares, which caps the achievable speedup */

  double windowEnd;  /**< CmiWallTimer() when the window closed */
  double windowWall; /**< seconds the window covered */

  /** Total useful work: time inside chare entry methods, summed over PEs.
   *
   * This is the quantity that does not change with width, so it is what lets
   * a reader predict a different width. Note it excludes runtime overhead and
   * communication progress, which is the whole point: those are counted as
   * waste, where an idle-based measure would count them as work. */
  double objWall;
  /** The largest single object's time, which bounds the achievable speedup at
   *  any width no matter how many workers are added. */
  double maxObjWall;
  /** Busy time of the busiest PE. The gap between this and objWall/numPes is
   *  imbalance, which a load balancer can fix at the current width; the gap
   *  between wall time and this is latency, which only gets worse with width.
   *  Distinguishing them is why both are reported. */
  double maxPeLoad;

  double idleTime; /**< summed over PEs */
  double bgTime;   /**< summed runtime and scheduler overhead */
  double gpuTime;  /**< summed device time; 0 unless CK_TELEMETRY_HAS_GPU */

  double interPeBytes; /**< bytes crossing PEs, which grow with width */
  double interPeMsgs;

  /** How long the most recent rescale stalled this job, in seconds; 0 if it
   *  has not rescaled.
   *
   * This is the stall alone -- the runtime's own measurement, from entering
   * the realloc path to the restore completing. It deliberately excludes the
   * wait for a boundary to arrive, because a scheduler pacing a job needs to
   * add its own bound on that wait rather than have one folded in here. */
  double lastRescaleSeconds;

  /** Total time objects spent waiting for their inputs, summed over objects.
   *
   * This is the *potential* wait, not the exposed one: it counts the gap
   * between an object finishing and its next input being produced, whether or
   * not the PE had other objects to run in the meantime. Idle time already
   * reports the exposed part.
   *
   * The two differ by exactly the overlap a job is getting, which is what
   * makes this worth measuring: overlap depends on how many objects share a
   * PE, so it shrinks as a job is spread wider. A reader that only sees idle
   * time cannot tell a job with little communication from one hiding a lot of
   * it, and those two behave very differently at a larger width.
   *
   * Zero unless CK_TELEMETRY_HAS_WAIT is set. */
  double objWaitSeconds;

  /** How many entries of `samples` are filled; 0 unless CK_TELEMETRY_HAS_SIM. */
  uint32_t numSamples;
  uint32_t reserved0; /**< zero */
  /** Seconds the prediction itself cost, on the balancer's PE.
   *
   * Reported rather than assumed negligible: it is work done inside the load
   * balancing step, so a reader that asks for a wide range of widths on a job
   * with a large communication graph can see what the answer cost. */
  double simSeconds;

  /** Predicted makespan at other widths, ascending by width.
   *
   * These come from re-packing this window's own objects onto a different
   * number of processors, not from extrapolating a fitted curve, so the
   * granularity ceiling and the imbalance steps are exact rather than smoothed
   * over. The entry whose numPes equals this snapshot's numPes is always
   * present: comparing its predWall against windowWall is the only available
   * check on whether the rest of the array can be believed.
   *
   * They model the makespan of the *work*. Load balancer statistics are
   * per-window totals with no ordering, so nothing here accounts for a
   * critical path or for latency that fails to overlap. */
  struct CkSchedSpeedupSample samples[CK_TELEMETRY_MAX_SAMPLES];
};

#ifdef __cplusplus
/* The wire layout is a contract with the scheduler, which decodes these bytes
 * without including this header. Pin it here so a field reordered or widened
 * fails the build rather than being misread at the far end. */
static_assert(sizeof(struct CkSchedSpeedupSample) == 16, "telemetry wire size changed");
static_assert(sizeof(struct CkSchedTelemetry) == 328, "telemetry wire size changed");
static_assert(offsetof(struct CkSchedTelemetry, windowEnd) == 24, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, interPeMsgs) == 96, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, lastRescaleSeconds) == 104, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, objWaitSeconds) == 112, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, numSamples) == 120, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, simSeconds) == 128, "telemetry layout changed");
static_assert(offsetof(struct CkSchedTelemetry, samples) == 136, "telemetry layout changed");
#endif

/** Record a snapshot from a completed load-balancing window. Called on the
 *  central load balancer's PE with the collected statistics. */
void CkTelemetryRecord(BaseLB::LDStats* stats);

/** Record how long the most recent rescale stalled the job. Called from the
 *  restore path, which is where the runtime already measures it. */
void CkTelemetryRescaleCost(double seconds);

/** Ask for makespan predictions over [lo, hi] processors at each window.
 *
 * Set from `+LBSimRange lo:hi`. Off unless asked for: the prediction is only
 * useful to an elastic scheduler, and it is that scheduler which knows the
 * range worth asking about -- the job's own floor at the bottom, and what the
 * cluster could ever give it at the top.
 */
void CkTelemetrySetSimRange(int lo, int hi);

/** Register the CCS handler. Called from manager_init, which runs once per
 *  process and whose registrations survive a rescale restart. */
void CkTelemetryInit(void);

#endif

/*@}*/
