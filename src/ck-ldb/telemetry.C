/**
 * \addtogroup CkLdb
 */
/*@{*/

#include "telemetry.h"

#include "speedupsim.h"

#include "charm++.h"
#include "conv-ccs.h"
#include "converse.h"

#include <string.h>

#if CMK_LBDB_ON

/** The most recent snapshot, on the load balancer's PE.
 *
 * Kept as a cached value that the CCS handler only copies out. The handler
 * runs in the Converse scheduler, so it must not start a reduction or block:
 * the work happens when a window closes, not when someone asks.
 *
 * File-scope state survives a rescale restart, which is what makes windowSeq
 * monotone across the whole run. PE 0 is never reclaimed, so the process that
 * holds this is the same one throughout.
 */
static CkSchedTelemetry lastSnapshot;
static double lastWindowEnd = 0.0;
static uint32_t lastGeneration = 0;
static uint32_t windowCounter = 0;
static double lastRescaleSeconds = 0.0;

/** The width range the scheduler asked to be predicted, or 0 for none.
 *
 * File-scope so it survives a rescale restart. That matters: `CmiGetArg*`
 * consumes the argument, and a survivor re-runs its argument parsing after the
 * longjmp with an argv the first pass already stripped. Keeping the value here
 * means the second pass finding nothing is harmless.
 */
static int simLo = 0;
static int simHi = 0;

extern int _rescaleGeneration;

void CkTelemetryRecord(BaseLB::LDStats* stats)
{
  if (stats == NULL) return;

  const double now = CmiWallTimer();
  CkSchedTelemetry snap;
  memset(&snap, 0, sizeof(snap));

  snap.version = CK_TELEMETRY_VERSION;
  snap.flags = CK_TELEMETRY_VALID;
  snap.generation = (uint32_t)_rescaleGeneration;
  snap.windowSeq = ++windowCounter;
  snap.numPes = (uint32_t)stats->nprocs();
  snap.windowEnd = now;
  // The first window of a run has no predecessor to measure from; fall back to
  // the PEs' own accounting rather than reporting a window that started at
  // time zero.
  snap.windowWall = (lastWindowEnd > 0.0) ? (now - lastWindowEnd) : 0.0;

  // A window that spans a rescale describes neither width. Say so.
  if (snap.generation != lastGeneration)
  {
    snap.flags |= CK_TELEMETRY_POST_RESCALE;
    lastGeneration = snap.generation;
  }

  for (size_t i = 0; i < stats->nprocs(); i++)
  {
    const BaseLB::ProcStats& proc = stats->procs[i];
    snap.idleTime += proc.idletime;
    snap.bgTime += proc.bg_walltime;
    const double busy = proc.total_walltime - proc.idletime;
    if (busy > snap.maxPeLoad) snap.maxPeLoad = busy;
    if (snap.windowWall == 0.0 && proc.total_walltime > snap.windowWall)
      snap.windowWall = proc.total_walltime;
  }

  snap.numObjs = (uint32_t)stats->objData.size();
  for (size_t i = 0; i < stats->objData.size(); i++)
  {
    const LDObjData& obj = stats->objData[i];
    snap.objWall += obj.wallTime;
    if (obj.wallTime > snap.maxObjWall) snap.maxObjWall = obj.wallTime;
#if CMK_CUDA || CMK_HIP
    snap.gpuTime += obj.gpuTime;
    snap.flags |= CK_TELEMETRY_HAS_GPU;
#endif
#if CMK_LB_WAIT_TIME
    snap.objWaitSeconds += obj.waitTime;
    snap.flags |= CK_TELEMETRY_HAS_WAIT;
#endif
  }

  // Only traffic that crosses PEs matters for scaling: intra-PE messages stay
  // free as the job widens, while these are what a wider job pays more of.
  int nmsgs = 0, nbytes = 0;
  stats->computeNonlocalComm(nmsgs, nbytes);
  snap.interPeMsgs = (double)nmsgs;
  snap.interPeBytes = (double)nbytes;
  snap.lastRescaleSeconds = lastRescaleSeconds;

  // What this window would have cost at other widths. Only when asked for:
  // this is real work on the balancer's PE, inside the load balancing step.
  if (simLo > 0)
  {
    const double simStart = CmiWallTimer();
    int widths[CK_TELEMETRY_MAX_SAMPLES];
    const int n = CkChooseWidths(simLo, simHi, (int)snap.numPes, (int)snap.numObjs, widths,
                                 CK_TELEMETRY_MAX_SAMPLES);
    CkWidthPrediction pred[CK_TELEMETRY_MAX_SAMPLES];
    const unsigned int tiers = CkPredictWidths(stats, widths, n, pred);
    if (tiers & CK_WIDTHMODEL_PACKING)
    {
      snap.flags |= CK_TELEMETRY_HAS_SIM;
      if (tiers & CK_WIDTHMODEL_COMM) snap.flags |= CK_TELEMETRY_SIM_COMM;
      snap.numSamples = (uint32_t)n;
      for (int i = 0; i < n; i++)
      {
        snap.samples[i].numPes = (uint32_t)pred[i].numPes;
        snap.samples[i].predWall = pred[i].predWall;
      }
    }
    snap.simSeconds = CmiWallTimer() - simStart;
  }

  lastSnapshot = snap;
  lastWindowEnd = now;
}

void CkTelemetryRescaleCost(double seconds)
{
  if (seconds > 0.0) lastRescaleSeconds = seconds;
}

void CkTelemetrySetSimRange(int lo, int hi)
{
  if (lo < 1 || hi < lo) return;
  simLo = lo;
  simHi = hi;
}

/** Serve the cached snapshot.
 *
 * Replies immediately and always. A caller that has not seen a window yet gets
 * a snapshot without CK_TELEMETRY_VALID, which is a real answer -- "this job
 * has not reported yet" -- and lets it distinguish that from a job it cannot
 * reach at all.
 */
static void telemetryHandler(char* msg)
{
  if (CcsIsRemoteRequest()) CcsSendReply(sizeof(lastSnapshot), &lastSnapshot);
  CmiFree(msg);
}

void CkTelemetryInit(void)
{
  memset(&lastSnapshot, 0, sizeof(lastSnapshot));
  lastSnapshot.version = CK_TELEMETRY_VERSION;
  CcsRegisterHandler("get_telemetry", (CmiHandler)telemetryHandler);
}

#else

void CkTelemetryRecord(BaseLB::LDStats* stats) {}
void CkTelemetryRescaleCost(double seconds) {}
void CkTelemetrySetSimRange(int lo, int hi) {}
void CkTelemetryInit(void) {}

#endif

/*@}*/
