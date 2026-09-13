#ifndef _DIFFUSION_COST_MODEL_H
#define _DIFFUSION_COST_MODEL_H

// Cost model for DiffusionLB's across-node phase.
//
// The across-node loop sheds load until its diffusion quota is met and never
// asks what a move costs: popBestObject chooses *which* object crosses, never
// *whether* one should. On a device-zerocopy run that is the whole ballgame. A
// cross-"node" move here is a cross-process move, and it converts a ghost
// exchange that was a same-process device memcpy into an IPC or inter-node
// transfer -- permanently, for every iteration until the next rebalance.
// Measured on pic2d (2048^2, 32 PEs, 2 nodes): balancing drops the same-process
// share of device transfers from 70.0% to ~61.6% and the run goes 8.1 -> 13.5 s,
// with run time monotonic in how much load is shed.
//
// The asymmetry that makes this worth modelling: migration cost is paid once,
// but the communication change is paid every interval until the next LB. Over K
// intervals the comparison is
//
//     K * loadGain   vs   migrateCost + K * commDelta
//
// so per interval it is
//
//     loadGain   vs   migrateCost/K + commDelta
//
// Note where the horizon sits. commDelta needs no multiplier: LBDatabase::Send
// records communication only while instrumentation is on (LBDatabase.C:134), the
// same window that bounds the load measurement, so both figures already cover
// exactly one balancer interval and are directly comparable. It is the one-off
// migration cost that must be amortised, over however many intervals the
// placement is expected to survive.

#include "lbdb.h"

// Transport tiers a device transfer can resolve to, cheapest first. These are
// the four the runtime actually distinguishes, and the four the calibration
// benchmark measures.
enum DiffusionTier
{
  DIFF_TIER_INTRA_PROCESS = 0,  // same process: device-to-device memcpy
  DIFF_TIER_IPC_SAME_GPU,       // separate processes sharing one device
  DIFF_TIER_IPC_CROSS_GPU,      // separate processes, different devices, one host
  DIFF_TIER_INTER_NODE,         // separate hosts: network RDMA
  DIFF_TIER_COUNT
};

// Per-tier linear transfer cost: alpha seconds per message, beta seconds per
// byte. Populated by DiffusionCostConfig, whose defaults are deliberately
// useless (see below) so that an uncalibrated run is obvious rather than
// quietly mis-costed.
struct DiffusionTierCost
{
  double alpha = 0.0;
  double beta = 0.0;
};

// Loaded once from the file named by +LBCostConfig. Written by the calibration
// benchmark in benchmarks/charm++/cuda/gpudirect/lbcalib.
class DiffusionCostConfig
{
public:
  // True once a config file has been read successfully. Without one the model
  // must not be used to reject moves: guessing constants here reproduces the
  // very regression the model exists to prevent, only with a false air of
  // authority. Callers fall back to the historical unconditional behaviour.
  bool calibrated = false;

  DiffusionTierCost tier[DIFF_TIER_COUNT];

  // Seconds per byte of device state a migration must pack, ship and unpack.
  // Distinct from the tier betas: a migration moves the object's whole device
  // footprint once, over a path that stages through the migration buffer rather
  // than the steady-state ghost path.
  double migrateBetaDevice = 0.0;
  // Same, for the host-side pupped bytes.
  double migrateBetaHost = 0.0;
  // Fixed per-migration overhead: the location update, the barrier the move
  // participates in, and the pack/unpack call itself.
  double migrateAlpha = 0.0;

  // How many balancer intervals a placement is assumed to survive, used to
  // amortise the one-off migration cost. Overwritten at runtime once the
  // balancer has seen enough steps to measure its own period.
  double placementLifetimeIntervals = 4.0;

  // Reads the key = value file written by the calibration benchmark. Returns
  // false (leaving calibrated == false) if the file cannot be read or omits a
  // required key; the reason is printed once on PE 0.
  bool load(const char* path);

  // Resolves which tier a transfer between two PEs would use. Same process is a
  // memcpy; same host splits on whether the two processes drive the same device;
  // otherwise the network.
  static DiffusionTier tierBetween(int pe1, int pe2);

  static const char* tierName(DiffusionTier t);
};

// Evaluates one candidate move. Holds no state of its own: every figure comes
// from the MetricComm arrays, which updateState keeps current as moves are
// accepted, so a greedy loop sees the partition its earlier choices created.
class DiffusionCostModel
{
public:
  DiffusionCostModel(const DiffusionCostConfig& cfg, DiffusionTier localTier)
      : cfg_(cfg), localTier_(localTier)
  {
  }

  // Change in this node's per-interval communication time if an object with the
  // given incident traffic moves to a neighbour reached over `destTier`.
  //
  //   internal*  traffic with objects that stay here -- becomes cut
  //   external*  traffic with objects on the destination -- becomes local
  //
  // Both are incident figures (both directions of every edge) on the same basis;
  // see the note in DiffusionMetric.C about how the external side's inbound half
  // is recovered. Positive means the move makes steady-state communication worse.
  double commDelta(double internalBytes, double internalMsgs, double externalBytes,
                   double externalMsgs, DiffusionTier destTier) const
  {
    const double dAlpha = cfg_.tier[destTier].alpha - cfg_.tier[localTier_].alpha;
    const double dBeta = cfg_.tier[destTier].beta - cfg_.tier[localTier_].beta;
    return dBeta * (internalBytes - externalBytes) +
           dAlpha * (internalMsgs - externalMsgs);
  }

  // One-off cost of moving the object, amortised over the intervals the
  // placement is expected to survive so it can be compared against per-interval
  // load and communication figures. Sizes the application never declared are
  // zero (LBObj), so such an object is priced at the per-migration constant
  // alone -- an under-estimate, and the reason declaring them matters.
  double migrateCost(const LDObjData& o) const
  {
    double hostBytes = (double)pup_decodeSize(o.pupSize);
    double devBytes = 0.0;
#if CMK_CUDA
    devBytes = (double)o.gpuPupSize;
#endif
    const double oneOff = cfg_.migrateAlpha + cfg_.migrateBetaHost * hostBytes +
                          cfg_.migrateBetaDevice * devBytes;
    const double k = (cfg_.placementLifetimeIntervals > 1.0)
                         ? cfg_.placementLifetimeIntervals
                         : 1.0;
    return oneOff / k;
  }

  // The benefit of shedding this object, capped at the node's remaining
  // obligation: load shed past the fair share buys nothing, so an object larger
  // than the remaining excess is credited only with the excess it removes.
  static double loadBenefit(double objLoad, double loadStillToShed)
  {
    return (objLoad < loadStillToShed) ? objLoad : loadStillToShed;
  }

private:
  const DiffusionCostConfig& cfg_;
  DiffusionTier localTier_;
};

#endif
