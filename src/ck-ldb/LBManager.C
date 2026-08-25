/**
 * \addtogroup CkLdb
 */
/*@{*/

#include "converse.h"
#include <charm++.h>
#include <ck.h>
#include "cksyncbarrier.h"

#if CMK_CUDA || CMK_HIP
#include "hapi_portable.h"
#endif

#include "DistributedLB.h"
#include "LBManager.h"
#include "LBSimulation.h"
#include "TreeLB.h"
#include "telemetry.h"
#include "rescalepoint.h"
#include "ckrescale.h"
#include "topology.h"

#include "json.hpp"

CkGroupID _lbmgr;

CkpvDeclare(LBUserDataLayout, lbobjdatalayout);
CkpvDeclare(int, _lb_obj_index);

CkpvDeclare(bool, lbmanagerInited); /**< true if lbdatabase is inited */

extern int quietModeRequested;

// command line options
CkLBArgs _lb_args;
bool _lb_predict = false;
int _lb_predict_delay = 10;
int _lb_predict_window = 20;
bool _lb_psizer_on = false;

SystemLoad::SystemLoad() {
  auto *activeRec = CkActiveLocRec();
  lbmgr = LBManagerObj();
  if (lbmgr && activeRec) {
    const LDObjHandle &runObj = activeRec->getLdHandle();
    lbmgr->ObjectStop(runObj);
  }
}

// registry class stores all load balancers linked and created at runtime
class LBDBRegistry
{
  friend class LBMgrInit;
  friend class LBManager;

 private:
  // table for all available LBs linked in
  struct LBDBEntry
  {
    std::string name;
    LBCreateFn cfn;
    LBAllocFn afn;
    std::string help;
    bool shown;  // if false, do not show in help page
    LBDBEntry() : name(""), cfn(0), afn(0), help(""), shown(true) {}
    LBDBEntry(int) {}
    LBDBEntry(std::string n, LBCreateFn cf, LBAllocFn af, std::string h, bool show = true)
        : name(n), cfn(cf), afn(af), help(h), shown(show){};
  };
  std::vector<LBDBEntry> lbtables;       // a list of available LBs linked
  std::vector<const char*> compile_lbs;  // load balancers at compile time
  std::vector<const char*> runtime_lbs;  // load balancers at run time
  // map of {index in runtime_lbs, name of legacy LB to instantiate TreeLB with}
  // for use with the legacy LBs (e.g. GreedyLB -> the predefined Greedy version of TreeLB)
  std::unordered_map<int, const char*> legacy_runtime_treelbs;
 public:
  LBDBRegistry() {}
  void displayLBs()
  {
    CmiPrintf("\nAvailable load balancers:\n");
    for (const auto& entry : lbtables)
    {
      if (entry.shown) CmiPrintf("* %s:\t%s\n", entry.name.c_str(), entry.help.c_str());
    }
    CmiPrintf("\n");
  }
  void addEntry(std::string name, LBCreateFn fn, LBAllocFn afn, std::string help,
                bool shown)
  {
    lbtables.emplace_back(name, fn, afn, help, shown);
  }
  bool hasBalancers() const { return !runtime_lbs.empty() || !compile_lbs.empty(); }
  void addCompiletimeBalancer(const char* name) { compile_lbs.push_back(name); }
  void addRuntimeBalancer(const char* name, const char* legacyLBName = nullptr)
  {
    if (legacyLBName != nullptr)
    {
      legacy_runtime_treelbs.emplace((int)runtime_lbs.size(), legacyLBName);
    }

    runtime_lbs.push_back(name);
  }
  LBCreateFn search(std::string name)
  {
    const auto index = name.find_first_of(":,");
    for (int i = 0; i < lbtables.size(); i++)
      if (0 == lbtables[i].name.compare(0, index, name))
        return lbtables[i].cfn;
    return nullptr;
  }
  LBAllocFn getLBAllocFn(std::string name)
  {
    const auto index = name.find_first_of(":,");
    for (int i = 0; i < lbtables.size(); i++)
      if (0 == lbtables[i].name.compare(0, index, name))
        return lbtables[i].afn;
    return nullptr;
  }
};

static LBDBRegistry lbRegistry;
static std::vector<std::string> lbNames;

void LBDefaultCreate(const char* lbname) { lbRegistry.addCompiletimeBalancer(lbname); }

// default is to show the helper
void LBRegisterBalancer(std::string name, LBCreateFn fn, LBAllocFn afn, std::string help,
                        bool shown)
{
  lbRegistry.addEntry(name, fn, afn, help, shown);
}

LBAllocFn getLBAllocFn(const char* lbname) { return lbRegistry.getLBAllocFn(lbname); }

bool LBHasBalancersRegistered()
{
  return lbRegistry.hasBalancers();
}

// create a load balancer group using the strategy name
static void createLoadBalancer(const std::string& lbname, const char* legacybalancer = nullptr)
{
  LBCreateFn fn = lbRegistry.search(lbname);
  if (!fn)
  {  // invalid lb name
    CmiPrintf("Abort: Unknown load balancer: '%s'!\n", lbname.c_str());
    lbRegistry.displayLBs();  // display help page
    if(lbname == "help")
      CkExit(0);
    else
      CkExit(1);
  }
  // invoke function to create load balancer
  int seqno = LBManagerObj()->getLoadbalancerTicket();
  fn(CkLBOptions(seqno, legacybalancer));
}

// mainchare
LBMgrInit::LBMgrInit(CkArgMsg* m)
{
#if CMK_LBDB_ON
  _lbmgr = CProxy_LBManager::ckNew();

  // runtime specified load balancer
  if (!lbRegistry.runtime_lbs.empty())
  {
    for (int i = 0; i < lbRegistry.runtime_lbs.size(); i++)
    {
      // If this is a legacy TreeLB, pass in the legacy LB name
      const char* legacybalancer = lbRegistry.legacy_runtime_treelbs.count(i) > 0
                                       ? lbRegistry.legacy_runtime_treelbs[i]
                                       : nullptr;
      createLoadBalancer(lbRegistry.runtime_lbs[i], legacybalancer);
    }
  }
  else if (!lbRegistry.compile_lbs.empty())
  {
    for (const auto& balancer : lbRegistry.compile_lbs)
    {
      createLoadBalancer(balancer);
    }
  }

  // simulation mode
  if (LBSimulation::doSimulation)
  {
    CmiPrintf("Charm++> Entering Load Balancer Simulation Mode ... \n");
    CProxy_LBManager(_lbmgr).ckLocalBranch()->StartLB();
  }
#endif
  delete m;
}

// called from init.C
/** How far ahead of its own iteration PE 0 opens the bidding. Set by
 *  +RescaleLead.
 *
 *  This is no longer a correctness margin -- safety comes from every PE
 *  installing its ceiling before reporting it, so the agreed iteration cannot
 *  be one any element has passed. It is a stall knob. A larger value puts the
 *  tentative ceiling further out, so elements keep working through the
 *  consensus round trip instead of holding immediately; zero is legal and
 *  simply means "hold at once". */
static int _rescaleLead = 3;

/** Fault-injection for the rescale consensus, off unless both are set.
 *
 * Delays the processing of RescaleTentative on one PE, deterministically
 * widening the race the protocol exists to survive: the delayed PE's elements
 * run on past the boundary where everyone else armed, so its reported ceiling
 * comes back higher than theirs and the commit must rewind their parked
 * elements. Test-only. */
static int _rescaleTestDelayPe = -1;
static int _rescaleTestDelayMs = 0;

/** Whether the round now heading for CheckForRealloc was started barrier-less
 *  (no declared boundaries; application still running). Read by CentralLB to
 *  decide whether the checkpoint needs a grace window first. PE 0 only. */
bool _rescaleBarrierlessRound = false;
/** Whether the application opted into barrier-less rescaling
 *  (+rescalebarrierless). Off by default: cutting an application at an
 *  arbitrary point obliges it to tolerate message reordering across the cut
 *  (a blind receive counter miscounts), which the runtime cannot verify --
 *  the flag is the application asserting that property. It also protects a
 *  boundary application from a request that arrives before its first element
 *  has declared a boundary, which would otherwise read as "no boundaries" and
 *  take the barrier-less cut. Set-once: a survivor's re-parse after the
 *  longjmp sees an argv the first pass already consumed. */
static bool _rescaleBarrierlessEnabled = false;
// Boundary-mode early release: after the rescale LB decision at an AtSync
// boundary, resume every chare before the cut (migrants resume on their
// destination) instead of holding the whole application through the drain
// and cut.
//
// Early release buys the application the drain -- a few milliseconds -- and
// costs it the guarantee that nothing is in flight when the transport is cut.
// A resumed chare can start new traffic toward a PE that is leaving: measured
// as roughly two failures in five on a four-rescale campaign, in a Charm++
// application as much as an AMPI one, appearing as a lost point-to-point
// message, a group send to a departed PE ("Destnode N out of range N"), or a
// job whose ranks simply exit. Holding instead is free by comparison: the
// stop window is if anything shorter, since it skips the drain.
//
// So: off by default, but a job with AtSync-only migratable objects turns it
// on for itself (see TCharm::procInit -- an AMPI rank cannot be moved anywhere
// but a quiescent point, which makes the early release's premise false for it).
// +rescaleholdboundary forces it on, +rescaleearlyrelease forces it off.
bool _rescaleHoldBoundary = false;
// Whether the setting above came from the command line. Without this, a
// library that prefers holding could not tell "the user did not say" from
// "the user said no".
bool _rescaleHoldBoundaryExplicit = false;
// Read-only accessor for other translation units (the flag itself stays
// file-local so the parse in _loadbalancerInit remains the single writer).
bool CkRescaleBarrierlessEnabled() { return _rescaleBarrierlessEnabled; }
/** Grace between evacuation and the cut on the barrier-less path, in ms.
 *
 * The application never stops on that path, so at MigrationDone the doomed
 * PEs still hold in-flight state their evacuated elements left behind --
 * queued ghosts to forward, reduction partials to flush upward -- and cutting
 * immediately loses it (observed as a post-rescale wedge: contributions
 * absorbed by a doomed PE's reduction manager died with it). The grace lets
 * that traffic land while everything keeps running: ghosts forward to the
 * elements' new homes, partials flow up, location updates stop new traffic
 * from targeting the leavers. A boundary rescale needs none of this -- the
 * app is quiescent at the cut, which is the boundary path's whole point. */
int _rescaleGraceMs = 100;

void _loadbalancerInit()
{
  CkpvInitialize(bool, lbmanagerInited);
  CkpvInitialize(LBUserDataLayout, lbobjdatalayout);
  CkpvInitialize(int, _lb_obj_index);
#if CMK_SHRINK_EXPAND
  // On a shrink/expand survivor restart the LBManager group instance is
  // preserved across the longjmp, so its constructor (which sets
  // lbmanagerInited=true) does not run again. Resetting these here would
  // leave LBManager::Object() permanently NULL and BuildStatsMsg would trap
  // on the next AtSync. Only initialize on a fresh start; survivors keep
  // their pre-rescale state.
  extern bool _reuseRegistrationStateOnRestart;
  if (!_reuseRegistrationStateOnRestart) {
    CkpvAccess(lbmanagerInited) = false;
    CkpvAccess(_lb_obj_index) = -1;
  }
#else
  CkpvAccess(lbmanagerInited) = false;
  CkpvAccess(_lb_obj_index) = -1;
#endif

  char** argv = CkGetArgv();
  char* balancer = NULL;
  CmiArgGroup("Charm++", "Load Balancer");

  CmiGetArgStringDesc(argv, "+TreeLBFile", &_lb_args.treeLBFile(), "TreeLB config file");

  // turn on MetaBalancer if set
  _lb_args.metaLbOn() = CmiGetArgFlagDesc(argv, "+MetaLB", "Turn on MetaBalancer");
  CmiGetArgStringDesc(argv, "+MetaLBModelDir", &_lb_args.metaLbModelDir(),
                      "Use this directory to read model for MetaLB");

  if (_lb_args.metaLbOn() && _lb_args.metaLbModelDir() != nullptr)
  {
#if CMK_USE_ZLIB
    if (CkMyRank() == 0)
    {
      lbRegistry.addRuntimeBalancer("TreeLB");
      lbNames.push_back("Greedy");
      lbNames.push_back("GreedyRefine");
      lbNames.push_back("DistributedLB");
      lbNames.push_back("Refine");
      lbNames.push_back("Hybrid");
      lbNames.push_back("MetisLB");
      lbNames.push_back("GreedyCentralLB");
      lbNames.push_back("GreedyRefineCentralLB");
      if (CkMyPe() == 0)
      {
        if (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
          CkPrintf(
              "Warning: Ignoring the +balancer option, since Meta-Balancer's model-based "
              "load balancer selection is enabled.\n");
        CkPrintf(
            "Warning: Automatic strategy selection in MetaLB is activated. This is an "
            "experimental feature.\n");
      }
      while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
        ;
    }
#else
    if (CkMyPe() == 0)
      CkAbort(
          "MetaLB random forest model not supported because Charm++ was built without "
          "zlib support.\n");
#endif
  }
  else
  {
    if (CkMyPe() == 0 && _lb_args.metaLbOn())
      CkPrintf(
          "Warning: MetaLB is activated. For Automatic strategy selection in MetaLB, "
          "pass directory of model files using +MetaLBModelDir.\n");
    if (CkMyRank() == 0)
    {
      while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
      {
        bool isLegacyTreeLB = true;
        const char* legacyBalancer;
        if (strcmp(balancer, "GreedyLB") == 0)
          legacyBalancer = "Greedy";
        else if (strcmp(balancer, "GreedyRefineLB") == 0)
          legacyBalancer = "GreedyRefine";
        else if (strcmp(balancer, "RefineLB") == 0)
          legacyBalancer = "RefineA";
        else if (strcmp(balancer, "RandCentLB") == 0)
          legacyBalancer = "Random";
        else if (strcmp(balancer, "DummyLB") == 0)
          legacyBalancer = "Dummy";
        else if (strcmp(balancer, "RotateLB") == 0)
          legacyBalancer = "Rotate";
        else
        {
          lbRegistry.addRuntimeBalancer(balancer); /* lbRegistry is a static */
          isLegacyTreeLB = false;
        }

        if (isLegacyTreeLB)
        {
          lbRegistry.addRuntimeBalancer("TreeLB", legacyBalancer);
        }
      }
    }
    else
    {
      // For other ranks, consume the +balancer arguments to avoid spuriously
      // passing them to the application
      while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
        ;
    }
    CmiNodeBarrier();
  }

  CmiGetArgDoubleDesc(
      argv, "+DistLBTargetRatio", &_lb_args.targetRatio(),
      "The max/avg load ratio that DistributedLB will attempt to achieve");
  CmiGetArgIntDesc(argv, "+DistLBMaxPhases", &_lb_args.maxDistPhases(),
                   "The maximum number of phases that DistributedLB will attempt");

  // Migration ceiling honored by GreedyRefineCentralLB (and any future LB
  // that consults _lb_args.percentMovesAllowed()). 0..100; default 100.
  // Treated as a soft target: if no parameter-sweep solution fits the cap,
  // GreedyRefine falls back to the lowest-migration solution instead.
  CmiGetArgIntDesc(argv, "+LBPercentMoves", &_lb_args.percentMovesAllowed(),
                   "Soft cap (0..100) on percent of chares that may migrate "
                   "per LB step; honored by GreedyRefineCentralLB.");

  // set up init value for LBPeriod time in seconds
  // it can also be set by calling LBSetPeriod/LBManager::SetLBPeriod
  CmiGetArgDoubleDesc(argv, "+LBPeriod", &_lb_args.lbperiod(),
                      "the minimum time period in seconds allowed for two consecutive "
                      "automatic load balancing");
  _lb_args.loop() = CmiGetArgFlagDesc(argv, "+LBLoop",
                                      "Use multiple load balancing strategies in loop");

  // now called in cldb.C: CldModuleGeneralInit()
  // registerLBTopos();
  CmiGetArgStringDesc(argv, "+LBTopo", &_lbtopo, "define load balancing topology");

  /**************** FUTURE PREDICTOR ****************/
  _lb_predict = CmiGetArgFlagDesc(argv, "+LBPredictor", "Turn on LB future predictor");
  CmiGetArgIntDesc(argv, "+LBPredictorDelay", &_lb_predict_delay,
                   "Number of balance steps before learning a model");
  CmiGetArgIntDesc(argv, "+LBPredictorWindow", &_lb_predict_window,
                   "Number of steps to use to learn a model");
  if (_lb_predict_window < _lb_predict_delay)
  {
    CmiPrintf(
        "LB> [%d] Argument LBPredictorWindow (%d) less than LBPredictorDelay (%d) , "
        "fixing\n",
        CkMyPe(), _lb_predict_window, _lb_predict_delay);
    _lb_predict_delay = _lb_predict_window;
  }

  /******************* SIMULATION *******************/
  // get the step number at which to dump the LB database
  CmiGetArgIntDesc(argv, "+LBVersion", &_lb_args.lbversion(),
                   "LB database file version number");
  CmiGetArgIntDesc(argv, "+LBCentPE", &_lb_args.central_pe(), "CentralLB processor");
  CmiGetArgIntDesc(argv, "+LBPercentMovesAllowed", &_lb_args.percentMovesAllowed(),
                   "For GreedyRefineCentralLB, the percentage of chares that can be moved");
  bool _lb_dump_activated = false;
  if (CmiGetArgIntDesc(argv, "+LBDump", &LBSimulation::dumpStep,
                       "Dump the LB state from this step"))
    _lb_dump_activated = true;
  if (_lb_dump_activated && LBSimulation::dumpStep < 0)
  {
    CmiPrintf("LB> Argument LBDump (%d) negative, setting to 0\n",
              LBSimulation::dumpStep);
    LBSimulation::dumpStep = 0;
  }
  CmiGetArgIntDesc(argv, "+LBDumpSteps", &LBSimulation::dumpStepSize,
                   "Dump the LB state for this amount of steps");
  if (LBSimulation::dumpStepSize <= 0)
  {
    CmiPrintf("LB> Argument LBDumpSteps (%d) too small, setting to 1\n",
              LBSimulation::dumpStepSize);
    LBSimulation::dumpStepSize = 1;
  }
  CmiGetArgStringDesc(argv, "+LBDumpFile", &LBSimulation::dumpFile,
                      "Set the LB state file name");
  // get the simulation flag and number. Now the flag can also be avoided by the presence
  // of the number
  LBSimulation::doSimulation =
      CmiGetArgIntDesc(argv, "+LBSim", &LBSimulation::simStep,
                       "Read LB state from LBDumpFile since this step");
  // check for stupid LBSim parameter
  if (LBSimulation::doSimulation && LBSimulation::simStep < 0)
  {
    CkAbort("LB> Argument LBSim (%d) invalid, should be >= 0\n", LBSimulation::simStep);
    return;
  }
  CmiGetArgIntDesc(argv, "+LBSimSteps", &LBSimulation::simStepSize,
                   "Read LB state for this number of steps");
  if (LBSimulation::simStepSize <= 0)
  {
    CmiPrintf("LB> Argument LBSimSteps (%d) too small, setting to 1\n",
              LBSimulation::simStepSize);
    LBSimulation::simStepSize = 1;
  }

  LBSimulation::simProcs = 0;
  CmiGetArgIntDesc(argv, "+LBSimProcs", &LBSimulation::simProcs,
                   "Number of target processors.");

  LBSimulation::showDecisionsOnly =
      CmiGetArgFlagDesc(argv, "+LBShowDecisions",
                        "Write to File: Load Balancing Object to Processor Map decisions "
                        "during LB Simulation");

  // force a global barrier after migration done
  _lb_args.syncResume() = CmiGetArgFlagDesc(
      argv, "+LBSyncResume", "LB performs a barrier after migration is finished");

  // both +LBDebug and +LBDebug level should work
  if (!CmiGetArgIntDesc(argv, "+LBDebug", &_lb_args.debug(),
                        "Turn on LB debugging printouts"))
    _lb_args.debug() =
        CmiGetArgFlagDesc(argv, "+LBDebug", "Turn on LB debugging printouts");

  // ask to print summary/quality of load balancer
  _lb_args.printSummary() =
      CmiGetArgFlagDesc(argv, "+LBPrintSummary", "Print load balancing result summary");

  // to ignore baclground load
  _lb_args.ignoreBgLoad() = CmiGetArgFlagDesc(
      argv, "+LBNoBackground", "Load balancer ignores the background load.");
  _lb_args.migObjOnly() = CmiGetArgFlagDesc(
      argv, "+LBObjOnly", "Only load balancing migratable objects, ignoring all others.");
  if (_lb_args.migObjOnly()) _lb_args.ignoreBgLoad() = true;

  // assume all CPUs are identical
  _lb_args.testPeSpeed() =
      CmiGetArgFlagDesc(argv, "+LBTestPESpeed", "Load balancer test all CPUs speed.");
  _lb_args.samePeSpeed() = CmiGetArgFlagDesc(
      argv, "+LBSameCpus", "Load balancer assumes all CPUs are of same speed.");
  if (!_lb_args.testPeSpeed()) _lb_args.samePeSpeed() = true;

  _lb_args.useCpuTime() = CmiGetArgFlagDesc(
      argv, "+LBUseCpuTime", "Load balancer uses CPU time instead of wallclock time.");

  // turn instrumentation off at startup
  _lb_args.statsOn() =
      !CmiGetArgFlagDesc(argv, "+LBOff", "Turn load balancer instrumentation off");

  // turn instrumentation of communication on at startup
  _lb_args.traceComm() = CmiGetArgFlagDesc(
    argv, "+LBCommOn", "Turn load balancer instrumentation of communication on");

  // +LBCommOff is deprecated as instrumentation of communication is off by default
  bool lbcommOff = CmiGetArgFlagDesc(
      argv, "+LBCommOff", "(No-op) Turn load balancer instrumentation of communication off");
  if(CkMyPe()==0 && lbcommOff)
    CmiPrintf("Warning: Ignoring the deprecated +LBCommOff option as communication is off by default.\n");

  // set alpha and beta
  _lb_args.alpha() = PER_MESSAGE_SEND_OVERHEAD_DEFAULT;
  _lb_args.beta() = PER_BYTE_SEND_OVERHEAD_DEFAULT;
  CmiGetArgDoubleDesc(argv, "+LBAlpha", &_lb_args.alpha(), "per message send overhead");
  CmiGetArgDoubleDesc(argv, "+LBBeta", &_lb_args.beta(), "per byte send overhead");

  // Widths to predict this job's speedup at, for an elastic scheduler deciding
  // how many workers to give it. Drained in a loop, and parsed here whether or
  // not anything consumes the value: an option the runtime recognises but does
  // not remove from argv shifts every positional argument the application
  // reads, which surfaces as nonsense inside the application rather than as a
  // complaint from the runtime.
  {
    char* simRange = NULL;
    while (CmiGetArgStringDesc(argv, "+LBSimRange", &simRange,
                               "Predict speedup over this width range, min:max"))
    {
      int simLo = 0, simHi = 0;
      if (simRange != NULL && sscanf(simRange, "%d:%d", &simLo, &simHi) == 2 && simLo >= 1 &&
          simHi >= simLo)
        CkTelemetrySetSimRange(simLo, simHi);
      else if (CkMyPe() == 0)
        CmiPrintf("Warning: ignoring malformed +LBSimRange '%s'; expected min:max.\n",
                  simRange != NULL ? simRange : "");
    }
  }

  // How far ahead PE 0 opens the bidding for the rescale iteration. Parsed
  // into the static directly rather than through a local: a survivor re-runs
  // this after the rescale longjmp with an argv the first pass already
  // stripped, and a local would reset the value to the default every time.
  CmiGetArgIntDesc(argv, "+RescaleLead", &_rescaleLead,
                   "Iterations to bid ahead when scheduling a rescale at an "
                   "application-declared boundary");
  if (_rescaleLead < 0) _rescaleLead = 0;

  CmiGetArgIntDesc(argv, "+RescaleTestDelayPe", &_rescaleTestDelayPe,
                   "TEST ONLY: PE whose RescaleTentative processing is delayed");
  CmiGetArgIntDesc(argv, "+RescaleTestDelayMs", &_rescaleTestDelayMs,
                   "TEST ONLY: milliseconds to delay it by");
  CmiGetArgIntDesc(argv, "+RescaleGraceMs", &_rescaleGraceMs,
                   "Grace between evacuation and the cut on a barrier-less rescale");
  if (CmiGetArgFlagDesc(argv, "+rescalebarrierless",
                        "Allow rescaling applications with no declared "
                        "iteration boundaries by cutting at an arbitrary "
                        "point (requires the app to tolerate message "
                        "reordering across the cut)"))
    _rescaleBarrierlessEnabled = true;

  if (CmiGetArgFlagDesc(argv, "+rescaleholdboundary",
                        "Hold every chare at the rescale boundary through "
                        "migration and the cut, so that nothing is in flight "
                        "when the transport is cut"))
  {
    _rescaleHoldBoundary = true;
    _rescaleHoldBoundaryExplicit = true;
  }
  if (CmiGetArgFlagDesc(argv, "+rescaleearlyrelease",
                        "Resume chares before the drain and cut, overriding a "
                        "library that asked to hold. Faster by the length of "
                        "the drain, at the cost of losing messages that are "
                        "in flight when the transport is cut"))
  {
    _rescaleHoldBoundary = false;
    _rescaleHoldBoundaryExplicit = true;
  }

  CkRescalePointInit();

  if (CkMyPe() == 0)
  {
    if (_lb_args.debug())
    {
      CmiPrintf("CharmLB> Verbose level %d, load balancing period: %g seconds\n",
                _lb_args.debug(), _lb_args.lbperiod());
    }
    if (_lb_args.debug() > 1)
    {
      CmiPrintf("CharmLB> Topology %s alpha: %es beta: %es.\n", _lbtopo, _lb_args.alpha(),
                _lb_args.beta());
    }
    if (_lb_args.printSummary())
      CmiPrintf("CharmLB> Load balancer print summary of load balancing result.\n");
    if (_lb_args.ignoreBgLoad())
      CmiPrintf("CharmLB> Load balancer ignores processor background load.\n");
    if (_lb_args.samePeSpeed())
      CmiPrintf("CharmLB> Load balancer assumes all CPUs are same.\n");
    if (_lb_args.useCpuTime())
      CmiPrintf("CharmLB> Load balancer uses CPU time instead of wallclock time.\n");
    if (LBSimulation::doSimulation)
      CmiPrintf(
          "CharmLB> Load balancer running in simulation mode on file '%s' version %d.\n",
          LBSimulation::dumpFile, _lb_args.lbversion());
    if (_lb_args.statsOn() == 0)
      CkPrintf("CharmLB> Load balancing instrumentation is off.\n");
    if (_lb_args.migObjOnly())
      CkPrintf("LB> Load balancing strategy ignores non-migratable objects.\n");
  }
}

bool LBManager::manualOn = false;
std::vector<char> LBManager::avail_vector;
bool LBManager::avail_vector_set = false;
CmiNodeLock avail_vector_lock;

static LBRealType* _expectedLoad = NULL;

void LBManager::initnodeFn()
{
  int proc;
  int num_proc = CkNumPes();
  avail_vector.clear();
  avail_vector.resize(num_proc, 1);
  avail_vector_lock = CmiCreateLock();

  _expectedLoad = new LBRealType[num_proc];
  for (proc = 0; proc < num_proc; proc++) _expectedLoad[proc] = 0.0;

  _registerCommandLineOpt("+balancer");
  _registerCommandLineOpt("+LBPeriod");
  _registerCommandLineOpt("+TreeLBFile");
  _registerCommandLineOpt("+LBLoop");
  _registerCommandLineOpt("+LBTopo");
  _registerCommandLineOpt("+LBPredictor");
  _registerCommandLineOpt("+LBPredictorDelay");
  _registerCommandLineOpt("+LBPredictorWindow");
  _registerCommandLineOpt("+LBVersion");
  _registerCommandLineOpt("+LBCentPE");
  _registerCommandLineOpt("+LBDump");
  _registerCommandLineOpt("+LBDumpSteps");
  _registerCommandLineOpt("+LBDumpFile");
  _registerCommandLineOpt("+LBSim");
  _registerCommandLineOpt("+LBSimSteps");
  _registerCommandLineOpt("+LBSimProcs");
  _registerCommandLineOpt("+LBShowDecisions");
  _registerCommandLineOpt("+LBSyncResume");
  _registerCommandLineOpt("+LBDebug");
  _registerCommandLineOpt("+LBPrintSummary");
  _registerCommandLineOpt("+LBNoBackground");
  _registerCommandLineOpt("+LBObjOnly");
  _registerCommandLineOpt("+LBTestPESpeed");
  _registerCommandLineOpt("+LBSameCpus");
  _registerCommandLineOpt("+LBUseCpuTime");
  _registerCommandLineOpt("+LBOff");
  _registerCommandLineOpt("+LBCommOn");
  _registerCommandLineOpt("+LBCommOff");
  _registerCommandLineOpt("+MetaLB");
  _registerCommandLineOpt("+LBAlpha");
  _registerCommandLineOpt("+LBBeta");
}

void LBManager::InvokeLB()
{
  // A barrier round has begun, so this PE's offer of a boundary has been
  // taken up. Disarm before the round decides anything. If it turns out not to
  // be a rescale after all -- an ordinary balancing step reached the barrier
  // first and consumed the pending request, say -- a PE left armed would offer
  // every subsequent iteration, and the job would balance on all of them.
  CkRescaleDisarm();

  if (loadbalancers.size() > 0)
  {
    loadbalancers[currentLBIndex]->InvokeLB();
  }
  else
  {
    ResumeClients();
  }
}

// Called at end of each load balancing cycle
void LBManager::periodicLB(void* in)
{
  auto* const manager = static_cast<LBManager*>(in);
  manager->isPeriodicQueued = false;
  manager->InvokeLB();
}

void LBManager::setTimer()
{
  if (!isPeriodicQueued)
  {
    isPeriodicQueued = true;
    CcdCallFnAfterOnPE((CcdVoidFn)periodicLB, (void*)this, 1000 * _lb_args.lbperiod(),
                       CkMyPe());
  }
}

// called my constructor
void LBManager::init(void)
{
  mystep = 0;
  new_ld_balancer = 0;
  lb_in_progress = false;
  chare_count = 0;
  metabalancer = nullptr;
  lbdb_obj = new LBDatabase();
  currentLBIndex = 0;
  reallocQueue.clear();
#if CMK_LB_CPUTIMER
  obj_cputime = 0;
#endif
  useBarrier = true;
  predictCBFn = nullptr;
  startLBFn_count = 0;

  CkpvAccess(lbmanagerInited) = true;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif
  if (CkMyPe()==0 && _lb_args.traceComm() == 0 && !quietModeRequested)
      CkPrintf("CharmLB> Load balancing instrumentation for communication is off.\n");

  if (_lb_args.lbperiod() > 0.0)
  {
    setTimer();
  }
  else
  {
    CkSyncBarrier::object()->addReceiver([this](void) { this->InvokeLB(); });
  }
}

/* Application-declared rescale points. See ../ck-ldb/rescalepoint.h. */

#if CMK_SHRINK_EXPAND
/* Opt-in trace of the rescale-point consensus (CHARM_RESCALE_TRACE=1). Cached,
   so the cost when it is off is a load and a branch. */
static bool CkRescaleTraceOn() {
  static int on = -1;
  if (on < 0) on = (getenv("CHARM_RESCALE_TRACE") != NULL) ? 1 : 0;
  return on != 0;
}
#else
static bool CkRescaleTraceOn() { return false; }
#endif

void LBManager::ArmRescalePoint()
{
  if (CkMyPe() != 0) return;

  // A request replayed by callRealloc() during a survivor restore arrives here
  // while this PE's peers are still rebuilding, so a broadcast issued now is
  // only buffered and a reduction started now races the restore. Defer to the
  // end of it, after the buffered-message drain has run.
  if (get_in_restart())
  {
    rescaleArmDeferred = true;
    return;
  }

  // The two mechanisms are exclusive by construction: with
  // +rescalebarrierless the boundary consensus never starts (no tentative is
  // broadcast, so no ceiling is ever installed and checkRescale() stays
  // false), and without it the barrier-less round never starts. Running both
  // for one request would race -- two paths consuming one
  // pending_realloc_state.
  if (_rescaleBarrierlessEnabled)
  {
    if (startLBFn_count > 0)
    {
      if (_lb_args.debug())
        CkPrintf("CharmLB> Barrier-less rescale: immediate LB round.\n");
      // The manual path skips InvokeLB, so raise the in-progress flag here or
      // a second request arriving mid-round would not be buffered.
      lb_in_progress = true;
      _rescaleBarrierlessRound = true;
      StartLB();
    }
    else
      CkPrintf("CharmLB> Warning: rescale requested but no load balancer can "
               "run a round; request will wait for one.\n");
    return;
  }

  // Only an opening bid. Every PE raises it to at least its own next boundary
  // in RescaleTentative, so PE 0 having no elements of its own, or lagging
  // behind the rest, costs nothing but a slightly longer hold.
  const int seen = CkRescaleIterSeen();
  int tentative;
  if (seen < 0)
    tentative = CK_RESCALE_ARM_NOW;
  else if (seen > 2147483647 - _rescaleLead)
    tentative = CK_RESCALE_ARM_NOW;
  else
    tentative = seen + _rescaleLead;

  extern int _rescaleGeneration;
  if (CkRescaleTraceOn())
    CmiPrintf("[%d] ArmRescalePoint: tentative=%d gen=%d\n", CkMyPe(),
              tentative, _rescaleGeneration);
  thisProxy.RescaleTentative(tentative, _rescaleGeneration);
}

void LBManager::ArmRescalePointIfDeferred()
{
  if (CkMyPe() != 0 || !rescaleArmDeferred) return;
  rescaleArmDeferred = false;
  ArmRescalePoint();
}

void LBManager::RescaleTentative(int tentative, int gen)
{
  extern int _rescaleGeneration;
  if (CkRescaleTraceOn())
    CmiPrintf("[%d] RescaleTentative(t=%d, gen=%d) myGen=%d %s\n", CkMyPe(),
              tentative, gen, _rescaleGeneration,
              gen != _rescaleGeneration ? "DROPPED" : "");
  // A stamp from a dead world. The rescale that killed it already reset every
  // PE's arming state; acting on its stragglers would re-arm a job that no
  // longer has a request pending.
  if (gen != _rescaleGeneration) return;

  if (_rescaleTestDelayMs > 0 && CkMyPe() == _rescaleTestDelayPe)
  {
    // Deterministically lose the race on this PE: its elements process their
    // next boundary before the arming lands, exactly the interleaving the
    // consensus has to survive.
    struct Delayed
    {
      LBManager* mgr;
      int tentative, gen;
      static void fire(void* p)
      {
        Delayed* d = (Delayed*)p;
        d->mgr->RescaleTentativeNow(d->tentative, d->gen);
        delete d;
      }
    };
    CcdCallFnAfterOnPE((CcdVoidFn)Delayed::fire, new Delayed{this, tentative, gen},
                       _rescaleTestDelayMs, CkMyPe());
    return;
  }
  RescaleTentativeNow(tentative, gen);
}

void LBManager::RescaleTentativeNow(int tentative, int gen)
{
  extern int _rescaleGeneration;
  if (gen != _rescaleGeneration) return;

  // Ceiling first, report second. Both happen inside this entry method, which
  // is not preemptible, so no element can advance between the read of iterSeen
  // and the arm -- and that is exactly what makes the reduced maximum a bound
  // nobody has passed. iterSeen is the last *completed* iteration, so the
  // furthest element here is already executing iterSeen+1 and will next stand
  // at that boundary; that is the earliest ceiling that can still catch it.
  if (CkRescaleTraceOn())
    CmiPrintf("[%d] RescaleTentativeNow(t=%d) iterSeen=%d\n", CkMyPe(),
              tentative, CkRescaleIterSeen());
  const int seen = CkRescaleIterSeen();
  int local = tentative;
  if (seen >= 0 && seen + 1 > local) local = seen + 1;
  CkRescaleArmAt(local);

  int v[3];
  v[0] = CkRescaleUsesBoundaries() ? 1 : 0;
  v[1] = local;
  v[2] = gen;
  contribute(3 * sizeof(int), v, CkReduction::max_int,
             CkCallback(CkReductionTarget(LBManager, RescaleFinal), thisProxy[0]));
}

void LBManager::RescaleFinal(int n, int* v)
{
  if (n < 3) return;
  extern int _rescaleGeneration;
  if (CkRescaleTraceOn())
    CmiPrintf("[%d] RescaleFinal(v0=%d v1=%d gen=%d) myGen=%d\n", CkMyPe(),
              v[0], v[1], v[2], _rescaleGeneration);
  if (v[2] != _rescaleGeneration) return;
  if (v[0] == 0)
  {
    // No PE has ever seen a declared boundary: the job has no iteration
    // boundaries to rescale at, and for an app that never calls AtSync no
    // ordinary load balancing step is coming either. Lift the ceilings that
    // were just installed (which also reopens the barrier on every PE), then
    // start a load balancing round directly, with the application running.
    //
    // The boundary consensus stays the preferred path whenever any element
    // declares boundaries, because entering at a boundary is also what makes
    // the collected loads describe whole iterations -- the strategy that
    // balances the shrunken world reads them. This round's stats instead
    // describe whatever window the instrumentation happened to cover.
    //
    // The round needs no barrier: CentralLB::StartLB goes straight to
    // ProcessAtSync, stats and migration work on running elements exactly as
    // manual load balancing always has, and MigrationDone contributes to
    // CheckForRealloc, which the pending rescale request then rides. Every PE
    // enters the rescale exit flow by processing a message, so elements are
    // between entry methods at the cut by non-preemptivity; survivors' queued
    // and in-flight messages are the preserved/buffered population the
    // restore machinery re-delivers.
    thisProxy.RescaleCommit(2147483647, v[2]);
    // Only reachable with +rescalebarrierless off (the flag skips the
    // consensus entirely in ArmRescalePoint). Keep the pre-existing behavior
    // of waiting for a boundary or an ordinary load balancing step -- and say
    // so, because for an application that never declares boundaries and never
    // calls AtSync, that wait is indefinite.
    CkPrintf("CharmLB> Warning: rescale requested but the application has "
             "declared no iteration boundaries; waiting for the next load "
             "balancing step. If the application has none, run with "
             "+rescalebarrierless to allow an arbitrary-point rescale.\n");
    return;
  }
  if (_lb_args.debug())
    CkPrintf("CharmLB> Rescale agreed for iteration %d.\n", v[1]);
  thisProxy.RescaleCommit(v[1], v[2]);
}

void LBManager::RescaleAnnounceDoom(std::vector<char> bitmap)
{
  // See the header. new_ld=0 mirrors what PE 0's realloc() does locally.
  set_avail_vector(bitmap.data(), 0);
}

void LBManager::RescaleCommit(int final, int gen)
{
  extern int _rescaleGeneration;
  if (CkRescaleTraceOn())
    CmiPrintf("[%d] RescaleCommit(final=%d, gen=%d) myGen=%d\n", CkMyPe(),
              final, gen, _rescaleGeneration);
  if (gen != _rescaleGeneration) return;
  CkRescaleCommit(final);
}

/** Called from the restore path once buffered messages have been drained, to
 *  run an arming that had to be deferred. Free function so ckcheckpoint.C need
 *  not include this header. */
void CkArmDeferredRescalePoint(void)
{
  LBManager* mgr = LBManager::Object();
  if (mgr != NULL) mgr->ArmRescalePointIfDeferred();
}

int LBManager::AddStartLBFn(std::function<void()> fn)
{
  // Save startLB function
  StartLBCB* callbk = new StartLBCB;

  callbk->fn = fn;
  callbk->on = true;
  CkPrintf("Registering StartLB function %p\n", (void*)callbk);
  startLBFnList.push_back(callbk);
  startLBFn_count++;
  return startLBFnList.size() - 1;
}

void LBManager::RemoveStartLBFn(int handle)
{
  StartLBCB* callbk = startLBFnList[handle];
  if (callbk)
  {
    delete callbk;
    startLBFnList[handle] = nullptr;
    startLBFn_count--;
  }
}

void LBManager::StartLB()
{
  CkPrintf("Start LB called, count %d\n", startLBFn_count);
  if (startLBFn_count == 0)
  {
    CmiAbort("StartLB is not supported in this LB");
  }
  for (int i = 0; i < startLBFnList.size(); i++)
  {
    StartLBCB* startLBFn = startLBFnList[i];
    CkPrintf("StartLB checking function %d: %p, %d\n", i, (void*)startLBFn, startLBFn->on);
    if (startLBFn && startLBFn->on) 
    {
      CkPrintf("Invoking StartLB function %p\n", (void*)&startLBFn->fn);
      startLBFn->fn();
    }
  }
}

int LBManager::AddMigrationDoneFn(std::function<void()> fn)
{
  // Save migrationDone callback function
  MigrationDoneCB* callbk = new MigrationDoneCB;

  callbk->fn = fn;
  migrationDoneCBList.push_back(callbk);
  return migrationDoneCBList.size() - 1;
}

void LBManager::RemoveMigrationDoneFn(int handle)
{
  MigrationDoneCB* callbk = migrationDoneCBList[handle];
  if (callbk)
  {
    delete callbk;
    migrationDoneCBList[handle] = nullptr;
  }
}

void LBManager::MigrationDone()
{
  for (int i = 0; i < migrationDoneCBList.size(); i++)
  {
    MigrationDoneCB* callbk = migrationDoneCBList[i];
    if (callbk) callbk->fn();
  }
}

void LBManager::DumpDatabase()
{
#ifdef DEBUG
  CmiPrintf("Database contains %d object managers\n", omCount);
  CmiPrintf("Database contains %d objects\n", objs.size());
#endif
}

void LBManager::Migrated(LDObjHandle h, int waitBarrier)
{
  // Object migrated, inform load balancers
  if (loadbalancers.size() > 0) loadbalancers[currentLBIndex]->Migrated(waitBarrier);
}

LBManager::LastLBInfo::LastLBInfo() { expectedLoad = _expectedLoad; }

void LBManager::get_avail_vector(char* bitmap) const
{
  CmiAssert(bitmap);
  const int num_proc = CkNumPes();
  CmiAssert(num_proc <= avail_vector.size());
  std::copy(avail_vector.begin(), avail_vector.begin() + num_proc, bitmap);
}

// new_ld == -1(default) : calcualte a new ld
//           -2 : ignore new ld
//           >=0: given a new ld
void LBManager::set_avail_vector(const char* bitmap, int new_ld)
{
  int assigned = 0;
  const int num_proc = CkNumPes();
  if (new_ld == -2)
    assigned = 1;
  else if (new_ld >= 0)
  {
    CmiAssert(new_ld < num_proc);
    new_ld_balancer = new_ld;
    assigned = 1;
  }
  CmiAssert(bitmap);
  CmiAssert(num_proc <= avail_vector.size());
  for (int count = 0; count < num_proc; count++)
  {
    avail_vector[count] = bitmap[count];
    if ((bitmap[count] == 1) && !assigned)
    {
      new_ld_balancer = count;
      assigned = 1;
    }
  }
}
void LBManager::set_avail_vector(const std::vector<char> & bitmap, int new_ld)
{
  int assigned = 0;
  const int num_proc = CkNumPes();
  if (new_ld == -2)
    assigned = 1;
  else if (new_ld >= 0)
  {
    CmiAssert(new_ld < num_proc);
    new_ld_balancer = new_ld;
    assigned = 1;
  }
  avail_vector = bitmap;
  for (int count = 0; count < num_proc; count++)
  {
    if (bitmap[count] == 1 && !assigned)
    {
      new_ld_balancer = count;
      assigned = 1;
    }
  }
}

// called in CreateFooLB() when multiple load balancers are created
// on PE0, BaseLB of each load balancer applies a ticket number
// and broadcast the ticket number to all processors
int LBManager::getLoadbalancerTicket()
{
  loadbalancers.push_back(nullptr);
  return loadbalancers.size() - 1;
}

void LBManager::addLoadbalancer(BaseLB* lb, int seq)
{
  //  CmiPrintf("[%d] addLoadbalancer for seq %d\n", CkMyPe(), seq);
  if (seq == -1) return;
  if (CkMyPe() == 0)
  {
    CmiAssert(seq < loadbalancers.size());
    if (loadbalancers[seq])
    {
      CmiPrintf("Duplicate load balancer created at %d\n", seq);
      CmiAbort("LBManager");
    }
  }
  if (loadbalancers.size() < seq + 1)
    loadbalancers.resize(seq + 1);
  loadbalancers[seq] = lb;
}

// switch strategy in order
void LBManager::nextLoadbalancer(int seq)
{
  if (seq == -1) return;  // -1 means this is the only LB
  currentLBIndex = seq + 1;
  if (_lb_args.loop())
  {
    if (currentLBIndex == loadbalancers.size()) currentLBIndex = 0;
  }
  else
  {
    if (currentLBIndex == loadbalancers.size()) currentLBIndex--;  // keep using the last one
  }
  if (seq != currentLBIndex)
  {
    loadbalancers[seq]->turnOff();
    CmiAssert(loadbalancers[currentLBIndex]);
    loadbalancers[currentLBIndex]->turnOn();
  }
}

// switch strategy
void LBManager::switchLoadbalancer(int switchFrom, int switchTo)
{
  if (lbNames[switchTo] != "DistributedLB" &&
    lbNames[switchTo] != "MetisLB" && 
    lbNames[switchTo] != "GreedyCentralLB" && 
    lbNames[switchTo] != "GreedyRefineCentralLB")
  {
    json config;
    if (lbNames[switchTo] == "Hybrid")
    {
      config["tree"] = "PE_Process_Root";
      config["Root"]["pe"] = 0;
      config["Root"]["step_freq"] = 3;
      config["Root"]["strategies"] = {"GreedyRefine"};
      config["Process"]["strategies"] = {"GreedyRefine"};
    }
    else
    {
      config["tree"] = "PE_Root";
      config["Root"]["pe"] = 0;
      config["Root"]["strategies"] = {lbNames[switchTo]};
    }
    configureTreeLB(config);
  }
  else
  {
    // TODO: Implement turn off / on for Distributed
  }
}

// return the seq-th load balancer string name of
// it can be specified in either compile time or runtime
// runtime has higher priority
const char* LBManager::loadbalancer(int seq)
{
  if (!lbRegistry.runtime_lbs.empty())
  {
    CmiAssert(seq < lbRegistry.runtime_lbs.size());
    return lbRegistry.runtime_lbs[seq];
  }
  else
  {
    CmiAssert(seq < lbRegistry.compile_lbs.size());
    return lbRegistry.compile_lbs[seq];
  }
}

void LBManager::pup(PUP::er& p)
{
  IrrGroup::pup(p);
  if (p.isUnpacking())
  {
    // Since avail_vector is static, only unpack one of them for real in SMP mode, the
    // rest to some tmp variable
    CmiLock(avail_vector_lock);
    if (!avail_vector_set)
    {
      avail_vector_set = true;
      p | avail_vector;
      // If we're restarting with more PEs, make the new ones available
      //if (avail_vector.size() < CkNumPes())
      //avail_vector.resize(CkNumPes(), 1);
      avail_vector = std::vector<char>(CkNumPes(), 1);
    }
    else
    {
      decltype(avail_vector) tmp;
      p | tmp;
    }
    CmiUnlock(avail_vector_lock);
  }
  else
  {
    p | avail_vector;
  }
  p | mystep;
  if (p.isUnpacking())
  {
    reallocQueue.clear();
    if (_lb_args.metaLbOn())
    {
      // if unpacking set metabalancer using the id
      metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
    }
  }
}

void configureTreeLB(const char* json_str)
{
  ((LBManager*)CkLocalBranch(_lbmgr))->configureTreeLB(json_str);
}

void LBManager::configureTreeLB(const char* json_str)
{
  json config = json::parse(json_str);
  configureTreeLB(config);
}

void LBManager::configureTreeLB(json& config)
{
  bool found = false;
  for (int i = 0; i < loadbalancers.size(); i++)
  {
    if (strcmp(loadbalancers[i]->lbName(), "TreeLB") == 0)
    {
      ((TreeLB*)loadbalancers[i])->configure(config);
      found = true;
      // break; // not sure if there could be more than one TreeLB
    }
  }
  if (!found) CkAbort("LBManager: TreeLB is not in my list of load balancers");
}

void LBManager::ResetAdaptive()
{
#if CMK_LBDB_ON
  if (_lb_args.metaLbOn())
  {
    if (metabalancer == NULL)
    {
      metabalancer = CProxy_MetaBalancer(_metalb).ckLocalBranch();
    }
    if (metabalancer != NULL)
    {
      metabalancer->ResetAdaptive();
    }
  }
#endif
}

void LBManager::ResumeClientsIfHeld()
{
#if CMK_LBDB_ON
  if (_lb_args.metaLbOn() && metabalancer) metabalancer->ResumeClients();
  if (_lb_args.lbperiod() != -1.0)
    setTimer();  // re-arm periodic stepping in the new world
  else
    CkSyncBarrier::object()->resumeClientsIfHeld();
#endif
}

void LBManager::ResumeClients()
{
#if CMK_LBDB_ON
#if CMK_SHRINK_EXPAND
  // A load balancing round has finished: whatever it was going to move has
  // moved, so the reduction settling window opened by the rescale restore can
  // close and the inactive-kid optimization can come back on. For an expand
  // this is the populate round that gave the newcomer its elements.
  {
    extern bool _rescaleReductionSettling;
    if (_rescaleReductionSettling)
    {
      _rescaleReductionSettling = false;
      // Closing the window is not enough: a PE this round left barren was not
      // allowed to say so while the window was open, and has no later occasion
      // to. Give every manager one chance to notice now.
      extern void CkReEvaluateReductionActivity(void);
      CkReEvaluateReductionActivity();
    }
  }
#endif
  if (_lb_args.metaLbOn())
  {
    if (metabalancer == NULL)
    {
      metabalancer = CProxy_MetaBalancer(_metalb).ckLocalBranch();
    }
    if (metabalancer != NULL)
    {
      metabalancer->ResumeClients();
    }
  }

  // If periodic is enabled, reset the timer and don't resume clients
  if (_lb_args.lbperiod() != -1.0)
  {
    setTimer();
  }
  else
  {
    CkSyncBarrier::object()->resumeClients();
  }
#endif
}

void LBManager::SetMigrationCost(double cost)
{
#if CMK_LBDB_ON
  if (_lb_args.metaLbOn())
  {
    if (metabalancer == NULL)
    {
      metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
    }
    if (metabalancer != NULL)
    {
      metabalancer->SetMigrationCost(cost);
    }
  }
#endif
}

void LBManager::SetStrategyCost(double cost)
{
#if CMK_LBDB_ON
  if (_lb_args.metaLbOn())
  {
    if (metabalancer == NULL)
    {
      metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
    }
    if (metabalancer != NULL)
    {
      metabalancer->SetStrategyCost(cost);
    }
  }
#endif
}

void LBManager::UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad)
{
#if CMK_LBDB_ON
  if (_lb_args.metaLbOn())
  {
    if (metabalancer == NULL)
    {
      metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
    }
    if (metabalancer != NULL)
    {
      metabalancer->UpdateAfterLBData(mLoad, mCpuLoad, avgLoad);
    }
  }
#endif
}

LDBarrierClient LBManager::AddLocalBarrierClient(Chare* obj, std::function<void()> fn)
{
  return CkSyncBarrier::object()->addClient(obj, fn);
}

void LBManager::RemoveLocalBarrierClient(LDBarrierClient h)
{
  CkSyncBarrier::object()->removeClient(h);
}

LDBarrierReceiver LBManager::AddLocalBarrierReceiver(std::function<void()> fn)
{
  return CkSyncBarrier::object()->addReceiver(fn);
}

void LBManager::RemoveLocalBarrierReceiver(LDBarrierReceiver h)
{
  CkSyncBarrier::object()->removeReceiver(h);
}

void LBManager::AtLocalBarrier(LDBarrierClient _n_c)
{
  if (useBarrier) CkSyncBarrier::object()->atBarrier(_n_c);
}

void LBManager::TurnOnBarrierReceiver(LDBarrierReceiver h)
{
  CkSyncBarrier::object()->turnOnReceiver(h);
}

void LBManager::TurnOffBarrierReceiver(LDBarrierReceiver h)
{
  CkSyncBarrier::object()->turnOffReceiver(h);
}

void LBManager::LocalBarrierOn(void) { CkSyncBarrier::object()->turnOn(); }
void LBManager::LocalBarrierOff(void) { CkSyncBarrier::object()->turnOff(); }

#if CMK_LBDB_ON
static void work(int iter_block, volatile int* result)
{
  int i;
  *result = 1;
  for (i = 0; i < iter_block; i++)
  {
    double b = 0.1 + 0.1 * *result;
    *result = (int)(sqrt(1 + cos(b * 1.57)));
  }
}

int LDProcessorSpeed()
{
  if (_lb_args.samePeSpeed() ||
      CkNumPes() == 1)  // I think it is safe to assume that we can
    return 1;           // skip this if we are only using 1 PE

  volatile int result = 0;

  int wps = 0;
  const double elapse = 0.2;
  // First, count how many iterations happen in "elapse" seconds.
  // Since we are doing lots of function calls, this will be rough
  const double end_time = CmiCpuTimer() + elapse;
  while (CmiCpuTimer() < end_time)
  {
    work(1000, &result);
    wps += 1000;
  }

  // Now we have a rough idea of how many iterations happen in
  // "elapse" seconds, so just perform a few cycles of correction
  // by running for what should take that long. Then correct the
  // number of iterations if needed.

  for (int i = 0; i < 2; i++)
  {
    const double start_time = CmiCpuTimer();
    work(wps, &result);
    const double end_time = CmiCpuTimer();
    const double correction = elapse / (end_time - start_time);
    wps = (int)((double)wps * correction + 0.5);
  }

  if (_lb_args.debug() > 1)
    CmiPrintf("LB> PE %d speed is %d\n", CkMyPe(), wps);

  return wps;
}

#else
int LDProcessorSpeed() { return 1; }
#endif  // CMK_LBDB_ON

int LBManager::ProcessorSpeed()
{
  static int peSpeed = LDProcessorSpeed();
  return peSpeed;
}

int LBManager::ProcessorGPUSpeed()
{
#if CMK_hapi || CMK_HIP
  static int gpuSpeed = -1; // Cache the result
  
  if (gpuSpeed != -1) {
    return gpuSpeed;
  }
  
  // Check if GPU is available
  int deviceCount = 0;
  if (hapiGetDeviceCount(&deviceCount) != hapiSuccess || deviceCount == 0) {
    CmiAbort("LB> PE %d: No GPU available, GPU speed = 0\n", CkMyPe());
  }
  
  // Get device for this PE (round-robin assignment)
  int deviceId = CkMyPe() % deviceCount;
  if (hapiSetDevice(deviceId) != hapiSuccess) {
    CmiAbort("LB> PE %d: Failed to set GPU device %d, GPU speed = 0\n", CkMyPe(), deviceId);
  }
  
  // Get device properties
  hapiDeviceProp prop;
  if (hapiGetDeviceProperties(&prop, deviceId) != hapiSuccess) {
    CmiAbort("LB> PE %d: Failed to get GPU device properties, GPU speed = 0\n", CkMyPe());
  }

  int clockRate = 0;
  if (hapiDeviceGetAttribute(&clockRate, hapiDevAttrClockRate, deviceId) != hapiSuccess) {
    CmiAbort("LB> PE %d: Failed to get GPU clock rate, GPU speed = 0\n", CkMyPe());
  }
  
  // Calculate theoretical peak single-precision FLOPS
  // Formula: multiProcessorCount * maxThreadsPerMultiProcessor * clockRate(KHz) * 2(FMA)
  // Convert to GFLOPS and then scale to integer for comparison with CPU speed
  long long peakFLOPS = (long long)prop.multiProcessorCount * 
                        prop.maxThreadsPerMultiProcessor * 
                        clockRate * 2LL; // 2 for FMA (multiply-add)
  
  // Convert from KHz*ops to GFLOPS, then scale to reasonable integer range
  double gflops = peakFLOPS / 1e6; // KHz to GHz conversion for GFLOPS
  
  // Scale to integer range similar to CPU ProcessorSpeed (typically 1-10000)
  // Use a scaling factor to make GPU speeds comparable to CPU speeds
  gpuSpeed = (int)(gflops / 100.0); // Scale down GFLOPS to reasonable range
  
  if (gpuSpeed < 1) gpuSpeed = 1; // Minimum speed
  
  if (_lb_args.debug() > 1) {
    CmiPrintf("LB> PE %d GPU %s: %d SMs, %d threads/SM, %d MHz, %.1f GFLOPS -> speed %d\n", 
              CkMyPe(), prop.name, prop.multiProcessorCount, 
              prop.maxThreadsPerMultiProcessor, clockRate/1000, gflops, gpuSpeed);
  }
  
  return gpuSpeed;
#else
  CmiAbort("LB> PE %d: ProcessorGPUSpeed() GPU support not enabled in this build\n", CkMyPe());
#endif
}

/*
  callable from user's code
*/
void TurnManualLBOn()
{
#if CMK_LBDB_ON
  LBManager* myLbdb = LBManager::Object();
  if (myLbdb)
  {
    myLbdb->TurnManualLBOn();
  }
  else
  {
    LBManager::manualOn = true;
  }
#endif
}

void TurnManualLBOff()
{
#if CMK_LBDB_ON
  LBManager* myLbdb = LBManager::Object();
  if (myLbdb)
  {
    myLbdb->TurnManualLBOff();
  }
  else
  {
    LBManager::manualOn = true;
  }
#endif
}

void LBTurnInstrumentOn()
{
#if CMK_LBDB_ON
  if (CkpvAccess(lbmanagerInited))
    LBManager::Object()->CollectStatsOn();
  else
    _lb_args.statsOn() = 1;
#endif
}

void LBTurnInstrumentOff()
{
#if CMK_LBDB_ON
  if (CkpvAccess(lbmanagerInited))
    LBManager::Object()->CollectStatsOff();
  else
    _lb_args.statsOn() = 0;
#endif
}

void LBTurnCommOn()
{
#if CMK_LBDB_ON
  _lb_args.traceComm() = 1;
#endif
}

void LBTurnCommOff()
{
#if CMK_LBDB_ON
  _lb_args.traceComm() = 0;
#endif
}

void LBClearLoads()
{
#if CMK_LBDB_ON
  LBManager::Object()->ClearLoads();
#endif
}

void LBTurnPredictorOn(LBPredictorFunction* model)
{
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOn(model);
#endif
}

void LBTurnPredictorOn(LBPredictorFunction* model, int wind)
{
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOn(model, wind);
#endif
}

void LBTurnPredictorOff()
{
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOff();
#endif
}

void LBChangePredictor(LBPredictorFunction* model)
{
#if CMK_LBDB_ON
  LBManager::Object()->ChangePredictor(model);
#endif
}

void LBSetPeriod(double period)
{
#if CMK_LBDB_ON
  LBManager::SetLBPeriod(period);
#endif
}

int LBRegisterObjUserData(int size) { return CkpvAccess(lbobjdatalayout).claim(size); }

#include "LBManager.def.h"

/*@}*/
