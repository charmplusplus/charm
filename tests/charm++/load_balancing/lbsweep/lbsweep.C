// Runs one load balancer (+balancer ...) on a deterministic, imbalanced load
// and checks the placement after each AtSync step.
//
// Elements report a modelled load (usesAutoMeasure = false, UserSetLBLoad),
// so the checks do not depend on timing. PackedMap starts 3/4 of the
// elements on PE 0 and the rest on PEs 1..P/2-1 (all on PE 0 when P < 4),
// leaving the upper half of the PEs empty. Every element exchanges a message with both ring
// neighbours each iteration, so communication-aware balancers see a graph.
// Elements with i % nonmig == 0 (-nonmig, default 16, 0 for none) are
// non-migratable and must never move.
//
// After each step one sum reduction gives element count and index sum, pup
// checksum failures, migrations, and the per-PE load of that step's loads on
// the placement before and after the step. PE 0 then checks the class given
// with -expect:
//   improve  max/avg after <= tol * lower bound; on the first step it must
//            also be strictly better than before; with -maxmig F, later
//            steps move at most F of the migratable elements (refinement)
//   rotate   every migratable element moved from PE p to (p+1)%P
//   dummy    no element moved
//   random   integrity checks only
// The lower bound is max(total/P, largest migratable load, largest
// non-migratable load on one PE). With -localfreq F the balancer is taken to
// balance only within each process except on steps s with (s+1)%F == 0 (a
// TreeLB process level with a root step_freq of F); on those other steps the
// bound is computed per process.
//
//   ./lbsweep +p4 -expect improve [-tol 1.34] [-n 64] [-steps 3]
//       [-profile heavy|ramp|uniform] [-nonmig 16] [-maxmig 0.5]
//       [-localfreq 1] +balancer GreedyLB
#include "lbsweep.decl.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int numElems;
/* readonly */ int numSteps;
/* readonly */ int numIters;
/* readonly */ int profile;
/* readonly */ int nonmigEvery;

enum { PROFILE_HEAVY, PROFILE_RAMP, PROFILE_UNIFORM };

// Large compared with the real busy time per step, which the balancers add
// as background load.
static const double UNIT = 10.0;

static double loadOf(int i, int step)
{
  switch (profile)
  {
    case PROFILE_HEAVY:
      return UNIT * (((i + 3 * step) % 8 == 0) ? 6.0 : 1.0);
    case PROFILE_RAMP:
      return UNIT * (1 + (i + 5 * step) % 8);
    default:
      return UNIT;
  }
}

static bool isNonmig(int i) { return nonmigEvery > 0 && i % nonmigEvery == 0; }

static int initialPe(int i)
{
  const int P = CkNumPes();
  const int half = P / 2 > 1 ? P / 2 : 1;
  if (half == 1) return 0;
  const int onZero = (3 * numElems) / 4;
  if (i < onZero) return 0;
  return 1 + (i - onZero) % (half - 1);
}

static double payloadValue(int i, int k) { return i * 1000.0 + k + 0.5; }
static double ghostValue(int from, int ref) { return from * 7.0 + ref * 0.25; }

// Reduction layout: fixed slots, then three blocks of CkNumPes() doubles.
enum { R_COUNT, R_INDEXSUM, R_BAD, R_MOVED, R_NONMIG_MOVED, R_ROTATE_BAD, R_FIXED };
static int rBefore() { return R_FIXED; }
static int rAfter() { return R_FIXED + CkNumPes(); }
static int rNonmig() { return R_FIXED + 2 * CkNumPes(); }
static int rSize() { return R_FIXED + 3 * CkNumPes(); }

class PackedMap : public CBase_PackedMap
{
 public:
  PackedMap() {}
  int procNum(int, const CkArrayIndex& idx) override { return initialPe(idx.data()[0]); }
};

class Elem : public CBase_Elem
{
  Elem_SDAG_CODE
  int step = 0, iter = 0, got = 0;
  int prevPE = -1;
  int bad = 0;
  std::vector<double> payload;

 public:
  Elem()
  {
    usesAtSync = true;
    usesAutoMeasure = false;
    if (isNonmig(thisIndex)) setMigratable(false);
    if (CkMyPe() != initialPe(thisIndex))
      CkAbort("lbsweep: element %d created on PE %d, map says %d\n", thisIndex,
              CkMyPe(), initialPe(thisIndex));
    payload.resize(4 + thisIndex % 5);
    for (size_t k = 0; k < payload.size(); k++) payload[k] = payloadValue(thisIndex, k);
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p)
  {
    p | step;
    p | iter;
    p | got;
    p | prevPE;
    p | bad;
    p | payload;
  }

  void UserSetLBLoad() { setObjTime(loadOf(thisIndex, step)); }

  void sendGhosts()
  {
    const int ref = step * numIters + iter;
    const double v = ghostValue(thisIndex, ref);
    thisProxy[(thisIndex + 1) % numElems].recvGhost(ref, thisIndex, v);
    thisProxy[(thisIndex + numElems - 1) % numElems].recvGhost(ref, thisIndex, v);
  }

  void checkGhost(int ref, int from, double v)
  {
    const bool neighbour = from == (thisIndex + 1) % numElems ||
                           from == (thisIndex + numElems - 1) % numElems;
    if (!neighbour || v != ghostValue(from, ref))
    {
      CkPrintf("lbsweep: element %d got ghost ref %d from %d value %g\n", thisIndex, ref,
               from, v);
      bad++;
    }
  }

  void report()
  {
    const int P = CkNumPes();
    const int pe = CkMyPe();
    size_t want = 4 + thisIndex % 5;
    bool ok = payload.size() == want;
    for (size_t k = 0; ok && k < want; k++) ok = payload[k] == payloadValue(thisIndex, k);
    if (!ok)
    {
      CkPrintf("lbsweep: element %d payload corrupted after step %d\n", thisIndex, step);
      bad++;
    }
    const bool moved = pe != prevPE;
    const double load = loadOf(thisIndex, step);
    std::vector<double> v(rSize(), 0.0);
    v[R_COUNT] = 1;
    v[R_INDEXSUM] = thisIndex;
    v[R_BAD] = bad;
    v[R_MOVED] = moved;
    if (isNonmig(thisIndex))
    {
      v[R_NONMIG_MOVED] = moved;
      v[rNonmig() + prevPE] += load;
      if (moved)
        CkPrintf("lbsweep: non-migratable element %d moved from PE %d to %d\n",
                 thisIndex, prevPE, pe);
    }
    else
      v[R_ROTATE_BAD] = pe != (prevPE + 1) % P;
    v[rBefore() + prevPE] += load;
    v[rAfter() + pe] += load;
    contribute(v, CkReduction::sum_double, CkCallback(CkReductionTarget(Main, stepDone), mainProxy));
  }
};

class Main : public CBase_Main
{
  std::string expect = "improve";
  double tol = 4.0 / 3.0;
  double maxMigFrac = 1.0;
  int localFreq = 1;
  int stepsDone = 0;
  int failures = 0;

  void fail(const char* what)
  {
    CkPrintf("lbsweep: FAIL step %d: %s\n", stepsDone, what);
    failures++;
  }

  // Lower bound over the PEs [lo, hi): average, largest movable load,
  // largest fixed load.
  double bound(const double* load, const double* fixed, int lo, int hi)
  {
    double total = 0, fixedMax = 0;
    for (int p = lo; p < hi; p++)
    {
      total += load[p];
      fixedMax = std::max(fixedMax, fixed[p]);
    }
    double migMax = 0;
    for (int i = 0; i < numElems; i++)
      if (!isNonmig(i)) migMax = std::max(migMax, loadOf(i, stepsDone));
    return std::max(std::max(total / (hi - lo), migMax), fixedMax);
  }

 public:
  Main(CkArgMsg* m)
  {
    numElems = 64;
    numSteps = 3;
    numIters = 2;
    profile = PROFILE_HEAVY;
    nonmigEvery = 16;
    for (int a = 1; a < m->argc; a++)
    {
      const char* o = m->argv[a];
      const char* val = a + 1 < m->argc ? m->argv[a + 1] : nullptr;
      if (!val) CkAbort("lbsweep: option %s needs a value\n", o);
      if (!strcmp(o, "-n")) numElems = atoi(val);
      else if (!strcmp(o, "-steps")) numSteps = atoi(val);
      else if (!strcmp(o, "-iters")) numIters = atoi(val);
      else if (!strcmp(o, "-nonmig")) nonmigEvery = atoi(val);
      else if (!strcmp(o, "-expect")) expect = val;
      else if (!strcmp(o, "-tol")) tol = atof(val);
      else if (!strcmp(o, "-maxmig")) maxMigFrac = atof(val);
      else if (!strcmp(o, "-localfreq")) localFreq = atoi(val);
      else if (!strcmp(o, "-profile"))
      {
        if (!strcmp(val, "heavy")) profile = PROFILE_HEAVY;
        else if (!strcmp(val, "ramp")) profile = PROFILE_RAMP;
        else if (!strcmp(val, "uniform")) profile = PROFILE_UNIFORM;
        else CkAbort("lbsweep: unknown profile %s\n", val);
      }
      else CkAbort("lbsweep: unknown option %s\n", o);
      a++;
    }
    if (expect != "improve" && expect != "rotate" && expect != "dummy" && expect != "random")
      CkAbort("lbsweep: unknown -expect %s\n", expect.c_str());
    if (numElems < 1 || numSteps < 1 || numIters < 1 || localFreq < 1)
      CkAbort("lbsweep: bad -n/-steps/-iters/-localfreq\n");
    CkPrintf("lbsweep: %d elements on %d PEs in %d processes, %d steps, expect %s, tol %.3f\n",
             numElems, CkNumPes(), CkNumNodes(), numSteps, expect.c_str(), tol);
    delete m;
    mainProxy = thisProxy;
    CkArrayOptions opts(numElems);
    opts.setMap(CProxy_PackedMap::ckNew());
    CProxy_Elem elems = CProxy_Elem::ckNew(opts);
    elems.run();
  }

  void stepDone(int n, double* v)
  {
    const int P = CkNumPes();
    CkAssert(n == rSize());
    const double* before = v + rBefore();
    const double* after = v + rAfter();
    const double* fixed = v + rNonmig();
    const long count = lround(v[R_COUNT]);
    const long moved = lround(v[R_MOVED]);
    long nonmigCount = 0;
    for (int i = 0; i < numElems; i++) nonmigCount += isNonmig(i);
    const long migratable = numElems - nonmigCount;

    char msg[256];
    if (count != numElems)
    {
      snprintf(msg, sizeof(msg), "%ld elements reported, expected %d", count, numElems);
      fail(msg);
    }
    if (lround(v[R_INDEXSUM]) != (long)numElems * (numElems - 1) / 2) fail("index sum wrong");
    if (lround(v[R_BAD]) != 0) fail("ghost or payload check failed");
    if (lround(v[R_NONMIG_MOVED]) != 0) fail("a non-migratable element moved");

    double total = 0, maxB = 0, maxA = 0;
    for (int p = 0; p < P; p++)
    {
      total += after[p];
      maxB = std::max(maxB, before[p]);
      maxA = std::max(maxA, after[p]);
    }
    const double avg = total / P;
    const bool local = (stepsDone + 1) % localFreq != 0;
    double lb = bound(after, fixed, 0, P);
    if (local)
      for (int node = 0; node < CkNumNodes(); node++)
        lb = std::max(lb, bound(before, fixed, CkNodeFirst(node),
                                CkNodeFirst(node) + CkNodeSize(node)));
    CkPrintf("lbsweep: step %d: max/avg %.3f -> %.3f, %s bound %.3f, migrations %ld of %ld\n",
             stepsDone, maxB / avg, maxA / avg, local ? "process" : "global", lb / avg, moved,
             migratable);

    if (expect == "improve")
    {
      if (maxA > tol * lb * (1 + 1e-6))
      {
        snprintf(msg, sizeof(msg), "max/avg %.3f above %.3f x bound %.3f", maxA / avg, tol,
                 lb / avg);
        fail(msg);
      }
      if (stepsDone == 0 && P > 1 && maxB > lb * (1 + 1e-6) && maxA >= maxB * (1 - 1e-6))
        fail("first step did not reduce the max load");
      if (stepsDone > 0 && moved > maxMigFrac * migratable)
      {
        snprintf(msg, sizeof(msg), "%ld migrations, more than %.2f of %ld", moved, maxMigFrac,
                 migratable);
        fail(msg);
      }
    }
    else if (expect == "rotate")
    {
      if (P > 1 && lround(v[R_ROTATE_BAD]) != 0)
      {
        snprintf(msg, sizeof(msg), "%ld migratable elements not moved to the next PE",
                 lround(v[R_ROTATE_BAD]));
        fail(msg);
      }
    }
    else if (expect == "dummy")
    {
      if (moved != 0) fail("elements moved under a balancer that must not move any");
    }

    if (++stepsDone == numSteps)
    {
      if (failures) CkAbort("lbsweep: %d check(s) failed\n", failures);
      CkPrintf("lbsweep: PASS\n");
      CkExit(0);
    }
  }
};

#include "lbsweep.def.h"
