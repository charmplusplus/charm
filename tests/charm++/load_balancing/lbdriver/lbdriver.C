/**
 * lbdriver -- run load balancer strategies on a synthetic chare-level comm
 * graph and record the object-to-PE mapping each one produces.
 *
 * This is not an application: the chares do no work. Each holds a declared
 * weight (setObjTime, with automatic instrumentation off) and sends fixed-size
 * messages to its four stencil neighbours so the runtime records a real
 * communication graph. That is exactly the input a load balancer consumes --
 * objData with loads and positions, plus commData edges -- so a strategy runs
 * against a graph you specify rather than one you have to provoke out of a
 * real application.
 *
 * Why a Charm program rather than a plain function call: the CentralLB
 * strategies do expose their decision as a pure function (work(LDStats*) reads
 * the stats and writes stats->to_proc), but DiffusionLB does not. It derives
 * from DistBaseLB and its mapping emerges from an asynchronous protocol across
 * PEs, so the only way to see what it decides is to let it run. Driving both
 * through AtSync keeps the two comparable, since both then see identical input.
 *
 * Phases, each ending in one LB step:
 *   0  initial mapping, the array's default block map (no LB has run)
 *   1  after the first balancer  -- uniform weights
 *   2  after the second balancer -- weights changed to a hot region
 *
 * Pass the balancers in that order, e.g.
 *   +balancer MetisLB +balancer DiffusionLB
 * Charm runs them in sequence, one per LB step (LBManager::nextLoadbalancer),
 * so step 0 uses the first and step 1 the second.
 *
 * Writes lbdriver.json: the grid dimensions, the weights used in each phase,
 * and the full object-to-PE map after each phase.
 */

#include "lbdriver.decl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Cell cells;
/*readonly*/ int nx;
/*readonly*/ int ny;
/*readonly*/ int itersPerPhase;
/*readonly*/ int ghostBytes;

// Weight patterns. Phase 1 is flat, so the first balancer partitions on
// communication alone and produces the compact blocks a stencil should get.
// Phase 2 puts a heavy disc off-centre, which is the imbalance the second
// balancer has to repair -- and because it straddles whatever partition the
// first one chose, repairing it means moving objects across several PEs.
static double weightUniform(int, int) { return 1.0; }

static double weightHotSpot(int i, int j)
{
  const double cx = nx * 0.30, cy = ny * 0.30;
  const double r = std::sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
  const double radius = 0.22 * (nx < ny ? nx : ny);
  return (r <= radius) ? 12.0 : 1.0;
}

class Main : public CBase_Main
{
private:
  int phase;
  int iters;
  std::vector<std::vector<int>> maps;
  std::vector<std::string> phaseNames;
  std::vector<std::vector<double>> weights;
  std::string lb1, lb2;

  void recordWeights(double (*fn)(int, int))
  {
    std::vector<double> w(nx * ny);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) w[i * ny + j] = fn(i, j);
    weights.push_back(w);
  }

public:
  Main(CkArgMsg* m)
  {
    nx = 32;
    ny = 32;
    itersPerPhase = 6;
    ghostBytes = 4096;
    if (m->argc > 1) nx = atoi(m->argv[1]);
    if (m->argc > 2) ny = atoi(m->argv[2]);
    if (m->argc > 3) itersPerPhase = atoi(m->argv[3]);
    if (m->argc > 4) ghostBytes = atoi(m->argv[4]);
    // The balancer names, for labelling only. Charm decides which balancer runs
    // at which step from the order of the +balancer arguments; nothing here can
    // read that back, so the caller passes the same order again.
    lb1 = m->argc > 5 ? m->argv[5] : "balancer 1";
    lb2 = m->argc > 6 ? m->argv[6] : "balancer 2";
    delete m;

    CkPrintf("lbdriver> %d x %d = %d objects on %d PEs, %d iterations per phase,"
             " %d-byte ghosts\n",
             nx, ny, nx * ny, CkNumPes(), itersPerPhase, ghostBytes);

    mainProxy = thisProxy;
    phase = 0;
    iters = 0;

    cells = CProxy_Cell::ckNew(nx, ny);

    // Phase 0 is the map the array was created with; no balancer has run.
    phaseNames.push_back("Initial (default block map)");
    recordWeights(weightUniform);
    cells.reportMap();
  }

  // One reduction per phase: every object has sent this iteration's ghosts and
  // declared its load.
  void iterDone()
  {
    if (++iters < itersPerPhase)
      cells.iterate();
    else
      cells.startLB();
  }

  // Every object is through AtSync and has resumed, so the mapping is final.
  void lbDone() { cells.reportMap(); }

  // The map, gathered as a max-reduction over one slot per object.
  void mapReport(int n, int peOf[])
  {
    maps.push_back(std::vector<int>(peOf, peOf + n));

    if (phase == 0)
    {
      // Uniform weights, then the first balancer.
      phase = 1;
      iters = 0;
      phaseNames.push_back(lb1 + " (uniform weights)");
      recordWeights(weightUniform);
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) cells(i, j).setWeight(weightUniform(i, j));
      cells.iterate();
    }
    else if (phase == 1)
    {
      // The weights change under the mapping the first balancer chose, then
      // the second balancer runs against that.
      phase = 2;
      iters = 0;
      phaseNames.push_back(lb2 + " (hot region added)");
      recordWeights(weightHotSpot);
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) cells(i, j).setWeight(weightHotSpot(i, j));
      cells.iterate();
    }
    else
    {
      writeJson();
      CkExit();
    }
  }

  void writeJson()
  {
    const char* path = "lbdriver.json";
    FILE* f = fopen(path, "w");
    if (f == NULL)
    {
      CkPrintf("lbdriver> could not open %s for writing\n", path);
      CkExit();
      return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"nx\": %d,\n  \"ny\": %d,\n  \"npes\": %d,\n", nx, ny, CkNumPes());
    fprintf(f, "  \"phases\": [\n");
    for (size_t p = 0; p < maps.size(); p++)
    {
      fprintf(f, "    {\n      \"name\": \"%s\",\n", phaseNames[p].c_str());

      fprintf(f, "      \"map\": [");
      for (size_t k = 0; k < maps[p].size(); k++)
        fprintf(f, "%s%d", k ? "," : "", maps[p][k]);
      fprintf(f, "],\n");

      fprintf(f, "      \"weights\": [");
      for (size_t k = 0; k < weights[p].size(); k++)
        fprintf(f, "%s%g", k ? "," : "", weights[p][k]);
      fprintf(f, "]\n");

      fprintf(f, "    }%s\n", p + 1 < maps.size() ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);

    CkPrintf("lbdriver> wrote %s (%zu phases)\n", path, maps.size());

    // Per-phase summary, so a mapping that looks plausible but is badly
    // balanced is visible without opening the plot.
    for (size_t p = 0; p < maps.size(); p++)
    {
      std::vector<double> peLoad(CkNumPes(), 0.0);
      std::vector<int> peCount(CkNumPes(), 0);
      for (size_t k = 0; k < maps[p].size(); k++)
      {
        const int pe = maps[p][k];
        if (pe < 0 || pe >= CkNumPes()) continue;
        peLoad[pe] += weights[p][k];
        peCount[pe]++;
      }
      double sum = 0.0, mx = 0.0;
      int minObjs = nx * ny, maxObjs = 0;
      for (int pe = 0; pe < CkNumPes(); pe++)
      {
        sum += peLoad[pe];
        if (peLoad[pe] > mx) mx = peLoad[pe];
        if (peCount[pe] < minObjs) minObjs = peCount[pe];
        if (peCount[pe] > maxObjs) maxObjs = peCount[pe];
      }
      const double avg = sum / CkNumPes();
      CkPrintf("lbdriver>   %-40s max/avg load %.3f, objects/PE %d..%d\n",
               phaseNames[p].c_str(), avg > 0 ? mx / avg : 0.0, minObjs, maxObjs);
    }
  }
};

class Cell : public CBase_Cell
{
private:
  double weight;
  std::vector<char> ghost;

public:
  Cell()
  {
    usesAtSync = true;
    // The weight is declared, not measured: setObjTime below is the object's
    // load as far as the balancer is concerned. Leaving instrumentation on
    // would add this chare's real (near-zero, noisy) execution time on top and
    // make the input non-deterministic.
    usesAutoMeasure = false;
    weight = 1.0;
    ghost.resize(ghostBytes, 0);
    // Object positions are deliberately not registered. CkMigratable::setObjPosition
    // exists in the source tree but not in this build's installed libck, and
    // rebuilding is blocked (see the README). Nothing here needs it: run
    // DiffusionLB with +LBDiffusionCommOn so it selects neighbours and objects by
    // communication rather than by centroid. Without positions its 1-D interval
    // rules also stand down, which is the correct behaviour for a 2-D stencil.
  }

  Cell(CkMigrateMessage* m) : CBase_Cell(m) { usesAtSync = true; usesAutoMeasure = false; }

  void pup(PUP::er& p)
  {
    CBase_Cell::pup(p);
    p | weight;
    p | ghost;
  }

  void setWeight(double w) { weight = w; }

  // With usesAutoMeasure off, the framework asks each element for its load at
  // the LB step instead of measuring it. This is the whole point of the driver:
  // the balancer's input is the weight pattern chosen here, identical on every
  // run and for every balancer, rather than whatever the machine happened to
  // measure.
  void UserSetLBLoad() { setObjTime(weight); }

  // One iteration: send to the four stencil neighbours, declare the load, and
  // report. The sends are what the runtime records as commData -- that is the
  // comm graph the balancer partitions. Nothing waits on the receives; a ghost
  // is delivered whenever it arrives and its contents are never read, since no
  // result depends on them.
  void iterate()
  {
    const int x = thisIndex.x, y = thisIndex.y;
    if (x > 0) thisProxy(x - 1, y).recvGhost(ghostBytes, ghost.data());
    if (x < nx - 1) thisProxy(x + 1, y).recvGhost(ghostBytes, ghost.data());
    if (y > 0) thisProxy(x, y - 1).recvGhost(ghostBytes, ghost.data());
    if (y < ny - 1) thisProxy(x, y + 1).recvGhost(ghostBytes, ghost.data());

    setObjTime(weight);

    contribute(CkCallback(CkReductionTarget(Main, iterDone), mainProxy));
  }

  void recvGhost(int, char*) {}

  void startLB() { AtSync(); }

  void ResumeFromSync()
  {
    contribute(CkCallback(CkReductionTarget(Main, lbDone), mainProxy));
  }

  // Each object writes its own PE into its own slot and -1 everywhere else; a
  // max reduction then leaves the complete map. Simple rather than scalable,
  // which is the right trade at the grid sizes this driver is for.
  void reportMap()
  {
    std::vector<int> slot(nx * ny, -1);
    slot[thisIndex.x * ny + thisIndex.y] = CkMyPe();
    contribute(slot.size() * sizeof(int), slot.data(), CkReduction::max_int,
               CkCallback(CkReductionTarget(Main, mapReport), mainProxy));
  }
};

#include "lbdriver.def.h"
