// ringflow_test -- the pseudo-LB rounds on a ring, against measured loads.
//
// DiffusionLB's neighbour graph on a job with no 1-D keys is the ring backbone
// by node id, and leanmd's x-slab decomposition under a density gradient puts a
// monotone load ramp around that ring. The loads below are the ones the chare
// printed at the second LB step of job 22137894 (2 nodes, 4 GPUs each, weak
// grid 16 8 8): node 6 holding 20.880676 with neighbours at 17.432060 and
// 18.175282, every edge comm-adjacent so the flow gate is open.
//
// The planner used to leave that state alone -- every node was inside the
// decision floor of its own two neighbours while the job spanned 11.68 to
// 20.88, so nothing came off the heaviest node and the imbalance the job paid
// never moved again. The floor is now held against the job-wide mean as well,
// which is the number DiffusionLB::pseudoLoadContribute gathers before the
// rounds. It applies in three places in one round -- the node's overload, the
// per-edge send, and the flow after momentum -- and all three take the same
// exception.
//
// No Charm runtime: diffusionRoundFlows is a pure function, and this runs the
// same lockstep loop the SDAG round does. Login-node safe.

#include "DiffusionFlow.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
// Job 22137894, leanmd weak 16 8 8, LB step 2, one entry per process.
const std::vector<double> kLeanmdStep2 = {17.71, 11.68, 11.68, 13.83,
                                          16.12, 18.18, 20.88, 17.43};
const double kFloor = 0.10;  // +LBDiffusionMinImbalance, the production default
const double kBeta = 1.0;    // first-order; momentum does not change the stop

struct RingResult
{
  double finalImbalance;  // max/avg of the notional loads the rounds settle on
  double planned;         // total load the plan moves off the node it is on
  int rounds;
};

// One pass of the chare's round loop over a ring of nodes, in lockstep, exactly
// as DiffusionLB.ci runs it: every node sees its neighbours' current notional
// loads, plans its flows, and the flows are committed on both ends.
RingResult runRing(const std::vector<double>& load, double globalAvgLoad)
{
  const int n = (int)load.size();
  std::vector<double> pseudo(load), prevPseudo(load);
  std::vector<std::vector<double>> toSend(n, std::vector<double>(2, 0.0));
  std::vector<std::vector<double>> prevRound(n, std::vector<double>(2, 0.0));
  // Neighbour 0 is the next node round the ring, neighbour 1 the previous one.
  const std::vector<char> adjacent(2, 1);

  RingResult r = {0.0, 0.0, 0};
  for (int itr = 0; itr < diffusionIterations(); itr++)
  {
    std::vector<std::vector<double>> flow(n, std::vector<double>(2, 0.0));
    for (int i = 0; i < n; i++)
    {
      const std::vector<double> nbors = {pseudo[(i + 1) % n], pseudo[(i + n - 1) % n]};
      diffusionRoundFlows(load[i], pseudo[i], kFloor, kBeta, nbors, adjacent, toSend[i],
                          prevRound[i], flow[i], false, globalAvgLoad);
    }
    for (int i = 0; i < n; i++)
      for (int d = 0; d < 2; d++)
      {
        const double f = flow[i][d];
        const int t = (d == 0) ? (i + 1) % n : (i + n - 1) % n;
        toSend[i][d] += f;
        prevRound[i][d] = f;
        pseudo[i] -= f;
        pseudo[t] += f;
        toSend[t][(d == 0) ? 1 : 0] -= f;
      }
    r.rounds = itr + 1;

    double maxRatio = 0.0;
    for (int i = 0; i < n; i++)
    {
      const double denom = (load[i] > 1e-12) ? load[i] : 1e-12;
      maxRatio = std::max(maxRatio, std::fabs(pseudo[i] - prevPseudo[i]) / denom);
      prevPseudo[i] = pseudo[i];
    }
    if (maxRatio <= diffusionPseudoConvergeRatio()) break;
  }

  double sum = 0.0, mx = 0.0;
  for (int i = 0; i < n; i++)
  {
    sum += pseudo[i];
    mx = std::max(mx, pseudo[i]);
    for (int d = 0; d < 2; d++)
      if (toSend[i][d] > 0.0) r.planned += toSend[i][d];
  }
  r.finalImbalance = (sum > 0.0) ? mx / (sum / n) : 0.0;
  return r;
}

double meanOf(const std::vector<double>& v)
{
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / v.size();
}

int failures = 0;
void check(bool ok, const char* what)
{
  if (!ok)
  {
    printf("ringflow_test FAILED: %s\n", what);
    failures++;
  }
}
}  // namespace

int main()
{
  const double mean = meanOf(kLeanmdStep2);

  // 1. The measured stall. The purely local floor does plan something -- the
  //    light end of the ramp shuffles, which is where the neighbour gaps are
  //    widest -- but nothing comes off the heaviest node, so the imbalance the
  //    job actually pays is exactly where it started. This is what the chare
  //    did at every LB step after the first (job 22137894).
  const RingResult local = runRing(kLeanmdStep2, 0.0);
  check(local.finalImbalance > 1.30,
        "local-only floor should leave the ramp's maximum where it is");

  // 2. With the job-wide mean the same state plans the hop down the ramp and
  //    the maximum comes down with it.
  const RingResult global = runRing(kLeanmdStep2, mean);
  check(global.planned > 2.0 * local.planned,
        "global floor should plan substantially more flow");
  check(global.finalImbalance < 1.15,
        "global floor should flatten the ramp to near the floor");
  check(global.finalImbalance < local.finalImbalance - 0.15,
        "global floor should improve the planned imbalance");

  // 3. A job that is already flat plans nothing either way: the floor still
  //    keeps step-to-step noise from moving work, which is why it exists.
  const std::vector<double> flat = {16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0};
  check(runRing(flat, meanOf(flat)).planned == 0.0, "flat job should plan nothing");

  // 4. Noise, not a ramp: one node 12% over the mean, the rest scattered under
  //    the floor. The node is over the line, so it is allowed to shed -- but
  //    only what it is over by, not its whole local difference, so the plan
  //    stays small. This is the case the floor was added for (commit 402913794).
  const std::vector<double> noisy = {17.92, 15.7, 16.4, 15.4, 16.3, 15.6, 16.2, 15.9};
  const RingResult noise = runRing(noisy, meanOf(noisy));
  check(noise.planned < 0.02 * meanOf(noisy) * noisy.size(),
        "noise should plan less than 2% of the job's load");

  printf("ringflow_test: leanmd step-2 ring, local floor -> %.3f (%.2f planned, %d rounds); "
         "global floor -> %.3f (%.2f planned, %d rounds); noise plans %.3f\n",
         local.finalImbalance, local.planned, local.rounds, global.finalImbalance,
         global.planned, global.rounds, noise.planned);
  if (failures == 0) printf("ringflow_test: all scenarios passed\n");
  return failures == 0 ? 0 : 1;
}
