// metis_replay -- replay a METIS call MetisLB wrote (CHARM_DEBUG_METIS=<dir>)
// against libckmetis, off the machine. No Charm++ runtime: runs on a login node.
//
// MetisLB partitions measured loads, so no two runs hand METIS the same graph,
// and the partition it keeps moves the step time by several percent (sph2d
// strong N=1: the priced step bound ranged 17.9 .. 21.3 over 20 runs of one
// command, and ms/step followed it). This answers, on ONE captured input, what
// the knobs MetisLB has would do about that:
//
//   seeds   how far does the result move with METIS's seed alone? (MetisLB
//           never sets one, so every run uses libmetis's default, 4321.)
//   ubvec   constraint 0 is the load, and METIS only promises it within the
//           tolerance -- does a tighter one narrow the spread, and what does it
//           cost in cut?
//
//   metis_replay <dump> [nseeds=32] [ubvec0 ...]
//
// For each tolerance (the dump's own first, then each ubvec0 given) it runs the
// default seed and seeds 1..nseeds and prints, per run, the edge cut and each
// constraint's worst part against its target (max_p w_p / (tpwgt_p * total));
// then the spread. Constraint 0's figure is what the balancer's priced step
// bound follows: the most loaded GPU group against the mean.
#include <metis.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

struct Call {
  idx_t nv = 0, ncon = 0, nparts = 0;
  int kway = 0;
  std::vector<idx_t> xadj, adjncy, adjwgt, vwgt, options;
  std::vector<real_t> tpwgts, ubvec;
  bool hasTp = false;
};

static bool readCall(const char* path, Call& c) {
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); return false; }
  long nv, ncon, ne, np; int kway;
  if (fscanf(f, "%ld %ld %ld %ld %d", &nv, &ncon, &ne, &np, &kway) != 5) return false;
  c.nv = (idx_t)nv; c.ncon = (idx_t)ncon; c.nparts = (idx_t)np; c.kway = kway;
  auto ints = [&](std::vector<idx_t>& v, size_t n) {
    v.resize(n);
    for (size_t i = 0; i < n; i++) { long x; if (fscanf(f, "%ld", &x) != 1) return false; v[i] = (idx_t)x; }
    return true;
  };
  auto reals = [&](std::vector<real_t>& v, size_t n) {
    v.resize(n);
    for (size_t i = 0; i < n; i++) { double x; if (fscanf(f, "%lf", &x) != 1) return false; v[i] = (real_t)x; }
    return true;
  };
  int tp = 0;
  if (!ints(c.xadj, (size_t)nv + 1) || !ints(c.adjncy, (size_t)ne) || !ints(c.adjwgt, (size_t)ne) ||
      !ints(c.vwgt, (size_t)nv * ncon) || fscanf(f, "%d", &tp) != 1) return false;
  c.hasTp = tp != 0;
  if (c.hasTp && !reals(c.tpwgts, (size_t)np * ncon)) return false;
  if (!reals(c.ubvec, (size_t)ncon) || !ints(c.options, METIS_NOPTIONS)) return false;
  fclose(f);
  return true;
}

struct Result { idx_t cut; std::vector<double> worst; std::vector<idx_t> parts; };

static Result run(const Call& c, idx_t seed, double ubvec0) {
  Call k = c;   // METIS takes non-const pointers
  k.options[METIS_OPTION_SEED] = seed;
  if (ubvec0 > 1.0) k.ubvec[0] = (real_t)ubvec0;
  Result r; r.cut = 0; r.parts.assign(k.nv, 0);
  idx_t nv = k.nv, ncon = k.ncon, np = k.nparts;
  real_t* tp = k.hasTp ? k.tpwgts.data() : nullptr;
  const int rc = k.kway
      ? METIS_PartGraphKway(&nv, &ncon, k.xadj.data(), k.adjncy.data(), k.vwgt.data(), nullptr,
                            k.adjwgt.data(), &np, tp, k.ubvec.data(), k.options.data(), &r.cut,
                            r.parts.data())
      : METIS_PartGraphRecursive(&nv, &ncon, k.xadj.data(), k.adjncy.data(), k.vwgt.data(), nullptr,
                                 k.adjwgt.data(), &np, tp, k.ubvec.data(), k.options.data(), &r.cut,
                                 r.parts.data());
  if (rc != METIS_OK) { fprintf(stderr, "METIS returned %d\n", rc); exit(2); }
  r.worst.assign(c.ncon, 0.0);
  for (idx_t con = 0; con < c.ncon; con++) {
    std::vector<double> w(c.nparts, 0.0);
    double total = 0.0;
    for (idx_t v = 0; v < c.nv; v++) {
      const double x = (double)c.vwgt[(size_t)v * c.ncon + con];
      w[r.parts[v]] += x; total += x;
    }
    for (idx_t p = 0; p < c.nparts; p++) {
      const double share = c.hasTp ? (double)c.tpwgts[(size_t)p * c.ncon + con] : 1.0 / c.nparts;
      if (total > 0.0 && share > 0.0) r.worst[con] = std::max(r.worst[con], w[p] / (share * total));
    }
  }
  return r;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <dump> [nseeds=32] [ubvec0 ...]\n", argv[0]); return 1; }
  Call c;
  if (!readCall(argv[1], c)) { fprintf(stderr, "%s: not a MetisLB call dump\n", argv[1]); return 1; }
  const int nseeds = argc > 2 ? atoi(argv[2]) : 32;
  std::vector<double> tols = {0.0};   // 0: the dump's own tolerance
  for (int a = 3; a < argc; a++) tols.push_back(atof(argv[a]));
  printf("%s: %d vertices, %d constraint(s), %zu edge entries, %d parts, %s; ubvec",
         argv[1], (int)c.nv, (int)c.ncon, c.adjncy.size(), (int)c.nparts,
         c.kway ? "k-way" : "recursive");
  for (real_t u : c.ubvec) printf(" %.3f", (double)u);
  printf("\n");
  for (double tol : tols) {
    printf("\n--- load tolerance (ubvec[0]) %s ---\n", tol > 1.0 ? std::to_string(tol).c_str() : "as dumped");
    printf("%8s %12s  worst part / target, per constraint\n", "seed", "cut");
    std::vector<double> load; std::vector<double> cuts;
    std::set<std::vector<idx_t>> distinct;
    for (int s = 0; s <= nseeds; s++) {
      const idx_t seed = s == 0 ? -1 : s;   // -1: libmetis's default, what MetisLB runs
      const Result r = run(c, seed, tol);
      printf("%8s %12ld ", s == 0 ? "default" : std::to_string(s).c_str(), (long)r.cut);
      for (double w : r.worst) printf(" %.4f", w);
      printf("\n");
      load.push_back(r.worst[0]); cuts.push_back((double)r.cut); distinct.insert(r.parts);
    }
    std::vector<double> sl = load; std::sort(sl.begin(), sl.end());
    std::vector<double> sc = cuts; std::sort(sc.begin(), sc.end());
    printf("load worst/target: min %.4f  median %.4f  max %.4f  (default seed %.4f);  "
           "cut: min %.0f median %.0f max %.0f;  %zu distinct partition(s) of %d\n",
           sl.front(), sl[sl.size() / 2], sl.back(), load[0], sc.front(), sc[sc.size() / 2],
           sc.back(), distinct.size(), nseeds + 1);
  }
  return 0;
}
