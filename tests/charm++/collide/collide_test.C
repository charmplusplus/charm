/*
 Collide library correctness test.

 nChunks contributors (a 1D array) each submit a deterministic set of boxes
 to one collider in every step.  The voxel array inside the library is
 created without bounds, so its 3D voxel elements use hashed object ids, and
 the voxels are demand-created through the [createhere] entry
 collideVoxel::add.

 Four collision steps are run.  Step s places its boxes in the 8x8x8 block of
 voxels that starts at x = 2*(s-1), so every step after the first sends boxes
 to voxels that did not exist before; collideMgr must count its local voxels
 again in each step.  Between steps every contributor migrates to another PE
 (re-registering with the collider on arrival).  With fewer chunks than PEs,
 some PEs have no registered contributor in a step and must still take part
 in the step.  For each step the list of collisions reported by the library
 is compared, pair by pair, with a brute-force O(N^2) overlap test computed
 here on the same boxes.

 By default the collider has a serial client, which delivers all collisions
 to Main in one reduction.  With "dist" it has a distributed client: each PE
 receives the collisions found in its own voxels, at the local branch of the
 Collector group, which forwards them to Main in a concat reduction.  Main
 checks the same complete list either way; the reduction completes only if
 every PE received exactly one list in the step.

 Usage: ./collide_test [nChunks] [dist]   (default 32 chunks, serial client)
 Needs the collidecharm module, which only a `./build LIBS ...` (or
 `./build charm++ ... ` followed by building ck-libs) provides; the Makefile
 skips the test when the module is missing.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include "collidecharm.h"
#include "collide_test.decl.h"

CProxy_Main mainProxy;
int nChunks;

static const int nSteps = 4;
static const int domain = 8;      // voxels per axis, voxel size 1.0
static const int stepShift = 2;   // voxels the block moves along x per step

// Deterministic generator shared by the contributors and the checker.
struct Rng {
  unsigned long long s;
  Rng(unsigned long long seed) : s(seed * 0x9E3779B97F4A7C15ULL + 12345) { next(); }
  unsigned long long next() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
  }
  double uniform(double lo, double hi) {
    return lo + (hi - lo) * ((next() >> 11) * (1.0 / 9007199254740992.0));
  }
};

static void makeBoxes(int step, int chunk, std::vector<bbox3d> &boxes)
{
  boxes.clear();
  Rng r(1000ULL * step + chunk);
  int n = 40 + (chunk * 7 + step * 13) % 30;
  double maxSz = (step % 3 == 1) ? 0.5 : (step % 3 == 2 ? 0.8 : 0.3);
  double origin[3] = {double(stepShift * (step - 1)), 0.0, 0.0};
  for (int i = 0; i < n; i++) {
    double lo[3], sz[3];
    for (int a = 0; a < 3; a++) {
      sz[a] = r.uniform(0.05, maxSz);
      lo[a] = origin[a] + r.uniform(0.0, domain - sz[a]);
    }
    bbox3d b; b.empty();
    b.add(CkVector3d(lo[0], lo[1], lo[2]));
    b.add(CkVector3d(lo[0] + sz[0], lo[1] + sz[1], lo[2] + sz[2]));
    boxes.push_back(b);
  }
}

typedef std::tuple<int, int, int, int> Pair;

static Pair normPair(int ca, int na, int cb, int nb)
{
  if (std::make_pair(ca, na) > std::make_pair(cb, nb)) {
    std::swap(ca, cb); std::swap(na, nb);
  }
  return Pair(ca, na, cb, nb);
}

class Main : public CBase_Main {
  CollideGrid3d grid;
  int step;
  int nPassed;
  CProxy_Chunk chunks;
 public:
  Main(CkArgMsg *m)
    : grid(CkVector3d(0, 0, 0), CkVector3d(1, 1, 1)), step(0), nPassed(0)
  {
    nChunks = 32;
    bool dist = false;
    for (int i = 1; i < m->argc; i++) {
      if (strcmp(m->argv[i], "dist") == 0) dist = true;
      else if (m->argv[i][0] != '+') nChunks = atoi(m->argv[i]);
    }
    delete m;
    mainProxy = thisProxy;
    CkPrintf("collide_test: %d chunks, %d steps on %d PEs, %s client\n",
             nChunks, nSteps, CkNumPes(), dist ? "distributed" : "serial");
    CollideHandle collide;
    if (dist) {
      CProxy_Collector coll = CProxy_Collector::ckNew();
      collide = CollideCreate(grid, CollideDistributedClient(
          CkCallback(CkIndex_Collector::colls(NULL), coll)));
    } else {
      collide = CollideCreate(grid, CollideSerialClient(
          CkCallback(CkIndex_Main::collisions(NULL), thisProxy)));
    }
    chunks = CProxy_Chunk::ckNew(collide, nChunks);
    step = 1;
    chunks.doStep(step);
  }

  void expected(std::vector<Pair> &out, int &nVox, int &nBoxes)
  {
    struct Rec { int chunk, num; bbox3d b; };
    std::vector<Rec> all;
    std::vector<bbox3d> boxes;
    for (int c = 0; c < nChunks; c++) {
      makeBoxes(step, c, boxes);
      for (int i = 0; i < (int)boxes.size(); i++) all.push_back(Rec{c, i, boxes[i]});
    }
    nBoxes = all.size();
    std::vector<std::tuple<int, int, int>> vox;
    for (const Rec &r : all) {
      iSeg1d s[3];
      for (int a = 0; a < 3; a++) s[a] = grid.world2grid(a, r.b.axis(a));
      for (int z = s[2].getMin(); z < s[2].getMax(); z++)
        for (int y = s[1].getMin(); y < s[1].getMax(); y++)
          for (int x = s[0].getMin(); x < s[0].getMax(); x++)
            vox.push_back(std::make_tuple(x, y, z));
    }
    std::sort(vox.begin(), vox.end());
    nVox = std::unique(vox.begin(), vox.end()) - vox.begin();
    out.clear();
    for (size_t i = 0; i < all.size(); i++)
      for (size_t j = i + 1; j < all.size(); j++) {
        if (all[i].chunk == all[j].chunk) continue;  // same priority never collides
        if (!all[i].b.intersectsOpen(all[j].b)) continue;
        out.push_back(normPair(all[i].chunk, all[i].num, all[j].chunk, all[j].num));
      }
    std::sort(out.begin(), out.end());
  }

  void collisions(CkReductionMsg *msg)
  {
    Collision *colls = (Collision *)msg->getData();
    int nColl = msg->getSize() / sizeof(Collision);
    std::vector<Pair> got;
    for (int i = 0; i < nColl; i++)
      got.push_back(normPair(colls[i].A.chunk, colls[i].A.number,
                             colls[i].B.chunk, colls[i].B.number));
    delete msg;
    std::sort(got.begin(), got.end());
    std::vector<Pair> want;
    int nVox, nBoxes;
    expected(want, nVox, nBoxes);
    int dups = got.end() - std::unique(got.begin(), got.end());
    if (dups == 0 && got == want) {
      nPassed++;
      CkPrintf("step %d: PASS %d collisions (%d boxes, %d voxels touched)\n",
               step, nColl, nBoxes, nVox);
    } else {
      int extra = 0, missing = 0;
      std::vector<Pair> d;
      std::set_difference(got.begin(), got.end(), want.begin(), want.end(),
                          std::back_inserter(d));
      extra = d.size(); d.clear();
      std::set_difference(want.begin(), want.end(), got.begin(), got.end(),
                          std::back_inserter(d));
      missing = d.size();
      CkPrintf("step %d: FAIL got %d collisions, expected %d "
               "(%d duplicates, %d extra, %d missing)\n",
               step, nColl, (int)want.size(), dups, extra, missing);
      CkAbort("collide_test: collision list mismatch");
    }
    if (step == nSteps) {
      CkPrintf("collide_test: all %d steps passed\n", nPassed);
      CkExit();
      return;
    }
    chunks.migrateAway(step);
  }

  void migrated()
  {
    step++;
    chunks.doStep(step);
  }
};

class Collector : public CBase_Collector {
 public:
  Collector() {}
  void colls(CkDataMsg *m)
  {
    contribute(m->getSize(), m->getData(), CkReduction::concat,
               CkCallback(CkIndex_Main::collisions(NULL), mainProxy));
    delete m;
  }
};

class Chunk : public CBase_Chunk {
  CollideHandle collide;
  bool reportArrival;
 public:
  Chunk(const CollideHandle &c) : collide(c), reportArrival(false)
  {
    CollideRegister(collide, thisIndex);
  }
  Chunk(CkMigrateMessage *m) : CBase_Chunk(m), reportArrival(false) {}
  ~Chunk() { CollideUnregister(collide, thisIndex); }

  void pup(PUP::er &p)
  {
    p | collide;
    p | reportArrival;
    if (p.isUnpacking()) CollideRegister(collide, thisIndex);
  }

  void doStep(int step)
  {
    std::vector<bbox3d> boxes;
    makeBoxes(step, thisIndex, boxes);
    CollideBoxesPrio(collide, thisIndex, boxes.size(), boxes.data(), NULL);
  }

  void migrateAway(int step)
  {
    int dest = (CkMyPe() + 1 + (thisIndex + step) % 2) % CkNumPes();
    if (dest == CkMyPe()) {
      arrived();
    } else {
      reportArrival = true;
      migrateMe(dest);
    }
  }

  void ckJustMigrated()
  {
    CBase_Chunk::ckJustMigrated();
    if (reportArrival) {
      reportArrival = false;
      arrived();
    }
  }

  void arrived()
  {
    contribute(CkCallback(CkReductionTarget(Main, migrated), mainProxy));
  }
};

#include "collide_test.def.h"
