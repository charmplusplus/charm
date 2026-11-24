#include <stdio.h>
#include <string.h>
#include <vector>
#include "detour.decl.h"

// A sender in a third process keeps an entry for an element's dead life; the
// element is re-created elsewhere at epoch 0, so every later send from that
// process detours through the dead life's PE. The delivery repairs the sender's
// entry; the test checks that the entry names the new life within a few rounds
// (doc/internals/process-location-cache.md, section 3.5).
//
// Roles: P holds the first life, B the second, T sends. T is in the last
// process, P in another process than B when there are three or more, else on
// the second PE of process 0. One element per index, for every home PE, so the
// home's process takes each role in turn.

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem arrProxy;
/*readonly*/ CProxy_Sender senderProxy;

/*readonly*/ int N;  // elements
/*readonly*/ bool verbose;

class Elem : public CBase_Elem
{
public:
  Elem() {}
  Elem(CkMigrateMessage*) {}
  void ping(int) {}
  void die() { ckDestroy(); }
};

static const int Rounds = 4;  // sends after the re-creation
static const int MustBy = 3;  // the entries name the new life after this round

class Main : public CBase_Main
{
  int P, B, T, phase = 0, failures = 0;

public:
  Main(CkArgMsg* m)
  {
    verbose = m->argc > 1 && strcmp(m->argv[1], "-v") == 0;
    delete m;
    mainProxy = thisProxy;
    const int nodes = CkNumNodes();
    T = CkNodeFirst(nodes - 1);
    B = 0;
    if (nodes >= 3)
      P = CkNodeFirst(1);
    else if (nodes == 2 && CkNodeSize(0) >= 2)
      P = 1;
    else
    {
      CkPrintf("objid_detour: needs two processes with two PEs in the first, or three"
               " processes; skipped\n");
      CkPrintf("All done\n");
      CkExit();
      return;
    }
    N = 4 * CkNumPes();
    CkPrintf("objid_detour: %d PEs in %d processes; first life on PE %d, second on"
             " PE %d, sender PE %d, %d elements\n", CkNumPes(), nodes, P, B, T, N);
    CkArrayOptions opts;
    opts.setBounds(N);  // packed kind: the re-created element has the same id
    arrProxy = CProxy_Elem::ckNew(opts);
    senderProxy = CProxy_Sender::ckNew();
    for (int i = 0; i < N; i++) arrProxy[i].insert(P);
    arrProxy.doneInserting();
    next();
  }

  void next() { CkStartQD(CkCallback(CkIndex_Main::step(), thisProxy)); }

  // Phases, each after quiescence: 0, 1 T sends (learning the first life);
  // 2 T checks; 3 the first life dies; 4 the second life is created on B;
  // then T sends and checks, Rounds times.
  void step()
  {
    const int p = phase++;
    if (p < 2)
    {
      senderProxy[T].ping(-1);
      next();
    }
    else if (p == 2)
      senderProxy[T].check(0, P);
    else if (p == 3)
    {
      for (int i = 0; i < N; i++) arrProxy[i].die();
      next();
    }
    else if (p == 4)
    {
      for (int i = 0; i < N; i++) arrProxy[i].insert(B);
      next();
    }
    else if ((p - 5) % 2 == 0)
    {
      senderProxy[T].ping((p - 5) / 2 + 1);
      next();
    }
    else
      senderProxy[T].check((p - 5) / 2 + 1, B);
  }

  void report(int r, std::vector<int> list, int checked)
  {
    const int stale = list.size() / 3;
    CkPrintf("objid_detour: round %d: %d of %d entries name neither PE %d nor (home"
             " there) nothing\n", r, stale, checked, r == 0 ? P : B);
    if (verbose || r == 0 || r == MustBy)
      for (int k = 0; k < stale; k++)
        CkPrintf("objid_detour:   index %d (home PE %d) names PE %d\n", list[3 * k],
                 list[3 * k + 1], list[3 * k + 2]);
    if ((r == 0 || r >= MustBy) && stale != 0) failures++;
    if (r == Rounds)
    {
      if (failures)
        CkAbort("objid_detour: %d checks failed (the first life not learned, or the"
                " dead life still named after round %d)\n", failures, MustBy - 1);
      CkPrintf("All done\n");
      CkExit();
      return;
    }
    next();
  }
};

class Sender : public CBase_Sender
{
public:
  Sender() {}
  void ping(int round)
  {
    for (int i = 0; i < N; i++) arrProxy[i].ping(round);
  }
  void check(int round, int expectPe)
  {
    CkLocMgr* lm = arrProxy.ckLocMgr();
    std::vector<int> stale;  // index, home PE, the PE the entry names
    for (int i = 0; i < N; i++)
    {
      // No entry is as good when the home holds the element: sends go there.
      const int pe = lm->whichPe(CkArrayIndex1D(i));
      const int home = lm->homePe(CkArrayIndex1D(i));
      if (pe != expectPe && !(pe == -1 && home == expectPe))
      {
        stale.push_back(i);
        stale.push_back(home);
        stale.push_back(pe);
      }
    }
    mainProxy.report(round, stale, N);
  }
};

#include "detour.def.h"
