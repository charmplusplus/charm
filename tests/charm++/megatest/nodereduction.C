/*
 * nodereduction: nodegroup reductions whose contributions reach a process
 * before the reduction they belong to has started there.
 *
 * Each process's nodegroup branch contributes NRED reductions back to back
 * (burst). Two phases, in this order:
 *   0. every process but process 0 bursts; after quiescence process 0 does.
 *      The root then holds the remote contributions to the later reductions
 *      before its own first one (CkNodeReductionMgr futureRemoteMsgs).
 *   1. process 0 bursts first; after quiescence the others do. The root's
 *      own contributions to the later reductions wait for the earlier ones
 *      to finish (futureMsgs), and finishReduction takes them off the queue.
 * Reduction r of a burst uses CkReduction::set when r == SETRED, sum_int
 * otherwise (the root reduces with a streamable and an unstreamable reducer).
 * The driver checks every result and that each reduction of a phase arrives
 * exactly once. With one process the phases still run; nothing arrives early.
 */
#include "megatest.h"
#include "nodereduction.decl.h"
#include <vector>

static const int NRED = 4;
static const int SETRED = 2;

class nodereduction_ng : public CBase_nodereduction_ng
{
 public:
  nodereduction_ng(void) {}
  nodereduction_ng(CkMigrateMessage* m) {}

  void burst(CProxy_nodereduction_driver driver, int phase)
  {
    const int node = CkMyNode();
    for (int r = 0; r < NRED; r++)
    {
      if (r == SETRED)
      {
        int v[2] = {phase * NRED + r, node};
        contribute(sizeof(v), v, CkReduction::set,
                   CkCallback(CkIndex_nodereduction_driver::setDone(NULL), driver));
      }
      else
      {
        // [contributors, reduction id, sum of node+1]
        int v[3] = {1, phase * NRED + r, node + 1};
        contribute(sizeof(v), v, CkReduction::sum_int,
                   CkCallback(CkReductionTarget(nodereduction_driver, sumDone), driver));
      }
    }
  }
};

class nodereduction_driver : public CBase_nodereduction_driver
{
  CProxy_nodereduction_ng ng;
  int phase = 0;
  std::vector<int> seen;
  int nSeen = 0;

  void got(int id)
  {
    if (id / NRED != phase || seen[id % NRED]++ != 0)
      CkAbort("nodereduction: reduction %d arrived twice or in phase %d\n", id, phase);
    if (++nSeen < NRED) return;
    if (++phase == 2)
    {
      megatest_finish();
      return;
    }
    startPhase();
  }

  void startPhase()
  {
    seen.assign(NRED, 0);
    nSeen = 0;
    if (phase == 0) startRest();
    else startRoot();
  }

 public:
  nodereduction_driver(void)
  {
    ng = CProxy_nodereduction_ng::ckNew();
    startPhase();
  }
  nodereduction_driver(CkMigrateMessage* m) {}

  // Phase 0 bursts on every process but 0, then on process 0 after
  // quiescence; phase 1 the other way round.
  void startRest(void)
  {
    for (int n = 1; n < CkNumNodes(); n++) ng[n].burst(thisProxy, phase);
    if (phase == 0)
      CkStartQD(CkCallback(CkIndex_nodereduction_driver::startRoot(), thisProxy));
  }
  void startRoot(void)
  {
    ng[0].burst(thisProxy, phase);
    if (phase == 1)
      CkStartQD(CkCallback(CkIndex_nodereduction_driver::startRest(), thisProxy));
  }

  void sumDone(int n, int* v)
  {
    const int N = CkNumNodes();
    if (n != 3 || v[0] != N || v[1] % N != 0 || v[2] != N * (N + 1) / 2)
      CkAbort("nodereduction: sum result n=%d [%d %d %d] on %d processes\n", n, v[0],
              n > 1 ? v[1] : -1, n > 2 ? v[2] : -1, N);
    const int id = v[1] / N;
    if (id % NRED == SETRED) CkAbort("nodereduction: sum result for the set reduction\n");
    got(id);
  }

  void setDone(CkReductionMsg* m)
  {
    const int N = CkNumNodes();
    std::vector<int> nodes(N, 0);
    int id = -1, count = 0;
    for (CkReduction::setElement* e = (CkReduction::setElement*)m->getData();
         e != NULL; e = e->next())
    {
      const int* v = (const int*)e->data;
      if (e->dataSize != 2 * sizeof(int) || v[1] < 0 || v[1] >= N || nodes[v[1]]++ != 0 ||
          (id >= 0 && v[0] != id))
        CkAbort("nodereduction: bad set element (size %d)\n", e->dataSize);
      id = v[0];
      count++;
    }
    delete m;
    if (count != N || id % NRED != SETRED)
      CkAbort("nodereduction: set result has %d elements for reduction %d\n", count, id);
    got(id);
  }
};

void nodereduction_init(void) { CProxy_nodereduction_driver::ckNew(0); }
void nodereduction_moduleinit(void) {}

MEGATEST_REGISTER_TEST(nodereduction, "kale", 0)
#include "nodereduction.def.h"
