// Nodegroup lookups (ckLocalBranch) and nodegroup messages while nodegroups
// are being created, after start-up.
//
// Main creates nPos nodegroups (default 800, spanning four 256-id blocks of
// the lookup table) from PE 0 in batches of 8, one batch per message. With
// more than one process, the first PE of process 1 creates nNeg = 200 more;
// ids created there are negative. Node 0 hands out ids in sequence, and so
// does node 1 (downwards), so after the first nodegroup of each creator the
// ids of the later ones are known in advance; the creators check that each
// ckNew returns the expected id. Every nodegroup stores its tag (its index,
// counting the negative-id ones after the positive-id ones).
//
// Meanwhile one Worker per PE runs `rounds` rounds, one message each. A
// round picks a target id at or ahead of creation, possibly in a block that
// is not allocated yet, sends a message to the target's branch on this node
// (it may arrive before the nodegroup exists and is then held), looks the
// target up (waiting inside ckLocalBranch until it exists) and then looks up
// up to 16 earlier ids. The same is done for a negative-id target. Every
// lookup checks the returned object's tag.
//
// The creator also looks up each nodegroup right after its ckNew returns,
// and sends it a message on every node; on another process that message can
// arrive before the creation message.
//
// After quiescence, each process checks that each nodegroup received every
// message sent to it exactly once, and Main checks the total numbers of
// lookups and messages.
#include "nodegroup_lookup.decl.h"
#include <algorithm>
#include <atomic>
#include <vector>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Worker workers;
/* readonly */ int nPos;
/* readonly */ int nNeg;
/* readonly */ int rounds;

enum { MAX_TAGS = 4096, BATCH = 8, POS_BACK = 16, POS_STRIDE = 23, NEG_BACK = 8, NEG_STRIDE = 7 };

// Messages received per tag in this process.
static std::atomic<int> hitCount[MAX_TAGS];

static int posTarget(int r, int pe)
{
  return std::min<long>(nPos - 1, (long)r * nPos / rounds + (pe * 37 + r * 11) % 300);
}
static int negTarget(int r, int pe)
{
  return std::min<long>(nNeg - 1, (long)r * nNeg / rounds + (pe * 13 + r * 7) % 80);
}
// Lookups made for a target t: t itself and t - k*stride for k = 1..back while >= 0.
static long lookupsFor(int t, int back, int stride) { return 1 + std::min(back, t / stride); }

static CkGroupID makeId(int idx)
{
  CkGroupID g;
  g.idx = idx;
  return g;
}

class Tagged : public CBase_Tagged
{
public:
  const int tag;
  explicit Tagged(int t) : tag(t)
  {
    // A lookup of itself from the constructor returns this object.
    if (thisProxy.ckLocalBranch() != this) CkAbort("constructor lookup returned another object");
  }
  void hit(int t)
  {
    if (t != tag) CkAbort("message for tag %d delivered to tag %d", t, tag);
    hitCount[tag]++;
  }
};

static void verify(int idx, int tag)
{
  Tagged* o = CProxy_Tagged(makeId(idx)).ckLocalBranch();
  if (o == nullptr || o->tag != tag)
    CkAbort("PE %d: lookup of nodegroup %d returned %d, expected tag %d", CkMyPe(), idx,
            o ? o->tag : -1, tag);
}

// Creates a nodegroup, checks its id (if expectIdx != 0), looks it up at
// once, and sends it a message on every node.
static int create(int tag, int expectIdx)
{
  CkGroupID id = CProxy_Tagged::ckNew(tag);
  if (expectIdx != 0 && id.idx != expectIdx)
    CkAbort("nodegroup %d got id %d, expected %d (another nodegroup created meanwhile?)", tag,
            id.idx, expectIdx);
  verify(id.idx, tag);
  CProxy_Tagged(id).hit(tag);
  return id.idx;
}

class Main : public CBase_Main
{
  int posFirst = 0, pending = 0;
  long lookups = 0;

public:
  explicit Main(CkArgMsg* m)
  {
    nPos = m->argc > 1 ? atoi(m->argv[1]) : 800;
    rounds = m->argc > 2 ? atoi(m->argv[2]) : 300;
    nNeg = CkNumNodes() > 1 ? 200 : 0;
    delete m;
    if (nPos < 1 || nPos + nNeg > MAX_TAGS || rounds < 1) CkAbort("bad arguments");
    mainProxy = thisProxy;
    workers = CProxy_Worker::ckNew();
    // readers, positive-id creation, negative-id creation
    pending = nNeg > 0 ? 3 : 2;
    thisProxy.start();
  }

  void start()
  {
    posFirst = create(0, 0);
    if (nNeg > 0)
      workers[CkNodeFirst(1)].createFirstNeg();
    else
      go(0);
  }

  void go(int negFirst)
  {
    workers.startReading(posFirst, negFirst);
    thisProxy.batch(1);
  }

  void batch(int first)
  {
    int last = std::min(first + BATCH, nPos);
    for (int i = first; i < last; i++) create(i, posFirst + i);
    if (last < nPos)
      thisProxy.batch(last);
    else
      doneOne();
  }

  void readersDone(long n)
  {
    lookups = n;
    doneOne();
  }

  void doneOne()
  {
    if (--pending == 0) CkStartQD(CkCallback(CkIndex_Main::quiet(), thisProxy));
  }

  void quiet() { workers.check(); }

  void finish(long hits)
  {
    long expectLookups = 0;
    for (int pe = 0; pe < CkNumPes(); pe++)
      for (int r = 0; r < rounds; r++)
      {
        expectLookups += lookupsFor(posTarget(r, pe), POS_BACK, POS_STRIDE);
        if (nNeg > 0) expectLookups += lookupsFor(negTarget(r, pe), NEG_BACK, NEG_STRIDE);
      }
    long expectHits = (long)CkNumNodes() * (nPos + nNeg) +
                      (long)CkNumPes() * rounds * (nNeg > 0 ? 2 : 1);
    if (lookups != expectLookups) CkAbort("%ld lookups, expected %ld", lookups, expectLookups);
    if (hits != expectHits) CkAbort("%ld messages delivered, expected %ld", hits, expectHits);
    CkPrintf("nodegroup_lookup: %d+%d nodegroups, %d rounds, %d PEs, %d processes, %ld lookups, "
             "%ld messages: passed\n",
             nPos, nNeg, rounds, CkNumPes(), CkNumNodes(), lookups, hits);
    CkExit();
  }
};

class Worker : public CBase_Worker
{
  int posFirst = 0, negFirst = 0;
  long lookups = 0;

public:
  Worker() {}

  // On the first PE of process 1.
  void createFirstNeg() { mainProxy.go(create(nPos, 0)); }

  void createNeg(int first)
  {
    int last = std::min(first + BATCH, nNeg);
    for (int j = first; j < last; j++) create(nPos + j, negFirst - j);
    if (last < nNeg)
      thisProxy[CkMyPe()].createNeg(last);
    else
      mainProxy.doneOne();
  }

  void startReading(int pos, int neg)
  {
    posFirst = pos;
    negFirst = neg;
    if (nNeg > 0 && CkMyPe() == CkNodeFirst(1)) thisProxy[CkMyPe()].createNeg(1);
    thisProxy[CkMyPe()].round(0);
  }

  void round(int r)
  {
    int t = posTarget(r, CkMyPe());
    CProxy_Tagged(makeId(posFirst + t))[CkMyNode()].hit(t);
    for (int k = 0; k <= POS_BACK && t - k * POS_STRIDE >= 0; k++, lookups++)
      verify(posFirst + t - k * POS_STRIDE, t - k * POS_STRIDE);
    if (nNeg > 0)
    {
      int u = negTarget(r, CkMyPe());
      CProxy_Tagged(makeId(negFirst - u))[CkMyNode()].hit(nPos + u);
      for (int k = 0; k <= NEG_BACK && u - k * NEG_STRIDE >= 0; k++, lookups++)
        verify(negFirst - (u - k * NEG_STRIDE), nPos + u - k * NEG_STRIDE);
    }
    if (r + 1 < rounds)
      thisProxy[CkMyPe()].round(r + 1);
    else
      contribute(sizeof(long), &lookups, CkReduction::sum_long,
                 CkCallback(CkReductionTarget(Main, readersDone), mainProxy));
  }

  // Each nodegroup on this node got one message from its creator plus one
  // per round of a Worker of this node that targeted it.
  void check()
  {
    long hits = 0;
    if (CkMyRank() == 0)
    {
      std::vector<int> expect(nPos + nNeg, 1);
      for (int pe = CkNodeFirst(CkMyNode()); pe < CkNodeFirst(CkMyNode()) + CkMyNodeSize(); pe++)
        for (int r = 0; r < rounds; r++)
        {
          expect[posTarget(r, pe)]++;
          if (nNeg > 0) expect[nPos + negTarget(r, pe)]++;
        }
      for (int i = 0; i < nPos + nNeg; i++)
      {
        if (hitCount[i] != expect[i])
          CkAbort("process %d: tag %d received %d messages, expected %d", CkMyNode(), i,
                  hitCount[i].load(), expect[i]);
        hits += expect[i];
      }
    }
    contribute(sizeof(long), &hits, CkReduction::sum_long,
               CkCallback(CkReductionTarget(Main, finish), mainProxy));
  }
};

#include "nodegroup_lookup.def.h"
