// Every PE looks up many nodegroups with ckLocalBranch() while the
// nodegroups are still being created on its node. The lookups of a
// nodegroup that is ready take no lock; the first lookups of each one may
// arrive before it exists and wait for it. Each lookup must return the
// object constructed with that nodegroup's tag. The count (default 600)
// spans several blocks of the lookup table (256 ids per block).
#include "nodegroup_lookup.decl.h"
#include <vector>

/* readonly */ CProxy_Main mainProxy;

class Tagged : public CBase_Tagged
{
public:
  const int tag;
  explicit Tagged(int t) : tag(t)
  {
    // A lookup of itself from the constructor returns this object.
    if (thisProxy.ckLocalBranch() != this) CkAbort("constructor lookup returned another object");
  }
};

class Main : public CBase_Main
{
  int count = 600, rounds = 200;

public:
  explicit Main(CkArgMsg* m)
  {
    if (m->argc > 1) count = atoi(m->argv[1]);
    if (m->argc > 2) rounds = atoi(m->argv[2]);
    delete m;
    mainProxy = thisProxy;

    CProxy_Reader readers = CProxy_Reader::ckNew();
    std::vector<CkGroupID> ids;
    for (int i = 0; i < count; i++)
    {
      ids.push_back(CProxy_Tagged::ckNew(i));
      // Lookups of the first nodegroups run while later ones are created.
      if (i % 100 == 99) readers.lookup(ids, rounds, count);
    }
    readers.lookup(ids, rounds, count);
  }

  void done(long sum)
  {
    long expected = 0;
    for (int n = 100; n <= count; n += 100) expected += (long)n * (n - 1) / 2;
    expected += (long)count * (count - 1) / 2;
    expected *= (long)rounds * CkNumPes();
    if (sum != expected) CkAbort("tag sum %ld, expected %ld", sum, expected);
    CkPrintf("nodegroup_lookup: %d nodegroups, %d rounds, %d PEs: passed\n", count, rounds,
             CkNumPes());
    CkExit();
  }
};

class Reader : public CBase_Reader
{
  int calls = 0;
  long sum = 0;

public:
  Reader() {}

  void lookup(std::vector<CkGroupID> ids, int rounds, int count)
  {
    for (int r = 0; r < rounds; r++)
      for (size_t i = 0; i < ids.size(); i++)
      {
        Tagged* t = CProxy_Tagged(ids[i]).ckLocalBranch();
        if (t == nullptr || t->tag != (int)i)
          CkAbort("lookup %d returned the wrong nodegroup", (int)i);
        sum += t->tag;
      }
    // Main sends one call per 100 nodegroups and a final one.
    if (++calls == count / 100 + 1)
      contribute(sizeof(long), &sum, CkReduction::sum_long,
                 CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

#include "nodegroup_lookup.def.h"
