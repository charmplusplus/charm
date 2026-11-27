// Every PE sends a [createhere] message to the same N indices of an empty
// array at once. Each index must be constructed exactly once and receive one
// message from every PE. Phase 0 uses an unbounded array (ids minted per PE),
// phase 1 a bounded one (ids computed from the index).
// Arguments: [N [first phase]], default 200 0.
#include "createhere_race.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Sender senderProxy;
/*readonly*/ CProxy_Elem unboundedProxy;
/*readonly*/ CProxy_Elem boundedProxy;
/*readonly*/ int numIndices;

class Main : public CBase_Main
{
  int phase = -1;

public:
  Main(CkArgMsg* m)
  {
    numIndices = m->argc > 1 ? atoi(m->argv[1]) : 200;
    phase = (m->argc > 2 ? atoi(m->argv[2]) : 0) - 1;
    delete m;
    mainProxy = thisProxy;
    unboundedProxy = CProxy_Elem::ckNew();
    unboundedProxy.doneInserting();
    CkArrayOptions opts;
    opts.setBounds(numIndices);
    boundedProxy = CProxy_Elem::ckNew(opts);
    boundedProxy.doneInserting();
    senderProxy = CProxy_Sender::ckNew();
    CkPrintf("createhere_race: %d indices, %d PEs\n", numIndices, CkNumPes());
    startPhase();
  }

  void quiet()
  {
    senderProxy.report();
  }

  void startPhase()
  {
    phase++;
    senderProxy.go(phase);
    CkStartQD(CkCallback(CkIndex_Main::quiet(), thisProxy));
  }

  void done(int n, int* counts)
  {
    const int constructions = counts[0], messages = counts[1];
    const char* kind = phase == 0 ? "unbounded" : "bounded";
    if (constructions != numIndices || messages != numIndices * CkNumPes())
      CkAbort("createhere_race (%s): %d constructions (expected %d), %d messages "
              "(expected %d)\n",
              kind, constructions, numIndices, messages, numIndices * CkNumPes());
    CkPrintf("createhere_race (%s): %d constructions, %d messages: passed\n", kind,
             constructions, messages);
    if (phase == 1) CkExit();
    else startPhase();
  }
};

class Sender : public CBase_Sender
{
public:
  int constructions = 0;
  int messages = 0;

  Sender() {}

  void go(int phase)
  {
    CProxy_Elem target = phase == 0 ? unboundedProxy : boundedProxy;
    for (int i = 0; i < numIndices; i++) target[i].ping(CkMyPe());
  }

  void report()
  {
    int counts[2] = {constructions, messages};
    constructions = messages = 0;
    contribute(sizeof(counts), counts, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

class Elem : public CBase_Elem
{
public:
  Elem() { senderProxy.ckLocalBranch()->constructions++; }
  Elem(CkMigrateMessage*) {}

  void ping(int) { senderProxy.ckLocalBranch()->messages++; }
};

#include "createhere_race.def.h"
