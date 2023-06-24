// This test creates two array chares: the Pingers and the Pingees

// Each Pinger performs a number of iterations in which it sends as a broadcast
// to the Pingee array the index of the Pinger and the current iteration.
// Notably, this test is called with periodic LB enabled, so chares can migrate
// concurrently with these broadcasts.

// Each Pingee starts with an empty std::unordered_map<int,
// std::unordered_multiset<int>>>. Each time the Pingee is called, it inserts
// the received array index of the Pinger into the std::unordered_multiset<int>
// retrieved from the std::unordered_map at the received iteration.

// At the end, it is checked that the std::unordered_map has an entry at each
// iteration, and that for each iteration, the std::unordered_multiset contains
// one entry for each Pinger.

#include "ping.decl.h"

#include <chrono>
#include <unordered_map>
#include <unordered_set>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Pingers pingersProxy;
/*readonly*/ CProxy_Pingees pingeesProxy;

namespace
{
constexpr std::chrono::milliseconds SPIN_TIME(10);
constexpr int NUM_PINGERS = 10;
constexpr int NUMBER_OF_PINGEES = 1;
constexpr int NUMBER_OF_ITERATIONS = 4;

// wait `time_to_spin` milliseconds
void spin(const std::chrono::milliseconds& timeToSpin)
{
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < timeToSpin)
  {
  }
}
}  // namespace

class Main : public CBase_Main
{
private:
  int migrations{0};

public:
  Main(CkArgMsg* msg)
  {
    delete msg;

    pingersProxy = CProxy_Pingers::ckNew(NUM_PINGERS);
    pingeesProxy = CProxy_Pingees::ckNew(NUMBER_OF_PINGEES);
    CkStartQD(CkCallback(CkIndex_Main::execute(), mainProxy));
  }

  void execute()
  {
    CkPrintf("Main is in phase execute\n");
    pingersProxy.sendPings();
    CkStartQD(CkCallback(CkIndex_Main::check(), mainProxy));
  }

  void check()
  {
    CkPrintf("Main is in phase check\n");
    pingeesProxy.check(migrations);
    CkStartQD(CkCallback(CkIndex_Main::exit(), mainProxy));
  }

  void exit()
  {
    CkPrintf("Main is in phase exit\n");
    CkExit();
  }

  void migrated()
  {
    ++migrations;
    CkPrintf("Migrations done: %i\n", migrations);
  }

  void countErrors(const int errors)
  {
    if (errors > 0)
    {
      CkPrintf("Errors: %i\n", errors);
      CkAbort("Test failed!\n");
    }
  }
};

class Pingers : public CBase_Pingers
{
public:
  Pingers() {}

  Pingers(CkMigrateMessage* /*msg*/) {}

  void sendPings()
  {
    for (int iteration = 1; iteration <= NUMBER_OF_ITERATIONS; ++iteration)
    {
      spin(SPIN_TIME);
      pingeesProxy.receivePing(iteration, thisIndex);
    }
  }
};

class Pingees : public CBase_Pingees
{
private:
  std::unordered_map<int, std::unordered_multiset<int>> pings{};
  int migrations{0};
  int initialProc{-1};

public:
  Pingees() : initialProc(CkMyPe()) {}

  Pingees(CkMigrateMessage* /*msg*/) {}

  void receivePing(const int iteration, const int indexOfPinger)
  {
    pings[iteration].insert(indexOfPinger);
    CkAssert(pings.at(iteration).count(indexOfPinger) == 1);
  }

  void check(const int migrationsRecordedByMain)
  {
    CkAssert(migrations == migrationsRecordedByMain);
    // RotateLB should increase proc at each migration
    CkAssert(CkMyPe() == ((initialProc + migrations) % CkNumPes()));

    int errors = 0;
    for (int i = 1; i <= NUMBER_OF_ITERATIONS; ++i)
    {
      CkAssert(pings.count(i) == 1);
      for (int p = 0; p < NUM_PINGERS; ++p)
      {
        if (pings.at(i).count(p) != 1)
        {
          ++errors;
          CkPrintf(
              "Pingee %i unexpected count %zu on iteration %i for pinger"
              " %i\n",
              thisIndex, pings.at(i).count(p), i, p);
        }
      }
    }

    contribute(sizeof(int), &errors, CkReduction::sum_int,
               CkCallback(CkReductionTarget(Main, countErrors), mainProxy));
  }

  void pup(PUP::er& p)
  {
    p | pings;
    p | migrations;
    p | initialProc;
  }

  void ckJustMigrated()
  {
    ++migrations;
    contribute(CkCallback(CkReductionTarget(Main, migrated), mainProxy));
  }
};

#include "ping.def.h"
