#include <cstdio>
#include <cstdlib>
#include <vector>

#include "main.decl.h"
#include "msgq.h"

#define RUN_TEST(f) f()

const int qSizeMin   = 1<<4;
const int qSizeMax   = 1<<12;
const int qBatchSize = 1<<4;
const int numIters   = 1<<12;
const int numMsgs    = 1<<7;

std::vector<char> msgs(qSizeMax + numMsgs);
std::vector<unsigned int> prios(qSizeMax + numMsgs);


double timePerOp_stlQ(int qBaseSize = 256)
{
  conv::msgQ<int> q;

  for (int i = 0; i < qBaseSize; i++)
      q.enq((conv::msg_t*)&msgs[i], prios[i]);

  double startTime = CmiWallTimer();
  for (int i = 0; i < numIters; i++)
  {
    for (int strt = qBaseSize; strt < qBaseSize + numMsgs; strt += qBatchSize)
    {
      for (int j = strt; j < strt + qBatchSize; j++)
        q.enq((conv::msg_t*)&msgs[j], prios[j]);
      void *m;
      for (int j = 0; j < qBatchSize; j++)
        q.deq();
    }
  }

  return 1000000 * (CmiWallTimer() - startTime) / (numIters * numMsgs * 2);
}


bool perftest_general_ififo()
{
  std::vector<double> timings;
  timings.reserve(256);
  // Charm applications typically have a small/moderate number of different message priorities
  for (int hl = 16; hl <= 128; hl *=2)
  {
    std::srand(42);
    for (int i = 0; i < qSizeMax + numMsgs; i++)
      prios[i] = std::rand() % hl;

    for (int i = qSizeMin; i <= qSizeMax; i *= 2)
      timings.push_back( timePerOp_stlQ(i) );
  }

  #if CMK_HAS_STD_UNORDERED_MAP
  CkPrintf("The STL variant of the msg q is using a std::unordered_map\n");
  #else
  CkPrintf("The STL variant of the msg q is using a std::map\n");
  #endif

  CkPrintf("Reporting time per enqueue / dequeue operation (us) for an STL-based msg Q\n"
           "Nprios (row) is the number of different priority values that are used.\n"
           "Qlen (col) is the base length of the queue on which the enq/deq operations are timed\n"
          );

  CkPrintf("\nversion  Nprios");
  for (int i = qSizeMin; i <= qSizeMax; i*=2)
    CkPrintf("%10d", i);

  for (int hl = 16, j=0; hl <= 128; hl *=2)
  {
    CkPrintf("\n    stl %7d", hl);
    for (int i = qSizeMin; i <= qSizeMax; i *= 2, j++)
      CkPrintf("%10.4f", timings[j]);
  }

  CkPrintf("\n");
  return true;
}

struct main : public CBase_main
{
  main(CkArgMsg *)
  {
    RUN_TEST(perftest_general_ififo); 
    CkExit();
  }
};

#include "main.def.h"

