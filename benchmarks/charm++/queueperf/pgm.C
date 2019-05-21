#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>

using std::cerr;
using std::endl;
using std::sprintf;

#include "queueing.h"
#include "main.decl.h"

#define CmiFree free

#define RUN_TEST(f) f()

const int qSizeMin   = 1<<4;
const int qSizeMax   = 1<<12;
const int qBatchSize = 1<<4;
const int numIters   = 1<<12;
const int numMsgs    = 1<<7;

std::vector<char> msgs(qSizeMax + numMsgs);
std::vector<unsigned int> prios(qSizeMax + numMsgs);

double timePerOp_general_ififo(int qBaseSize = 256)
{
  Queue q = CqsCreate();

  for (int i = 0; i < qBaseSize; i++)
      CqsEnqueueGeneral(q, (void*)&msgs[i], CQS_QUEUEING_IFIFO, 8*sizeof(int), &prios[i]);

  double startTime = CmiWallTimer();
  for (int i = 0; i < numIters; i++)
  {
    for (int strt = qBaseSize; strt < qBaseSize + numMsgs; strt += qBatchSize)
    {
      for (int j = strt; j < strt + qBatchSize; j++)
        CqsEnqueueGeneral(q, (void*)&msgs[j], CQS_QUEUEING_IFIFO, 8*sizeof(int), &prios[j]);
      void *m;
      for (int j = 0; j < qBatchSize; j++)
        CqsDequeue(q, &m);
    }
  }

  CqsDelete(q);
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
      timings.push_back( timePerOp_general_ififo(i) );
  }

  CkPrintf("Reporting time per enqueue / dequeue operation (us) for charm's underlying mixed priority queue\n"
           "Nprios (row) is the number of different priority values that are used.\n"
           "Qlen (col) is the base length of the queue on which the enq/deq operations are timed\n"
          );

  CkPrintf("\nversion  Nprios");
  for (int i = qSizeMin; i <= qSizeMax; i*=2)
    CkPrintf("%10d", i);

  for (int hl = 16, j=0; hl <= 128; hl *=2)
  {
    CkPrintf("\n  charm %7d", hl);
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
