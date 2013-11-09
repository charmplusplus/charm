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

#define RUN_TEST(f) do { \
  ++tests; \
  if (f()) { \
    ++success; \
  } else { \
    ++fail; \
    cerr << "Test \"" #f "\" failed" << endl; \
  } \
} while (0)

// A newly created Queue should be empty, which corresponds to a
// length of 0
bool test_empty()
{
  Queue q = CqsCreate();
  bool result = (0 == CqsLength(q)) && (1 == CqsEmpty(q));
  CqsDelete(q);
  return result;
}

// Enqueueing an element should show that there is an element
// present. We should get the same thing back when we dequeue
//
// The queue is not allowed to dereference the void* we give it
bool test_one()
{
  Queue q = CqsCreate();
  void *p = 0;
  CqsEnqueue(q, p);
  bool result = (1 == CqsLength(q)) && (0 == CqsEmpty(q));
  void *r;
  CqsDequeue(q, &r);
  result &= (r == p) 
    && (0 == CqsLength(q)) 
    && (1 == CqsEmpty(q));
  CqsDelete(q);
  return result;
}

// Put two in, see if we get the same two back. Order unspecified
bool test_two()
{
  Queue q = CqsCreate();
  void *i = 0, *j = (char *)1;
  void *r, *s;
  CqsEnqueue(q, i);
  CqsEnqueue(q, j);
  bool result = (2 == CqsLength(q));
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  result &= (r == i && s == j) || (r == j && s == i);
  result &= 1 == CqsEmpty(q);
  CqsDelete(q);
  return result;
}

bool test_fifo()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  CqsEnqueueFifo(q, i);
  CqsEnqueueFifo(q, j);
  CqsEnqueueFifo(q, k);
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  CqsDequeue(q, &t);
  bool result = (r == i) && (s == j) && (t == k);
  CqsDelete(q);
  return result;
}

bool test_lifo()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  CqsEnqueueLifo(q, i);
  CqsEnqueueLifo(q, j);
  CqsEnqueueLifo(q, k);
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  CqsDequeue(q, &t);
  bool result = (r == k) && (s == j) && (t == i);
  CqsDelete(q);
  return result;
}

bool test_enqueue_mixed()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  CqsEnqueueFifo(q, i);
  CqsEnqueueFifo(q, j);
  CqsEnqueueLifo(q, k);
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  CqsDequeue(q, &t);
  bool result = (r == k) && (s == i) && (t == j);
  CqsDelete(q);
  return result;
}

static bool findEntry(void ** const e, int num, void * const t)
{
    for (int i = 0; i < num; ++i)
    {
	if (e[i] == t)
	    return true;
    }
    return false;
}

bool test_enumerate()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void **e;
  CqsEnqueueFifo(q, i);
  CqsEnqueueFifo(q, j);
  CqsEnqueueLifo(q, k);
  CqsEnumerateQueue(q, &e);
  int n = CqsLength(q);
  bool result = findEntry(e, n, i) && findEntry(e, n, j) && findEntry(e, n, k);
  CmiFree(e);
  CqsDelete(q);
  return result;
}

bool test_general_fifo()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  CqsEnqueueGeneral(q, i, CQS_QUEUEING_FIFO, 1, 0);
  CqsEnqueueGeneral(q, j, CQS_QUEUEING_FIFO, 2, 0);
  CqsEnqueueGeneral(q, k, CQS_QUEUEING_FIFO, 42, 0);
  void *r, *s, *t;
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  CqsDequeue(q, &t);
  bool result = (r == i) && (s == j) && (t == k);
  CqsDelete(q);
  return result;
}

bool test_general_ififo()
{
  Queue q = CqsCreate();
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  unsigned int a = -1, b = 0, c = 1;
  CqsEnqueueGeneral(q, i, CQS_QUEUEING_IFIFO, 8*sizeof(int), &c);
  CqsEnqueueGeneral(q, j, CQS_QUEUEING_IFIFO, 8*sizeof(int), &b);
  CqsEnqueueGeneral(q, k, CQS_QUEUEING_IFIFO, 8*sizeof(int), &a);
  void *r, *s, *t;
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  CqsDequeue(q, &t);
  bool result = (r == k) && (s == j) && (t == i);
  CqsDelete(q);
  return result;
}

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


#if 0
// Template for new harness-driven tests
bool test_foo()
{
  Queue q = CqsCreate();
  
  bool result = ;
  CqsDelete(q);
  return result;
}
#endif

struct main : public CBase_main
{
  main(CkArgMsg *)
  {
    #if CMK_USE_STL_MSGQ
    CkPrintf("Charm has CMK_USE_STL_MSGQ enabled\n");
    #if CMK_HAS_STD_UNORDERED_MAP
    CkPrintf("... and its using a msg q implemented with a std::unordered_map\n");
    #else
    CkPrintf("... and its using a msg q implemented with a std::map\n");
    #endif
    #endif
    int tests = 0, success = 0, fail = 0;
    char message[100];

    RUN_TEST(test_empty);
    RUN_TEST(test_one);
    RUN_TEST(test_two);
#if ! CMK_RANDOMIZED_MSGQ
    RUN_TEST(test_fifo);
    RUN_TEST(test_lifo);
    RUN_TEST(test_enqueue_mixed);
    RUN_TEST(test_general_fifo);
    RUN_TEST(test_general_ififo);
    RUN_TEST(test_enumerate);
#endif
    RUN_TEST(perftest_general_ififo);

    if (fail) {
      sprintf(message, "%d/%d tests failed\n", fail, tests);
      CkAbort(message);
    }
    else {
      CkPrintf("All %d tests passed\n", tests);
      CkExit();
    }
  }

};

#include "main.def.h"
