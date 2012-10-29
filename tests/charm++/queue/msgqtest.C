#include <cstdio>
#include <cstdlib>
#include <vector>

using std::sprintf;

#include "main.decl.h"
#include "msgq.h"

#define RUN_TEST(f) do { \
  ++tests; \
  if (f()) { \
    ++success; \
  } else { \
    ++fail; \
    CkPrintf("Test \"" #f "\" failed\n"); \
  } \
} while (0)

// A newly created msgQ should be empty, which corresponds to a
// length of 0
bool test_empty()
{
  conv::msgQ<int> q;
  bool result = (0 == q.size()) && (q.empty());
  return result;
}

// Enqueueing an element should show that there is an element
// present. We should get the same thing back when we dequeue
//
// The queue is not allowed to dereference the void* we give it
bool test_one()
{
  conv::msgQ<int> q;
  void *p = 0;
  q.enq(p);
  bool result = (1 == q.size()) && (!q.empty());
  void *r;
  r = (void*)q.deq();
  result &= (r == p)
    && (0 == q.size())
    && (q.empty());
  return result;
}

// Put two in, see if we get the same two back. Order unspecified
bool test_two()
{
  conv::msgQ<int> q;
  void *i = 0, *j = (char *)1;
  void *r, *s;
  q.enq(i);
  q.enq(j);
  bool result = (2 == q.size());
  r = (void*)q.deq();
  s = (void*)q.deq();
  result &= (r == i && s == j) || (r == j && s == i);
  result &= 1 == q.empty();
  return result;
}

bool test_fifo()
{
  conv::msgQ<int> q;
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  q.enq(i);
  q.enq(j);
  q.enq(k);
  r = (void*)q.deq();
  s = (void*)q.deq();
  t = (void*)q.deq();
  bool result = (r == i) && (s == j) && (t == k);
  return result;
}

bool test_lifo()
{
  conv::msgQ<int> q;
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  q.enq(i, 0, false);
  q.enq(j, 0, false);
  q.enq(k, 0, false);
  r = (void*)q.deq();
  s = (void*)q.deq();
  t = (void*)q.deq();
  bool result = (r == k) && (s == j) && (t == i);
  return result;
}

bool test_enqueue_mixed()
{
  conv::msgQ<int> q;
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  void *r, *s, *t;
  q.enq(i);
  q.enq(j);
  q.enq(k, 0, false);
  r = (void*)q.deq();
  s = (void*)q.deq();
  t = (void*)q.deq();
  bool result = (r == k) && (s == i) && (t == j);
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
  conv::msgQ<int> q;
  void *i = (char *)1, *j = (char *)2, *k = (char *)3;
  q.enq(i);
  q.enq(j);
  q.enq(k, 0, false);
  void **e = new void*[q.size()];
  q.enumerate(e, e + q.size());
  int n = q.size();
  bool result = findEntry(e, n, i) && findEntry(e, n, j) && findEntry(e, n, k);
  return result;
}

bool test_stl_ififo()
{
  conv::msgQ<int> q;
  void *i = (char *)1, *j = (char *)2, *k = (char *)3, *l = (char*)4;
  unsigned int a = -1, b = 0, c = 1, d = -1;
  q.enq(i, d, true);
  q.enq(j, c);
  q.enq(k, b);
  q.enq(l, a);
  void *r, *s, *t, *u;
  r = (void*)q.deq();
  s = (void*)q.deq();
  t = (void*)q.deq();
  u = (void*)q.deq();
  bool result = (r == i) && (s == l) && (t == k) && (u == j);
  return result;
}

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


#if 0
// Template for new harness-driven tests
bool test_foo()
{
  conv::msgQ<int> q;

  bool result = ;
  return result;
}
#endif


struct main : public CBase_main
{
 main(CkArgMsg *)
 {
  int tests = 0, success = 0, fail = 0;
  char message[100];

  RUN_TEST(test_empty);
  RUN_TEST(test_one);
  RUN_TEST(test_two);
#if ! CMK_RANDOMIZED_MSGQ
  RUN_TEST(test_fifo);
  RUN_TEST(test_lifo);
  RUN_TEST(test_enqueue_mixed);
  RUN_TEST(test_stl_ififo);
#endif
  RUN_TEST(test_enumerate);
  RUN_TEST(perftest_general_ififo);
  if (fail) {
    sprintf(message, "%d/%d tests failed\n", fail, tests);
    CkAbort(message);
  }
  else {
    CkPrintf("All %d stl msgQ tests passed\n", tests);
    CkExit();
  }
 }
};

#include "main.def.h"

