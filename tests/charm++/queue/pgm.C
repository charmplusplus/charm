#include <iostream>
#include <cstdlib>

using std::cerr;
using std::endl;
using std::sprintf;

#include "queueing.h"
#include "main.decl.h"

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
  result &= CqsMaxLength(q) >= 0;
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
  result &= CqsMaxLength(q) >= 1;
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
  result &= CqsMaxLength(q) >= 2;
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
  CqsDelete(q);
  return result;
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
    int tests = 0, success = 0, fail = 0;
    char message[100];

    RUN_TEST(test_empty);
    RUN_TEST(test_one);
    RUN_TEST(test_two);
    RUN_TEST(test_fifo);
    RUN_TEST(test_lifo);
    RUN_TEST(test_enqueue_mixed);
    RUN_TEST(test_enumerate);

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
