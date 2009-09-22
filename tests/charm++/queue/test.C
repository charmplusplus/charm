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
    cerr << "Test " #f " failed" << endl; \
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
  CqsEnqueue(q, i);
  CqsEnqueue(q, j);
  bool result = (2 == CqsLength(q));
  void *r, *s;
  CqsDequeue(q, &r);
  CqsDequeue(q, &s);
  result &= (r == i && s == j) || (r == j && s == i);
  result &= 1 == CqsEmpty(q);
  CqsDelete(q);
  return result;
}
#if 0
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
    RUN_TEST(test_empty);
    RUN_TEST(test_one);
    RUN_TEST(test_two);
    
    if (fail) {
      char message[100];
      sprintf(message, "%d/%d tests failed\n", fail, tests);
      CkAbort(message);
    }
    else
      CkExit();
  }

};

#include "main.def.h"
