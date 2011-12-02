 /**************************************************************************
 * DESCRIPTION:
 *
 * To add a test to megatest, you have to:
 *
 *   1. write a testname_moduleinit function that initializes the module.
 *   2. write a testname_init function that starts the test.
 *   3. declare the testname_init function inside this module.
 *   4. extend the tests[] table in this module to include the new test.
 *   5. add extern module statement to megatest.ci
 *
 **************************************************************************/

#include <stdio.h>
#include "megatest.h"

/******************************************************************************
 *
 * Test Configuration Section
 *
 *****************************************************************************/

struct testinfo
{
  const char *name;
  const char *author;
  megatest_init_fn initiator;
  megatest_moduleinit_fn initializer;
  int reentrant;
  megatest_register_fn regfn;
} tests[100]={0};
int nTests=0;

/**
 This routine is called from MEGATEST_REGISTER_TEST, which 
 means it may be called before global variable initialization is complete.
*/
void megatest_register(const char *moduleName,const char *author,
	megatest_init_fn init, megatest_moduleinit_fn moduleinit,
	int reentrant, megatest_register_fn regfn)
{
	// printf("Registering module %s \n",moduleName);
	tests[nTests].name=moduleName;
	tests[nTests].author=author;
	tests[nTests].initiator=init;
	tests[nTests].initializer=moduleinit;
	tests[nTests].reentrant=reentrant;
	tests[nTests].regfn=regfn;
	nTests++;
}

/******************************************************************************
 *
 * Central Control Section
 *
 *****************************************************************************/
#include "megatest.decl.h"

CkChareID mainhandle;

class main : public CBase_main {
 private:
  int test_bank_size;
  double test_start_time;
  int next_test_index;
  int next_test_number;
  int acks_expected;
  int acks_received;
  
// Command line arguments:
  int test_negate_skip; // boolean: *only* run listed tests
  int test_repeat;      // boolean: keep running tests
  char **tests_to_skip; // argv
  int num_tests_to_skip; // argc
  int megatest_skip(const char *);
  
  void megatest_next(void);
 public:

  main(CkArgMsg *);
  main(CkMigrateMessage *m) {}
  void start(void);
  void finish(void);
};

void megatest_finish(void)
{
  CProxy_main mainproxy(mainhandle);
  mainproxy.finish();
}

int main::megatest_skip(const char *test)
{
  int i;
  int num_skip = num_tests_to_skip;
  char **skip;
  skip = tests_to_skip;
  for (i=0; i<num_skip; i++) {
    if ((skip[i][0]=='-')&&(strcmp(skip[i]+1, test)==0))
      return 1 - test_negate_skip;
  }
  return test_negate_skip;
}

void main::megatest_next(void)
{
  int i, pos, idx, num, bank;

  bank = test_bank_size;
  num = next_test_number;

nextidx:
  idx = next_test_index;
  if (idx < bank) {
    if (megatest_skip(tests[idx].name)) {
      next_test_index++;
      goto nextidx;
    }
    test_start_time = CkWallTimer();
    CkPrintf("test %d: initiated [%s (%s)]\n", num, tests[idx].name, 
                                               tests[idx].author);
    acks_expected = 1;
    acks_received = 0;
    (tests[idx].initiator)();
    return; 
  }
  if (idx < (2*bank)) {
    pos = idx - bank;
    if (!tests[pos].reentrant||(megatest_skip(tests[pos].name))||
        test_negate_skip) {
      next_test_index++;
      goto nextidx;
    }
    acks_expected = 5;
    acks_received = 0;
    test_start_time = CkWallTimer();
    CmiPrintf("test %d: initiated [multi %s (%s)]\n", num, tests[pos].name,
                                                      tests[pos].author);
    for (i=0; i<5; i++) (tests[pos].initiator)();
    return;
  }
  if (idx== (2*bank)) {
    acks_expected = 1;
    acks_received = 0;
    test_start_time = CkWallTimer();
    CmiPrintf("test %d: initiated [all-at-once]\n", num);
    for (i=0; i<bank; i++) {
      if (!megatest_skip(tests[i].name)) {
	acks_expected++;
	(tests[i].initiator)();
      }
    }
    megatest_finish();
    return;
  }
  if (idx== ((2*bank)+1)) {
    if (test_repeat) { 
      next_test_index=0;
      goto nextidx;
    } else { /* no repeat, normal exit */
      CkPrintf("All tests completed, exiting\n");
      CkExit();
    }
  }
}

void main::finish(void)
{
    acks_received++;
    if(acks_expected != acks_received)
      return;
    CkPrintf("test %d: completed (%1.2f sec)\n",
	      next_test_number,
	      CkWallTimer() - test_start_time);
    next_test_number++;
    next_test_index++;
    megatest_next();
}

main::main(CkArgMsg *msg)
{
  CmiPrintf("Megatest is running on %d nodes %d processors. \n", CkNumNodes(), CkNumPes());
  int argc = msg->argc;
  char **argv = msg->argv;
  int numtests, i;
  delete msg;
  mainhandle = thishandle;
  if (nTests<=0)
    CkAbort("Megatest: No tests registered-- is MEGATEST_REGISTER_TEST malfunctioning?");
  for (i=0; i<nTests; i++)
    (tests[i].initializer)();
  test_bank_size = nTests;
  next_test_index = 0;
  next_test_number = 0;
  test_negate_skip=0;
  test_repeat = 0;
  for (i=1; i<argc; i++) {
    if (strcmp(argv[i],"-only")==0)
      test_negate_skip = 1;
    if (strcmp(argv[i],"-repeat")==0)
      test_repeat = 1;
  }
  num_tests_to_skip = argc;
  tests_to_skip = argv;
  CProxy_main(thishandle).start();
}

void main::start()
{
  megatest_next();
}

void megatest_initnode(void)
{
	for (int i=0; i<nTests; i++)
		 if (tests[i].regfn)
		 	tests[i].regfn();
}

#include "megatest.def.h"
