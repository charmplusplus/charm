 /**************************************************************************
 * DESCRIPTION:
 *
 * To add a test to megacon, you have to:
 *
 *   1. write a testname_moduleinit function that initializes the module.
 *   2. write a testname_init function that starts the test.
 *   3. declare the testname_init function inside this module.
 *   4. extend the tests[] table in this module to include the new test.
 *
 **************************************************************************/

#include <stdio.h>
#include <converse.h>
#include "megacon.cpm.h"

/******************************************************************************
 *
 * Test Configuration Section
 *
 *****************************************************************************/

void blkinhand_init(void);
void posixth_init(void);
void future_init(void);
void bigmsg_init(void);
void vecsend_init(void);
void nodenum_init(void);
void specmsg_init(void);
void vars_init(void);
void priotest_init(void);
void ringsimple_init(void);
void ring_init(void);
void fibobj_init(void);
void fibthr_init(void);
void broadc_init(void);
void multicast_init(void);
void deadlock_init(void);
void multisend_init(void);
void handler_init(void);
void reduction_init(void);

void blkinhand_moduleinit(void);
void posixth_moduleinit(void);
void future_moduleinit(void);
void bigmsg_moduleinit(void);
void vecsend_moduleinit(void);
void nodenum_moduleinit(void);
void specmsg_moduleinit(void);
void vars_moduleinit(void);
void priotest_moduleinit(void);
void ringsimple_moduleinit(void);
void ring_moduleinit(void);
void fibobj_moduleinit(void);
void fibthr_moduleinit(void);
void broadc_moduleinit(void);
void multicast_moduleinit(void);
void deadlock_moduleinit(void);
void multisend_moduleinit(void);
void handler_moduleinit(void);
void reduction_moduleinit(void);

struct testinfo
{
  char *name;
  void (*initiator)(void);
  void (*initializer)(void);
  int  reentrant;
  int  numacks;
}
tests[] = {
  { "blkinhand", blkinhand_init, blkinhand_moduleinit,  1,  1 },
  { "posixth",   posixth_init,   posixth_moduleinit,    0,  1 },
  { "future",    future_init,    future_moduleinit,     1,  1 },
  { "bigmsg",    bigmsg_init,    bigmsg_moduleinit,     1,  1 },
  { "vecsend",   vecsend_init,   vecsend_moduleinit,    0,  1 },
  { "nodenum",   nodenum_init,   nodenum_moduleinit,    0,  1 },
  { "specmsg",   specmsg_init,   specmsg_moduleinit,    0,  0 },
  { "vars",      vars_init,      vars_moduleinit,       0,  1 },
#if ! CMK_RANDOMIZED_MSGQ
  { "priotest",  priotest_init,  priotest_moduleinit,   1,  0 },
#endif
  { "ringsimple",ringsimple_init,ringsimple_moduleinit, 0, 10 },
  { "ring",      ring_init,      ring_moduleinit,       1,  1 },
  { "fibobj",    fibobj_init,    fibobj_moduleinit,     1,  1 },
  { "fibthr",    fibthr_init,    fibthr_moduleinit,     1,  1 },
  { "broadc",    broadc_init,    broadc_moduleinit,     1,  1 },
  { "multicast", multicast_init, multicast_moduleinit,  1,  1 },
  { "deadlock",  deadlock_init,  deadlock_moduleinit,   0,  2 },
  { "handler",  handler_init,  handler_moduleinit,   1,  1 },
  { "multisend", multisend_init, multisend_moduleinit,  0,  1 },
  { "reduction", reduction_init, reduction_moduleinit, 0, 1 },
  { 0,0,0,0 },
};

/******************************************************************************
 *
 * Central Control Section
 *
 *****************************************************************************/

CpvDeclare(int, test_bank_size);
CpvDeclare(int, test_negate_skip);
CpvDeclare(char **, tests_to_skip);
CpvDeclare(int, num_tests_to_skip);
CpvDeclare(double, test_start_time);
CpvDeclare(int, next_test_index);
CpvDeclare(int, next_test_number);
CpvDeclare(int, acks_expected);
CpvDeclare(int, acks_received);

/* The megacon shutdown sequence is to idle for a while, then exit.  */
/* the idling period makes it possible to detect extra runaway msgs. */

CpmInvokable megacon_stop()
{
  CsdExitScheduler();
}

CpmInvokable megacon_shutdown(int n)
{
  if (n==0) {
    CmiPrintf("exiting.\n");
    Cpm_megacon_stop(CpmSend(CpmALL));
  } else {
    Cpm_megacon_shutdown(CpmEnqueueIFIFO(0, 1), n-1);
  }
}

int megacon_skip(char *test)
{
  int i;
  int num_skip = CpvAccess(num_tests_to_skip);
  char **skip;
  skip = CpvAccess(tests_to_skip);
  for (i=0; i<num_skip; i++) {
    if ((skip[i][0]=='-')&&(strcmp(skip[i]+1, test)==0))
      {
	/*	CmiPrintf("skipping test %s\n",skip[i]);*/
	return 1 - CpvAccess(test_negate_skip);
      }
  }
  return CpvAccess(test_negate_skip);
}

void megacon_next()
{
  int i, pos, idx, num, bank, numacks;

  bank = CpvAccess(test_bank_size);
  num = CpvAccess(next_test_number);
nextidx:
  idx = CpvAccess(next_test_index);
  if (idx < bank) {
    numacks = tests[idx].numacks;
    if (megacon_skip(tests[idx].name)) {
      /*      CmiPrintf("skipping test %s\n",tests[idx].name);*/
      CpvAccess(next_test_index)++;
      goto nextidx;
    }
    CpvAccess(acks_expected) = numacks ? numacks : CmiNumPes();
    CpvAccess(acks_received) = 0;
    CpvAccess(test_start_time) = CmiWallTimer();
    CmiPrintf("test %d: initiated [%s]\n", num, tests[idx].name);
    (tests[idx].initiator)();
    return; 
  }
  if (idx < (2*bank)) {
    pos = idx - bank;
    numacks = tests[pos].numacks;
    if ((tests[pos].reentrant == 0)||(megacon_skip(tests[pos].name))||
	CpvAccess(test_negate_skip)) {
      CpvAccess(next_test_index)++;
      goto nextidx;
    }
    CpvAccess(acks_expected) = 5 * (numacks ? numacks : CmiNumPes());
    CpvAccess(acks_received) = 0;
    CpvAccess(test_start_time) = CmiWallTimer();
    CmiPrintf("test %d: initiated [multi %s]\n", num, tests[pos].name);
    for (i=0; i<5; i++) (tests[pos].initiator)();
    return;
  }
  if (idx== (2*bank)) {
    CpvAccess(acks_expected) = 0;
    CpvAccess(acks_received) = 0;
    CpvAccess(test_start_time) = CmiWallTimer();
    CmiPrintf("test %d: initiated [all-at-once]\n", num);
    for (i=0; i<bank; i++) {
      numacks = tests[i].numacks;
      if (!megacon_skip(tests[i].name)) {
	CpvAccess(acks_expected) += (numacks ? numacks : CmiNumPes());
	(tests[i].initiator)();
      }
    }
    return;
  }
  if (idx== ((2*bank)+1)) {
    CmiPrintf("All tests completed, verifying quiescence...\n");
    Cpm_megacon_shutdown(CpmSend(0), 50000);
    return;
  }
  CmiPrintf("System should have been quiescent, but it wasnt.\n");
  exit(1);
}

CpmInvokable megacon_ack()
{
  CpvAccess(acks_received)++;
  if (CpvAccess(acks_received) == CpvAccess(acks_expected)) {
    CmiPrintf("test %d: completed (%1.2f sec)\n",
	      CpvAccess(next_test_number),
	      CmiWallTimer() - CpvAccess(test_start_time));
    CpvAccess(next_test_number)++;
    CpvAccess(next_test_index)++;
    megacon_next();
  }
}

void megacon_init(int argc, char **argv)
{
  int numtests, i;
  CpmModuleInit();
  CfutureModuleInit();
  CpthreadModuleInit();
  CpmInitializeThisModule();
  for (i=0; (tests[i].initializer); i++)
    (tests[i].initializer)();
  CpvInitialize(int, test_bank_size);
  CpvInitialize(int, test_negate_skip);
  CpvInitialize(double, test_start_time);
  CpvInitialize(int, num_tests_to_skip);
  CpvInitialize(char **, tests_to_skip);
  CpvInitialize(int, next_test_index);
  CpvInitialize(int, next_test_number);
  CpvInitialize(int, acks_expected);
  CpvInitialize(int, acks_received);
  for (numtests=0; tests[numtests].name; numtests++);
  CpvAccess(test_bank_size) = numtests;
  CpvAccess(next_test_index) = 0;
  CpvAccess(next_test_number) = 0;
  CpvAccess(test_negate_skip)=0;
  for (i=1; i<argc; i++)
    if (strcmp(argv[i],"-only")==0)
      CpvAccess(test_negate_skip)=1;
  CpvAccess(num_tests_to_skip) = argc;
  /*    if(CpvAccess(test_negate_skip)) {
    CpvAccess(num_tests_to_skip)--;
  }
  */
  CpvAccess(tests_to_skip) = argv;
  if (CmiMyPe()==0)
    megacon_next();
}

int main(int argc, char **argv)
{
  ConverseInit(argc,argv,megacon_init,0,0);
}
