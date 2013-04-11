 /**************************************************************************
 * DESCRIPTION:
 *
 * To add a test to commbench, you have to:
 *
 *   1. write a testname_moduleinit function that initializes the module.
 *   2. write a testname_init function that starts the test.
 *   3. declare the testname_init function inside this module.
 *   4. extend the tests[] table in this module to include the new test.
 *
 **************************************************************************/

#include <stdio.h>
#include <converse.h>
#include "commbench.h"

/******************************************************************************
 *
 * Test Configuration Section
 *
 *****************************************************************************/

extern void memoryAccess_init(void);
extern void overhead_init(void);
extern void timer_init(void);
extern void proc_init(void);
extern void smputil_init(void);
extern void pingpong_init(void);
extern void flood_init(void);
extern void broadcast_init(void);
extern void reduction_init(void);
extern void ctxt_init(void);

extern void memoryAccess_moduleinit(void);
extern void overhead_moduleinit(void);
extern void timer_moduleinit(void);
extern void proc_moduleinit(void);
extern void smputil_moduleinit(void);
extern void pingpong_moduleinit(void);
extern void flood_moduleinit(void);
extern void broadcast_moduleinit(void);
extern void reduction_moduleinit(void);
extern void ctxt_moduleinit(void);

struct testinfo
{
  char *name;
  void (*initiator)(void);
  void (*initializer)(void);
} tests[] = {
  { "memoryAccess",  memoryAccess_init,  memoryAccess_moduleinit },
  { "overhead",  overhead_init,  overhead_moduleinit },
  { "timer",     timer_init,     timer_moduleinit },
  { "proc",      proc_init,      proc_moduleinit },
  { "smputil",   smputil_init,   smputil_moduleinit },
  { "pingpong",  pingpong_init,  pingpong_moduleinit },
  { "flood",    flood_init,     flood_moduleinit},
  { "broadcast", broadcast_init, broadcast_moduleinit },
  { "reduction", reduction_init, reduction_moduleinit },
  { "ctxt",      ctxt_init,      ctxt_moduleinit },
  { 0,0,0 },
};

/******************************************************************************
 *
 * Central Control Section
 *
 *****************************************************************************/

CpvStaticDeclare(int, test_bank_size);
CpvStaticDeclare(int, test_negate_skip);
CpvStaticDeclare(char **, tests_to_skip);
CpvStaticDeclare(int, num_tests_to_skip);
CpvStaticDeclare(int, next_test_index);
CpvStaticDeclare(int, shutdown_handler);
CpvDeclare(int, ack_handler);

void commbench_shutdown(void *msg)
{
  CmiFree(msg);
  CsdExitScheduler();
}

int commbench_skip(char *test)
{
  int i;
  int num_skip = CpvAccess(num_tests_to_skip);
  char **skip;
  /* default mode where no tests are skipped */
  if(num_skip==0) return 0;

  skip = CpvAccess(tests_to_skip);
  for (i=0; i<num_skip; i++) {
    if (strcmp(skip[i+2], test)==0)
      return 0; 
  }
  return 1;
}

void commbench_next()
{
  int idx, bank;
  EmptyMsg msg;

  bank = CpvAccess(test_bank_size);
nextidx:
  idx = CpvAccess(next_test_index);
  if (idx < bank) {
    if (commbench_skip(tests[idx].name)) {
      CpvAccess(next_test_index)++;
      goto nextidx;
    }
    CmiPrintf("[%s] initiated\n", tests[idx].name);
    (tests[idx].initiator)();
    return; 
  }
  if (idx==bank) {
    CmiPrintf("All benchmarks completed, exiting...\n");
    CmiSetHandler(&msg, CpvAccess(shutdown_handler));
    CmiSyncBroadcastAll(sizeof(EmptyMsg), &msg);
    return;
  }
}

void commbench_ack(void *msg)
{
  CmiFree(msg);
  CpvAccess(next_test_index)++;
  commbench_next();
}

void commbench_init(int argc, char **argv)
{
  int numtests, i;
  CpvInitialize(int, shutdown_handler);
  CpvInitialize(int, ack_handler);
  CpvAccess(shutdown_handler) = CmiRegisterHandler((CmiHandler)commbench_shutdown);
  CpvAccess(ack_handler) = CmiRegisterHandler((CmiHandler)commbench_ack);
  for (i=0; (tests[i].initializer); i++)
    (tests[i].initializer)();
  CpvInitialize(int, test_bank_size);
  CpvInitialize(int, test_negate_skip);
  CpvInitialize(int, num_tests_to_skip);
  CpvInitialize(char **, tests_to_skip);
  CpvInitialize(int, next_test_index);
  for (numtests=0; tests[numtests].name; numtests++);
  CpvAccess(test_bank_size) = numtests;
  CpvAccess(next_test_index) = 0;
  CpvAccess(test_negate_skip)=0;
  for (i=1; i<argc; i++)
    if (strcmp(argv[i],"-only")==0)
      CpvAccess(test_negate_skip)=1;
  CpvAccess(num_tests_to_skip) = 0;
  if(CpvAccess(test_negate_skip)) {
    CpvAccess(num_tests_to_skip) = argc-2;
  }
  CpvAccess(tests_to_skip) = argv;

  if (CmiMyPe()==0)
    commbench_next();
}

int main(int argc, char **argv)
{
  ConverseInit(argc,argv,commbench_init,0,0);
}
