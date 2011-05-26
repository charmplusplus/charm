#define SUPPRESS_PTHREADS
#include <stdio.h>
#include <converse.h>
#include <cpthreads.h>
#include "posixth.cpm.h"

void Cpm_megacon_ack();

CpvStaticDeclare(Cpthread_attr_t,  joinable);
CpvStaticDeclare(Cpthread_attr_t,  detached);

CpvStaticDeclare(Cpthread_mutexattr_t, mutexattrs);
CpvStaticDeclare(Cpthread_condattr_t,  condattrs);

CpvStaticDeclare(Cpthread_mutex_t, total_mutex);
CpvStaticDeclare(int,              total);
CpvStaticDeclare(Cpthread_mutex_t, leaves_mutex);
CpvStaticDeclare(int,              leaves);
CpvStaticDeclare(Cpthread_mutex_t, fibs_mutex);
CpvStaticDeclare(int,              fibs);

CpvStaticDeclare(Cpthread_cond_t,  donecond);

static void posixth_fail()
{
  CmiError("error detected in posix threads.\n");
  exit(1);
}

static void errck(int code)
{
  if (code != 0) posixth_fail();
}

void posixth_add(Cpthread_mutex_t *mutex, int *var, int val)
{
  int n;
  Cpthread_mutex_lock(mutex);
  n = *var;
  if (CrnRand()&1) CthYield();
  *var = n + val;
  Cpthread_mutex_unlock(mutex);
}

void *posixth_fib(void *np)
{
  Cpthread_t t1, t2; void *r1, *r2;
  CmiIntPtr n = (size_t)np, total;
  if (n<2) {
    if (CrnRand()&1) CthYield();
    posixth_add(&CpvAccess(leaves_mutex), &CpvAccess(leaves), 1);
    return (void*)n;
  }
  if (CrnRand()&1) CthYield();
  errck(Cpthread_create(&t1, &CpvAccess(joinable), posixth_fib, (void*)(n-1)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_create(&t2, &CpvAccess(joinable), posixth_fib, (void*)(n-2)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_join(t1, &r1));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_join(t2, &r2));
  if (CrnRand()&1) CthYield();
  total = ((size_t)r1) + ((size_t)r2);
  return (void*)total;
}

void *posixth_top(void *x)
{
  Cpthread_t t; void *result; int n;
  if (CrnRand()&1) CthYield();
  errck(Cpthread_create(&t, &CpvAccess(joinable), posixth_fib, (void*)6));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_join(t, &result));
  if (CrnRand()&1) CthYield();
  if ((size_t)result != 8) posixth_fail();
  if (CrnRand()&1) CthYield();
  posixth_add(&CpvAccess(total_mutex), &CpvAccess(total), (size_t)result);
  if (CrnRand()&1) CthYield();
  posixth_add(&CpvAccess(fibs_mutex), &CpvAccess(fibs), -1);
  if (CrnRand()&1) CthYield();
  if (CpvAccess(fibs)==0)
    errck(Cpthread_cond_signal(&CpvAccess(donecond)));
  if (CrnRand()&1) CthYield();
}

void posixth_main(int argc, char **argv)
{
  Cpthread_mutex_t dummymutex; int i; Cpthread_t t;

  if (CrnRand()&1) CthYield();
  errck(Cpthread_attr_init(&CpvAccess(joinable)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_attr_setdetachstate(&CpvAccess(joinable),CPTHREAD_CREATE_JOINABLE));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_attr_init(&CpvAccess(detached)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_attr_setdetachstate(&CpvAccess(detached),CPTHREAD_CREATE_DETACHED));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutexattr_init(&CpvAccess(mutexattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_condattr_init(&CpvAccess(condattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_init(&CpvAccess(total_mutex), &CpvAccess(mutexattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_init(&CpvAccess(leaves_mutex), &CpvAccess(mutexattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_init(&CpvAccess(fibs_mutex), &CpvAccess(mutexattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_cond_init(&CpvAccess(donecond), &CpvAccess(condattrs)));
  if (CrnRand()&1) CthYield();
  CpvAccess(total) = 0;
  CpvAccess(fibs) = 20;
  CpvAccess(leaves) = 0;

  for (i=0; i<20; i++) {
    if (CrnRand()&1) CthYield();
    Cpthread_create(&t, &CpvAccess(detached), posixth_top, 0);
  }

  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_init(&dummymutex, &CpvAccess(mutexattrs)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_lock(&dummymutex));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_cond_wait(&CpvAccess(donecond), &dummymutex));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_unlock(&dummymutex));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_destroy(&dummymutex));
  if (CrnRand()&1) CthYield();
  
  if (CpvAccess(total)!=160) posixth_fail();
  if (CpvAccess(leaves)!=260) posixth_fail();
  
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_destroy(&CpvAccess(total_mutex)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_destroy(&CpvAccess(leaves_mutex)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_mutex_destroy(&CpvAccess(fibs_mutex)));
  if (CrnRand()&1) CthYield();
  errck(Cpthread_cond_destroy(&CpvAccess(donecond)));
  if (CrnRand()&1) CthYield();
  
  Cpm_megacon_ack(CpmSend(0));
}

void posixth_init(void)
{
  Cpthread_start_main(posixth_main, 0, 0);
}

void posixth_moduleinit()
{
  CpmInitializeThisModule();

  CpvInitialize(Cpthread_attr_t, joinable);
  CpvInitialize(Cpthread_attr_t, detached);
  CpvInitialize(Cpthread_mutexattr_t, mutexattrs);
  CpvInitialize(Cpthread_condattr_t,  condattrs);
  CpvInitialize(Cpthread_mutex_t, total_mutex);
  CpvInitialize(int,              total);
  CpvInitialize(Cpthread_mutex_t, leaves_mutex);
  CpvInitialize(int,              leaves);
  CpvInitialize(Cpthread_mutex_t, fibs_mutex);
  CpvInitialize(int,              fibs);
  CpvInitialize(Cpthread_cond_t,  donecond);
  
}
