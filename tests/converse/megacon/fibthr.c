#include <stdio.h>
#include <converse.h>
void Cpm_megacon_ack();

/* an accumulator datatype, which can have one pending thread */

#define STACKSIZE_DEFAULT  (51200)

typedef struct accum
{
  int total; int countdown;
  CthThread pending;
}
*accum;

CpmDeclareSimple(accum);
#define CpmPack_accum(x)
#define CpmUnpack_accum(x)

#include "fibthr.cpm.h"

int randpe()
{
  /* return ((rand()&0x7FFFFFFF)>>11) % CmiNumPes(); */
  return ((CrnRand()&0x7FFFFFFF)>>11) % CmiNumPes();
}

/* a function to add a number to an accumulator */

CpmInvokable accum_add(accum a, int val)
{
  a->total += val;
  a->countdown --;
  if ((a->countdown==0)&&(a->pending))
    CthAwaken(a->pending);
}

/* The fib function: calculate fib of N, then add it to the specified accum */

CpmInvokable fibthr(int n, int pe, accum resp)
{
  int result;
  if (n<2) result = n;
  else {
    struct accum acc;
    acc.total = 0; acc.countdown = 2; acc.pending = CthSelf();
    Cpm_fibthr(CpmMakeThreadSize(randpe(),STACKSIZE_DEFAULT), n-1, CmiMyPe(), &acc);
    Cpm_fibthr(CpmMakeThreadSize(randpe(),STACKSIZE_DEFAULT), n-2, CmiMyPe(), &acc);
    CthSuspend();
    result = acc.total;
  }
  Cpm_accum_add(CpmSend(pe), resp, result);
}

/* The top-level function */

CpmInvokable fibtop(int n)
{
  struct accum acc;
  acc.total = 0; acc.countdown = 1; acc.pending = CthSelf();
  Cpm_fibthr(CpmMakeThreadSize(randpe(),STACKSIZE_DEFAULT), n, CmiMyPe(), &acc);
  CthSuspend();
  if (acc.total != 144) {
    CmiPrintf("Failure in fibtop\n");
    exit(1);
  }
  Cpm_megacon_ack(CpmSend(0));
}

void fibthr_init()
{
  Cpm_fibtop(CpmMakeThreadSize(0,STACKSIZE_DEFAULT), 12);
}

void fibthr_moduleinit()
{
  CpmInitializeThisModule();
}
