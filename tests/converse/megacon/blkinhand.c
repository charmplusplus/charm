#include <stdio.h>
#include <converse.h>
void Cpm_megacon_ack();

/* an accumulator datatype, which can have one pending thread */

typedef struct accum
{
  int total; int countdown;
  CthThread pending;
}
*accum;

CpmDeclareSimple(accum);
#define CpmPack_accum(x)
#define CpmUnpack_accum(x)

#include "blkinhand.cpm.h"

int blk_randpe()
{
  /* return ((rand()&0x7FFFFFFF)>>11) % CmiNumPes(); */
  return ((CrnRand()&0x7FFFFFFF)>>11) % CmiNumPes();
}

/* a function to add a number to an accumulator */

CpmInvokable blk_accum_add(accum a, int val)
{
  a->total += val;
  a->countdown --;
  if ((a->countdown==0)&&(a->pending))
    CthAwaken(a->pending);
}

/* The fib function: calculate fib of N, then add it to the specified accum */

CpmInvokable blk_fibthr(int n, int pe, accum resp)
{
  int result;
  if (n<2) result = n;
  else {
    struct accum acc;
    acc.total = 0; acc.countdown = 2; acc.pending = CthSelf();
    Cpm_blk_fibthr(CpmMakeThread(blk_randpe()), n-1, CmiMyPe(), &acc);
    Cpm_blk_fibthr(CpmMakeThread(blk_randpe()), n-2, CmiMyPe(), &acc);
    CthSuspend();
    result = acc.total;
  }
  Cpm_blk_accum_add(CpmSend(pe), resp, result);
}

/* The top-level function */

CpmInvokable blk_fibtop(int n)
{
  struct accum acc;
  acc.total = 0; acc.countdown = 1; acc.pending = CthSelf();
  Cpm_blk_fibthr(CpmMakeThread(blk_randpe()), n, CmiMyPe(), &acc);
  CthSuspend();
  if (acc.total != 21) {
    CmiPrintf("blkinhand failed. %08x\n", CthSelf());
    CmiPrintf("Failure in blk_fibtop\n");
    exit(1);
  }
  Cpm_megacon_ack(CpmSend(0));
}

void blkinhand_init()
{
  Cpm_blk_fibtop(CpmMakeThread(0), 8);
}

void blkinhand_moduleinit()
{
  CpmInitializeThisModule();
}
