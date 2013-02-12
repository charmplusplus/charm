#include <stdio.h>
#include <converse.h>

/*
 *  Input:    10   11   12   13   14   15   16   17   18   19   20
 *  Output:   55   89  144  233  377  610  987 1597 2584 4181 6765
 */

#define FIB_INPUT 17
#define FIB_OUTPUT 1597

void Cpm_megacon_ack();

typedef struct fibobj_chare
{
  int ppe;
  struct fibobj_chare *ppos;
  int count, total;
}
*fibobj_chare;

/* We can declare this a simple type, as long as valid pointers */
/* don't leave the address space */

CpmDeclareSimple(fibobj_chare);

void CpmPack_fibobj_chare(fibobj_chare v)
{
  return;
}

void CpmUnpack_fibobj_chare(fibobj_chare v)
{
  return;
}

#include "fibobj.cpm.h"

CpmDestination CpmLDB()
{
  int pe = ( (CrnRand() & 0x7FFFFFFF) >>8 ) % CmiNumPes();
  return CpmSend(pe);
}

CpmInvokable fibobj_result(int n, fibobj_chare cpos)
{
  fibobj_chare c = cpos;
  c->total += n; c->count ++;
  if (c->count == 2) {
    if (c->ppe >= 0)
      Cpm_fibobj_result(CpmSend(c->ppe), c->total, c->ppos);
    else {
      if (c->total != FIB_OUTPUT) {
	CmiPrintf("Fib: results incorrect.\n");
	exit(1);
      }
      Cpm_megacon_ack(CpmSend(0));
    }
    free(c);
  }
}

CpmInvokable fibobj_calc(int n, int ppe, fibobj_chare ppos)
{
  if (n<2) Cpm_fibobj_result(CpmSend(ppe), n, ppos);
  else {
    fibobj_chare c = (fibobj_chare)malloc(sizeof(struct fibobj_chare));
    c->ppe = ppe, c->ppos = ppos; c->count=c->total=0;
    Cpm_fibobj_calc(CpmLDB(), n-1, CmiMyPe(), c);
    Cpm_fibobj_calc(CpmLDB(), n-2, CmiMyPe(), c);
  }
}

void fibobj_init()
{
  fibobj_chare c = (fibobj_chare)malloc(sizeof(struct fibobj_chare));
  c->ppe = -1, c->ppos = (fibobj_chare)FIB_INPUT; c->total=0; c->count=1;
  Cpm_fibobj_calc(CpmLDB(), FIB_INPUT, CmiMyPe(), c);
}

void fibobj_moduleinit()
{
  CpmInitializeThisModule();
}



