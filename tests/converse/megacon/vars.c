#include <stdio.h>
#include <converse.h>

void Cpm_megacon_ack();

typedef struct vars_chare
{
  int countdown;
  CthThread pending;
}
*vars_chare;

CpmDeclareSimple(vars_chare);
#define CpmPack_vars_chare(x) (0)
#define CpmUnpack_vars_chare(x) (0)

#include "vars.cpm.h"

CtvDeclare(int, ctv1);
CpvDeclare(int, cpv1);
CsvDeclare(int, csv1);

CpmInvokable vars_ack(vars_chare c)
{
  c->countdown--;
  if ((c->countdown==0)&&(c->pending))
    CthAwaken(c->pending);
}

void vars_check_ctv_privacy(vars_chare c)
{
  int me = (size_t)CthSelf();
  CtvAccess(ctv1) = me;
  vars_ack(c);
  CthSuspend();
  if (CtvAccess(ctv1) != me) {
    CmiPrintf("ctv privacy test failed.\n");
    exit(1);
  }
  vars_ack(c);
  CthFree(CthSelf());
  CthSuspend();
}

CpmInvokable vars_set_cpv_and_csv(vars_chare c)
{
  CpvAccess(cpv1) = CmiMyPe();
  if (CmiMyRank() == 0)
    CsvAccess(csv1) = 0x12345678;
  Cpm_vars_ack(CpmSend(0), c);
}

CpmInvokable vars_check_cpv_and_csv(vars_chare c)
{
  if (CpvAccess(cpv1) != CmiMyPe()) {
    CmiPrintf("cpv privacy test failed.\n");
    exit(1);
  }
  if (CsvAccess(csv1) != 0x12345678) {
    CmiPrintf("csv sharing test failed.\n");
    exit(1);
  }
  Cpm_vars_ack(CpmSend(0), c);
}

CpmInvokable vars_control()
{
  struct vars_chare c; CthThread t1,t2;

  t1 = CthCreate(vars_check_ctv_privacy, (void *)&c, 0);
  t2 = CthCreate(vars_check_ctv_privacy, (void *)&c, 0);
  CthSetStrategyDefault(t1);
  CthSetStrategyDefault(t2);

  CthAwaken(t1); CthAwaken(t2);
  c.countdown = 2; c.pending = CthSelf(); CthSuspend();
  
  CthAwaken(t1); CthAwaken(t2);
  c.countdown = 2; c.pending = CthSelf(); CthSuspend();
  
  Cpm_vars_set_cpv_and_csv(CpmSend(CpmALL), &c);
  c.countdown = CmiNumPes(); c.pending = CthSelf(); CthSuspend();
  
  Cpm_vars_check_cpv_and_csv(CpmSend(CpmALL), &c);
  c.countdown = CmiNumPes(); c.pending = CthSelf(); CthSuspend();
  
  Cpm_megacon_ack(CpmSend(0));
}

void vars_init()
{
  Cpm_vars_control(CpmMakeThreadSize(0,0));
}

void vars_moduleinit()
{
  CpmInitializeThisModule();
  CtvInitialize(int, ctv1);
  CpvInitialize(int, cpv1);
  CsvInitialize(int, csv1);
}

