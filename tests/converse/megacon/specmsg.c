#include <stdio.h>
#include <converse.h>

typedef struct specmsg_chare
{
  int next;
}
*specmsg_chare;

CpmDeclareSimple(specmsg_chare);
#define CpmPack_specmsg_chare(x) (0)
#define CpmUnpack_specmsg_chare(x) (0)

#include "specmsg.cpm.h"

void Cpm_megacon_ack();

void specmsg_fail()
{
  CmiError("specmsg: CmiDeliverSpecificMsg failed.\n");
  exit(1);
}

CpmInvokable specmsg_step1(specmsg_chare c)
{
  if (c->next != 1) specmsg_fail();
  c->next++;
}

CpmInvokable specmsg_step2(specmsg_chare c)
{
  if (c->next != 2) specmsg_fail();
  c->next++;
}

CpmInvokable specmsg_step3(specmsg_chare c)
{
  if (c->next != 3) specmsg_fail();
  c->next++;
}

CpmInvokable specmsg_step4(specmsg_chare c)
{
  if (c->next != 4) specmsg_fail();
  c->next++;
}

CpmInvokable specmsg_begin()
{
  int i; struct specmsg_chare c;
  Cpm_specmsg_request(CpmSend(0), CmiMyPe(), &c);
  if (CmiMyPe()==0)
    for (i=0; i<CmiNumPes(); i++)
      CmiDeliverSpecificMsg(CpvAccess(CpmIndex_specmsg_request));
  c.next = 1;
  CmiDeliverSpecificMsg(CpvAccess(CpmIndex_specmsg_step1));
  CmiDeliverSpecificMsg(CpvAccess(CpmIndex_specmsg_step2));
  CmiDeliverSpecificMsg(CpvAccess(CpmIndex_specmsg_step3));
  CmiDeliverSpecificMsg(CpvAccess(CpmIndex_specmsg_step4));
  Cpm_megacon_ack(CpmSend(0));
}

CpmInvokable specmsg_request(int pe, specmsg_chare c)
{
  Cpm_specmsg_step3(CpmSend(pe), c);
  Cpm_specmsg_step2(CpmSend(pe), c); 
  Cpm_specmsg_step4(CpmSend(pe), c);
  Cpm_specmsg_step1(CpmSend(pe), c);
}

void specmsg_init()
{
  int pe;
  for (pe=1; pe<CmiNumPes(); pe++)
    Cpm_specmsg_begin(CpmSend(pe));
  specmsg_begin();
}

void specmsg_moduleinit()
{
  CpmInitializeThisModule();
}

