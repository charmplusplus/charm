#include "priomsg.h"

void priomsg_init(void)
{ 
  priomsg_Msg *m = new (8*sizeof(int)) priomsg_Msg;
  *((int *)CkPriorityPtr(m)) = 100;
  CkSetQueueing(m, CK_QUEUEING_IFIFO);
  CProxy_priomsg_test pri=CProxy_priomsg_test::ckNew();
  pri.donow(m);
}

void priomsg_moduleinit (void) {}

void priomsg_test::donow(priomsg_Msg *m)
{
  if(!m->check()) 
    CkAbort("priomsg: message corrupted!\n");
  delete m;
  megatest_finish();   
}

priomsg_Msg::priomsg_Msg(void)
{
  for(int i=0; i<10; i++)
    data[i] = i*i;
}

int priomsg_Msg::check(void)
{
  for(int i=0; i<10; i++)
    if(data[i] != i*i)
      return 0;
  return 1;
}


MEGATEST_REGISTER_TEST(priomsg,"fang",1)
#include "priomsg.def.h"
