#include "varsizetest2.h"

static int nextseq = 0;

void varsizetest2_init(void) 
{
    CProxy_varsizetest2_main::ckNew(0);
}

void varsizetest2_moduleinit(void) {}

varsizetest2_main::varsizetest2_main(void)
{
  int sizes[2];
  sizes[0] = 100; sizes[1] = 300;
  varsizetest2_Msg *m = new (sizes,0) varsizetest2_Msg();
  CkAssert(m->iarray && m->farray);
  m->myMain = thishandle;
  CProxy_varsizetest2_test::ckNew(m);
}

void varsizetest2_main::exit(varsizetest2_Msg *m)
{
  if(!m->check())
    CkAbort("varsizetest2 failed!\n");
  delete m;
  delete this;
  megatest_finish();
}

int varsizetest2_Msg::check(void)
{
  int i;
  for(i=0; i<10; i++)
    if(iarray[i] != 2*i*i*seqnum) {
      return 1;
    }
  for(i=0; i<10; i++)
    if(farray[i] != i*i*seqnum) {
      return 1;
    }
  return 1;
}

varsizetest2_test::varsizetest2_test(varsizetest2_Msg *m)
{
  CkChareID mhandle = m->myMain;
  int currentSeqnum = m->seqnum;
  if(!m->check()) {
    CkAbort("varsizetest2 failed!\n");
    megatest_finish();
    return;
  }
  delete m;
  int sizes[2];
  sizes[0] = 300;
  sizes[1] = 100;
  m = new (sizes,0) varsizetest2_Msg();
  CProxy_varsizetest2_main mainproxy(mhandle);
  mainproxy.exit(m); 
  delete this;
}

MEGATEST_REGISTER_TEST(varsizetest2,"phil",1)
#include "varsizetest2.def.h"
