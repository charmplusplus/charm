#include "varsizetest2.h"

static int nextseq = 0;

void varsizetest2_init(void) 
{
    CProxy_varsizetest2_main::ckNew(0);
}

void varsizetest2_moduleinit(void) {}

varsizetest2_Msg::varsizetest2_Msg(int s, int _isize, int _fsize ): seqnum(s), isize(_isize), fsize(_fsize)
{
  int i;
  for(i=0; i<isize; i++)
    iarray[i] = i*i*seqnum;
  for(i=0; i<fsize;i++)
    farray[i] = 2.0*i*i*seqnum;
}


varsizetest2_main::varsizetest2_main(void)
{
  varsizetest2_Msg *m = new (300, 100) varsizetest2_Msg(nextseq++, 300, 100);
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
    if(iarray[i] != i*i*seqnum) {
      CkPrintf("iarray[%d] should be %d, is instead %d\n",i,i*i*seqnum,iarray[i]);
      return 0;
    }
  for(i=0; i<10; i++)
    if(fabs(farray[i] - 2.0*i*i*seqnum)>10.0) {
      CkPrintf("farray[%d] should be %0.8f, is instead %0.8f\n",i,2.0*i*i*seqnum,farray[i]);
      return 0;
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
  m = new (300, 100) varsizetest2_Msg(currentSeqnum, 300, 100);
  CProxy_varsizetest2_main mainproxy(mhandle);
  mainproxy.exit(m); 
  delete this;
}

MEGATEST_REGISTER_TEST(varsizetest2,"phil",1)
#include "varsizetest2.def.h"
