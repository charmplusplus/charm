#include "varraystest.h"

static int nextseq = 0;

void varraystest_init(void) 
{
  if(CkNumPes()<2) {
    CkError("varraystest: requires at least 2 processors\n");
    megatest_finish();
  } else
    CProxy_varraystest_main::ckNew(0);
}

void varraystest_moduleinit(void) {}

varraystest_main::varraystest_main(void)
{
  varraystest_Msg *m = new (100, 300, 0) varraystest_Msg(100, 300, nextseq++);
  m->myMain = thishandle;
  CProxy_varraystest_test::ckNew(m, 1);
}

void varraystest_main::exit(varraystest_Msg *m)
{
  if(!m->check())
    CkAbort("varraystest failed!\n");
  delete m;
  delete this;
  megatest_finish();
}

varraystest_Msg::varraystest_Msg(int is, int fs, int s)
{
  isize = is;
  fsize = fs;
  seqnum = s;
  int i;
  for(i=0; i<isize; i++)
    iarray[i] = i*i*seqnum;
  for(i=0; i<fsize;i++)
    farray[i] = 2.0*i*i*seqnum;
}

int varraystest_Msg::check(void)
{
  int i;
  for(i=0; i<isize; i++)
    if(iarray[i] != i*i*seqnum) {
      CkPrintf("iarray[%d] should be %d, is instead %d\n",i,iarray[i],i*i*seqnum);
      return 0;
    }
  for(i=0; i<fsize; i++)
    //    if((fabs(fabs(farray[i]) - fabs(2.0*i*i*seqnum)))/1000000.0>0.00001) {
    if((fabs(farray[i]) - fabs(2.0*i*i*seqnum))>10.0) {
      CkPrintf("farray[%d] should be %0.8f, is instead %0.8f\n",i,farray[i],2.0*i*i*seqnum);
      return 0;
    }
  return 1;
}

varraystest_test::varraystest_test(varraystest_Msg *m)
{
  CkChareID mhandle = m->myMain;
  int currentSeqnum = m->seqnum;
  if(!m->check()) {
    CkAbort("varraystest failed!\n");
    megatest_finish();
    return;
  }
  delete m;
  m = new (300, 100, 0) varraystest_Msg(300, 100, currentSeqnum);
  CProxy_varraystest_main mainproxy(mhandle);
  mainproxy.exit(m); 
  delete this;
}

MEGATEST_REGISTER_TEST(varraystest,"milind",1)
#include "varraystest.def.h"
