#include "rotest.h"

readonly<CkGroupID> rotest_groupid;
roarray<int,10> rotest_iarray_num;
roarray<int,ROTEST_SIZE> rotest_iarray_sz;
romsg<rotest_msg> rmsg;

void rotest_init(void)
{
  CProxy_rotest_group pgg(rotest_groupid);
  pgg.start();
}

void rotest_moduleinit(void) 
{
  int i;
  for(i=0;i<10;i++) {
    rotest_iarray_num[i] = i*i+1023;
  }
  for(i=0;i<ROTEST_SIZE;i++) {
    rotest_iarray_sz[i] = i*i+511;
  }
  rmsg = new rotest_msg(1024);
  rotest_groupid = CProxy_rotest_group::ckNew();
}

static int rotest_check(void)
{
  int i;
  for(i=0;i<10;i++) {
    if(rotest_iarray_num[i] != i*i+1023)
      return 1;
  }
  for(i=0;i<ROTEST_SIZE;i++) {
    if(rotest_iarray_sz[i] != i*i+511)
      return 1;
  }
  return rmsg->check();
}

void rotest_group::start(void)
{
  if(rotest_check())
    CkAbort("rotest failed");
  CProxy_rotest_group rog(rotest_groupid);
  rog[0].done();
}

void rotest_group::done(void)
{
  numdone++;
  if(numdone == CkNumPes()) {
    numdone = 0;
    megatest_finish();
  }
}

MEGATEST_REGISTER_TEST(rotest,"milind",0)
#include "rotest.def.h"
