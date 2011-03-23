#include "groupcast.h"

void groupcast_moduleinit(void) {}

void groupcast_init(void) 
{
  CProxy_groupcast_main::ckNew(0);
}

groupcast_main::groupcast_main(void) 
{
  count = 0;
  groupcast_SimpleMsg *sm = new groupcast_SimpleMsg;
  sm->mainhandle = thishandle;
  gid=CProxy_groupcast_group::ckNew(sm);
}

void groupcast_main::groupReady() {
  count++;
  if (count < 3) {
    CProxy_groupcast_group groupproxy(gid);
    groupcast_BCMsg *g_bcm = new groupcast_BCMsg;
    groupproxy.doBroadcast(g_bcm);
  } else {
      delete this;
      megatest_finish();
  }
}

groupcast_group::groupcast_group(groupcast_SimpleMsg *sm) {
  myMain = sm->mainhandle;
  CkCallback cb(CkIndex_groupcast_main::groupReady(),myMain);
  contribute(0,0,CkReduction::sum_int,cb);
}

void groupcast_group::doBroadcast(groupcast_BCMsg *bcm) {
  if (!bcm->check()) {
    CkAbort("Broadcasted message fails check.\n");
  }
  delete bcm;
  CkCallback cb(CkIndex_groupcast_main::groupReady(),myMain);
  contribute(0,0,CkReduction::nop,cb);
}

MEGATEST_REGISTER_TEST(groupcast,"mjlang",1)
#include "groupcast.def.h"
