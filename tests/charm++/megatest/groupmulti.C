#include "groupmulti.h"

CmiGroup cgrp;

void groupmulti_moduleinit(void) {}

void groupmulti_init(void) 
{
  CProxy_groupmulti_main::ckNew(0);
}

groupmulti_main::groupmulti_main(void) 
{
  count = 0;
  int npes = CkNumPes();
  int *pes = new int[npes];
  for (int i=0; i<npes; i++) pes[i] = i;
  cgrp = CmiEstablishGroup(npes, pes);
  groupmulti_SimpleMsg *sm = new groupmulti_SimpleMsg;
  sm->mainhandle = thishandle;
  gid=CProxy_groupmulti_group::ckNew(sm);
}

void groupmulti_main::groupReady() {
  count++;
  if (count < 3) {
    CProxy_groupmulti_group groupproxy(gid);
    groupmulti_BCMsg *g_bcm = new groupmulti_BCMsg;
//      CkSendMsgBranchGroup(CkIndex_groupmulti_group::doBroadcast(NULL), g_bcm, gid, cgrp, 0);
      groupproxy.doBroadcast(g_bcm, cgrp);
  } else {
      delete this;
      megatest_finish();
  }
}

groupmulti_group::groupmulti_group(groupmulti_SimpleMsg *sm) {
  myMain = sm->mainhandle;
  CkCallback cb(CkIndex_groupmulti_main::groupReady(),myMain);
  contribute(0,0,CkReduction::sum_int,cb);
}

void groupmulti_group::doBroadcast(groupmulti_BCMsg *bcm) {
  if (!bcm->check()) {
    CkAbort("Broadcasted message fails check.\n");
  }
  delete bcm;
  CkCallback cb(CkIndex_groupmulti_main::groupReady(),myMain);
  contribute(0,0,CkReduction::sum_int,cb);
}

MEGATEST_REGISTER_TEST(groupmulti,"gengbin",1)
#include "groupmulti.def.h"
