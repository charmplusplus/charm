#include "nodecast.h"

void nodecast_moduleinit(void) {}

void nodecast_init(void) 
{
  CProxy_nodecast_main::ckNew();
}

nodecast_main::nodecast_main(void) 
{
  count = 0;
  nodecast_SimpleMsg *sm = new nodecast_SimpleMsg;
  sm->mainhandle = thishandle;
  CProxy_nodecast_group::ckNew(sm);
}

void nodecast_main::reply(nodecast_SimpleMsg *em) {
  gid = em->gid;
  delete em;
  count++;
  if (count == CkNumNodes()) {
    CProxy_nodecast_group groupproxy(gid);
    nodecast_BCMsg *g_bcm = new nodecast_BCMsg;
    groupproxy.doBroadcast(g_bcm);
  } else if (count == CkNumNodes()*2) {
      delete this;
      megatest_finish();
  }
}

nodecast_group::nodecast_group(nodecast_SimpleMsg *sm) {
  myMain = sm->mainhandle;
  sm->gid = thisgroup;
  CProxy_nodecast_main mainproxy(myMain);
  mainproxy.reply(sm);
}

void nodecast_group::doBroadcast(nodecast_BCMsg *bcm) {
  if (!bcm->check()) {
    CkAbort("nodecast: broadcast message is corrupted!\n");
  }
  delete bcm;
  CProxy_nodecast_main mainproxy(myMain);
  nodecast_SimpleMsg *g_em = new nodecast_SimpleMsg;
  mainproxy.reply(g_em);
}

MEGATEST_REGISTER_TEST(nodecast,"milind",1)
#include "nodecast.def.h"
