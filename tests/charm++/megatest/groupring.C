#include "groupring.h"

static int nextseq = 0;

void groupring_init(void)
{
  CkGroupID groupring_thisgroup = CProxy_groupring_group::ckNew();
  CProxy_groupring_group pgg(groupring_thisgroup);
  groupring_message *msg = new groupring_message(nextseq++,0);
  pgg[0].start(msg);
}

void groupring_moduleinit(void) {}

void groupring_group::start(groupring_message *msg)
{
  CProxyElement_groupring_group pgg(thisgroup,(CkMyPe()+1)%CkNumPes());
  if(CkMyPe()==0 && msg->iter==NITER) {
    delete msg;
    megatest_finish();
  } else if(CkMyPe()==0) {
    msg->iter++;
    pgg.start(msg);
  } else {
    pgg.start(msg);
  }
}

MEGATEST_REGISTER_TEST(groupring,"milind",1)
#include "groupring.def.h"
