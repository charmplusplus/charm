#include "nodering.h"

readonly<CkGroupID> nodering_thisgroup;
static int nextseq = 0;

void nodering_init(void)
{
  nodering_message *msg = new nodering_message(nextseq++,0);
  CProxy_nodering_group pgg(nodering_thisgroup);
  pgg[0].start(msg);
}

void nodering_moduleinit(void) {}

void nodering_group::start(nodering_message *msg)
{
  CProxyElement_nodering_group pgg(
    nodering_thisgroup, (CkMyNode()+1)%CkNumNodes());
  if(CkMyNode()==0 && msg->iter==NITER) {
    delete msg;
    megatest_finish();
  } else if(CkMyNode()==0) {
    msg->iter++;
    pgg.start(msg);
  } else {
    pgg.start(msg);
  }
}

MEGATEST_REGISTER_TEST(nodering,"milind",1)
#include "nodering.def.h"
