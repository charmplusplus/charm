#include "listsend.h"
#include <numeric>

#define VALUE 0x01ABCDEF

void listsend_moduleinit(void) {}

void listsend_init(void) { CProxy_listsend_main::ckNew(0); }

listsend_main::listsend_main(void) : count(0), checkedIn(CkNumPes(), false)
{
  groupProxy = CProxy_listsend_group::ckNew(thishandle);
}

void listsend_main::initDone(void)
{
  std::vector<int> pes(CkNumPes());
  std::iota(pes.begin(), pes.end(), 0);
  groupProxy.multicast(VALUE, pes.size(), pes.data());
}

void listsend_main::check(int sender, unsigned int val)
{
  count++;
  CmiEnforce(val == VALUE);
  checkedIn[sender] = true;

  if (count == CkNumPes())
  {
    for (int i = 0; i < CkNumPes(); i++)
    {
      if (!checkedIn[i]) CkAbort("PE %d didn't check in.\n", i);
    }
    megatest_finish();
  }
}

listsend_group::listsend_group(CProxy_listsend_main mProxy)
{
  mainProxy = mProxy;
  CkCallback cb(CkIndex_listsend_main::initDone(), mainProxy);
  contribute(cb);
}

void listsend_group::multicast(unsigned int val) { mainProxy.check(thisIndex, val); }

MEGATEST_REGISTER_TEST(listsend, "ronak", 1)
#include "listsend.def.h"
