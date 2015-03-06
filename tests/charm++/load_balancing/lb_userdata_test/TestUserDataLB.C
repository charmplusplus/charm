#include <charm++.h>
#include "TestUserDataLB.h"
#include "userdata_struct.h"

CkpvExtern(int, _lb_obj_index);

CreateLBFunc_Def(TestUserDataLB, "to test user lbdata.")

TestUserDataLB::TestUserDataLB(const CkLBOptions &opt): CBase_TestUserDataLB(opt) {
  lbname = "TestUserDataLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] TestUserDataLB created\n",CkMyPe());
  #if CMK_LB_USER_DATA
  CkpvAccess(_lb_obj_index) = LBRegisterObjUserData(sizeof(LBUserDataStruct));
  #endif
}

bool TestUserDataLB::QueryBalanceNow(int _step) {
  return true;
}

void TestUserDataLB::work(LDStats* stats) {
  CkPrintf("[%d] TestUserDataLB work\n", CkMyPe());
  stats->makeCommHash();
  for (int i = 0; i < stats->n_objs; i++) {
    LDObjData &odata = stats->objData[i];
  #if CMK_LB_USER_DATA
    LBUserDataStruct* udata = (LBUserDataStruct *)odata.getUserData(
        CkpvAccess(_lb_obj_index));
    LDObjHandle &handle = udata->handle;

    int tag = stats->getHash(handle.id,handle.omhandle.id);
    if (tag == -1) {
      CkAbort("tag is -1\n");
    }
    CkPrintf("[%d] tag %d idx %d\n", i, tag, udata->idx);
    CkAssert(tag == i);
  #endif
  }

}

#include "TestUserDataLB.def.h"
