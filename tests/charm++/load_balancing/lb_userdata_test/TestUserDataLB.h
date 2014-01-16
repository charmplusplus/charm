#ifndef _TESTUSERDATALB_H_
#define _TESTUSERDATALB_H_

#include "CentralLB.h"
#include "TestUserDataLB.decl.h"

void CreateTestUserDataLB();
BaseLB * AllocateTestUserDataLB();

class TestUserDataLB : public CentralLB {

public:

  TestUserDataLB(const CkLBOptions &);
  TestUserDataLB(CkMigrateMessage *m):CentralLB(m) { lbname = "TestUserDataLB"; }
  void work(LDStats* stats);
private:	
  bool        QueryBalanceNow(int step);
};

#endif /* _TESTUSERDATALB_H_ */
