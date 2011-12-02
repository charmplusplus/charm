#ifndef _VARSIZETEST_H
#define _VARSIZETEST_H

#include "megatest.h"
#include "varraystest.decl.h"


class varraystest_Msg : public CMessage_varraystest_Msg {
 public:
  varraystest_Msg(int, int, int s);
  int check(void);
  int isize;
  int fsize;
  int *iarray;
  float *farray;
  int seqnum;
  CkChareID myMain;
}; 

class varraystest_main : public CBase_varraystest_main {
 public:
  varraystest_main(void);
  varraystest_main(CkMigrateMessage *m) {}
  void exit(varraystest_Msg *m);
};

class varraystest_test : public CBase_varraystest_test {
 public:
  varraystest_test(varraystest_Msg *m);
  varraystest_test(CkMigrateMessage *m) {}
};

#endif
