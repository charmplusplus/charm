#ifndef _VARSIZETEST2_H
#define _VARSIZETEST2_H

#include "megatest.h"
#include "varsizetest2.decl.h"


class varsizetest2_Msg : public CMessage_varsizetest2_Msg {
 public:
  varsizetest2_Msg(int, int, int);
  int check(void);
  int seqnum;
  int isize;
  int fsize;
  int *iarray;
  float *farray;
  CkChareID myMain;
}; 

class varsizetest2_main : public CBase_varsizetest2_main {
 public:
  varsizetest2_main(void);
  varsizetest2_main(CkMigrateMessage *m) {}
  void exit(varsizetest2_Msg *m);
};

class varsizetest2_test : public CBase_varsizetest2_test {
 public:
  varsizetest2_test(varsizetest2_Msg *m);
  varsizetest2_test(CkMigrateMessage *m) {}
};

#endif
