#ifndef _VARSIZETEST2_H
#define _VARSIZETEST2_H

#include "megatest.h"
#include "varsizetest2.decl.h"


class varsizetest2_Msg : public CMessage_varsizetest2_Msg {
 public:
  
#if NOCRASH
    varsizetest2_Msg() {}
#endif
  int check(void);
  int isize;
  int fsize;
  int *iarray;
  float *farray;
  int seqnum;
  CkChareID myMain;
}; 

class varsizetest2_main : public Chare {
 public:
  varsizetest2_main(void);
  varsizetest2_main(CkMigrateMessage *m) {}
  void exit(varsizetest2_Msg *m);
};

class varsizetest2_test : public Chare {
 public:
  varsizetest2_test(varsizetest2_Msg *m);
  varsizetest2_test(CkMigrateMessage *m) {}
};

#endif
