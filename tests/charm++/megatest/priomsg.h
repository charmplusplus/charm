#ifndef _PRIOMSG_H
#define _PRIOMSG_H

#include "megatest.h"
#include "priomsg.decl.h"

class priomsg_Msg : public CMessage_priomsg_Msg {
 public:
  priomsg_Msg(void);
  int check(void);
  int data[10];
}; 

class priomsg_test : public CBase_priomsg_test {
 public:
  priomsg_test(void) {};
  priomsg_test(CkMigrateMessage *m) {}
  void donow(priomsg_Msg *m);
};

#endif
