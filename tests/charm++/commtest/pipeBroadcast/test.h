# include "test.decl.h"

class TheMain : public CBase_TheMain {
 public:
  TheMain(CkArgMsg *msg);
  void exit();
 private:
  int called;
};

class MyMess : public CkMcastBaseMsg, public CMessage_MyMess {
 public:
  double *data;
};

class Test : public CBase_Test {
 public:
  Test();
  Test(CkMigrateMessage *msg);
  void send();
  void receive(MyMess *msg);
};
