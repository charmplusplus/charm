# include "benchmark.decl.h"

class TheMain : public CBase_TheMain {
 public:
  TheMain(CkArgMsg *msg);
  void exit();
 private:
  int called;
};

class MyMess : public CkMcastBaseMsg, public CMessage_MyMess {
 public:
  char *data;
};

class BTest : public CBase_BTest {
  double startTime;
  double totalTime;
  int length;
  int type;
  int count;
  int iter;
  CkCallback cb;
  CkCallback cball;
 public:
  BTest();
  BTest(CkMigrateMessage *msg);
  void sendall();
  void send();
  void send(CkReductionMsg *msg);
  void receive(MyMess *msg);
};
