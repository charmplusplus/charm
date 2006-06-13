#include "onesided.decl.h"


void doneOp(void *tmp);

CpvDeclare(void *, _cmvar);

class charMsg : public CMessage_charMsg {
 public:
  char *addr;
};

class Main : public CBase_Main {
 public:
  Main(CkMigrateMessage *m) {}
  Main(CkArgMsg *);
  void done(void);
};

class commtest : public CBase_commtest {
 private:
  int idx;
  char *srcAddr, *destAddr;
  unsigned int size, dest;
  char srcChar, destChar;
  int operation; //get or put (current operation)
  int callb;
  void *pend;
 public:
  commtest(CkMigrateMessage *m) {}
  commtest(void);
  void startRMA(int op, int cb);
  void remoteRMA(int len, int op, int cb);
  void recvAddr(charMsg *cm);
  void verifyCorrectRMA(char c);
  void doJnkWork(void);
  void testDone(void);
  void initializeMem(char *addr, int len, char c);
  void testForCompletion(void);
  void testForCorrectness(void);
};
