#include "Counter.decl.h"

extern CkGroupID counterInit();
extern int counterReport();
extern void counterIncrement();

class countMsg : public CMessage_countMsg {
public:
  int count;
  countMsg(int c) : count(c) {};
};

class DUMMY : public CMessage_DUMMY {
};

class counter: public Group {
private:
  CkGroupID mygrp;
  int myCount;
  int totalCount;
  int waitFor;
  CthThread threadId;
public:
  counter(CkMigrateMessage *m) {}
  counter(DUMMY *);
  void childCount(countMsg *);
  void increment();
  void sendCounts(DUMMY *);
  int  getTotalCount();
};
