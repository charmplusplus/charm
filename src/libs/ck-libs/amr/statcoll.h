#ifndef _Stat_Coll__H
#define _Stat_Coll__H
#include "charm++.h"
#include "amr.h"

class _DummyMsg : public CMessage__DummyMsg {
 public:
  
};

class _CreateStatCollMsg: public CMessage__CreateStatCollMsg
{
 public:
  CkChareID coordHandle;
  // _CreateStateCollMsg(){ };
  _CreateStatCollMsg(CkChareID handle){
  coordHandle = handle;}
};

class _StatCollMsg :public CMessage__StatCollMsg
{
 public:
  int aRefineCount;
  int refineCount;
  int msgExpected;
  int migCount;
  int pnum;
  _StatCollMsg(){}
  _StatCollMsg(int ac, int rc, int msg, int mig, int pe){
      aRefineCount = ac;
      refineCount = rc;
      msgExpected = msg;
      migCount = mig;
      pnum = pe;
    }
};

class StatCollector :public Group {
 private:
  int aRefineCount;
  int refineCount;
  int msgExpected;
  int migCount;
  //CProxy_AmrCoordinator coordProxy;
  CkChareID coordHandle;
  
 public:
  StatCollector(){}
  StatCollector(_CreateStatCollMsg* m) {
    aRefineCount = 0;
    refineCount = 0;
    msgExpected = 0;
    migCount = 0;
    //    coordProxy = CProxy_AmrCoordinator(m->coordHandle);
    coordHandle = m->coordHandle;
    delete m;
  }
  
  void registerMe(void);
  void unRegisterMe(void);
  void migrating();
  void incrementAutorefine(void);
  void incrementRefine(void);
  void sendStat(_DummyMsg *m);
};

#endif
