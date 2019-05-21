#include "immediateEM.decl.h"

CProxy_main mainProxy;

class immMessage : public CMessage_immMessage{
public:
  int i;
  double d;
  immMessage() {i=0; d=0;}
  immMessage(int i_, double d_) {
    i = i_;
    d = d_;
  }
};

class main : public CBase_main{
  int counter;
  CProxy_NG1 ng1Proxy;

  public:
  main(CkArgMsg *m){
    CkPrintf("\n[PE: %d][Node: %d][Rank: %d] ****************** Invoking Nodegroup Constructor ****************\n", CkMyPe(), CkMyNode(), CkMyRank());
    mainProxy = thisProxy;
    counter = 0;
    // Create a nodegroup
    ng1Proxy = CProxy_NG1::ckNew();
  }

  void invokeRegularEntryMethod() {
    if(++counter != CkNumNodes()) return;
    counter = 0;
    CkPrintf("\n[PE: %d][Node: %d][Rank: %d] ****************** Invoking Regular Entry Method *****************\n", CkMyPe(), CkMyNode(), CkMyRank());
    ng1Proxy.regularEntryMethod();
  }

  void invokeImmediateEntryMethod() {
    if(++counter != CkNumNodes()) return;
    counter = 0;
    CkPrintf("\n[PE: %d][Node: %d][Rank: %d] *********** Invoking Immediate Entry Method through proxy ********\n", CkMyPe(), CkMyNode(), CkMyRank());
    ng1Proxy.immediateEntryMethod();
  }

  void invokeImmediateEntryMethodCb() {
    if(++counter != CkNumNodes()) return;
    counter = 0;
    CkPrintf("\n[PE: %d][Node: %d][Rank: %d] ******** Invoking Immediate Entry Method Through Callback *******\n", CkMyPe(), CkMyNode(), CkMyRank());
    CkCallback cb(CkIndex_NG1::immediateEntryMethod(), ng1Proxy);
    cb.send();
  }

  void invokeImmediateEntryMethodWithMessage() {
    if(++counter != CkNumNodes()) return;
    counter = 0;
    CkPrintf("\n[PE: %d][Node: %d][Rank: %d] ******** Invoking Immediate Entry Method with immediate message *******\n", CkMyPe(), CkMyNode(), CkMyRank());
    immMessage *iMsg = new immMessage(20,8.66);
    ng1Proxy.immediateEntryMethodWithMessage(iMsg);
  }

  void done() {
    if(++counter != CkNumNodes()) return;
    CkExit();
  }
};

class NG1 : public CBase_NG1{
  bool immCalled;
  public:
  NG1(){
    immCalled = false;
    CkPrintf("[PE: %d][Node: %d][Rank: %d] Invoked Nodegroup constructor\n", CkMyPe(), CkMyNode(), CkMyRank());
    mainProxy.invokeRegularEntryMethod();
  }

  void regularEntryMethod(){
    CkPrintf("[PE: %d][Node: %d][Rank: %d] Invoked regular entry method\n", CkMyPe(), CkMyNode(), CkMyRank());
    mainProxy.invokeImmediateEntryMethod();
  }

  void immediateEntryMethod(){
    if(!immCalled) {
      immCalled = true;
      CkPrintf("[PE: %d][Node: %d][Rank: %d] Invoked immediate entry method through regular proxy call\n", CkMyPe(), CkMyNode(), CkMyRank());
      mainProxy.invokeImmediateEntryMethodCb();
    } else {
      CkPrintf("[PE: %d][Node: %d][Rank: %d] Invoked immediate entry method through callback\n", CkMyPe(), CkMyNode(), CkMyRank());
      mainProxy.invokeImmediateEntryMethodWithMessage();
    }
  }

  void immediateEntryMethodWithMessage(immMessage *iMsg) {
    CkPrintf("[PE: %d][Node: %d][Rank: %d] Invoked immediate entry method with immediate Message\n", CkMyPe(), CkMyNode(), CkMyRank());
    CkAssert(iMsg->i == 20);
    CkAssert(fabs(iMsg->d - 8.66)< 0.000001);
    delete iMsg;
    mainProxy.done();
  }
};

#include "immediateEM.def.h"
