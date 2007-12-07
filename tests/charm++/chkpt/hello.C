#include <stdio.h>
#include "hello.decl.h"

CProxy_Main mainProxy;
CProxy_Hello helloProxy;
int nElements;

class Main : public CBase_Main {
  int step;
  int a;
  int b[2];
public:
  Main(CkArgMsg* m){
    step=0;	
    a=123;b[0]=456;b[1]=789;
    nElements=8;
    delete m;

    CkPrintf("Running Hello on %d processors for %d elements\n",CkNumPes(),nElements);
    mainProxy = thisProxy;
    helloProxy = CProxy_Hello::ckNew(nElements);
    helloProxy.SayHi();
  }
  
  Main(CkMigrateMessage *m) : CBase_Main(m) { 
    mainProxy = thisProxy;
    a=987;b[0]=654;b[1]=321;
    CkPrintf("Main's MigCtor. a=%d(%p), b[0]=%d(%p), b[1]=%d.\n",a,&a,b[0],b,b[1]);
  }

  void myClient(CkReductionMsg *m){
    step++;
    int stepInc = *((int *)m->getData());
    CkAssert(step == stepInc);
    CkPrintf("myClient. a=%d(%p), b[0]=%d(%p), b[1]=%d.\n",a,&a,b[0],b,b[1]);
    if(step == 3){
      CkCallback cb(CkIndex_Hello::SayHi(),helloProxy);
      CkStartCheckpoint("log",cb);
    }else{
      helloProxy.SayHi();
    }
    delete m;
  }

  void pup(PUP::er &p){
    CBase_Main::pup(p);
    p|step;
    p|a; p(b,2);
    CkPrintf("Main's PUPer. a=%d(%p), b[0]=%d(%p), b[1]=%d.\n",a,&a,b[0],b,b[1]);
  }
};

class Hello : public CBase_Hello
{
  int step;
public:
  Hello(){ step = 0; }
  Hello(CkMigrateMessage *m) : CBase_Hello(m) {}
  
  void SayHi(){
    step++;
    if(step < 10){
      CkCallback cb(CkIndex_Main::myClient(0),mainProxy);
      contribute(sizeof(int),(void*)&step,CkReduction::max_int,cb);
    }else{
      contribute(sizeof(int),(void*)&step,CkReduction::max_int,CkCallback(CkCallback::ckExit));
    }
  }
  
  void pup(PUP::er &p){
    CBase_Hello::pup(p);
    p|step;
  }
};

#include "hello.def.h"

