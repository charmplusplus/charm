#include <stdio.h>
#include "hello.decl.h"

CProxy_Main mainProxy;
CProxy_Hello helloProxy;
CProxy_CHello chelloProxy;
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

    chelloProxy = CProxy_CHello::ckNew(0);
    chelloProxy.SayHi(0);
  }
  
  Main(CkMigrateMessage *m) : CBase_Main(m) { 
    if (m!=NULL) {
      CkArgMsg *args = (CkArgMsg *)m;
      CkPrintf("Received %d arguments: { ",args->argc);
      for (int i=0; i<args->argc; ++i) {
        CkPrintf("|%s| ",args->argv[i]);
      }
      CkPrintf("}\n");
    } else {
      CkPrintf("Arguments null\n");
    }
      // subtle: Chare proxy readonly needs to be updated manually because of
      // the object pointer inside it.
    mainProxy = thisProxy;
    a=987;b[0]=654;b[1]=321;
    CkPrintf("Main's MigCtor. a=%d(%p), b[0]=%d(%p), b[1]=%d.\n",a,&a,b[0],b,b[1]);
  }

  void myClient(CkReductionMsg *m){
    step++;

      // if restarted from a different num of pes, we have to ignore the chare
    if (!_restarted || _chareRestored) chelloProxy.SayHi(step);

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

class CHello : public Chare 
{
  int step;
public:
  CHello(){ step = 0; }
  CHello(CkMigrateMessage *m): Chare(m) { step = 0; }

  void SayHi(int s) {
    step = s;
    printf("step %d done\n", step);
  }

  void pup(PUP::er &p){
    Chare::pup(p);
    p|step;
    printf("CHello's PUPer. step=%d.\n", step);
  }
};

#include "hello.def.h"

