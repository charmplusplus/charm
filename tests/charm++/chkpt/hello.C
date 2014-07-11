#include <stdio.h>
#include "hello.decl.h"

CProxy_Main mainProxy;
CProxy_Hello helloProxy;
CProxy_CHello chelloProxy;
CProxy_HelloGroup helloGroupProxy;
CProxy_HelloNodeGroup helloNodeGroupProxy;
int nElements;
int chkpPENum;
int chkpNodeNum;

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
    
    chkpPENum = CkNumPes();
    chkpNodeNum = CkNumNodes();
    CkPrintf("Running Hello on %d processors for %d elements\n",CkNumPes(),nElements);
    mainProxy = thisProxy;
    CkArrayOptions helloOpts(nElements);
    helloOpts.setBounds(10000);
    helloProxy = CProxy_Hello::ckNew(helloOpts);
    helloProxy.SayHi();

    chelloProxy = CProxy_CHello::ckNew(0);
    chelloProxy.SayHi(0);
  
    helloGroupProxy = CProxy_HelloGroup::ckNew();
    helloNodeGroupProxy = CProxy_HelloNodeGroup::ckNew();
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
    CkPrintf("Main's MigCtor. a=%d(%p), b[0]=%d(%p), b[1]=%d, old PE number %d\n",a,&a,b[0],b,b[1], chkpPENum);
  }

  void myClient(CkReductionMsg *m){
    step++;

      // if restarted from a different num of pes, we have to ignore the chare
    if (!_restarted || _chareRestored) chelloProxy.SayHi(step);

    int stepInc = *((int *)m->getData());
    CkAssert(step == stepInc);
    CkPrintf("myClient. a=%d(%p), b[0]=%d(%p), b[1]=%d\n",a,&a,b[0],b,b[1]);
    if(step == 3){
      CkCallback cb(CkIndex_Hello::SayHi(),helloProxy);
      CkStartCheckpoint("log",cb);
    }else{
      helloProxy.SayHi();
    }
    delete m;
  }

  void pup(PUP::er &p){
    p|step;
    p|a; p(b,2);
    CkPrintf("Main's PUPer. a=%d(%p), b[0]=%d(%p), b[1]=%d\n",a,&a,b[0],b,b[1]);
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
    p|step;
  }
};

class CHello : public CBase_CHello
{
  int step;
public:
  CHello(){ step = 0; }
  CHello(CkMigrateMessage *m): CBase_CHello(m) { step = 0; }

  void SayHi(int s) {
    step = s;
    printf("step %d done\n", step);
  }

  void pup(PUP::er &p){
    p|step;
    printf("CHello's PUPer. step=%d.\n", step);
  }
};

class HelloGroup: public CBase_HelloGroup
{
  int data;
public:
  HelloGroup(){data = CkMyPe();}
  HelloGroup(CkMigrateMessage *m): CBase_HelloGroup(m) {}

  void pup(PUP::er &p)
  {
    p|data;
    if(p.isUnpacking())
    {
      CkPrintf("[%d] data on Group %d\n", CkMyPe(), data);
      if(chkpPENum == CkNumPes())
      {
        if(data!=CkMyPe())
        {
          CkAbort("data not recovered for Group");
        }
      }
    }
  }
};

class HelloNodeGroup: public CBase_HelloNodeGroup
{
  int data;
public:
  HelloNodeGroup(){data = CkMyNode();}
  HelloNodeGroup(CkMigrateMessage *m): CBase_HelloNodeGroup(m) {}

  void pup(PUP::er &p)
  {
    p|data;
    if(p.isUnpacking())
    {
      CkPrintf("[%d] data on NOdeGroup %d\n", CkMyNode(), data);
      if(chkpNodeNum == CkNumNodes())
      {
        if(data!=CkMyNode())
        {
          CkAbort("data not recovered for NodeGroup");
        }
      }
    }
  }
};

#include "hello.def.h"

