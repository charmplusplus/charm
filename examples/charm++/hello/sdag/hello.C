#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Hello helloProxy;
/*readonly*/ int nElements;

class HelloMsg : public CMessage_HelloMsg{
public:
    int data;
    HelloMsg(int d):data(d){}
};

/*mainchare*/
class Main : public CBase_Main
{
public:
  int counter;
  Main(CkArgMsg* m)
  {
    delete m;
    counter=0;
    nElements=5;

    CkPrintf("Running Hello on %d processors for %d elements\n",CkNumPes(),nElements);
    mainProxy = thisProxy;
    helloProxy = CProxy_Hello::ckNew(nElements);
    helloProxy.SayHi(new HelloMsg(0));
  }

  void Done(void)
  {
    counter++;
    if(counter==nElements){
      CkPrintf("All done\n");
      CkExit();
    }
  }
};

/*array [1D]*/
class Hello : public CBase_Hello
{
  Hello_SDAG_CODE
public:
  int i,sum;
  Hello()
  {
    sum=0;
    CkPrintf("Hello %d created\n",thisIndex);
  }
  Hello(CkMigrateMessage *m) {}
};

#include "hello.def.h"
