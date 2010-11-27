#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

class MsgPointer : public CMessage_MsgPointer {
public:
  int *ptr;
  MsgPointer(int *p) : ptr(p) { }
};

/*mainchare*/
class Main : public Chare
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thishandle;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17);
  };

  void done(int i)
  {
    CkPrintf("All done\n");
    //CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
  int myInteger;
  char pad[200];
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    if (hiNo>30) *((int*)NULL) = 0;
    CkPrintf("Hi[%d] from element %d: %d\n",hiNo,thisIndex,myInteger);
    if (thisIndex < nElements-1) {
      //Pass the hello on:
      CkPrintf("sending pointer %p (%p)\n",&myInteger,this);
      thisProxy[thisIndex+1].passPointer(new MsgPointer(&myInteger));
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    } else 
      //We've been around once-- we're done.
      mainProxy.done(1);
  }

  void passPointer(MsgPointer *msg) {
    int *ptr = msg->ptr;
    CkPrintf("receiving pointer %p (%p)\n",ptr,this);
    // this next line writes into the memory of another chare!
    *ptr = 10;
  }
};

#include "hello.def.h"
