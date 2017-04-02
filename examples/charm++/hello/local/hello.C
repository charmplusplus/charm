#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_HelloGroup groupProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
  int counter;
  CProxy_Hello arr;
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
    mainProxy = thisProxy;

    counter = 0;
    groupProxy = CProxy_HelloGroup::ckNew();
    arr = CProxy_Hello::ckNew(nElements);
  };
    
  void startIter() {
    // Since the array contact the group with a local method, we need to
    // guarantee that the group if present everywhere before we can allow
    // the array to run.
    if (++counter < CkNumPes()) return; 
    arr[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    int groupValue;
    groupProxy[CkMyPe()].request(&groupValue);
    CkPrintf("Hi[%d] from element %d on group %d\n",hiNo,thisIndex,groupValue);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

class HelloGroup : public CBase_HelloGroup {
public:
  HelloGroup() {
    mainProxy.startIter();
  }
  
  void request(int *ret) {
    *ret = CkMyPe();
  }
};

#include "hello.def.h"
