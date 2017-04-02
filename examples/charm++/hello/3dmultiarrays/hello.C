#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  int count;
  Main(CkArgMsg* m)
  {
    count = 2;
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    //Allocate elements scattered down a sparse 3D line
    CProxy_Hello arr = CProxy_Hello::ckNew();
    for (int y=0;y<nElements;y++)
        arr(37,y,2*y+1).insert();
    arr.doneInserting();

    CProxy_Hello2 arr2 = CProxy_Hello2::ckNew();
    for (int y=0;y<nElements;y++)
        arr2(37,y,2*y+1).insert();
    arr2.doneInserting();

    arr(37,0,1).SayHi(17);
    arr2(37,0,1).SayHi2(17);
  };

  void done(void)
  {
    count--;
    if (count == 0) {
      CkPrintf("All done\n");
      CkExit();
    }
  };
};

/*array [3D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex.y);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi[%d] from element (%d,%d,%d)\n",hiNo,
	thisIndex.x,thisIndex.y,thisIndex.z);
    int y=thisIndex.y+1;
    if (y < nElements) {
      //Pass the hello on:
      thisProxy(37,y,2*y+1).SayHi(hiNo+1);
    }
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

/*array [3D]*/
class Hello2 : public CBase_Hello2
{
public:
  Hello2()
  {
    CkPrintf("Hello2 %d created\n",thisIndex.y);
  }

  Hello2(CkMigrateMessage *m) {}
  
  void SayHi2(int hiNo)
  {
    CkPrintf("Hi 2 [%d] from element (%d,%d,%d)\n",hiNo,
	thisIndex.x,thisIndex.y,thisIndex.z);
    int y=thisIndex.y+1;
    if (y < nElements) {
      //Pass the hello on:
      thisProxy(37,y,2*y+1).SayHi2(hiNo+1);
    }
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
