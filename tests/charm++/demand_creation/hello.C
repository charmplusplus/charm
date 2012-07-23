/** \file hello.C
 *  Author: Ehsan Totoni 
 *  Date Created: July 20, 2012
 */

#include "hello.decl.h"
#include <stdio.h>

/*readonly*/ CProxy_Main mainProxy;

/** \class Main
 *
 */
class Main : public CBase_Main
{
public:
  CProxy_Hello arr;
  int counter;

  Main(CkArgMsg* m)
  {
    //Start the computation
    CkPrintf("Running Hello on %d processors \n",
	     CkNumPes());
    mainProxy = thisProxy;

    arr = CProxy_Hello::ckNew();
    arr.doneInserting();
    counter = 0;
    CkPrintf("Array created\n");

    arr(0, 0, 0).SayHi(0);
    arr(0, 4, 8).SayHi(1);
    arr(20, 3, 7).SayHi(2);
    arr(8, 0, 4).SayHi(3);
  };

  void done(void)
  {
    counter++;
    if(counter == 4) {
      CkPrintf("All done\n");
      CkExit();
    }
  };
};

/** \class Hello
 *
 */
class Hello : public CBase_Hello 
{
public:
  Hello()
  {
    CkPrintf("Hello %d %d %d created\n", thisIndex.x, thisIndex.y, thisIndex.z);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi [%d] from element %d %d %d\n", hiNo, thisIndex.x, thisIndex.y, thisIndex.z);
    mainProxy.done();
  }

};

#include "hello.def.h"
