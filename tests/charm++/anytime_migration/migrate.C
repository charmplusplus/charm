#include <stdio.h>
#include "migrate.decl.h"

// This program is meant to test anytime migration and messaging in a simple context
// as a part of debugging the dynamic deletion-followed-by-reinsertion bug (#3660)
// It should be run on exactly 2 PEs with 3 chare array elements
// it is expected that A[0] lives on pe 0, A[2] on pe 1, and A[1] will alternate its location
// every MIGRATION_PERIOD steps.
// If the mapping is not right, then we should manually insert it on specific PEs.
// (but seems like the default map is doing the right thing)

// The baseline program does the following:
//     each element (in parallel) in its sayHi Method, sends ackPlease() to its next element
//     ackPlease sends sayHi() to its predecessor.
//     a count, passed as a parameter, is decremented by sayHi
//     program stops via quiescence detection after counts reach 0.

#define MIGRATION_PERIOD 5
/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Hello arrProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=3;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    arrProxy = CProxy_Hello::ckNew(nElements);
    arrProxy.SayHi(12);
    CkStartQD(CkCallback(CkIndex_Main::done(),thisProxy));
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
    CkPrintf("Hello[%d] on PE %d: created.\n",thisIndex, CkMyPe());
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int count)
  {
    CkPrintf("Hello[%d] on PE %d : Saying Hi with count = %d\n",thisIndex,CkMyPe(),count);
    if (count > 0)
      thisProxy[(thisIndex+1)%nElements].ackPlease(thisIndex, count-1);
    if (1 == (count % MIGRATION_PERIOD)) migrateMe(1-CkMyPe());
    // using 1 because no point migrating when you are about to exit
    // 1 - CkMyPe() ensures it migrates between PE 0 and 1 (assuming it started on one of them..
    // This code is meant to be used on exactly 2 PEs anyway.
  }
  void ackPlease(int prev,int c) {
    CkPrintf("Hello[%d] on PE %d: acking with count= %d\n",thisIndex,CkMyPe(),c);
    thisProxy[prev].SayHi(c);
  }

};

#include "migrate.def.h"
