#include <stdio.h>
#include "migrateHello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numIterations;

/*mainchare*/
class Main : public CBase_Main
{
    CProxy_MigrateHello arr;
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    numIterations=1000;
    if(m->argc >1 ) numIterations=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running MigrateHello on %d processors for %d iterations \n",
	     CkNumPes(),numIterations);
    mainProxy = thisProxy;
    arr =  CProxy_MigrateHello::ckNew(1);
    arr[0].SayHi(0);
  };

};

double startTimer;
/*array [1D]*/
class MigrateHello : public CBase_MigrateHello
{
public:
  MigrateHello()
  {
    //CkPrintf("MigrateHello %d created\n",thisIndex);
  }

  MigrateHello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
      if(hiNo <2 )
          startTimer = CkWallTimer();
      else if(hiNo >= numIterations)
      {
          double time = CkWallTimer() - startTimer;
          CkPrintf(" migration cost total : %f sec single migration cost: %f us\n", time, time/(hiNo-1)*1000000); 
          CkExit();
      }
      //CkPrintf("executing  %d  %d\n", CkMyPe(), hiNo);
      thisProxy[thisIndex].SayHi(hiNo+1);
      migrateMe(1-CkMyPe());
  }
};

#include "migrateHello.def.h"
