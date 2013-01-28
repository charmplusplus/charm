#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;
/*readonly*/ int msgSize;

/*mainchare*/
class dataMsg : public CMessage_dataMsg {

public :
    char *data;
};
class Main : public CBase_Main
{
    double startTimer;
    int back;
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    msgSize = 2048;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    if(m->argc >2 ) msgSize =atoi(m->argv[2]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;
    back = 0;
    
    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
    dataMsg *datamsg = new (msgSize) dataMsg();
    startTimer = CmiWallTimer();
    arr.SayHi(datamsg);
  };

  void done(void)
  {
      back++;
      if(back == nElements)
          CkPrintf("All done %d %d  cost  %lf \n", nElements, msgSize, CmiWallTimer()-startTimer);
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
  
  void SayHi(dataMsg *dm)
  {
    CkPrintf("Hi[%d] from element \n",thisIndex);
    delete dm;  
    mainProxy.done();
  }
};

#include "hello.def.h"
