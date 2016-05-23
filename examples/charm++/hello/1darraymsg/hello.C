#include <stdio.h>
#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;
/*readonly*/ int msgSize;
/*readonly*/ int maxIter;

/*mainchare*/
class dataMsg : public CMessage_dataMsg {

public :
    char *data;
};
class Main : public CBase_Main
{
    double startTimer;
    int back;
    int iter;
    CProxy_Hello arr;
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    msgSize = 2048;
    maxIter = 50;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    if(m->argc >2 ) msgSize =atoi(m->argv[2]);
    if(m->argc >3 ) maxIter=atoi(m->argv[3]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;
    back = 0;
    iter = 0; 
    arr = CProxy_Hello::ckNew(nElements);
    dataMsg *datamsg = new (msgSize) dataMsg();
    startTimer = CkWallTimer();
    arr.SayHi(datamsg);
  };

  void done(void)
  {
      back++;
      if(back == nElements)
      {
          iter++;
          if(iter == maxIter)
          {
              CkPrintf("All done %d %d  cost  %lf \n", nElements, msgSize, (CkWallTimer()-startTimer)/maxIter);
              CkExit();
          }else
          {
              back = 0;
              dataMsg *datamsg = new (msgSize) dataMsg();
              arr.SayHi(datamsg);
          }
      }
  };
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    //CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(dataMsg *dm)
  {
    //CkPrintf("Hi[%d] from element \n",thisIndex);
    delete dm;  
    mainProxy.done();
  }
};

#include "hello.def.h"
