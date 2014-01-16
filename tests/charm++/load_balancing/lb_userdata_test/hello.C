#include "hello.decl.h"

#include <stdio.h>
#include <math.h>

#include "userdata_struct.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;
/*readonly*/ int nSteps;
/*readonly*/ int lbstep;

CkpvExtern(int, _lb_obj_index);

/*mainchare*/
class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    nSteps = 100;
    if (m->argc > 2) nSteps = atoi(m->argv[2]);
    lbstep = 10;
    if (m->argc > 3) lbstep = atoi(m->argv[3]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements for %d steps lb %d\n",
	     CkNumPes(),nElements, nSteps, lbstep);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
    arr.doWork();
  };

  void done(void) {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello {
public:
  Hello() {
    usesAtSync = true;
    iteration = 0;
    res = 0.0;
  }

  Hello(CkMigrateMessage *m) {}

  void pup(PUP::er &p) {
    p|iteration;
    p|res;
  }
  
  void doWork() {
    if (iteration < nSteps) {
      for (int i = 0; i < 1000; i++) {
        res += sqrt(i);
      }
      iteration++;
      if (iteration%lbstep == 0) {
        LBUserDataStruct udata;
        udata.idx = thisIndex;
        udata.handle = myRec->getLdHandle();
#if CMK_LB_USER_DATA
        void* data = getObjUserData(CkpvAccess(_lb_obj_index));
        *(LBUserDataStruct *) data = udata;
#else
        CkAbort("CMK_LB_USER_DATA is not enabled.\n");
#endif
        AtSync();
      } else {
        thisProxy[thisIndex].doWork();
      }
    } else { 
      mainProxy.done();
    }
  }

  void ResumeFromSync() {
    thisProxy[thisIndex].doWork();
  }
private:
  int iteration;
  double res;
};

#include "hello.def.h"
