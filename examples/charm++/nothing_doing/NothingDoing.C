#include "NothingDoing.decl.h"

#define ARR_SIZE 24

CProxy_Main mainChare;  /* readonly */

class Main : public CBase_Main
{
public:
  CProxy_NothingDoing arrayProxy;
  int count;

  Main(CkArgMsg* m) {
    count = 0;
    mainChare = thisProxy;

        arrayProxy = CProxy_NothingDoing::ckNew(ARR_SIZE);  
        arrayProxy.doNothing();
    //	 CkExit();
  }

  void done(void) {
    count++;
    if (count == ARR_SIZE) {
      CkExit();
    }
  }
};

class NothingDoing: public CBase_NothingDoing {
public:
  NothingDoing() {
    CkPrintf("Invoking Constructor\n");
  }
  NothingDoing(CkMigrateMessage* m) {
    CkPrintf("Invoking Migrate Constructor\n");
  }
  ~NothingDoing() {
  }
  void doNothing(void) {
    CkPrintf("Invoking doNothing\n");
    mainChare.done();
  }
};

#include "NothingDoing.def.h"
