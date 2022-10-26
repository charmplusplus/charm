#include "driver.decl.h"
#include "driver.h"
#include "htram.h"
/* readonly */ CProxy_Master mainProxy;
/* readonly */ CProxy_Histogram histArray;
/* readonly */ CProxy_HTram htramProxy;
/* readonly */ CProxy_HTramRecv nodeGrpProxy;

class Master : public CBase_Master {
  Master_SDAG_CODE
  public:
    Master(CkArgMsg* msg){
      nodeGrpProxy = CProxy_HTramRecv::ckNew();
      htramProxy = CProxy_HTram::ckNew(CkCallback(CkReductionTarget(Master, done), thisProxy));
      histArray = CProxy_Histogram::ckNew();
    }
    void done() {
      CkPrintf("\nFinished sending data");
      CkExit();
    }
};

Histogram::Histogram(){
  data[CkMyRank()] = new int[BUFSIZE*10];
  index[CkMyRank()] = 0;
  HTram* htram = htramProxy.ckLocalBranch();
  htram->setCb(&userDeliver);
  contribute(CkCallback(CkReductionTarget(Histogram, insertSend), thisProxy));
}

Histogram::Histogram(CkMigrateMessage* msg){}

void Histogram:: insertSend() {
  for(int i=0;i<1024*10;i++)
  htramProxy[thisIndex].insertValue(CkMyRank(), ((CkMyNode()+1)%CkNumNodes())*CkNodeSize(0)+CkMyRank());
}

void Histogram:: userDeliver(int val) {
  data[CkMyRank()][index[CkMyRank()]++] = val;
}    

#include "driver.def.h"

