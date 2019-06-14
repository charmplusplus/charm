#include "ro_qd_test.decl.h"

#define ELEMS_PER_PE 10

int arr_size;
int vec_size;

int num_arr1[2000000];
std::vector<int> num_vec1;
CProxy_Main mProxy;

class Main : public CBase_Main {

  int numElements;
  bool reductionCompleted;

  public:
  Main(CkArgMsg *msg) {
    CkPrintf("[%d][%d][%d] Hello, inside main\n", CmiMyPe(), CmiMyNode(), CmiMyRank());

    arr_size = 2000000;
    vec_size = 2000000;
    mProxy = thisProxy;

    for(int i=0; i<arr_size; i++) num_arr1[i] = i;
    for(int i=0; i<vec_size; i++) num_vec1.push_back(i);

    reductionCompleted = false;

    CProxy_arr1::ckNew(ELEMS_PER_PE * CkNumPes());

    CkStartQD(CkCallback(CkIndex_Main::qdReached(), mProxy));
  }

  void done() {
    CkPrintf("[%d][%d][%d] Reduction completed\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
    reductionCompleted = true;
  }

  void qdReached() {
    CkPrintf("[%d][%d][%d] Quiescence has been reached\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
    CkAssert(reductionCompleted == true);
    CkExit();
  }
};

class arr1 : public CBase_arr1 {
  public:
  arr1() {

    for(int i=0; i<arr_size; i++) CkAssert(num_arr1[i] == i);
    for(int i=0; i<vec_size; i++) CkAssert(num_vec1[i] == i);

    CkCallback cb(CkReductionTarget(Main, done), mProxy);
    contribute(cb);
  }
};

#include "ro_qd_test.def.h"
