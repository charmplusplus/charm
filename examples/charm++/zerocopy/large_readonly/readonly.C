#include "ro_example.decl.h"

int arr_size;
int vec_size;

int num_arr1[2000000];
std::vector<int> num_vec1;
CProxy_Main mProxy;

class Main : public CBase_Main {

  public:
  Main(CkArgMsg *msg) {
    CkPrintf("[%d][%d][%d] Hello, inside main\n", CmiMyPe(), CmiMyNode(), CmiMyRank());

    arr_size = 2000000;
    vec_size = 2000000;
    mProxy = thisProxy;

    for(int i=0; i<arr_size; i++) num_arr1[i] = i;
    for(int i=0; i<vec_size; i++) num_vec1.push_back(i);

    CProxy_arr1::ckNew(2*CkNumPes());
  }

  void done() {
    CkPrintf("[%d][%d][%d] Verified successful Readonly transfer\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
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

#include "ro_example.def.h"
