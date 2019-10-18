#include "readonlyBcast.decl.h"

std::vector<char> num_vec1;
CProxy_Main mProxy;

#undef CMK_ONESIDED_RO_THRESHOLD
#define CMK_ONESIDED_RO_THRESHOLD 0

class Main : public CBase_Main {

  int vec_size;
  bool isZcpy;
  double start_time, end_time, bcast_time;

  public:
  Main(CkArgMsg *msg) {

    if(msg->argc != 2) {
      CkPrintf("Usage: ./readonlyBcast <data-size>\n");
      CkExit(1);
    }

    vec_size = atoi(msg->argv[1]);
    isZcpy = false;

#if !CMK_ONESIDED_RO_DISABLE && CMK_ONESIDED_IMPL
    if(vec_size >= CMK_ONESIDED_RO_THRESHOLD) {
      isZcpy = true;
      CkPrintf("[%d][%d][%d] Bcasting using Zerocopy Bcast for size:%d and threshold:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), vec_size, CMK_ONESIDED_RO_THRESHOLD);
    }
    else
#endif
      CkPrintf("[%d][%d][%d] Bcasting using Regular Bcast for size:%d \n", CmiMyPe(), CmiMyNode(), CmiMyRank(), vec_size);

    mProxy = thisProxy;
    for(int i=0; i<vec_size; i++) num_vec1.push_back(0 + rand()%255);

    CProxy_group1::ckNew();

    start_time = CkWallTimer();
  }

  void done() {
    end_time = CkWallTimer();

    bcast_time = end_time - start_time;
    if(isZcpy)
      CkPrintf("[%d][%d][%d] ZC, Result: Size:%d bytes, Time taken:%lf us\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), vec_size, bcast_time*1.0e6);
    else
      CkPrintf("[%d][%d][%d] Regular, Result: Size:%d bytes, Time taken:%lf us\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), vec_size, bcast_time*1.0e6);
    CkExit();
  }
};

class group1 : public CBase_group1 {
  public:
  group1() {
    CkCallback cb(CkReductionTarget(Main, done), mProxy);
    contribute(cb);
  }
};

#include "readonlyBcast.def.h"
