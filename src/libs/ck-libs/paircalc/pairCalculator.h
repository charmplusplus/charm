#ifndef _pairCalculator_h
#define _pairCalculator_h
#include "ckPairCalculator.h"

class PairCalcID {
 public:
  CkArrayID Aid;
  CkGroupID Gid;
  int GrainSize;
  int BlkSize;
  int S;
  bool Symmetric;
  PairCalcID() {}
  ~PairCalcID() {}
  void Init(CkArrayID aid,CkArrayID gid, int grain, int blk, int s, bool sym) {
    Aid = aid;
    Gid = gid;
    GrainSize = grain;
    BlkSize = blk;
    S = s;
    Symmetric = sym;
  }
  void pup(PUP::er &p) {
    p|Aid;
    p|Gid;
    p|GrainSize;
    p|BlkSize;
    p|S;
    p|Symmetric;
  }
};

extern "C" void createPairCalculator(bool sym, int w, int numZ, int* z, int op1, FuncType f1, int op2, FuncType f2, const CkCallback cb, PairCalcID* aid);

void startPairCalcLeft(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRight(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

extern "C" void finishPairCalc(PairCalcID* aid, int n, complex*ptr, const CkCallback cb);

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

#endif
