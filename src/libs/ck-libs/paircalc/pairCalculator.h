#ifndef _pairCalculator_h
#define _pairCalculator_h
#include "ckPairCalculator.h"

class PairCalcID {
 public:
  CkArrayID Aid;
  int GrainSize;
  int BlkSize;
  int S;
  bool Symmetric;
  PairCalcID() {}
  ~PairCalcID() {}
  void Init(CkArrayID aid, int grain, int blk, int s, bool sym) {
    Aid = aid;
    GrainSize = grain;
    BlkSize = blk;
    S = s;
    Symmetric = sym;
  }
  void pup(PUP::er &p) {
    p|Aid;
    p|GrainSize;
    p|BlkSize;
    p|S;
    p|Symmetric;
  }
};

void createPairCalculator(bool sym, int w, int numZ, int* z, int op1, FuncType f1, int op2, FuncType f2, const CkCallback cb, PairCalcID* aid);

void startPairCalcLeft(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRight(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void finishPairCalc(PairCalcID* aid, int n, complex*ptr, int myS, int myZ, const CkCallback cb);

#endif
