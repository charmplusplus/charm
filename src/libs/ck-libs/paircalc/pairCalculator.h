
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
  int instid;
  CkGroupID dmid;
  bool useComlib;
  PairCalcID() {}
  ~PairCalcID() {}
  void Init(CkArrayID aid, CkGroupID gid, int grain, int blk, int s, bool sym, bool _useComlib, int _instid, CkGroupID _dmid) {
    Aid = aid;
    Gid = gid;
    GrainSize = grain;
    BlkSize = blk;
    S = s;
    Symmetric = sym;
    useComlib = _useComlib;
    instid = _instid; 
    dmid = _dmid; 
  }
  void pup(PUP::er &p) {
    p|Aid;
    p|Gid;
    p|GrainSize;
    p|BlkSize;
    p|S;
    p|Symmetric;
    p|instid;
    p|dmid;
    p|useComlib;
  }
};

extern "C" void createPairCalculator(bool sym, int w, int grainSize, int numZ, int* z, int op1, FuncType f1, int op2, FuncType f2, const CkCallback cb, PairCalcID* aid, int flag=0, CkGroupID *gid = 0);

void startPairCalcLeft(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRight(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

extern "C" void finishPairCalc(PairCalcID* aid, int n, double *ptr, const CkCallback cb);

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

#endif
