
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

  ComlibInstanceHandle cinst;

  bool useComlib;
  bool isDoublePacked;
  bool conserveMemory;
  PairCalcID() {}
  ~PairCalcID() {}

  void Init(CkArrayID aid, CkGroupID gid, int grain, int blk, int s, bool sym, bool _useComlib,  ComlibInstanceHandle h, bool _dp, bool _conserveMemory) {
      
    Aid = aid;
    Gid = gid;
    GrainSize = grain;
    BlkSize = blk;
    S = s;
    Symmetric = sym;
    useComlib = _useComlib;
    conserveMemory = _conserveMemory;
    cinst = h;
    
    isDoublePacked = _dp;
  }

  void set(PairCalcID pid) {
    Init(pid.Aid, pid.Gid, pid.GrainSize, pid.BlkSize, pid.S, pid.Symmetric, pid.useComlib, pid.cinst, pid.isDoublePacked, pid.conserveMemory);
  }

  void pup(PUP::er &p) {
    p|Aid;
    p|Gid;
    p|GrainSize;
    p|BlkSize;
    p|S;
    p|Symmetric;
    p|cinst;
    p|useComlib;
    p|isDoublePacked;
    p|conserveMemory;
  }
};

extern "C" void createPairCalculator(bool sym, int w, int grainSize, int numZ, int* z, int op1, FuncType f1, int op2, FuncType f2, CkCallback cb, PairCalcID* aid, int ep, CkArrayID cbid, int flag=0, CkGroupID *gid = 0, int flag_dp=0, int conserveMemory=1);

void startPairCalcLeft(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRight(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

extern "C" void finishPairCalc(PairCalcID* aid, int n, double *ptr);

extern "C" void finishPairCalc2(PairCalcID* pcid, int n, double *ptr1, double *ptr2);

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

#endif
