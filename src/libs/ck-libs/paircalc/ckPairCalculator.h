#ifndef _ckPairCalculator_h_
#define _ckPairCalculator_h_

#include "pairutil.h"
#include "cksparsecontiguousreducer.h"

typedef void (*FuncType) (complex a, complex b);
PUPmarshallBytes(FuncType);

#include "ckPairCalculator.decl.h"

class mySendMsg : public CMessage_mySendMsg {
 public:
  int N;
  complex *data;
  friend class CMessage_mySendMsg;
  mySendMsg(int N, complex *data) {
    this->N = N;
    memcpy(this->data, data, N*sizeof(complex));
  }
};

class PairCalculator: public CBase_PairCalculator {
 public:
  PairCalculator(bool, int, int, int, int op1, FuncType fn1, int op2, FuncType fn2, CkCallback cb, CkGroupID gid);
  PairCalculator(CkMigrateMessage *);
  ~PairCalculator();
  void calculatePairs(int, complex *, int, bool); 
  void acceptEntireResult(int size, complex *matrix);
  void acceptEntireResult(int size, complex *matrix, CkCallback cb);
  void acceptResult(int size, complex *matrix, int rowNum, CkCallback cb);
  void sumPartialResult(int size, complex *result, int offset, CkCallback cb);
  void pup(PUP::er &);
  inline complex compute_entry(int n, complex *psi1, complex *psi2, int op);
 private:
  int numRecd, numExpected, grainSize, S, blkSize, N;
  int op1, op2;
  FuncType fn1, fn2;
  complex **inDataLeft, **inDataRight;
  complex *outData, *newData;
  int sumPartialCount;
  bool symmetric;
  CkCallback cb;
  CkSparseContiguousReducer<CkTwoDoubles> r;
};

class PairCalcReducer : public Group {
 public:
  PairCalcReducer(CkMigrateMessage *m) { }
  PairCalcReducer(){ acceptCount = 0; numRegistered[0] = 0; numRegistered[1] = 0;}
  ~PairCalcReducer() {}
  void acceptPartialResult(int size, complex* matrix, int fromRow, int fromCol, CkCallback cb);
  void broadcastEntireResult(int size, complex* matrix, bool symmtype, CkCallback cb);
  void doRegister(PairCalculator *, bool);

 private:
  CkVec<PairCalculator *> localElements[2];
  int numRegistered[2];
  int acceptCount;
}; 

//#define  _DEBUG_

#endif
