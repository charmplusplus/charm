#ifndef _ckPairCalculator_h_
#define _ckPairCalculator_h_

#include "pairutil.h"
#include "cksparsecontiguousreducer.h"
#include "PipeBroadcastStrategy.h"
#include "BroadcastStrategy.h"

#if USE_BLAS
extern "C" complex zdotu_( const int *N,  complex *X, const int *incX,			       complex *Y, const int *incY);

#endif

typedef void (*FuncType) (complex a, complex b);
PUPmarshallBytes(FuncType);

#include "ckPairCalculator.decl.h"

class mySendMsg : public CMessage_mySendMsg {
 public:
  int N;
  complex *data;
  friend class CMessage_mySendMsg;

  /*
  mySendMsg(int N, complex *data) {
    this->N = N;
    memcpy(this->data, data, N*sizeof(complex));
  }
  */
};

class partialResultMsg : public CMessage_partialResultMsg {
 public:
  int N;
  complex *result;
  int priority;
  CkCallback cb;

  friend class CMessage_partialResultMsg;
  /*  
  partialResultMsg(unsigned int iN,   complex *iresult, int ipriority, CkCallback icb) : N(iN),  priority(ipriority), cb(icb)
    {
      memcpy(this->result,iresult,N*sizeof(complex));
    }
  */
};

class priorSumMsg : public CMessage_priorSumMsg {
 public:
  int N;
  complex *result;
  int priority;
  CkCallback cb;

  friend class CMessage_priorSumMsg;
  /*  priorSumMsg(unsigned int iN,  complex *iresult,int ipriority,CkCallback icb) :  N(iN), priority(ipriority), cb(icb)
    {
      memcpy(this->result,iresult,N*sizeof(complex));
    }
  */
};

class PairCalculator: public CBase_PairCalculator {
 public:
  PairCalculator(bool, int, int, int, int op1, FuncType fn1, int op2, FuncType fn2, CkCallback cb, CkGroupID gid);
  PairCalculator(CkMigrateMessage *);
  ~PairCalculator();
  void calculatePairs(int, complex *, int, bool); 
  void acceptEntireResult(int size, double *matrix);
  void acceptEntireResult(int size, double *matrix, CkCallback cb);
  void acceptResult(int size, double *matrix, int rowNum, CkCallback cb);
  void sumPartialResult(int size, complex *result, int offset, CkCallback cb);
  void sumPartialResult(priorSumMsg *msg);
  void sumPartialResult(partialResultMsg *msg);
  void pup(PUP::er &);
  inline double compute_entry(int n, complex *psi1, complex *psi2, int op) 
    {
        /*
          double re=0, im = 0;
          double *ptr1 = (double*)psi1;
          double *ptr2 = (double*)psi2;
          for (int i = 0; i < 2*n; i+=2){
          re += ptr1[i]*ptr2[i] - ptr1[i+1]*ptr2[i+1];
          im += ptr1[i+1]*ptr2[i] + ptr1[i]*ptr2[i+1];
          }
          complex sum(re,im);
        */
#ifdef USE_BLAS
      int incx=1;
      int incy=1;
      complex output=zdotu_(&n, psi1, &incx,  psi2, &incy);
      return(output.re);
#else
        
        int i;
        register double sum = 0;
        for (i = 0; i < n; i++) {
            sum += psi1[i].re * psi2[i].re +  psi1[i].im * psi2[i].im;
        }
        
        return sum;
#endif
  }
 private:
  int numRecd, numExpected, grainSize, S, blkSize, N;
  int kRightCount, kLeftCount, kUnits;
  int *kLeftOffset;
  int *kRightOffset;
  int *kLeftMark;
  int *kRightMark;
  int op1, op2;
  FuncType fn1, fn2;
  complex **inDataLeft, **inDataRight;
  double *outData;
  complex *newData;
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
  void broadcastEntireResult(int size, double* matrix, bool symmtype, CkCallback cb);
  void doRegister(PairCalculator *, bool);

 private:
  CkVec<PairCalculator *> localElements[2];
  int numRegistered[2];
  int acceptCount;
}; 



//#define  _DEBUG_

#endif
