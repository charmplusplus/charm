#ifndef CKLAMBDA_H
#define CKLAMBDA_H

#include <functional>

#if     CMK_SMP && USE_CKLOOP

#include "CkLoopAPI.h"

static void CkLoop_LambdaHelperFn(int first, int last, void *result, int paramNum, void *param) {
  (*static_cast<std::function<void(int,int,void*)>*>(param))(first,last,result);
}

static void CkLoop_LambdaCallerFn(int paramNum, void *param) {
  (*static_cast<std::function<void()>*>(param))();
}

// non-sync is not supported because there is no way to preserve std::function object
// reductions are not implemented but could be

inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange,
  std::function<void(int,int,void*)> func /* the function that finishes a partial work on another thread */
) {
  if ( numChunks < 2 || lowerRange == upperRange ) {  // disable with numChunks == 0 or 1
    func(lowerRange, upperRange, NULL);
  } else {
    CkLoop_Parallelize(CkLoop_LambdaHelperFn, 1, (void *)&func,
      numChunks, lowerRange, upperRange);
  }
}

inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange,
  std::function<void(int,int,void*)> func, /* the function that finishes a partial work on another thread */
  // int sync=1, /* whether the flow will continue unless all chunks have finished */
  void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE, /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
  std::function<void()> cfunc=NULL /* caller PE will call this function before ckloop is done and before starting to work on its chunks */
) {

  if ( numChunks < 1 ) {  // disable with numChunks == 0
    if ( cfunc ) cfunc();
    func(lowerRange, upperRange, redResult);
  } else {
    if ( cfunc )
      CkLoop_Parallelize(CkLoop_LambdaHelperFn, 1, (void *)&func,
        numChunks, lowerRange, upperRange, 1, redResult, type,
        CkLoop_LambdaCallerFn, 1, (void *)&cfunc);
    else
      CkLoop_Parallelize(CkLoop_LambdaHelperFn, 1, (void *)&func,
        numChunks, lowerRange, upperRange, 1, redResult, type);
  }
}

#else // CMK_SMP && USE_CKLOOP

template< class F >
inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange, F func ) {
  func(lowerRange, upperRange);
}

template< class F, class C >
inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange, F func, C cfunc ) {
  cfunc();
  func(lowerRange, upperRange);
}

inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange,
  std::function<void(int,int)> func
) {
  func(lowerRange, upperRange);
}

inline void CkLoop_Parallelize( int numChunks, int lowerRange, int upperRange,
  std::function<void(int,int)> func,
  std::function<void()> cfunc=nullptr
) {
  if ( cfunc ) cfunc();
  func(lowerRange, upperRange);
}

#endif // CMK_SMP && USE_CKLOOP

#endif // CKLAMBDA_H
