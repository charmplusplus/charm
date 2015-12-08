#ifndef _CKLOOPAPI_H
#define _CKLOOPAPI_H

#include "CkLoop.decl.h"

/* "result" is the buffer for reduction result on a single simple-type variable */
typedef void (*HelperFn)(int first,int last, void *result, int paramNum, void *param);
/* Function that will be executed by the caller PE before ckloop is done */
typedef void (*CallerFn)(int paramNum, void *param);

typedef enum REDUCTION_TYPE {
    CKLOOP_NONE=0,
    CKLOOP_INT_SUM,
    CKLOOP_FLOAT_SUM,
    CKLOOP_DOUBLE_SUM,
    CKLOOP_DOUBLE_MAX
} REDUCTION_TYPE;

class CProxy_FuncCkLoop;
/*
 * "numThreads" argument is intended to be used in non-SMP mode to specify
 * the number of pthreads to be spawned. In SMP mode, this argument is
 * ignored. This function should be called only on one PE, say PE 0.
 **/
extern CProxy_FuncCkLoop CkLoop_Init(int numThreads=0);

/* used to free resources if using the library in non-SMP mode. It should be called on just one PE, say PE 0 */
extern void CkLoop_Exit(CProxy_FuncCkLoop ckLoop); 

extern void CkLoop_Parallelize(
    HelperFn func, /* the function that finishes a partial work on another thread */
    int paramNum, void * param, /* the input parameters for the above func */
    int numChunks, /* number of chunks to be partitioned */
    int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
    int sync=1, /* whether the flow will continue unless all chunks have finished */
    void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE, /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
    CallerFn cfunc=NULL, /* caller PE will call this function before ckloop is done and before starting to work on its chunks */
    int cparamNum=0, void *cparam=NULL /* the input parameters to the above function */
);

extern void CkLoop_DestroyHelpers();
#endif
