#ifndef _CKLOOPAPI_H
#define _CKLOOPAPI_H

#include "CkLoop.decl.h"

/* "result" is the buffer for reduction result on a single simple-type variable */
typedef void (*HelperFn)(int first,int last, void *result, int paramNum, void *param);

typedef enum REDUCTION_TYPE {
    CKLOOP_NONE=0,
    CKLOOP_INT_SUM,
    CKLOOP_FLOAT_SUM,
    CKLOOP_DOUBLE_SUM
} REDUCTION_TYPE;

#define CKLOOP_USECHARM 1
#define CKLOOP_PTHREAD 2

class CProxy_FuncCkLoop;
/*
 * The default mode is intended to be used in SMP mode
 * The next mode that uses pthread is intended to be used in a restricted mode where
 * a node just have one charm PE!
 **/
extern CProxy_FuncCkLoop CkLoop_Init(int mode=CKLOOP_USECHARM, int numThreads=0);

extern void CkLoop_Exit(CProxy_FuncCkLoop ckLoop); /* used to free resources if using pthread mode. It should be called on just one PE, say PE 0 */

extern void CkLoop_Parallelize(
    CProxy_FuncCkLoop ckLoop, /* the proxy to the FuncCkLoop instance */
    HelperFn func, /* the function that finishes a partial work on another thread */
    int paramNum, void * param, /* the input parameters for the above func */
    int numChunks, /* number of chunks to be partitioned */
    int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
    int sync=1, /* whether the flow will continue unless all chunks have finished */
    void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
);
extern void CkLoop_Parallelize(
    HelperFn func, /* the function that finishes a partial work on another thread */
    int paramNum, void * param, /* the input parameters for the above func */
    int numChunks, /* number of chunks to be partitioned */
    int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
    int sync=1, /* whether the flow will continue unless all chunks have finished */
    void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
);
#endif
