#ifndef _NODEHELPERAPI_H
#define _NODEHELPERAPI_H

#include "NodeHelper.decl.h"

/* "result" is the buffer for reduction result on a single simple-type variable */
typedef void (*HelperFn)(int first,int last, void *result, int paramNum, void *param);

typedef enum REDUCTION_TYPE {
    NODEHELPER_NONE=0,
    NODEHELPER_INT_SUM,
    NODEHELPER_FLOAT_SUM,
    NODEHELPER_DOUBLE_SUM
} REDUCTION_TYPE;

#define NODEHELPER_USECHARM 1
#define NODEHELPER_PTHREAD 2

class CProxy_FuncNodeHelper;
/*
 * The default mode is intended to be used in SMP mode
 * The next mode that uses pthread is intended to be used in a restricted mode where
 * a node just have one charm PE!
 **/
extern CProxy_FuncNodeHelper NodeHelper_Init(int mode=NODEHELPER_USECHARM, int numThreads=0);

extern void NodeHelper_Exit(CProxy_FuncNodeHelper nodeHelper); /* used to free resources if using pthread mode. It should be called on just one PE, say PE 0 */

extern void NodeHelper_Parallelize(
    CProxy_FuncNodeHelper nodeHelper, /* the proxy to the FuncNodeHelper instance */
    HelperFn func, /* the function that finishes a partial work on another thread */
    int paramNum, void * param, /* the input parameters for the above func */
    int numChunks, /* number of chunks to be partitioned */
    int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
    int sync=1, /* whether the flow will continue unless all chunks have finished */
    void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
);
extern void NodeHelper_Parallelize(
    HelperFn func, /* the function that finishes a partial work on another thread */
    int paramNum, void * param, /* the input parameters for the above func */
    int numChunks, /* number of chunks to be partitioned */
    int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */
    int sync=1, /* whether the flow will continue unless all chunks have finished */
    void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
);
#endif
