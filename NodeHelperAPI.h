#ifndef _NODEHELPERAPI_H
#define _NODEHELPERAPI_H

#include "NodeHelper.decl.h"

/* "result" is the buffer for reduction result on a single simple-type variable */
typedef void (*HelperFn)(int first,int last, void *result, int paramNum, void *param);

#define NODEHELPER_PTHREAD 0
#define NODEHELPER_DYNAMIC  1
#define NODEHELPER_STATIC 2

typedef enum REDUCTION_TYPE{
    NODEHELPER_NONE=0,
    NODEHELPER_INT_SUM,
    NODEHELPER_FLOAT_SUM,
    NODEHELPER_DOUBLE_SUM
}REDUCTION_TYPE;

class CProxy_FuncNodeHelper;
extern CProxy_FuncNodeHelper NodeHelper_Init(int mode, /* indicates the nodehelper running mode, pthread of non-SMP, dynamic/static of SMP */
                                            int numThds /* only valid in non-SMP mode, indicating how many pthreads are going to be created*/);
extern void NodeHelper_Parallelize(
						CProxy_FuncNodeHelper nodeHelper, /* the proxy to the FuncNodeHelper instance */
						HelperFn func, /* the function that finishes a partial work on another thread */
                        int paramNum, void * param, /* the input parameters for the above func */
                        int msgPriority, /* the priority of the intra-node msg, and node-level msg */
                        int numChunks, /* number of chunks to be partitioned */
                        int lowerRange, int upperRange, /* the loop-like parallelization happens in [lowerRange, upperRange] */                        
                        void *redResult=NULL, REDUCTION_TYPE type=NODEHELPER_NONE /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
                        );
#endif
