#ifndef _NODEHELPERAPI_H
#define _NODEHELPERAPI_H

#include "NodeHelper.decl.h"

/* "result" is the buffer for reduction result on a single simple-type variable */
typedef void (*HelperFn)(int first,int last, void *result, int paramNum, void *param);

typedef enum REDUCTION_TYPE{
    NODEHELPER_NONE=0,
    NODEHELPER_INT_SUM,
    NODEHELPER_FLOAT_SUM,
    NODEHELPER_DOUBLE_SUM
}REDUCTION_TYPE;

class CProxy_FuncNodeHelper;
/* currently only thinking of SMP mode */
extern CProxy_FuncNodeHelper NodeHelper_Init();
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
