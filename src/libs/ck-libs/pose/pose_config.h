
/// POSE confgiuration parameters
/** This code provides all the switches for control over adaptivity,
    communication, statistics gathering, load balancing, etc. */
#ifndef POSE_CONFIG_H
#define POSE_CONFIG_H
#include <limits.h>
/// Uncomment to gather and print POSE statistics set
#define POSE_STATS_ON 1
/// Insane level of analysis
//#define POSE_DOP_ON 1
// Set to 1 to use 64 bit timestamps
#define USE_LONG_TIMESTAMPS 0
// Uncomment to save time on memory allocation and freeing
#define MSG_RECYCLING 1
/// Uncomment to make use of the Streaming Communication Library optimizations
//#define POSE_COMM_ON 1
/// Uncomment to turn on POSE load balancer
//#define LB_ON 1
#ifdef POSE_COMM_ON
#include <StreamingStrategy.h>
#include <DummyStrategy.h> 
#define COMM_TIMEOUT 5
#define COMM_MAXMSG 5
#endif 

/// Synchronization strategy constants
#define MAX_ITERATIONS 100  // maximum forward executions per Step call
#define STORE_RATE 20      // default checkpoint rate: 1 for every n events
#define SPEC_WINDOW 100      // speculative event window size
#define MIN_LEASH 10        // min speculative window for adaptive strategy
#define MAX_LEASH 50        // max  "     "     "     "        "     "
#define LEASH_FLEX 1        // leash increment

/// Load balancer constants
#define LB_SKIP 51          // LB done 1/LB_SKIP times GVT iterations
#define LB_THRESHOLD 2000   // 20 heavy objects
#define LB_DIFF 10000       // min diff between min and max load PEs

// MISC
#define MAX_POOL_SIZE 40    // maximum size of a memory pool
#define MAX_RECYCLABLE 1000 // maximum size of a recyclable block
#define SEND 0
#define RECV 1
#define OPTIMISTIC 0
#define CONSERVATIVE 1

#if USE_LONG_TIMESTAMPS 
typedef CmiInt8 POSE_TimeType;
const POSE_TimeType POSE_UnsetTS=-1LL;
const POSE_TimeType POSE_TimeMax=9223372036854775807LL;
#else
typedef int POSE_TimeType;
const POSE_TimeType POSE_UnsetTS=-1;
const POSE_TimeType POSE_TimeMax=INT_MAX;
#endif



#endif
