/// POSE confgiuration parameters
/** This code provides all the switches for control over adaptivity,
    communication, statistics gathering, load balancing, etc. */
#ifndef POSE_CONFIG_H
#define POSE_CONFIG_H
#include <limits.h>

/// Uncomment to run POSE is sequential mode
//#define SEQUENTIAL_POSE 1

/// Uncomment to gather and print POSE statistics set
#define POSE_STATS_ON 1
/// Insane level of analysis
//#define POSE_DOP_ON 1
/// Projections analysis
//#define TRACE_DETAIL 1

/// Set this to use 64 bit timestamps
//#define USE_LONG_TIMESTAMPS 1

/// Uncomment to force determinism in event ordering
#define DETERMINISTIC_EVENTS 1

/// Uncomment this to turn on coarse memory management
//#define MEM_COARSE
#define MAX_USAGE 10   // maximum uncommits per object for coarse mem. man.

/// Uncomment to save time on memory allocation and freeing
#define MSG_RECYCLING 1

/// Uncomment to make use of the Streaming Communication Library optimizations
//#define POSE_COMM_ON 1

/// Uncomment to turn on POSE load balancer
//#define LB_ON 1

#include <StreamingStrategy.h>
#include <PrioStreaming.h>
#define COMM_TIMEOUT 2
#define COMM_MAXMSG 20

/// Synchronization strategy constants
#define MAX_ITERATIONS 100  // maximum forward executions per Step call
#define STORE_RATE 10       // default checkpoint rate: 1 for every n events
#define SPEC_WINDOW 10      // speculative event window size
#define MIN_LEASH 10        // min speculative window for adaptive strategy
#define MAX_LEASH 100        // max  "     "     "     "        "     "
#define LEASH_FLEX 10        // leash increment

/// Load balancer constants
#define LB_SKIP 51          // LB done 1/LB_SKIP times GVT iterations
#define LB_THRESHOLD 4000   // 20 heavy objects
#define LB_DIFF 20000       // min diff between min and max load PEs

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
