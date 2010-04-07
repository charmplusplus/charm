/// POSE configuration parameters
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
#define USE_LONG_TIMESTAMPS 1

/// Uncomment to force determinism in event ordering
#define DETERMINISTIC_EVENTS 1

/// Uncomment this to turn on coarse memory management
#define MEM_COARSE
#define MAX_USAGE 5   // maximum checkpoints per object for coarse mem. man.

/// Uncomment to save time on memory allocation and freeing
#define MSG_RECYCLING 1

/// Uncomment to use temporally-blocked memory management
//#define MEM_TEMPORAL

/// Uncomment to make use of the Streaming Communication Library optimizations
//#define POSE_COMM_ON 1

/// Uncomment to turn on POSE load balancer
//#define LB_ON 1

#ifdef POSE_COMM_ON
#include <StreamingStrategy.h>
#include <MeshStreamingStrategy.h>
#include <PrioStreaming.h>
#endif

#define COMM_TIMEOUT 1
#define COMM_MAXMSG 20

/// Synchronization strategy constants
#define MAX_ITERATIONS 100  // maximum forward executions per Step call
#define STORE_RATE 5       // default checkpoint rate: 1 for every n events
#define SPEC_WINDOW 10      // speculative event window size
#define MIN_LEASH 10        // min speculative window for adaptive strategy
#define MAX_LEASH 100        // max  "     "     "     "        "     "
#define LEASH_FLEX 10        // leash increment

/// Load balancer constants
#define LB_SKIP 51          // LB done 1/LB_SKIP times GVT iterations
#define LB_THRESHOLD 4000   // 20 heavy objects
#define LB_DIFF 2000       // min diff between min and max load PEs

/// Checkpointing constants
#define POSE_CHECKPOINT_DIRECTORY "__pose_chkpt_files" // directory where checkpoint files are stored

// MISC
#define MAX_POOL_SIZE 40    // maximum size of a memory pool
#define MAX_RECYCLABLE 1000 // maximum size of a recyclable block
#define SEND 0
#define RECV 1
#define OPTIMISTIC 0
#define CONSERVATIVE 1

#if USE_LONG_TIMESTAMPS 
typedef CmiInt8 POSE_TimeType;
//we'd like to set UnsetTS to a very large negative value with some
//wiggle room for underflow.  But there are many maddeningly hard to
//find things which quietly break if its not -1.

#ifdef LLONG_MAX
const POSE_TimeType POSE_TimeMax=LLONG_MAX;
//const POSE_TimeType POSE_UnsetTS=LLONG_MIN+10LL;
const POSE_TimeType POSE_UnsetTS=-1LL;
#else
const POSE_TimeType POSE_TimeMax=9223372036854775807LL;
//const POSE_TimeType POSE_UnsetTS=(-POSE_TimeMax-1LL)+10LL;
const POSE_TimeType POSE_UnsetTS=-1LL;
#endif
#else
typedef int POSE_TimeType;
const POSE_TimeType POSE_TimeMax=INT_MAX;
const POSE_TimeType POSE_UnsetTS=-1;
#endif



// POSE Command line struct
//glorified struct
enum POSE_Commlib_strat {nostrat, stream, mesh, prio};

class POSE_Config 
{
 public:
  bool stats;
  int start_proj;
  int end_proj;
  bool trace;
  bool dop;
  int max_usage;
  bool msg_pool;
  int msg_pool_size;
  int max_pool_msg_size;
  POSE_Commlib_strat commlib_strat;
  int commlib_timeout;
  int commlib_maxmsg;
  bool lb_on;
  int lb_skip;
  int lb_threshold;
  int lb_diff;
  int store_rate;
  int max_iter;
  int spec_window;
  int min_leash;
  int max_leash;
  int leash_flex;
  bool deterministic;
  int checkpoint_gvt_interval;
  int checkpoint_time_interval;
  int lb_gvt_interval;
  /* one very long initializer line */
  POSE_Config() :
#ifdef POSE_STATS_ON                   //w
    stats(true),
#else
    stats(false), 
#endif
    start_proj(-1),
    end_proj(-1),
#ifdef TRACE_DETAIL                    //w
    trace(true),
#else      
    trace(false),
#endif
#ifdef POSE_DOP_ON
    dop(true),
#else
    dop(false),
#endif
    max_usage(MAX_USAGE),
/** MSG POOLING AND COMMLIB OPTIONS NOT SUPPORTED YET **/
#ifdef MSG_RECYCLING
    msg_pool(true),
#else
    msg_pool(false),
#endif
    msg_pool_size(40),
    max_pool_msg_size(1000),
#ifdef POSE_COMM_ON
    commlib_strat(nostrat),
#else
    commlib_strat(stream),
#endif
    commlib_timeout(COMM_TIMEOUT),
    commlib_maxmsg(COMM_MAXMSG),
#ifdef LB_ON
    lb_on(true),                  //w
#else
    lb_on(false),                 //w
#endif
    lb_skip(LB_SKIP),             //w
    lb_threshold(LB_THRESHOLD),   //w
    lb_diff(LB_DIFF),             //w
    store_rate(STORE_RATE),       //w
    max_iter(MAX_ITERATIONS),     //apparently defunct 
    spec_window(SPEC_WINDOW),     //w
    min_leash(MIN_LEASH),         //w
    max_leash(MAX_LEASH),         //w
    leash_flex(LEASH_FLEX),       //w
#ifdef DETERMINISTIC_EVENTS        //w
    deterministic(true),
#else
    deterministic(false),
#endif
    checkpoint_gvt_interval(0),
    checkpoint_time_interval(0),
    lb_gvt_interval(0)
    {// all handled in initializer
    }
};
PUPbytes(POSE_Config)

#endif
