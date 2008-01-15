#ifndef __SPE_RUNTIME_H__
#define __SPE_RUNTIME_H__


#include "spert_common.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// Work Request Flags
#define WORK_REQUEST_FLAGS_NONE            (0x00)  // No flags
#define WORK_REQUEST_FLAGS_RW_IS_RO        (0x01)  // (Standard Only) Treat the buffer that readWritePtr points to as a readOnly buffer
#define WORK_REQUEST_FLAGS_RW_IS_WO        (0x02)  // (Standard Only) Tread the buffer that readWritePtr points to as a writeOnly buffer
#define WORK_REQUEST_FLAGS_LIST            (0x04)  // (List Only) The work request uses a dma list instead of a single set of buffers
#define WORK_REQUEST_FLAGS_BOTH_CALLBACKS  (0x08)  // (Groups Only) If the work request is part of a group, setting this flag indicates that the individual work requests callback should also be called in addition to the group callback when the entire group is finished.  By default, only the group callback will be called.

// Right shift amounts to bring flag checks into lsb of register value
#define WORK_REQUEST_FLAGS_RW_IS_RO_SHIFT (0)
#define WORK_REQUEST_FLAGS_RW_IS_WO_SHIFT (1)
#define WORK_REQUEST_FLAGS_LIST_SHIFT     (2)

// NOTE : This should be "unsigned long" for 32-bit and "unsigned long long" for 64-bit
#define PPU_POINTER_TYPE   unsigned long

// Defines that describe the message queue between the PPU and SPE
#define SPE_MESSAGE_QUEUE_LENGTH      8   // DO NOT SET ABOVE 31 (because of the way tags are used with the DMA engines)
#define SPE_MESSAGE_QUEUE_BYTE_COUNT  (SIZEOF_128(SPEMessage) * SPE_MESSAGE_QUEUE_LENGTH)
#define DOUBLE_BUFFER_MESSAGE_QUEUE   1   // Set to non-zero to make the SPE Runtime double buffer the message queue
#define SPE_NOTIFY_VIA_MAILBOX        0
#define SPE_NOTIFY_QUEUE_BYTE_COUNT   (ROUNDUP_128(sizeof(SPENotify) * SPE_MESSAGE_QUEUE_LENGTH))

// The number of dma list entries in a pre-allocated dma list.
#define SPE_DMA_LIST_LENGTH                16       // Per message in message queue (NOTE: Must be an even # >= 4: 4, 10, 22, etc.)
#define SPE_DMA_LIST_ENTRY_MAX_LENGTH      0x4000   // Maximum length of a buffer pointed to by a single dma list entry (should be a power of 2)

// Scheduler controls
#define SPE_USE_STATE_LOOKUP_TABLE  1
#define LIMIT_READY  5

// Memory Settings
#define SPE_TOTAL_MEMORY_SIZE   (256 * 1024)  // Defined by the architecture
#define SPE_USE_OWN_MEMSET               (0)  // Set to 1 to force a local version of memset to be used (to try to remove C/C++ runtime dependence)
#define SPE_USE_OWN_MALLOC               (1)  // Set to 1 to force a local version of malloc and free to be used
#define SPE_MEMORY_BLOCK_SIZE     (1024 * 4)  // !!! IMPORTANT !!! : NOTE : SPE_MEMORY_BLOCK_SIZE should be a power of 2.
#define SPE_RESERVED_STACK_SIZE  (1024 * 48)  // Reserve this much memory for the stack
#define SPE_MINIMUM_HEAP_SIZE    (1024 * 16)  // Require at least this amount of heap (or the SPE Runtime will exit)
#define SPE_ZERO_WRITE_ONLY_MEMORY       (0)  // Set to non-zero if the write-only buffer should be zero-ed out on the SPE before being filled in

// The maximum number of work requests that can be serviced in a single SPE scheduler loop iteration
#define SPE_MAX_GET_PER_LOOP       10
#define SPE_MAX_EXECUTE_PER_LOOP   2
#define SPE_MAX_PUT_PER_LOOP       10

// Defines for SPEMessage::state
#define SPE_MESSAGE_STATE_MIN                 0
#define SPE_MESSAGE_STATE_CLEAR               0
#define SPE_MESSAGE_STATE_SENT                1
#define SPE_MESSAGE_STATE_PRE_FETCHING        2
#define SPE_MESSAGE_STATE_PRE_FETCHING_LIST   3  // NOTE: code in processMsgState_send requires 'PRE_FETCHING_LIST = PRE_FETCHING + 1'
#define SPE_MESSAGE_STATE_FETCHING            4
#define SPE_MESSAGE_STATE_LIST_READY_LIST     5
#define SPE_MESSAGE_STATE_FETCHING_LIST       6
#define SPE_MESSAGE_STATE_READY               7
#define SPE_MESSAGE_STATE_EXECUTED            8
#define SPE_MESSAGE_STATE_EXECUTED_LIST       9  // NOTE: code in processMsgState_ready requires 'EXECUTED_LIST = EXECUTED + 1'
#define SPE_MESSAGE_STATE_COMMITTING          10
#define SPE_MESSAGE_STATE_FINISHED            11
#define SPE_MESSAGE_STATE_ERROR               12
#define SPE_MESSAGE_STATE_MAX                 12
#define SPE_MESSAGE_NUM_STATES                (SPE_MESSAGE_STATE_MAX - SPE_MESSAGE_STATE_MIN + 1)

// SPE Function Indexes
#define SPE_FUNC_INDEX_INIT       (-2)
#define SPE_FUNC_INDEX_CLOSE      (-1)
#define SPE_FUNC_INDEX_USER       (0)

// SPE Commands
#define SPE_MESSAGE_COMMAND_MIN          0
#define SPE_MESSAGE_COMMAND_NONE         0
#define SPE_MESSAGE_COMMAND_EXIT         1
#define SPE_MESSAGE_COMMAND_RESET_CLOCK  2
#define SPE_MESSAGE_COMMAND_MAX          2

// SPE Error Codes
#define SPE_MESSAGE_OK                       (0x0000)
#define SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY  (0x0001)

// Tracing
#define ENABLE_TRACE        0  // Set to non-zero to enable trance statements for work requests that have tracing enabled

// DEBUG Display Level
#define SPE_DEBUG_DISPLAY   0  // Set to 0 to save on LS memory usage (all printf's should be wrapped in this!)
#define SPE_DEBUG_DISPLAY_STILL_ALIVE  0 // If > 0 then display a "still alive" message every SPE_DEBUG_DISPLAY_STILL_ALIVE iterations
#define SPE_DEBUG_DISPLAY_NO_PROGRESS  0 // If non-zero, warn when no messages changes state for this many iterations
#define SPE_REPORT_END      1  // Have each SPE report the address of it's _end variable (end of data segment; will be printed by PPE during spe thread creation)
#define SPE_NOTIFY_ON_MALLOC_FAILURE   0  // Set to 1 to force the SPE to notify the user when a pointer returned by malloc/new returns an un-usable pointer (message will retry malloc/new later)

#define OFFLOAD_API_FULL_CHECK  1

// STATS Data Collection
#define PPE_STATS    0  // Set to have stat data collected during execution for the PPE side of the Offload API

// NOTE : Only a single SPE_TIMING/STATS should be enabled at a time
//   !!! (e.g. - if SPE_STATS enabled, then SPE_STATS1, SPE_STATS2, and SPE_TIMING should be disabled) !!!
#define SPE_TIMING   1  // Set to have timing data on the WRs sent back to the PPE
#define SPE_STATS    0  // Set to have stat data collected during execution for the SPE side of the Offload API (SPE Runtime)
#define SPE_STATS1   0
#define SPE_STATS2   0  // 0: unset; >0: message queue index to track; <0: track all message queue entries

// The lower and upper bounds of tags that are available to the user's code (incase the user's code needs to
//   do DMA transactions directly and needs to use tags in doing so).
#define SPE_USER_TAG_MIN   SPE_MESSAGE_QUEUE_LENGTH  // NOTE: 0 through SPE_MESSAGE_QUEUE_LENGTH are used for work request DMA transactions
#define SPE_USER_TAG_MAX   29  // NOTE: 31 and 30 are reserved for message and notify queues
#define SPE_NUM_USER_TAGS  (SPE_USER_TAG_MAX - SPE_USER_TAG_MIN + 1)


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures

/** \addtogroup OffloadAPI
 *  @{
 */

/** DMAListEntry : A structure that contains a single entry of a DMA list. */
//   NOTE: The actual layout of this data structure should include the notify and reserved fields
//   (as shown below).  However, to make this a bit easier on the user (calling sendWorkRequest_list)
//   size will be treated as a 32-bit number and the upper 16 bits will be zero-ed out.  This will
//   also prevent the user from setting the notify flag.
typedef struct __dma_list_entry {
  //unsigned int notify   : 1;   // Notify when finished
  //unsigned int reserved : 15;  // Reserved (set to all zeros)
  unsigned int size;             ///< ( : 16 ) The size of the DMA transfer (actually three values (from MSB): notify:1, reserved:15, size:16)
  unsigned int ea;               ///< ( : 32 ) Effective address of the data (lower 32 bits only)
} DMAListEntry;

/* @} */


// SPE Message: The structure that defines a message being passed to an SPE
typedef struct __SPE_MESSAGE {

  volatile int counter0;
  volatile int state;              // Current state of the message (see SPE_MESSAGE_STATE_xxx)
  volatile unsigned int flags;
  volatile int funcIndex;          // Indicates what "function" the SPE should perform

  volatile PPU_POINTER_TYPE readWritePtr;
  volatile PPU_POINTER_TYPE readOnlyPtr;
  volatile PPU_POINTER_TYPE writeOnlyPtr;
  volatile int readWriteLen;

  volatile int readOnlyLen;
  volatile int writeOnlyLen;
  volatile unsigned int totalMem;  // The total amount of memory that will be needed on the SPE for the request
  volatile int traceFlag;          // DEBUG

  // NOTE : !!! VERY IMPORTANT !!! : The dmaList address must be 16 byte aligned in the SPE's LS.  The SPEMessage
  //   data structures get 16 byte aligned so the fields in this data structure must be order in such a way
  //   that dmaList starts a multiple of 16 bytes away from the start of the overall structure.
  volatile DMAListEntry dmaList[SPE_DMA_LIST_LENGTH];

  volatile int command;             // A control command that the PPU can use to send commands to the SPE runtime (see SPE_MESSAGE_COMMAND_xxx)
  volatile PPU_POINTER_TYPE wrPtr;  // A pointer to userData specified in the sendWorkRequest call that will be passed to the callback function
  volatile int counter1;            // A counter used to uniquely identify this message from the message previously held in this slot
  volatile int checksum;            // A checksum of the contents of the data structure (NOTE: Code assumes that checksum is the last field and is an int)

} SPEMessage;


// SPE Notify: The structure that defines a notification beind passed from the SPE to the PPE notifying the
//   the PPE that a given work request has completed.
// NOTE : Size of this structure should be a multiple of 16 bytes
typedef struct __SPE_NOTIFY {

  //volatile unsigned long long int startTime;   // The time the Work Request entered user code
  //volatile unsigned int runTime;               // The amount of time the Work Request spent in user code
  //volatile unsigned short errorCode;           // The error code for the Work Request
  //volatile unsigned short counter;             // The counter value (when completed, should match corresponding counter in Message Queue)

  volatile unsigned long long int recvTimeStart; // The time the SPE Runtime first "noticed" the Work Request entry
  volatile unsigned int recvTimeEnd;
  volatile unsigned int __padding0__[1];

  volatile unsigned int preFetchingTimeStart;
  volatile unsigned int preFetchingTimeEnd;
  volatile unsigned int fetchingTimeStart;
  volatile unsigned int fetchingTimeEnd;

  volatile unsigned int readyTimeStart;
  volatile unsigned int readyTimeEnd;
  volatile unsigned int userTimeStart;
  volatile unsigned int userTimeEnd;

  volatile unsigned int executedTimeStart;
  volatile unsigned int executedTimeEnd;
  //volatile unsigned int __padding1__[2];
  volatile unsigned int userTime0Start;
  volatile unsigned int userTime0End;

  // NOTE : Important to keep the commit timing fields, errorCode, and counter fields together in the same
  //   cache line (they are all written at the same time and this will ensure that the other cache lines in
  //   the same structure are in the LS ... i.e. loads cannot go out of order).
  volatile unsigned int commitTimeStart;
  volatile unsigned int commitTimeEnd;
  volatile unsigned short errorCode;           // The error code for the Work Request
  volatile unsigned short counter;             // The counter value (when completed, should match corresponding counter in Message Queue)
  volatile unsigned int __padding2__[1];

  volatile unsigned int userTime1Start;
  volatile unsigned int userTime1End;
  volatile unsigned int userTime2Start;
  volatile unsigned int userTime2End;

  volatile unsigned long long int userAccumTime0;
  volatile unsigned long long int userAccumTime1;

  volatile unsigned long long int userAccumTime2;
  volatile unsigned long long int userAccumTime3;

} SPENotify;


// Define a structure that will be passed to each SPE thread when it is created
typedef struct __SPE_DATA {
  volatile PPU_POINTER_TYPE messageQueue;  // Pointer to the message queue's location in main memory
  #if SPE_NOTIFY_VIA_MAILBOX == 0
    volatile PPU_POINTER_TYPE notifyQueue;
  #endif
  volatile int messageQueueLength;         // Length of the message queue (the number of messages)
  volatile unsigned short vID;             // The virtual SPE number
} SPEData;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes for SPE Functions
// TODO : NOTE : These should probably be moved so they are only declared for the SPE (i.e. - the PPE code
//   may need to include this file, but these functions aren't available, nor make sense, on the PPE).

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned short getSPEID();
extern int isTracing();
extern void debug_dumpSPERTState();

extern void startUserTime0();
extern void endUserTime0();
extern void startUserTime1();
extern void endUserTime1();
extern void startUserTime2();
extern void endUserTime2();

extern void clearUserAccumTime(int index);
extern void startUserAccumTime(int index);
extern void endUserAccumTime(int index);

#ifdef __cplusplus
}
#endif


#endif //__SPE_RUNTIME_H__
