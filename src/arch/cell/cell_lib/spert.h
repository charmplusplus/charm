#ifndef __SPE_RUNTIME_H__
#define __SPE_RUNTIME_H__


#include "spert_common.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// Work Request Flags
#define WORK_REQUEST_FLAGS_NONE      (0x00)  // No flags
#define WORK_REQUEST_FLAGS_RW_IS_RO  (0x01)  // (Standard Only) Treat the buffer that readWritePtr points to as a readOnly buffer
#define WORK_REQUEST_FLAGS_RW_IS_WO  (0x02)  // (Standard Only) Tread the buffer that readWritePtr points to as a writeOnly buffer
#define WORK_REQUEST_FLAGS_LIST      (0x04)  // (List Only) The work request uses a dma list instead of a single set of buffers

// NOTE : This should be "unsigned long" for 32-bit and "unsigned long long" for 64-bit
#define PPU_POINTER_TYPE   unsigned long

// Defines that describe the message queue between the PPU and SPE
#define SPE_MESSAGE_QUEUE_LENGTH      10   // DO NOT SET ABOVE 31 (because of the way tags are used with the DMA engines)
#define SPE_MESSAGE_QUEUE_BYTE_COUNT  (SIZEOF_16(SPEMessage) * SPE_MESSAGE_QUEUE_LENGTH)

// Set to non-zero if the write-only buffer should be zero-ed out on the SPE before being filled in
#define SPE_ZERO_WRITE_ONLY_MEMORY    0

// The number of dma list entries in a pre-allocated dma list.
#define SPE_DMA_LIST_LENGTH                8        // Per message in message queue (NOTE: Must be even: 2, 4, 10, etc.)
#define SPE_DMA_LIST_ENTRY_MAX_LENGTH      0x4000   // Maximum length of a buffer pointed to by a single dma list entry (should be a power of 2)

// The reserved area for stack
#define SPE_TOTAL_MEMORY_SIZE    (256 * 1024)  // Defined by the architecture
#define SPE_RESERVED_STACK_SIZE  (1024 * 40)   // Reserve this much memory for the stack
#define SPE_MINIMUM_HEAP_SIZE    (1024 * 16)   // Require at least this amount of heap (or the SPE Runtime will exit)

// The maximum number of work requests that can be serviced in a single SPE scheduler loop iteration
#define SPE_MAX_GET_PER_LOOP       10
#define SPE_MAX_EXECUTE_PER_LOOP   2
#define SPE_MAX_PUT_PER_LOOP       10

// Defines for SPEMessage::state
#define SPE_MESSAGE_STATE_MIN                 0
#define SPE_MESSAGE_STATE_CLEAR               0
#define SPE_MESSAGE_STATE_SENT                1
#define SPE_MESSAGE_STATE_PRE_FETCHING        2
#define SPE_MESSAGE_STATE_FETCHING            3
#define SPE_MESSAGE_STATE_PRE_FETCHING_LIST   4
#define SPE_MESSAGE_STATE_LIST_READY_LIST     5
#define SPE_MESSAGE_STATE_FETCHING_LIST       6
#define SPE_MESSAGE_STATE_READY               7
#define SPE_MESSAGE_STATE_EXECUTED            8
#define SPE_MESSAGE_STATE_EXECUTED_LIST       9
#define SPE_MESSAGE_STATE_COMMITTING          10
#define SPE_MESSAGE_STATE_FINISHED            11
#define SPE_MESSAGE_STATE_ERROR               12
#define SPE_MESSAGE_STATE_MAX                 12

// SPE Function Indexes
#define SPE_FUNC_INDEX_INIT       (-2)
#define SPE_FUNC_INDEX_CLOSE      (-1)
#define SPE_FUNC_INDEX_USER       (0)

// SPE Commands
#define SPE_MESSAGE_COMMAND_MIN   0
#define SPE_MESSAGE_COMMAND_NONE  0
#define SPE_MESSAGE_COMMAND_EXIT  1
#define SPE_MESSAGE_COMMAND_MAX   1

// SPE Error Codes
#define SPE_MESSAGE_OK                       (0x0000)
#define SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY  (0x0001)

// DEBUG Display Level
#define SPE_DEBUG_DISPLAY   0  // Set to 0 to save on LS memory usage (all printf's should be wrapped in this!)
#define SPE_DEBUG_DISPLAY_STILL_ALIVE  0 // If > 0 then display a "still alive" message every SPE_DEBUG_DISPLAY_STILL_ALIVE iterations
#define SPE_DEBUG_DISPLAY_NO_PROGRESS  0 // If non-zero, warn when no messages changes state for this many iterations
#define SPE_REPORT_END      1  // Have each SPE report the address of it's _end variable (end of data segment; will be printed by PPE during spe thread creation)
#define SPE_USE_OWN_MEMSET  0  // Set to 1 to force a local version of memset to be used (to try to remove C/C++ runtime dependence)
#define SPE_NOTIFY_ON_MALLOC_FAILURE   0  // Set to 1 to force the SPE to notify the user when a pointer returned by malloc/new returns an un-usable pointer (message will retry malloc/new later)

// STATS Data Collection
#define SPE_STATS    0  // Set to have stat data collected during execution

// The lower and upper bounds of tags that are available to the user's code (incase the user's code needs to
//   do DMA transactions directly and needs to use tags in doing so).
#define SPE_USER_TAG_MIN   SPE_MESSAGE_QUEUE_LENGTH  // NOTE: 0 through SPE_MESSAGE_QUEUE_LENGTH are used for work request DMA transactions
#define SPE_USER_TAG_MAX   30  // NOTE: 31 is reserved for message queue dma transactions (restricted by hardware)
#define SPE_NUM_USER_TAGS  (SPE_USER_TAG_MAX - SPE_USER_TAG_MIN + 1)


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures

// SPE Message: The structure that defines a message being passed to an SPE
typedef struct __SPE_MESSAGE {
  volatile int funcIndex;          // Indicates what "function" the SPE should perform
  volatile PPU_POINTER_TYPE readWritePtr;
  volatile int readWriteLen;
  volatile PPU_POINTER_TYPE readOnlyPtr;
  volatile int readOnlyLen;
  volatile PPU_POINTER_TYPE writeOnlyPtr;
  volatile int writeOnlyLen;
  volatile unsigned int flags;
  volatile unsigned int totalMem;  // The total amount of memory that will be needed on the SPE for the request
  volatile int state;              // Current state of the message (see SPE_MESSAGE_STATE_xxx)
  volatile int counter;            // A counter used to uniquely identify this message from the message previously held in this slot
  volatile int command;            // A control command that the PPU can use to send commands to the SPE runtime (see SPE_MESSAGE_COMMAND_xxx)
  volatile PPU_POINTER_TYPE wrPtr;  // A pointer to userData specified in the sendWorkRequest call that will be passed to the callback function
} SPEMessage;


// Define a structure that will be passed to each SPE thread when it is created
typedef struct __SPE_DATA {
  volatile PPU_POINTER_TYPE messageQueue;  // Pointer to the message queue's location in main memory
  volatile int messageQueueLength;         // Length of the message queue (the number of messages)
} SPEData;


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

#endif //__SPE_RUNTIME_H__
