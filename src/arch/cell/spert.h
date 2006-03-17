#ifndef __SPE_RUNTIME_H__
#define __SPE_RUNTIME_H__


#include "general.h"
#include "common.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// NOTE : This should be "unsigned long" for 32-bit and "unsigned long long" for 64-bit
#define PPU_POINTER_TYPE   unsigned long

// Defines that describe the message queue between the PPU and SPE
#define SPE_MESSAGE_QUEUE_LENGTH      10   // DO NOT SET ABOVE 31 (because of the way tags are used with the DMA engines)
#define SPE_MESSAGE_QUEUE_BYTE_COUNT  (SIZEOF_16(SPEMessage) * SPE_MESSAGE_QUEUE_LENGTH)

// Defines for SPEMessage::state
#define SPE_MESSAGE_STATE_MIN            0
#define SPE_MESSAGE_STATE_CLEAR          0
#define SPE_MESSAGE_STATE_SENT           1
#define SPE_MESSAGE_STATE_PRE_FETCHING   2
#define SPE_MESSAGE_STATE_FETCHING       3
//#define SPE_MESSAGE_STATE_FETCHING_DATA  2
//#define SPE_MESSAGE_STATE_FETCHING_MSG   3
#define SPE_MESSAGE_STATE_READY          4
#define SPE_MESSAGE_STATE_EXECUTED       5
#define SPE_MESSAGE_STATE_COMMITTING     6
#define SPE_MESSAGE_STATE_FINISHED       7
#define SPE_MESSAGE_STATE_MAX            7

// SPE Commands
#define SPE_MESSAGE_COMMAND_MIN   0
#define SPE_MESSAGE_COMMAND_NONE  0
#define SPE_MESSAGE_COMMAND_EXIT  1
#define SPE_MESSAGE_COMMAND_MAX   1


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures


// SPE Message: The structure that defines a message (work request) being passed to an SPE
typedef struct __SPE_MESSAGE {
  volatile int funcIndex;          // Indicates what "function" the SPE should perform
  volatile PPU_POINTER_TYPE data;  // Pointer to caller defined data location
  volatile int dataLen;            // Size of data location pointed to by data (in bytes)
  volatile PPU_POINTER_TYPE msg;   // Pointer to caller defined message
  volatile int msgLen;             // Size of message pointed to by msg (in bytes)
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


#endif //__SPE_RUNTIME_H__
