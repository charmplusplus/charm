#ifdef __cplusplus
extern "C" {
#endif
  #include <stdlib.h>
  #include <unistd.h>
  #include <stdio.h>
  #include <free_align.h>
  #include <sim_printf.h>
#ifdef __cplusplus
}
#endif

#define SPE_USE_OWN_MALLOC  0

#if SPE_USE_OWN_MALLOC <= 0
  #ifdef __cplusplus
  extern "C" {
  #endif
    #include <malloc_align.h>
  #ifdef __cplusplus
  }
  #endif
#else
void initMem();
  void* _malloc_align(int size, int alignment);
  void _free_align(void* addr);
#endif

#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <cbe_mfc.h>

#if SPE_USE_OWN_MEMSET == 0
  #include <string.h>
#endif

#include "spert_common.h"
#include "spert.h"


#define USE_PRINT_BLOCK  1
void print_block_table();


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// DEBUG
#define USE_SCHEDULE_LOOP  0

#define MESSAGE_RETURN_CODE(i, ec)   ((((unsigned int)ec << 16) & 0xFFFF0000) | ((unsigned int)i & 0x0000FFFF))


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Required External Function(s) that the User Needs to Provide

extern void funcLookup(int funcIndex,
                       void* readWritePtr, int readWriteLen,
                       void* readOnlyPtr, int readOnlyLen,
                       void* writeOnlyPtr, int writeOnlyLen,
                       DMAListEntry* dmaList
                      );


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures

#if SPE_STATS != 0

typedef struct __stat_data {
  unsigned long long schedulerLoopCount;
  unsigned long long numWorkRequestsExecuted;
} StatData;

StatData statData = { 0, 0 };

#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

#define DOUBLE_BUFFER_MESSAGE_QUEUE  1
#if DOUBLE_BUFFER_MESSAGE_QUEUE == 0
  volatile char msgQueueRaw[SPE_MESSAGE_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
  volatile SPEMessage* msgQueue[SPE_MESSAGE_QUEUE_LENGTH];
#else
  volatile char msgQueueRaw0[SPE_MESSAGE_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
  volatile char msgQueueRaw1[SPE_MESSAGE_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
  volatile char* msgQueueRaw;
  volatile char* msgQueueRawAlt;
  volatile SPEMessage* msgQueue0[SPE_MESSAGE_QUEUE_LENGTH];
  volatile SPEMessage* msgQueue1[SPE_MESSAGE_QUEUE_LENGTH];
  volatile SPEMessage** msgQueue;
  volatile SPEMessage** msgQueueAlt;
#endif


#if SPE_NOTIFY_VIA_MAILBOX == 0
  volatile char notifyQueueRaw[SPE_NOTIFY_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
#endif

// NOTE: Allocate two per entry (two buffers are read in from memory, two are
//   written out to memory, read write does not overlap for a given message).
//volatile DMAListEntry dmaListEntry[SPE_DMA_LIST_LENGTH * SPE_MESSAGE_QUEUE_LENGTH] __attribute__((aligned(16)));
int dmaListSize[SPE_MESSAGE_QUEUE_LENGTH];


int msgState[SPE_MESSAGE_QUEUE_LENGTH];
int msgCounter[SPE_MESSAGE_QUEUE_LENGTH];
void* localMemPtr[SPE_MESSAGE_QUEUE_LENGTH];
void* readWritePtr[SPE_MESSAGE_QUEUE_LENGTH];
void* readOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
void* writeOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
int errorCode[SPE_MESSAGE_QUEUE_LENGTH];
DMAListEntry* dmaList[SPE_MESSAGE_QUEUE_LENGTH];


// Location of the end of the 'data segment'
extern unsigned int _end;

unsigned short vID = 0xFFFF;


// TRACE
#if ENABLE_TRACE != 0
  int isTracingFlag = 0;
#endif

// DEBUG
int execIndex = -1;


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Timing

#define startTimer() {                                                         \
                       spu_writech(SPU_WrDec, 0xFFFFFFFF);                     \
                       spu_writech(SPU_WrEventMask, MFC_DECREMENTER_EVENT);    \
                     }

#define getTimer(var) {                                                         \
                        register unsigned int cntr = spu_readch(SPU_RdDec);     \
                        var = ((unsigned int)(0xFFFFFFFF)) - cntr;              \
                      }

#define stopTimer(var) {                                                         \
                         register unsigned int cntr = spu_readch(SPU_RdDec);     \
                         spu_writech(SPU_WrEventMask, 0);                        \
                         spu_writech(SPU_WrEventAck, MFC_DECREMENTER_EVENT);     \
                         var = ((unsigned int)(0xFFFFFFFF)) - cntr;              \
                       }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tracing

// TRACE
#if ENABLE_TRACE != 0

int stateTrace[SPE_MESSAGE_QUEUE_LENGTH][SPE_MESSAGE_STATE_MAX - SPE_MESSAGE_STATE_MIN + 1];
int stateTrace_counter[SPE_MESSAGE_QUEUE_LENGTH];
char stateTraceBuf[1024];

#define STATETRACE_CLEAR(index) {                                                                                  \
                                  int iii = 0;                                                                     \
                                  for (iii = 0; iii < (SPE_MESSAGE_STATE_MAX - SPE_MESSAGE_STATE_MIN + 1); iii++)  \
                                    stateTrace[index][iii] = SPE_MESSAGE_STATE_CLEAR;                              \
                                  stateTrace_counter[index] = 0;                                                   \
                                }

#define STATETRACE_UPDATE_ALL {                                                                     \
                                int iii = 0;                                                        \
                                for (iii = 0; iii < SPE_MESSAGE_QUEUE_LENGTH; iii++) {              \
                                  if (stateTrace[iii][stateTrace_counter[iii]] != msgState[iii]) {  \
                                    stateTrace[iii][stateTrace_counter[iii]] = msgState[iii];       \
                                    stateTrace_counter[iii]++;                                      \
                                    stateTrace[iii][stateTrace_counter[iii]] = msgState[iii];       \
                                  }                                                                 \
                                }                                                                   \
                              }

#define STATETRACE_UPDATE(index) {                                                                         \
                                   if (stateTrace[index][stateTrace_counter[index]] != msgState[index]) {  \
                                     stateTrace[index][stateTrace_counter[index]] = msgState[index];       \
                                     stateTrace_counter[index]++;                                          \
                                     stateTrace[index][stateTrace_counter[index]] = msgState[index];       \
                                   }                                                                       \
                                 }

#define STATETRACE_OUTPUT(index) {                                                                                      \
                                   int iii = 0, execed = 0; stateTraceBuf[0] = '\0';                                    \
                                   for (iii = 0; iii < (SPE_MESSAGE_STATE_MAX - SPE_MESSAGE_STATE_MIN + 1); iii++) {    \
                                     if (stateTrace[index][iii] == SPE_MESSAGE_STATE_EXECUTED) execed = 1;              \
                                   }                                                                                    \
                                   if (execed == 0) {                                                                   \
                                     sprintf(stateTraceBuf, "SPE :: STATE TRACE : ");                                   \
                                     for (iii = 0; iii < (SPE_MESSAGE_STATE_MAX - SPE_MESSAGE_STATE_MIN + 1); iii++) {  \
                                       sprintf(stateTraceBuf + strlen(stateTraceBuf), "%2d ", stateTrace[index][iii]);  \
                                     }                                                                                  \
                                     sprintf(stateTraceBuf + strlen(stateTraceBuf), "\n");                              \
                                     printf(stateTraceBuf);                                                             \
                                   }                                                                                    \
                                 }
#endif



/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

void speScheduler(SPEData *speData, unsigned long long id);
void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str);
void debug_displayStateHistogram(unsigned long long id, int* msgState, char* str);

#if SPE_USE_OWN_MEMSET != 0
void memset(void* ptr, char val, int len);
#endif

unsigned short getSPEID() { return vID; }

// DEBUG
void displayStatsData();


// TRACE
int isTracing() {
  #if ENABLE_TRACE != 0
    return isTracingFlag;
  #else
    return 0;
  #endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies


// STATS1
#if SPE_STATS1 != 0

  int timingIndex = -1;
  long long int wrClocks = 0;
  long long int wrClocksCounter = 0;

  long long int wrClocksDetail[SPE_MESSAGE_NUM_STATES];
  long long int wrClocksDetailSum[SPE_MESSAGE_NUM_STATES];

#endif

// TRACE
#if SPE_STATS != 0
  long long int sentClocks_msg = 0;
  long long int sentClocks_noMsg = 0;
  long long int sentClocksCounter_msg = 0;
  long long int sentClocksCounter_noMsg = 0;
#endif

// SPE_MESSAGE_STATE_SENT
inline void processMsgState_sent(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
    register int msgFoundFlag = 0;
  #endif

  //// DEBUG
  //printf("SPE_%d :: DEBUG :: msgQueue[%d] = { counter: %d, %d, state = %d }, msgState[%d] = %d...\n",
  //       (int)getSPEID(), msgIndex,
  //       msgQueue[msgIndex]->counter0, msgQueue[msgIndex]->counter1,
  //       msgQueue[msgIndex]->state,
  //       msgState[msgIndex]
  //      );

  // Check for a new message in this slot
  // Conditions... 1) msgQueue[i]->counter1 != msgCounter[i]          // Entry has not already been processed (most likely not true, i.e. - test first to reduce cost of overall condition check)
  //               2) msgQueue[i]->counter0 == msgQueue[i]->counter1  // Entire entry has been written by PPE
  //               3) msgQueue[i]->state == SPE_MESSAGE_STATE_SENT    // PPE wrote an entry and is waiting for the result
  //               4) msgState[i] == SPE_MESSAGE_STATE_CLEAR          // SPE isn't currently processing this slot

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_counter1 = tmp_msgQueueEntry->counter1;

  if (__builtin_expect(tmp_counter1 != msgCounter[msgIndex], 0)) {

    register int tmp_counter0 = tmp_msgQueueEntry->counter0;
    register int tmp_msgState = tmp_msgQueueEntry->state;
    register int tmp_localState = msgState[msgIndex];

    if (__builtin_expect(tmp_counter0 == tmp_counter1, 1)) {
      if (__builtin_expect(tmp_msgState == SPE_MESSAGE_STATE_SENT, 1)) {
        if (__builtin_expect(tmp_localState == SPE_MESSAGE_STATE_CLEAR, 1)) {

          // STATS1
          #if SPE_STATS1 != 0
	    if (timingIndex == -1) {
              startTimer();
              timingIndex = msgIndex;
	    }
          #endif

          // TRACE
          #if ENABLE_TRACE != 0
            STATETRACE_CLEAR(msgIndex);
          #endif

          // Update the state of the message queue entry (locally)
          if ((tmp_msgQueueEntry->flags & WORK_REQUEST_FLAGS_LIST) == 0x00)
            msgState[msgIndex] = SPE_MESSAGE_STATE_PRE_FETCHING;
          else
            msgState[msgIndex] = SPE_MESSAGE_STATE_PRE_FETCHING_LIST;

          // DEBUG (moved from _committing)
          //localMemPtr[msgIndex] = NULL;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            printf("SPE_%d :: msg %d's state going from %d ->%d\n", (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_SENT, msgState[msgIndex]);
          #endif

          // Update the local counter to refect the PPE's counter's value
	  msgCounter[msgIndex] = tmp_counter1; //msgQueue[msgIndex]->counter1;

          // TRACE
          #if ENABLE_TRACE != 0
            if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
              printf("SPE_%d :: [TRACE] :: Tracing entry at index %d...\n", (int)getSPEID(), msgIndex);
              debug_displayActiveMessageQueue(0x0, msgState, "(*)");
            }
          #endif

	  // STATS
          #if SPE_STATS != 0
            msgFoundFlag = 1;
          #endif

          // STATS1
          #if SPE_STATS1 != 0
            if (timingIndex == msgIndex) {
              register unsigned int clocks;
              getTimer(clocks);
              wrClocksDetail[SPE_MESSAGE_STATE_SENT] = clocks;
            }
          #endif

        }
      }
    }
  }

  // STATS
  #if SPE_STATS
    register unsigned int clocks;
    stopTimer(clocks);
    if (msgFoundFlag != 0) {
      sentClocks_msg += clocks;
      sentClocksCounter_msg++;
    } else {
      sentClocks_noMsg += clocks;
      sentClocksCounter_noMsg++;
    }
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

}


// TRACE
#if SPE_STATS != 0
  long long int preFetchingClocks = 0;
  long long int preFetchingClocksCounter = 0;
#endif

// STATS
#if SPE_STATS != 0
  int wrTryAllocCount = 0;
  int wrAllocCount = 0;
  int wrNoAllocCount = 0;
#endif

// SPE_MESSAGE_STATE_PRE_FETCHING
inline void processMsgState_preFetching(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_PRE_FETCHING for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x00, msgState, "(*)");
    }
  #endif

  // Allocate the memory for the message queue entry (if need be)
  // NOTE: First check to see if it is non-null.  What might have happened was that there was enough memory
  //   last time but the DMA queue was full and a retry was needed.  So this time, the memory is there and
  //   only the DMA needs to be retried.
  if (localMemPtr[msgIndex] == NULL) {

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Allocating LS Memory for index %d...\n", (int)getSPEID(), msgIndex);
        printf("SPE_%d :: [TRACE] ::   msgQueue[%d] = { readOnlyPtr = 0x%08lx, readOnlyLen = %d,\n"
               "                                      readWritePtr = 0x%08lx, readWritelen = %d,\n"
               "                                      writeOnlyPtr = 0x%08lx, writeOnlyLen = %d, ...}...\n",
               (int)getSPEID(), msgIndex,
               msgQueue[msgIndex]->readOnlyPtr, msgQueue[msgIndex]->readOnlyLen,
               msgQueue[msgIndex]->readWritePtr, msgQueue[msgIndex]->readWriteLen,
               msgQueue[msgIndex]->writeOnlyPtr, msgQueue[msgIndex]->writeOnlyLen
              );
        debug_displayActiveMessageQueue(0x0, msgState, "(*)");
      }
    #endif

    // Allocate the memory and place a pointer to the allocated buffer into localMemPtr.  This buffer will
    //   be divided up into three regions: readOnly, readWrite, writeOnly (IN THAT ORDER!; the regions may be
    //   empty if no pointer for the region was supplied by the original sendWorkRequest() call on the PPU).
    register int memNeeded = ROUNDUP_16(msgQueue[msgIndex]->readWriteLen) +
                             ROUNDUP_16(msgQueue[msgIndex]->readOnlyLen) +
                             ROUNDUP_16(msgQueue[msgIndex]->writeOnlyLen);
    register int offset = 0;

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: Allocating Memory for index %d :: memNeeded = %d, msgQueue[%d].totalMem = %d\n",
             (int)getSPEID(), msgIndex, memNeeded, msgIndex, msgQueue[msgIndex]->totalMem
            );
    #endif

    // Check the size of the memory needed.  If it is too large for the SPE's LS, then stop this
    //   message with an error code because the memory allocation will never work.
    register int heapSize = ((int)0x40000) - ((int)(&_end)) - SPE_RESERVED_STACK_SIZE;
    if (__builtin_expect(memNeeded >= heapSize, 0)) {

      // Move the message into the error state
      errorCode[msgIndex] = SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY;
      msgState[msgIndex] = SPE_MESSAGE_STATE_ERROR;

      // STATS
      #if SPE_STATS != 0
        register unsigned int clocks;
        stopTimer(clocks);
        preFetchingClocks += clocks;
        preFetchingClocksCounter++;
      #endif

      return;
    }

    // Allocate the memory
    #ifdef __cplusplus
      try {
    #endif
        //localMemPtr[msgIndex] = (void*)(new char[memNeeded]);
        localMemPtr[msgIndex] = (void*)(_malloc_align(memNeeded, 4));
    #ifdef __cplusplus
      } catch (...) {
        localMemPtr[msgIndex] = NULL;
      }
    #endif

    // STATS
    #if SPE_STATS != 0
      wrTryAllocCount++;
    #endif

    // Check the pointer (if it is bad, then skip this message for now and try again later)
    if (__builtin_expect((localMemPtr[msgIndex] == NULL) || 
                         (((unsigned int)localMemPtr[msgIndex]) < ((unsigned int)(&_end))) ||
                         (((unsigned int)localMemPtr[msgIndex] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                         0
			)
       ) {
      #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	printf("SPE_%d :: ERROR :: Failed to allocate memory for localMemPtr[%d] (2)... will try again later...\n",
               (int)getSPEID(), msgIndex
              );
        printf("SPE_%d :: ERROR :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMemPtr[msgIndex]);
      #endif
      localMemPtr[msgIndex] = NULL;

      // STATS
      #if SPE_STATS != 0
        register unsigned int clocks;
        stopTimer(clocks);
        preFetchingClocks += clocks;
        preFetchingClocksCounter++;
      #endif

      // STATS
      #if SPE_STATS != 0
        wrNoAllocCount++;
      #endif

      // DEBUG
      //print_block_table();

      return;
    }

    // STATS
    #if SPE_STATS != 0
      wrAllocCount++;
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMemPtr[msgIndex]);
    #endif

    // Assign the local pointers to the various buffers within the memory just allocated
    // NOTE : Order matters here.  Need to allocate the read buffers next to each other and the write
    //   buffers next to each other (i.e. - read-only, then read-write, then write-only).

    // read-only buffer
    if (msgQueue[msgIndex]->readOnlyPtr != (PPU_POINTER_TYPE)NULL) {
      readOnlyPtr[msgIndex] = localMemPtr[msgIndex];
      offset += ROUNDUP_16(msgQueue[msgIndex]->readOnlyLen);
    } else {
      readOnlyPtr[msgIndex] = NULL;
    }

    // read-write buffer
    if (msgQueue[msgIndex]->readWritePtr != (PPU_POINTER_TYPE)NULL) {
      readWritePtr[msgIndex] = (void*)((char*)localMemPtr[msgIndex] + offset);
      offset += ROUNDUP_16(msgQueue[msgIndex]->readWriteLen);
    } else {
      readWritePtr[msgIndex] = NULL;
    }

    // write-only buffer
    if (msgQueue[msgIndex]->writeOnlyPtr != (PPU_POINTER_TYPE)NULL) {
      writeOnlyPtr[msgIndex] = (void*)((char*)localMemPtr[msgIndex] + offset);
      #if SPE_ZERO_WRITE_ONLY_MEMORY != 0
        memset(writeOnlyPtr[msgIndex], 0, ROUNDUP_16(msgQueue[msgIndex]->writeOnlyLen));
      #endif
    } else {
      writeOnlyPtr[msgIndex] = NULL;
    }

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] ::   localMemPtr[%d] = %p...\n", (int)getSPEID(), msgIndex, localMemPtr[msgIndex]);
        printf("SPE_%d :: [TRACE] ::   readOnlyPtr[%d] = %p, readWritePtr[%d] = %p, writeOnlyPtr[%d] = %p...\n",
               (int)getSPEID(),
               msgIndex, readOnlyPtr[msgIndex],
               msgIndex, readWritePtr[msgIndex],
               msgIndex, writeOnlyPtr[msgIndex]
              );
      }
    #endif

  } // end if (localMemPtr[msgIndex] == NULL)

  // NOTE : If execution reaches here, localMemPtr[msgIndex] is set to a valid value

  // Check to see if this message does not needs to fetch any data (no read-only and read-write
  //   is actually write-only).
  // NOTE : This should be done after the memory for the work request has been allocated so the
  //   output buffers are created.
  if (__builtin_expect(((msgQueue[msgIndex]->readWritePtr == (PPU_POINTER_TYPE)NULL) ||
                        ((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == WORK_REQUEST_FLAGS_RW_IS_WO)
                       ) &&
                       (msgQueue[msgIndex]->readOnlyPtr == (PPU_POINTER_TYPE)NULL),
                       0
		      )
     ) {

    // Update the state (to ready to execute)
    msgState[msgIndex] = SPE_MESSAGE_STATE_READY;

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: msg %d's state going from %d -> %d\n",
             (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_PRE_FETCHING, msgState[msgIndex]
            );
    #endif

    // STATS
    #if SPE_STATS != 0
      register unsigned int clocks;
      stopTimer(clocks);
      preFetchingClocks += clocks;
      preFetchingClocksCounter++;
    #endif

    // Done with this one
    return;
  }

  // Create the DMA list
  if (dmaListSize[msgIndex] < 0) {

    // Count the number of DMA entries needed for the read DMA list
    register int entryCount = 0;
    if (__builtin_expect((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == 0x00, 0))
      entryCount += (ROUNDUP_16(msgQueue[msgIndex]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                    (((msgQueue[msgIndex]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
    entryCount += (ROUNDUP_16(msgQueue[msgIndex]->readOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                  (((msgQueue[msgIndex]->readOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

    // Check to see if the DMA list is too large for the static DMA list
    if (__builtin_expect(entryCount > SPE_DMA_LIST_LENGTH, 0)) {

      // Allocate memory for the larger-than-normal DMA list
      #ifdef __cplusplus
        try {
      #endif
          //dmaList[msgIndex] = new DMAListEntry[entryCount];
          dmaList[msgIndex] = (DMAListEntry*)(_malloc_align(entryCount * sizeof(DMAListEntry), 4));
      #ifdef __cplusplus
	} catch (...) {
          dmaList[msgIndex] = NULL;
	}
      #endif

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, dmaList[msgIndex]);
      #endif

      // Check the returned pointer
      if (__builtin_expect((dmaList[msgIndex] == NULL) || 
                           (((unsigned int)dmaList[msgIndex]) < ((unsigned int)(&_end))) ||
                           (((unsigned int)dmaList[msgIndex] + (entryCount * sizeof(DMAListEntry))) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                           0
			  )
         ) {
        #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	  printf("SPE_%d :: SPE :: Failed to allocate memory for dmaList[%d] (2)... will try again later...\n", (int)getSPEID(), msgIndex);
          printf("SPE_%d :: SPE :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, dmaList[msgIndex]);
	#endif
        dmaList[msgIndex] = NULL;

        // STATS
        #if SPE_STATS != 0
          register unsigned int clocks;
          stopTimer(clocks);
          preFetchingClocks += clocks;
          preFetchingClocksCounter++;
        #endif

        // DEBUG
        //print_block_table();

        return;  // Skip for now... try again later
      }

      // Clear the allocated DMA list
      //memset(dmaList[msgIndex], 0, sizeof(DMAListEntry) * entryCount);

    } else {

      // Point the DMA list pointer at the message queue entry's dma list
      dmaList[msgIndex] = (DMAListEntry*)(msgQueue[msgIndex]->dmaList);

      //dmaList[msgIndex] = (DMAListEntry*)(&(dmaListEntry[msgIndex * SPE_DMA_LIST_LENGTH]));
    }
    dmaListSize[msgIndex] = entryCount;

    // Construct the DMA list
    register int listIndex = 0;

    // Create the portion of the DMA list needed by the read-only buffer
    if (readOnlyPtr[msgIndex] != NULL) {
      register int bufferLeft = msgQueue[msgIndex]->readOnlyLen;
      register unsigned int srcOffset = (unsigned int)(msgQueue[msgIndex]->readOnlyPtr);

      while (bufferLeft > 0) {
        dmaList[msgIndex][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
        dmaList[msgIndex][listIndex].size = ROUNDUP_16(dmaList[msgIndex][listIndex].size);
        dmaList[msgIndex][listIndex].ea = srcOffset;

        bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
        listIndex++;
        srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
      }
    }

    // Create the portion of the DMA list needed by the read-write buffer
    if ((
         __builtin_expect((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == 0x00, 0)
        ) &&
        (readWritePtr[msgIndex] != NULL)
       ) {

      register int bufferLeft = msgQueue[msgIndex]->readWriteLen;
      register unsigned int srcOffset = (unsigned int)(msgQueue[msgIndex]->readWritePtr);

      while (bufferLeft > 0) {
        dmaList[msgIndex][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
        dmaList[msgIndex][listIndex].size = ROUNDUP_16(dmaList[msgIndex][listIndex].size);
        dmaList[msgIndex][listIndex].ea = srcOffset;

        bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
        listIndex++;
        srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
      }
    }

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Created DMA List for index %d...\n", (int)getSPEID(), msgIndex);
        printf("SPE_%d :: [TRACE] ::   dmaList[%d] = %p, dmaListSize[%d] = %d...\n",
               (int)getSPEID(), msgIndex, dmaList[msgIndex], msgIndex, dmaListSize[msgIndex]
              );
        register int _j0;
        for (_j0 = 0; _j0 < dmaListSize[msgIndex]; _j0++) {
          printf("SPE_%d :: [TRACE] ::    DMA Entry %d = { ea = 0x%08x, size = %d }\n",
                 (int)getSPEID(), _j0, dmaList[msgIndex][_j0].ea, dmaList[msgIndex][_j0].size
                );
        }
      }
    #endif

  } // end if (dmaListSize[msgIndex] < 0)

  // NOTE : If execution reaches here, dmaListSize[msgInde] is set to a valid value

  // Get the number of free DMA queue entries
  register int numDMAQueueEntries = mfc_stat_cmd_queue();
  
  // Initiate the DMA command if there is at least one free DMA queue entry
  if (__builtin_expect(numDMAQueueEntries > 0, 1)) {  // Stay positive

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: Pre-GETL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
             (int)getSPEID(), msgIndex, localMemPtr[msgIndex], 0,
             (unsigned int)(dmaList[msgIndex]), dmaListSize[msgIndex] * sizeof(DMAListEntry), msgIndex
            );

      {
        int j;
        for (j = 0; j < dmaListSize[msgIndex]; j++) {
          printf("SPE_%d :: dmaList[%d][%d].size = %d (0x%x), dmaList[%d][%d].ea = 0x%08x (0x%08x)\n",
                 (int)getSPEID(),
                 msgIndex, j, dmaList[msgIndex][j].size, ((DMAListEntry*)(localMemPtr[msgIndex]))[j].size,
                 msgIndex, j, dmaList[msgIndex][j].ea, ((DMAListEntry*)(localMemPtr[msgIndex]))[j].ea
	        );
        }
      }
    #endif

    // Queue the DMA command
    spu_mfcdma64(localMemPtr[msgIndex],
                 0,
                 (unsigned int)(dmaList[msgIndex]),
                 dmaListSize[msgIndex] * sizeof(DMAListEntry),
                 msgIndex,
                 MFC_GETL_CMD
		);

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: DMA transaction queued for input data for index %d...\n", (int)getSPEID(), msgIndex);
      }
    #endif

    // Update the state of the message queue entry now that the data should be in-flight
    msgState[msgIndex] = SPE_MESSAGE_STATE_FETCHING;

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: msg %d's state going from %d -> %d\n",
             (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_PRE_FETCHING, msgState[msgIndex]
            );
    #endif

  } // end if (numDMAQueueEntries > 0)

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    preFetchingClocks += clocks;
    preFetchingClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_PRE_FETCHING] = clocks;
    }
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int preFetchingListClocks = 0;
  long long int preFetchingListClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_PRE_FETCHING_LIST
inline void processMsgState_preFetchingList(int msgIndex) {

  // NOTE: Optimize for the case where the DMA list will fit within the static DMA list within the message queue.

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_PRE_FETCHING_LIST for index %d...\n",
             (int)getSPEID(), msgIndex
            );
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_dmaListSize = dmaListSize[msgIndex];

  // Ckeck the size of the dmaList.  If it is less than SPE_DMA_LIST_LENGTH then it will fit
  //   in the static DMA list.  Otherwise, malloc memory to receive the DMA list from main memory.
  // NOTE : Optimize for the case where the DMA list will fit in the static list (i.e. - this function
  //   should not have to allocate memory or do a DMA transaction to fetch the list and, thus, will
  //   only be called once to setup the list).
  if (__builtin_expect(tmp_dmaListSize < 0, 1)) {

    // Calculate the number of entries needed in the DMA list
    tmp_dmaListSize = tmp_msgQueueEntry->readOnlyLen +
                      tmp_msgQueueEntry->readWriteLen +
                      tmp_msgQueueEntry->writeOnlyLen;


    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: (pre-check) tmp_dmaListSize = %d\n", (int)getSPEID(), tmp_dmaListSize);
      }
    #endif

    // Check to see if the DMA list will fit in the static DMA list
    if (__builtin_expect(tmp_dmaListSize > SPE_DMA_LIST_LENGTH, 0)) {

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Dyanmically allocating DMA List memory...\n", (int)getSPEID());
        }
      #endif

      // Calculate the memory needed 
      register int memNeeded = ROUNDUP_16(tmp_dmaListSize * sizeof(DMAListEntry));

      // Allocate the memory needed by large DMA list
      #ifdef __cplusplus
        try {
      #endif
          //dmaList[msgIndex] = (DMAListEntry*)(new char[memNeeded]);
          dmaList[msgIndex] = (DMAListEntry*)(_malloc_align(memNeeded, 4));
      #ifdef __cplusplus
	} catch (...) {
          dmaList[msgIndex] = NULL;
	}
      #endif

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, dmaList[msgIndex]);
      #endif

      // Verify the pointer returned
      if (__builtin_expect((dmaList[msgIndex] == NULL) || 
                           (((unsigned int)dmaList[msgIndex]) < ((unsigned int)(&_end))) ||
                           (((unsigned int)dmaList[msgIndex] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                           0
                          )
         ) {
        #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
          printf("SPE_%d :: Failed to allocate memory for dmaList[%d] (1)... will try again later...\n",
                 (int)getSPEID(), msgIndex
                );
          printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, dmaList[msgIndex]);
	#endif

        dmaList[msgIndex] = NULL;
        dmaListSize[msgIndex] = -1;  // Try allocating again next time

        // STATS
        #if SPE_STATS != 0
          register unsigned int clocks;
          stopTimer(clocks);
          preFetchingListClocks += clocks;
          preFetchingListClocksCounter++;
        #endif

        // DEBUG
        //print_block_table();

        return;  // Skip for now, try again later
      }

      // Zero-out the dmaList
      //memset(dmaList[msgIndex], 0, memNeeded);

    } else {  // DMA list will fit in the static DMA list

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Using static DMA List...\n", (int)getSPEID());
        }
      #endif

      // Point the DMA list pointer at the message queue entry's dma list
      dmaList[msgIndex] = (DMAListEntry*)(tmp_msgQueueEntry->dmaList);

      //// DEBUG
      //printf("SPE_%d :: USING STATIC DMA LIST !!!\n", (int)getSPEID());
      //register int k;
      //for (k = 0; k < dmaListSize[msgIndex]; k++)
      //  printf("SPE_%d ::   dmaList[%d] = { ea = 0x%08x, size = %u }\n",
      //         (int)getSPEID(), k, dmaList[msgIndex][k].ea, dmaList[msgIndex][k].size
      //        );

      //// Set the dmaList pointer
      //dmaList[msgIndex] = (DMAListEntry*)(&(dmaListEntry[msgIndex * SPE_DMA_LIST_LENGTH]));
      //// Copy the list from the message queue entry

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Created DMA List for index %d...\n", (int)getSPEID(), msgIndex);
          printf("SPE_%d :: [TRACE] ::   dmaList[%d] = %p, dmaListSize[%d] = %d...\n",
                 (int)getSPEID(), msgIndex, dmaList[msgIndex], msgIndex, tmp_dmaListSize
                );
          register int _j0;
          for (_j0 = 0; _j0 < tmp_dmaListSize; _j0++) {
            printf("SPE_%d :: [TRACE] ::    DMA Entry %d = { ea = 0x%08x, size = %d }\n",
                   (int)getSPEID(), _j0, dmaList[msgIndex][_j0].ea, dmaList[msgIndex][_j0].size
                  );
          }
        }
      #endif

      // Update the message queue's state
      msgState[msgIndex] = SPE_MESSAGE_STATE_LIST_READY_LIST;
    }

    // Store the DMA list size to the LS
    dmaListSize[msgIndex] = tmp_dmaListSize;

  } // end if (dmaListSize[msgIndex] < 0)

  // NOTE : If execution reaches here, dmaListSize[msgIndex] contains a valid value
  // NOTE : Don't combine this code with the code above incase the DMA list was allocated but there
  //   were no free DMA Queue entries last time this function was called.

  // Only issue the DMA-Get for the WR's DMA list if the list is large
  if (__builtin_expect(tmp_dmaListSize > SPE_DMA_LIST_LENGTH, 0)) {

    // Get the number of free DMA queue entries
    register int numDMAQueueEntries = mfc_stat_cmd_queue();
  
    // Intiate the DMA transfer for the DMA list into dmaList
    if (__builtin_expect(numDMAQueueEntries > 0, 1)) {  // Stay positive

      // DEBUG
      //printf("SPE_%d :: ISSUING DMA-GET FOR DMA LIST !!!\n", (int)getSPEID());

      spu_mfcdma32(dmaList[msgIndex],
                   (unsigned int)(msgQueue[msgIndex]->readWritePtr),
                   ROUNDUP_16(dmaListSize[msgIndex] * sizeof(DMAListEntry)),
                   msgIndex,
                   MFC_GET_CMD
                  );

      // Update the state of the message queue entry now that the data should be in-flight
      msgState[msgIndex] = SPE_MESSAGE_STATE_FETCHING_LIST;
    }
  }

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    preFetchingListClocks += clocks;
    preFetchingListClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_PRE_FETCHING_LIST] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int fetchingListClocks = 0;
  long long int fetchingListClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_FETCHING_LIST
inline void processMsgState_fetchingList(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_FETCHING_LIST for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  // Read the tag status to see if the data has arrived for the fetching message entry
  mfc_write_tag_mask(0x1 << msgIndex);
  mfc_write_tag_update_immediate();
  register int tagStatus = mfc_read_tag_status();

  // Check if the data has arrived
  if (tagStatus != 0) {

    register int j;

    // Update the state to show that this message queue entry has its DMA list in the LS
    msgState[msgIndex] = SPE_MESSAGE_STATE_LIST_READY_LIST;

    // Roundup all of the sizes to the next highest multiple of 16
    for (j = 0; j < dmaListSize[msgIndex]; j++)
      dmaList[msgIndex][j].size = ROUNDUP_16(dmaList[msgIndex][j].size);
  }

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    fetchingListClocks += clocks;
    fetchingListClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_FETCHING_LIST] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int listReadyListClocks = 0;
  long long int listReadyListClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_LIST_READY_LIST
inline void processMsgState_listReadyList(int msgIndex) {

  // NOTE : Optimize as if this function is called once and everything goes smoothly (memory allocated and DMA
  //   transaction issued).

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_LIST_READY_LIST for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register void* tmp_localMemPtr = localMemPtr[msgIndex];
  register int tmp_dmaListSize = dmaListSize[msgIndex];

  // DEBUG - STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_ERROR] = clocks;
    }
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: (pre-check) tmp_localMemPtr = %p\n", (int)getSPEID(), tmp_localMemPtr);
    }
  #endif

  // Allocate the memory needed in the LS for this work request
  // NOTE : Assumes the common case is that this function is called only once.
  if (__builtin_expect(tmp_localMemPtr == NULL, 1)) {

    // NOTE : Format : The allocated memory will contain a list of pointers (to the buffers) along
    //   with the memory for the buffers themselves following the list (in order).

    // Setup the list pointers for the buffer types (make sure they are NULL)
    // NOTE : Moved to here so the compiler can mix these instructions in with the following instructions
    readOnlyPtr[msgIndex] = NULL;
    readWritePtr[msgIndex] = NULL;
    writeOnlyPtr[msgIndex] = NULL;

    // Determine the number of bytes needed
    register int tmp_evenDMAListSize = tmp_dmaListSize + (tmp_dmaListSize & 0x01);
    //register unsigned int memNeeded = sizeof(DMAListEntry) * tmp_dmaListSize;
    register unsigned int memNeeded = ROUNDUP_128(sizeof(DMAListEntry) * tmp_evenDMAListSize);
    register int j0;
    #if 0
      for (j0 = 0; j0 < tmp_dmaListSize; j0++) {
        memNeeded += dmaList[msgIndex][j0].size;
      }
    #else
      register int tmp_0 = 0;
      register const DMAListEntry* tmp_dmaList = dmaList[msgIndex];
      for (j0 = 0; j0 < tmp_dmaListSize; j0++) {
        memNeeded += tmp_0;
        tmp_0 = tmp_dmaList[j0].size;
      }
      memNeeded += tmp_0;
    #endif
    //if ((tmp_dmaListSize & 0x01) != 0x00)  // Force even number of dmaListEntry structures
    //  memNeeded += sizeof(DMAListEntry);

    // Check the size of the memory needed.  If it is too large for the SPE's LS, then stop this
    //   message with an error code because the memory allocation will never work.
    register unsigned int heapSize = ((int)0x40000) - ((int)(&_end)) - SPE_RESERVED_STACK_SIZE;
    if (__builtin_expect(memNeeded >= heapSize, 0)) {

      // Free the dmaList if it was allocated (in SPE_MESSAGE_STATE_PRE_FETCHING_LIST or
      //   processMsgState_preFetchingList()).
      if (__builtin_expect(dmaListSize[msgIndex] > SPE_DMA_LIST_LENGTH, 0)) {
        if (__builtin_expect(dmaList[msgIndex] != NULL, 1)) {
          //delete [] dmaList[msgIndex];
          _free_align(dmaList[msgIndex]);
	}
        dmaList[msgIndex] = NULL;
        dmaListSize[msgIndex] = -1;
      }

      // Move the message into the error state
      errorCode[msgIndex] = SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY;
      msgState[msgIndex] = SPE_MESSAGE_STATE_ERROR;

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
          printf("SPE_%d :: Unable to allocated memory for WR at index %d (too large)...\n", (int)getSPEID(), msgIndex);
        }
      #endif

      // STATS
      #if SPE_STATS != 0
        register unsigned int clocks;
        stopTimer(clocks);
        listReadyListClocks += clocks;
        listReadyListClocksCounter++;
      #endif

      return;
    }

    // Try to allocate that memory
    #ifdef __cplusplus
      try {
    #endif
        //localMemPtr[msgIndex] = (void*)(new char[memNeeded]);
        //localMemPtr[msgIndex] = (void*)(_malloc_align(memNeeded, 4));
        tmp_localMemPtr = (void*)(_malloc_align(memNeeded, 4));
    #ifdef __cplusplus
      } catch (...) {
        localMemPtr[msgIndex] = NULL;
      }
    #endif

    // STATS
    #if SPE_STATS != 0
      wrTryAllocCount++;

      #if SPE_STATS >= 2
        if ((wrTryAllocCount % (64 * 1024)) == 0) {
          printf("SPE_%d :: [STATS] :: Alloc Rate = %f\n", (int)getSPEID(), (float)wrAllocCount / (float)wrTryAllocCount);
          //print_block_table();
          wrAllocCount = 0;
          wrNoAllocCount = 0;
          wrTryAllocCount = 0;
        }
      #endif
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMemPtr[msgIndex]);
    #endif

    //// Check the pointer that was returned
    //if (__builtin_expect((localMemPtr[msgIndex] == NULL) || 
    //                     (((unsigned int)localMemPtr[msgIndex]) < ((unsigned int)(&_end))) ||
    //                     (((unsigned int)localMemPtr[msgIndex] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
    //                     0
    //                    )
    //   ) {
    // Check the pointer that was returned

    if (__builtin_expect((tmp_localMemPtr == NULL) || 
                         (((unsigned int)tmp_localMemPtr) < ((unsigned int)(&_end))) ||
                         (((unsigned int)tmp_localMemPtr + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                         0
                        )
       ) {

      #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	printf("SPE_%d :: SPE :: Failed to allocate memory for localMemPtr[%d] (1)... will try again later...\n",
               (int)getSPEID(), msgIndex
              );
        printf("SPE_%d :: SPE :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMemPtr[msgIndex]);
      #endif

      // NOTE : localMemPtr[msgIndex] is already NULL (was NULL to begin with if this code reached, the invalid
      //   pointer returned by the malloc call is in tmp_localMemPtr)
      //localMemPtr[msgIndex] = NULL;

      // STATS
      #if SPE_STATS != 0
        register unsigned int clocks;
        stopTimer(clocks);
        listReadyListClocks += clocks;
        listReadyListClocksCounter++;
      #endif

      // STATS
      #if SPE_STATS != 0
        wrNoAllocCount++;
      #endif

      // DEBUG
      //print_block_table();

      return; // Try again next time
    }

    // STATS
    #if SPE_STATS != 0
      wrAllocCount++;
    #endif

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: Allocated memory for WR at index %d... tmp_localMemPtr = %p\n",
               (int)getSPEID(), msgIndex, tmp_localMemPtr
              );
      }
    #endif

    #if 0

      // Setup pointers to the buffers
      register unsigned int initOffset = ROUNDUP_128(dmaListSize[msgIndex] * sizeof(DMAListEntry));
      register unsigned int offset = initOffset;
      register int j1;
      //register DMAListEntry* localDMAList = (DMAListEntry*)(localMemPtr[msgIndex]);
      register DMAListEntry* localDMAList = (DMAListEntry*)(tmp_localMemPtr);
      register const DMAListEntry* remoteDMAList = dmaList[msgIndex];

      //for (j1 = 0; j1 < dmaListSize[msgIndex]; j1++) {
      for (j1 = 0; j1 < tmp_dmaListSize; j1++) {
        register unsigned int size = ((remoteDMAList[j1].size) & 0x0000FFFF);
        localDMAList[j1].ea = ((unsigned int)localMemPtr[msgIndex]) + offset;
        localDMAList[j1].size = size;
        offset += size;

        // DEBUG
        //printf("SPE_%d :: DMA List Entry %d :: { ea = 0x%08x, size = %u } -=> { ea = 0x%08x, size = %u }\n",
        //       (int)getSPEID(), j1, remoteDMAList[j1].ea, remoteDMAList[j1].size, localDMAList[j1].ea, localDMAList[j1].size
        //      );

      }

    #elif 1

      // NOTE : The DMA list has an even number of elements (so the first buffer is 16 byte aligned)

      {
        // NOTE : This code works on an even number of DMA elements (unrolled 2 iterations).
        //   The format of a DMAListEntry is { unsigned int size, unsigned int ea } therefor
        //   each 'vector unsigned int' holds 'size+0, ea+0, size+1, ea+1'.
        //
        //   First iteration:
        //     remote input   :: { ea[0], size[0], ea[1], size[1] }
        //     result (local) :: { initOffset, size[0], initOffset+size[0], size[1] }
        //   Second iteration should result in:
        //     { initOffset+size[0]+size[1], size[2], initOffset+size[0]+size[1]+size[2], size[3] }

        register unsigned int initOffset = ROUNDUP_128(dmaListSize[msgIndex] * sizeof(DMAListEntry));
        initOffset += (unsigned int)(tmp_localMemPtr);
        register vector unsigned int* localDMAList = (vector unsigned int*)(tmp_localMemPtr);
        register vector unsigned int* remoteDMAList = (vector unsigned int*)(dmaList[msgIndex]);
        //register int tmp_evenDMAListSize = tmp_dmaListSize + (tmp_dmaListSize & 0x01);

        register int j1;
        register unsigned int offset = initOffset;
        register vector unsigned int tmp_offset = spu_splats(initOffset);

        register vector unsigned char mask2 = (vector unsigned char){ 0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x00, 0x01, 0x02, 0x03 };
        register vector unsigned char mask3 = (vector unsigned char){ 0x00, 0x01, 0x02, 0x03,  0x14, 0x15, 0x16, 0x17,  0x08, 0x09, 0x0a, 0x0b,  0x1c, 0x1d, 0x1e, 0x1f };

        // TRACE
        #if ENABLE_TRACE >= 2
          if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
            printf("SPE_%d :: [DEBUG] :: Creating Local DMA List...\n", (int)getSPEID());
            printf("SPE_%d :: [DEBUG] ::   tmp_evenDMAListSize = %d\n", (int)getSPEID(), tmp_evenDMAListSize);
	  }
        #endif

        for (j1 = 0; j1 < tmp_evenDMAListSize; j1 += 2) {

          // Read the remote DMA list entry (2 entries: size+0, ea+0, size+1, ea+1)
          register vector unsigned int tmp_0 = *(remoteDMAList);  // contains { 'size+0', 'ea+0', 'size+1', 'ea+1' }
          remoteDMAList++;

          // TRACE
          #if ENABLE_TRACE >= 2
            if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
              printf("SPE_%d :: [DEBUG] ::   tmp_0 = { %u, 0x%08x, %u, 0x%08x }\n",
                     (int)getSPEID(), spu_extract(tmp_0, 0), spu_extract(tmp_0, 1), spu_extract(tmp_0, 2), spu_extract(tmp_0, 3)
                    );
              printf("SPE_%d :: [DEBUG] ::   tmp_offset = { %u, %u, %u, %u }\n",
                     (int)getSPEID(), spu_extract(tmp_offset, 0), spu_extract(tmp_offset, 1), spu_extract(tmp_offset, 2), spu_extract(tmp_offset, 3)
                    );
	    }
          #endif

          // Update the offset (add first DMA list entry's size to second's ea)
          register vector unsigned int tmp_1 = spu_shuffle(tmp_0, tmp_0, mask2);   // contains { xx, 0x00, xx, 'size+0' }
          register vector unsigned int tmp_offset_1 = spu_add(tmp_offset, tmp_1);  // contains { xx, offset, xx, offset + 'size+0' }

          // TRACE
          #if ENABLE_TRACE >= 2
            if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
              printf("SPE_%d :: [DEBUG] ::   tmp_1 = { xx, %u, xx, %u }\n",
                     (int)getSPEID(), spu_extract(tmp_1, 1), spu_extract(tmp_1, 3)
                    );
              printf("SPE_%d :: [DEBUG] ::   tmp_offset_1 = { xx, %u, xx, %u }\n",
                     (int)getSPEID(), spu_extract(tmp_offset_1, 1), spu_extract(tmp_offset_1, 3)
                    );
	    }
          #endif

          // Update offset to be the sum of the current offset plus the sizes of both DMA list elements
          offset += (spu_extract(tmp_0, 0) + spu_extract(tmp_0, 2));
          tmp_offset = spu_splats(offset);

          // Write the result into the localDMAList (ea's change, sizes remain the same)
          register vector unsigned int tmp_entry = spu_shuffle(tmp_0, tmp_offset_1, mask3);  // contains { 'size+0', offset, 'size+1', offset+'size+0' }
          *(localDMAList) = tmp_entry;
          localDMAList++;

          // TRACE
          #if ENABLE_TRACE >= 2
            if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
              printf("SPE_%d :: [DEBUG] ::   tmp_entry = { %u, 0x%08x, %u, 0x%08x }\n",
                     (int)getSPEID(), spu_extract(tmp_entry, 0), spu_extract(tmp_entry, 1), spu_extract(tmp_entry, 2), spu_extract(tmp_entry, 3)
                    );
	    }
          #endif

	}
      }

    #else

      // Setup pointers to the buffers
      register unsigned int initOffset = ROUNDUP_128(tmp_dmaListSize * sizeof(DMAListEntry));
      register DMAListEntry* localDMAList = (DMAListEntry*)(tmp_localMemPtr);
      register const DMAListEntry* remoteDMAList = dmaList[msgIndex];

      {
        register int j1;
        register unsigned int tmp_ea = ((unsigned int)tmp_localMemPtr) + initOffset;
        register unsigned int tmp_size = remoteDMAList[0].size;

        for (j1 = 0; j1 < tmp_dmaListSize; j1++) {

          // Store the values into the localDMAList
          localDMAList[j1].ea = tmp_ea;
          localDMAList[j1].size = tmp_size;
          
          // Compute the next ea and size
          tmp_ea += tmp_size;
          tmp_size = (remoteDMAList[j1 + 1].size) & 0x0000FFFF;
        }
      }

    #endif

    // Zero the memory if needed
    #if SPE_ZERO_WRITE_ONLY_MEMORY != 0
      //if (msgQueue[msgIndex]->writeOnlyLen > 0) {
      if (tmp_msgQueueEntry->writeOnlyLen > 0) {
	register unsigned int writeSize = 0;
        register int j2;
        //for (j2 = dmaListSize[msgIndex] - msgQueue[msgIndex]->writeOnlyLen; j2 < dmaListSize[msgIndex]; j2++)
        register const DMAListEntry* tmp_dmaList = dmaList[msgIndex];
        for (j2 = tmp_dmaListSize - tmp_msgQueueEntry->writeOnlyLen; j2 < tmp_dmaListSize; j2++)
          //writeSize += dmaList[msgIndex][j2].size;
          writeSize += tmp_dmaList[j2].size;
        //memset(((char*)(localMemPtr[msgIndex])) + memNeeded - writeSize, 0, writeSize);
        memset(((char*)(tmp_localMemPtr)) + memNeeded - writeSize, 0, writeSize);
      }
    #endif


    //// DEBUG
    // {
    //  register int j2;
    //  printf("SPE_%d :: dmaListSize[%d] = %d, tmp_dmaListSize = %d\n",
    //         (int)getSPEID(), msgIndex, dmaListSize[msgIndex], tmp_dmaListSize
    //        );
    //  printf("SPE_%d :: localMemPtr[%d] = %p, tmp_localMemPtr = %p\n",
    //         (int)getSPEID(), msgIndex, localMemPtr[msgIndex], tmp_localMemPtr
    //        );
    //  printf("SPE_%d :: DMA List = {\n", (int)getSPEID());
    //  register DMAListEntry* localDMAList = (DMAListEntry*)(tmp_localMemPtr);
    //  for (j2 = 0; j2 < tmp_dmaListSize; j2++) {
    //    printf("SPE_%d ::   %d :: { ea = 0x%08x, size = %u }\n",
    //           (int)getSPEID(), j2, localDMAList[j2].ea, localDMAList[j2].size
    //          );
    //  }
    //}

    // Store the local memory pointer (DEBUG : moved below)
    //localMemPtr[msgIndex] = tmp_localMemPtr;

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
        register int j4;
        register DMAListEntry* localDMAList = (DMAListEntry*)(tmp_localMemPtr);
        printf("SPE_%d :: [TRACE] :: Local DMA List\n", (int)getSPEID());
        for (j4 = 0; j4 < tmp_dmaListSize; j4++) {
          printf("SPE_%d :: [TRACE] ::   entry %d = { ea = %u, size = %u }\n",
                 (int)getSPEID(), j4, localDMAList[j4].ea, localDMAList[j4].size
                );
	}
      }
    #endif

    // DEBUG - STATS1
    #if SPE_STATS1 != 0
      if (timingIndex == msgIndex) {
        register unsigned int clocks;
        getTimer(clocks);
        wrClocksDetail[SPE_MESSAGE_STATE_FINISHED] = clocks;
      }
    #endif

  } // end if (localMemPtr[msgIndex] == NULL)

  // NOTE : If execution reaches here, tmp_localMemPtr contains a valid value (store it to localMemPtr[msgIndex])
  localMemPtr[msgIndex] = tmp_localMemPtr;

  // Check to see if there are buffers to read into the LS for this work request
  //register int tmp_readCount = msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen;
  register int tmp_readCount = tmp_msgQueueEntry->readOnlyLen + tmp_msgQueueEntry->readWriteLen;

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: tmp_readCount = %d\n", (int)getSPEID(), tmp_readCount);
    }
  #endif

  if (__builtin_expect(tmp_readCount > 0, 1)) {

    // Get the number of free DMA queue entries
    register int numDMAQueueEntries = mfc_stat_cmd_queue();

    // Check to see if there is a free DMA queue entry
    if (__builtin_expect(numDMAQueueEntries > 0, 1)) {

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: Pre-GETL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
               (int)getSPEID, msgIndex, ((char*)(localMemPtr[msgIndex])) + ROUNDUP_16(dmaListSize[msgIndex] * sizeof(DMAListEntry)),
               (unsigned int)msgQueue[msgIndex]->readOnlyPtr,
               (unsigned int)(dmaList[msgIndex]),
               (msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen) * sizeof(DMAListEntry), msgIndex
              );

        register int j4;
        for (j4 = 0; j4 < dmaListSize[i]; j4++) {
          printf("SPE_%d :: dmaList[%d][%d].size = %d (0x%x), dmaList[%d][%d].ea = 0x%08x (0x%08x)\n",
                 (int)getSPEID(),
                 msgIndex, j4, dmaList[msgIndex][j4].size, ((DMAListEntry*)(localMemPtr[msgIndex]))[j4].size,
                 msgIndex, j4, dmaList[msgIndex][j4].ea, ((DMAListEntry*)(localMemPtr[msgIndex]))[j4].ea
	        );
        }
      #endif

      //// Queue the DMA transaction
      //spu_mfcdma64(((char*)(localMemPtr[msgIndex])) + ROUNDUP_16(dmaListSize[msgIndex] * sizeof(DMAListEntry)),
      //             (unsigned int)msgQueue[msgIndex]->readOnlyPtr,  // eah
      //             (unsigned int)(dmaList[msgIndex]),
      //             (msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen) * sizeof(DMAListEntry),
      //             msgIndex,
      //             MFC_GETL_CMD
      //            );

      // Queue the DMA transaction
      //register unsigned int lsPtr = (unsigned int)(localMemPtr[msgIndex]);
      register unsigned int lsPtr = (unsigned int)(tmp_localMemPtr);
      //register unsigned int lsOffset = dmaListSize[msgIndex] * sizeof(DMAListEntry);
      register unsigned int lsOffset = tmp_dmaListSize * sizeof(DMAListEntry);
      lsOffset = ROUNDUP_128(lsOffset);
      lsPtr += lsOffset;
      //spu_mfcdma64((void*)lsPtr,
      //             (unsigned int)msgQueue[msgIndex]->readOnlyPtr,  // eah
      //             (unsigned int)(dmaList[msgIndex]),
      //             (msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen) * sizeof(DMAListEntry),
      //             msgIndex,
      //             MFC_GETL_CMD
      //            );
      spu_mfcdma64((void*)lsPtr,
                   (unsigned int)tmp_msgQueueEntry->readOnlyPtr,  // eah
                   (unsigned int)(dmaList[msgIndex]),
                   (tmp_readCount * sizeof(DMAListEntry)),
                   msgIndex,
                   MFC_GETL_CMD
	          );

      // Update the state of the message queue entry now that the data should be in-flight 
      msgState[msgIndex] = SPE_MESSAGE_STATE_FETCHING;

    } // end if (numDMAQueueEntries > 0)

  } else {  // No input buffers so this work request is ready to execute

    // Update the state (to ready to execute)
    msgState[msgIndex] = SPE_MESSAGE_STATE_READY;
  }

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    listReadyListClocks += clocks;
    listReadyListClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_LIST_READY_LIST] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int fetchingClocks = 0;
  long long int fetchingClocksCounter = 0;
  long long int fetchingPassCounter = 0;
#endif

// SPE_MESSAGE_STATE_FETCHING
inline void processMsgState_fetching(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_FETCHING for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  // Read the tag status to see if the data has arrived for the fetching message entry
  mfc_write_tag_mask(0x1 << msgIndex);
  mfc_write_tag_update_immediate();
  register int tagStatus = mfc_read_tag_status();

  // Check if the data has arrived
  if (tagStatus != 0) {


    // DEBUG
    //static int fCntr[SPE_MESSAGE_QUEUE_LENGTH] = { 0 };
    //fCntr[msgIndex]++;
    //register int speID = (int)getSPEID();
    //if (speID == 0)
    //  printf("SPE_%d :: msgIndex = %d,  { %4d, %4d, %4d, %4d, %4d, %4d, %4d, %4d }\n",
    //         speID, msgIndex, fCntr[0], fCntr[1], fCntr[2], fCntr[3], fCntr[4], fCntr[5], fCntr[6], fCntr[7]
    //        );


    // Update the state to show that this message queue entry is ready to be executed
    msgState[msgIndex] = SPE_MESSAGE_STATE_READY;

    // STATS
    #if SPE_STATS != 0
      fetchingPassCounter++;
    #endif

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Input data for index %d has arrived...\n", (int)getSPEID(), msgIndex);
      }
    #endif

    // Clean up the DMA list (only if this is NOT a list type work request; if it is a
    //   list type work request (flag has LIST bit set) then do not delete the DMA list).
    if ((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {
      if (dmaListSize[msgIndex] > SPE_DMA_LIST_LENGTH) {
        //delete [] dmaList[msgIndex];
        _free_align(dmaList[msgIndex]);
        dmaList[msgIndex] = NULL;
      }
      dmaListSize[msgIndex] = -1;  // NOTE: Clear this so data that the DMA list looks like it has not been set now
    }

  } // end if (tagStatus != 0)

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    fetchingClocks += clocks;

    #if SPE_STATS >= 2
      // DEBUG
      static long long int lastPassCnt = 0;
      static long long int lastCnt = 0;
      if (__builtin_expect(fetchingClocksCounter > 0, 1) &&
          __builtin_expect(fetchingPassCounter != lastPassCnt, 0) &&
          __builtin_expect((fetchingPassCounter % (1 * 1024)) == 0, 0)
         ) {
        printf("SPE_%d :: [STATS] :: Fetching Pass Rate = %f :: ( %lld / %lld ) :: diffs = %lld / %lld\n",
               (int)getSPEID(),
               (float)fetchingPassCounter / (float)fetchingClocksCounter,
               fetchingPassCounter, fetchingClocksCounter,
               fetchingPassCounter - lastPassCnt, fetchingClocksCounter - lastCnt
              );
        lastPassCnt = fetchingPassCounter;
        lastCnt = fetchingClocksCounter;

        //printf("\n");
        //displayStatsData();
        //printf("\n");
      }
    #endif

    fetchingClocksCounter++;

  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_FETCHING] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int userClocks = 0;
  long long int userClocksCounter = 0;
  long long int readyClocks = 0;
  long long int readyClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_READY
inline void processMsgState_ready(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_READY for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  // Create a pointer to the message queue entry
  register volatile SPEMessage* msg = msgQueue[msgIndex];

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    if (msgQueue[msgIndex]->traceFlag)
      printf("SPE_%d :: >>>>> Entering User Code (index %d)...\n", (int)getSPEID(), msgIndex);
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Entring user code for index %d...\n", (int)getSPEID(), msgIndex);
    }
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    isTracingFlag = ((msg->traceFlag) ? (-1) : (0));
  #endif

  // DEBUG
  execIndex = msgIndex;

  // Pre-load values needed in either case (list or standard) to help the compiler schedule instructions
  register int tmp_funcIndex = msg->funcIndex;
  register int tmp_readWriteLen = msg->readWriteLen;
  register int tmp_readOnlyLen = msg->readOnlyLen;
  register int tmp_writeOnlyLen = msg->writeOnlyLen;

  // Execute the work request
  if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      if (msgQueue[msgIndex]->traceFlag)
	printf("SPE_%d :: Executing message queue entry as standard entry... fi = %d...\n", (int)getSPEID(), msg->funcIndex);
    #endif

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Executing index %d as STANDARD work request...\n", (int)getSPEID(), msgIndex);
      }
    #endif

    // STATS
    #if SPE_STATS != 0
      register unsigned int userStartClocks;
      getTimer(userStartClocks);
    #endif

    // Make the call to funcLookup()
    funcLookup(tmp_funcIndex,
               readWritePtr[msgIndex], tmp_readWriteLen,
               readOnlyPtr[msgIndex], tmp_readOnlyLen,
               writeOnlyPtr[msgIndex], tmp_writeOnlyLen,
               NULL
              );

    // STATS
    #if SPE_STATS != 0
      register unsigned int userEndClocks;
      getTimer(userEndClocks);
      userClocks += (userEndClocks - userStartClocks);
      userClocksCounter++;
    #endif

    // Update the state of the message queue entry
    msgState[msgIndex] = SPE_MESSAGE_STATE_EXECUTED;

  } else {

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      if (msgQueue[msgIndex]->traceFlag)
	printf("SPE_%d :: Executing message queue entry as list entry... fi = %d...\n", (int)getSPEID(), msg->funcIndex);
    #endif

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Executing index %d as LIST work request...\n", (int)getSPEID(), msgIndex);
      }
    #endif

    // STATS
    #if SPE_STATS != 0
      register unsigned int userStartClocks;
      getTimer(userStartClocks);
    #endif

    // Make the call to funcLookup()
    funcLookup(tmp_funcIndex,
               NULL, tmp_readWriteLen,
               NULL, tmp_readOnlyLen,
               NULL, tmp_writeOnlyLen,
               (DMAListEntry*)(localMemPtr[msgIndex])
              );

    // STATS
    #if SPE_STATS != 0
      register unsigned int userEndClocks;
      getTimer(userEndClocks);
      userClocks += (userEndClocks - userStartClocks);
      userClocksCounter++;
    #endif

    // Update the state of the message queue entry
    msgState[msgIndex] = SPE_MESSAGE_STATE_EXECUTED_LIST;
  }

  // DEBUG
  execIndex = -1;

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    if (msgQueue[msgIndex]->traceFlag)
      printf("SPE_%d :: <<<<< Leaving User Code...\n", (int)getSPEID());
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Exiting user code for index %d...\n", (int)getSPEID(), msgIndex);
    }
  #endif

  // SPE Stats Code
  #if SPE_STATS != 0
    statData.numWorkRequestsExecuted++;
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    readyClocks += clocks;
    readyClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_READY] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int executedClocks = 0;
  long long int executedClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_EXECUTED
inline void processMsgState_executed(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_EXECUTED for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  // Check to see if this message does not need to fetch any data
  if (__builtin_expect(((msgQueue[msgIndex]->readWritePtr == (PPU_POINTER_TYPE)NULL) ||
                        ((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == WORK_REQUEST_FLAGS_RW_IS_RO)
                       ) &&
                       (msgQueue[msgIndex]->writeOnlyPtr == (PPU_POINTER_TYPE)NULL),
                       0
		      )
     ) {

    // Update the state of the message queue entry
    // NOTE : Even though the work request is basically finished, the message queue index still needs to be
    //   passed back to the PPE, i.e. go to the committing state.
    msgState[msgIndex] = SPE_MESSAGE_STATE_COMMITTING;

    // STATS
    #if SPE_STATS != 0
      register unsigned int clocks;
      stopTimer(clocks);
      executedClocks += clocks;
      executedClocksCounter++;
    #endif

    return;
  }

  // Create the DMA list
  // NOTE: Check to see if the DMA list has already been created yet or not (if dmaListSize[msgIndex] < 0, not created)
  if (dmaListSize[msgIndex] < 0) {

    // Count the number of DMA entries needed for the read DMA list
    register int entryCount = 0;
    if ((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == 0x00)
      entryCount += (ROUNDUP_16(msgQueue[msgIndex]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                    (((msgQueue[msgIndex]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
    entryCount += (ROUNDUP_16(msgQueue[msgIndex]->writeOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                  (((msgQueue[msgIndex]->writeOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

    // Allocate a larger DMA list if needed
    if (__builtin_expect(entryCount > SPE_DMA_LIST_LENGTH, 0)) {

      // Allocate the memory
      #ifdef __cplusplus
        try {
      #endif
          //dmaList[msgIndex] = new DMAListEntry[entryCount];
          dmaList[msgIndex] = (DMAListEntry*)(_malloc_align(entryCount * sizeof(DMAListEntry), 4));
      #ifdef __cplusplus
        } catch (...) {
          dmaList[msgIndex] = NULL;
	}
      #endif

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, dmaList[msgIndex]);
      #endif

      // Verify the pointer that was returned
      if (__builtin_expect((dmaList[msgIndex] == NULL) || 
                           (((unsigned int)dmaList[msgIndex]) < ((unsigned int)(&_end))) ||
                           (((unsigned int)dmaList[msgIndex] + (entryCount * sizeof(DMAListEntry))) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                           0
			  )
         ) {
        #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	  printf("SPE_%d :: SPE :: Failed to allocate memory for dmaList[%d] (3)... will try again later...\n", (int)getSPEID(), msgIndex);
          printf("SPE_%d :: SPE :: dmaList[%d] = %p\n", (int)getSPEId(), msgIndex, dmaList[msgIndex]);
	#endif
        dmaList[msgIndex] = NULL;

        // STATS
        #if SPE_STATS != 0
          register unsigned int clocks;
          stopTimer(clocks);
          executedClocks += clocks;
          executedClocksCounter++;
        #endif

        // DEBUG
        //print_block_table();

        return;  // Skip for now, try again later
      }

      // Clear the newly allocated DMA list
      //memset(dmaList[msgIndex], 0, sizeof(DMAListEntry) * entryCount);

    } else {  // Otherwise, the static DMA list is large enough (just use it)

      // Point the DMA list pointer at the message queue entry's dma list
      dmaList[msgIndex] = (DMAListEntry*)(msgQueue[msgIndex]->dmaList);

      // Place the pointer of the static DMA list
      //dmaList[msgIndex] = (DMAListEntry*)(&(dmaListEntry[msgIndex * SPE_DMA_LIST_LENGTH]));

    }

    // Fill in the size of the DMA list
    dmaListSize[msgIndex] = entryCount;

    // Fill in the list
    readOnlyPtr[msgIndex] = NULL;   // Use this pointer to point to the first buffer to be written to memory (don't nead readOnlyPtr data anymore)
    register int listIndex = 0;

    // Read-write buffer
    if (__builtin_expect((msgQueue[msgIndex]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == 0x00, 0) &&
        (readWritePtr[msgIndex] != NULL)
       ) {

      register int bufferLeft = msgQueue[msgIndex]->readWriteLen;
      register unsigned int srcOffset = (unsigned int)(msgQueue[msgIndex]->readWritePtr);

      while (bufferLeft > 0) {
        dmaList[msgIndex][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ?
                                               (SPE_DMA_LIST_ENTRY_MAX_LENGTH) :
                                               (bufferLeft)
                                            );
        dmaList[msgIndex][listIndex].size = ROUNDUP_16(dmaList[msgIndex][listIndex].size);
        dmaList[msgIndex][listIndex].ea = srcOffset;

        bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
        listIndex++;
        srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
      }

      // Store the start of the write portion of the localMem buffer in readOnlyPtr
      readOnlyPtr[msgIndex] = readWritePtr[msgIndex];
    }

    // Write-only buffer
    if (writeOnlyPtr[msgIndex] != NULL) {

      register int bufferLeft = msgQueue[msgIndex]->writeOnlyLen;
      register unsigned int srcOffset = (unsigned int)(msgQueue[msgIndex]->writeOnlyPtr);

      while (bufferLeft > 0) {
        dmaList[msgIndex][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
        dmaList[msgIndex][listIndex].size = ROUNDUP_16(dmaList[msgIndex][listIndex].size);
        dmaList[msgIndex][listIndex].ea = srcOffset;

        bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
        listIndex++;
        srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
      }

      // Store the start of the write portion of the localMem buffer in readOnlyPtr (if
      //   it is not set already... i.e. - this buffer isn't first)
      if (readOnlyPtr[msgIndex] == NULL) readOnlyPtr[msgIndex] = writeOnlyPtr[msgIndex];

    }

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Created DMA List for index %d...\n", (int)getSPEID(), msgIndex);
        printf("SPE_%d :: [TRACE] ::   dmaList[%d] = %p, dmaListSize[%d] = %d...\n",
               (int)getSPEID(), msgIndex, dmaList[msgIndex], msgIndex, dmaListSize[msgIndex]
              );
        register int _j0;
        for (_j0 = 0; _j0 < dmaListSize[msgIndex]; _j0++) {
          printf("SPE_%d :: [TRACE] ::    DMA Entry %d = { ea = 0x%08x, size = %d }\n",
                 (int)getSPEID(), _j0, dmaList[msgIndex][_j0].ea, dmaList[msgIndex][_j0].size
                );
        }
      }
    #endif

  } // end if (dmaListSize[msgIndex] < 0)

  // NOTE : If execution reaches here, the DMA list is setup (dmaListSize[msgIndex] set to a valid value)

  // Get the number of free DMA queue entries
  register int numDMAQueueEntries = mfc_stat_cmd_queue();

  // DEBUG
  //{
  //  static long long int qel = 0;
  //  static long long int qelCnt = 0;
  //  qel += numDMAQueueEntries;
  //  qelCnt++;
  //
  //  if ((qelCnt % 1024) == 0)
  //    printf("SPE_%d :: from EXECUTED - qel avg = %lf (%lld samples)\n",
  //           (int)getSPEID(), (double)qel / (double)qelCnt, qelCnt
  //          );
  //}

  // Check to see if there is a free DMA queue entry
  if (numDMAQueueEntries > 0) {

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: Pre-PUTL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
             (int)getSPEID(), msgIndex, readOnlyPtr[msgIndex], 0, (unsigned int)(dmaList[msgIndex]),
             dmaListSize[msgIndex] * sizeof(DMAListEntry), msgIndex
            );
    #endif

    // Queue the DMA transaction
    spu_mfcdma64(readOnlyPtr[msgIndex],  // NOTE : This pointer is being used to point to the start of the write portion of localMem
                 0,
                 (unsigned int)(dmaList[msgIndex]),
                 dmaListSize[msgIndex] * sizeof(DMAListEntry),
                 msgIndex,
                 MFC_PUTL_CMD
		);

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: DMA transaction queued for output data for index %d...\n", (int)getSPEID(), msgIndex);
      }
    #endif

    // Update the state of the message queue entry now that the data should be in-flight
    msgState[msgIndex] = SPE_MESSAGE_STATE_COMMITTING;

  } // end if (numDMAQueueEntries > 0)

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    executedClocks += clocks;
    executedClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_EXECUTED] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int executedListClocks = 0;
  long long int executedListClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_EXECUTED_LIST
inline void processMsgState_executedList(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_EXECUTED_LIST for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

    register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_readOnlyLen = tmp_msgQueueEntry->readOnlyLen;
  register int tmp_writeCount = tmp_msgQueueEntry->readWriteLen + tmp_msgQueueEntry->writeOnlyLen;
  register void* tmp_localMemPtr = localMemPtr[msgIndex];

  // Check to see if there is output data that needs to be placed into system memory
  //if ((msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) > 0 && localMemPtr[msgIndex] != NULL) {
  if (__builtin_expect(tmp_writeCount > 0, 1) && __builtin_expect(tmp_localMemPtr != NULL, 1)) {

    // Get the number of free DMA queue entries
    register int numDMAQueueEntries = mfc_stat_cmd_queue();

    // If there is a free DMA queue entry, initiate the DMA transfer of the readWrite and
    //   writeOnly buffer back to main memory
    if (numDMAQueueEntries > 0) {

      // Get the offsets in the DMA list and the localMemPtr for the readWrite section
      register unsigned int readWriteOffset = ROUNDUP_128(dmaListSize[msgIndex] * sizeof(DMAListEntry));
      register int j0;
      register DMAListEntry* tmp_dmaList = dmaList[msgIndex];
      for (j0 = 0; j0 < tmp_readOnlyLen; j0++)
        readWriteOffset += tmp_dmaList[j0].size;

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: Pre-PUTL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
               (int)getSPEID(), msgIndex, ((char*)localMemPtr[msgIndex]) + (readWriteOffset),
               (unsigned int)msgQueue[msgIndex]->readOnlyPtr,
               (unsigned int)(&(dmaList[msgIndex][msgQueue[msgIndex]->readOnlyLen])),
               (msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) * sizeof(DMAListEntry), msgIndex
              );
      #endif

      // Queue the DMA transfer
      //spu_mfcdma64(((char*)localMemPtr[msgIndex]) + (readWriteOffset),
      //             (unsigned int)msgQueue[msgIndex]->readOnlyPtr,  // eah
      //             (unsigned int)(&(dmaList[msgIndex][msgQueue[msgIndex]->readOnlyLen])),
      //             (msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) * sizeof(DMAListEntry),
      //             msgIndex,
      //             MFC_PUTL_CMD
      //          );
      spu_mfcdma64(((char*)tmp_localMemPtr) + (readWriteOffset),
                   (unsigned int)tmp_msgQueueEntry->readOnlyPtr,  // eah
                   (unsigned int)(&(tmp_dmaList[tmp_readOnlyLen])),
                   (tmp_writeCount) * sizeof(DMAListEntry),
                   msgIndex,
                   MFC_PUTL_CMD
	          );

      // Update the state of the message queue entry now that the data should be in-flight
      msgState[msgIndex] = SPE_MESSAGE_STATE_COMMITTING;

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: msg %d's state going from %d -> %d\n",
               (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_EXECUTED_LIST, msgState[msgIndex]
              );
      #endif

    } // end if (numDMAQueueEntries > 0)

  } else {  // Otherwise, there is no output data

    // Update the state of the message queue entry now that the data should be in-flight
    msgState[msgIndex] = SPE_MESSAGE_STATE_COMMITTING;

  }

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    executedListClocks += clocks;
    executedListClocksCounter++;
  #endif

  // STATS1
  #if SPE_STATS1 != 0
    if (timingIndex == msgIndex) {
      register unsigned int clocks;
      getTimer(clocks);
      wrClocksDetail[SPE_MESSAGE_STATE_EXECUTED_LIST] = clocks;
    }
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int committingClocks = 0;
  long long int committingClocksCounter = 0;
  long long int committingPassCounter = 0;
#endif

// DEBUG
#if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
int wrCompletedCounter = 0;
#endif

// SPE_MESSAGE_STATE_COMMITTING
inline void processMsgState_committing(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_COMMITTING for index %d...\n", (int)getSPEID(), msgIndex);
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
    }
  #endif

  // Read the tag status to see if the data was sent for the committing message entry
  mfc_write_tag_mask(0x1 << msgIndex);
  mfc_write_tag_update_immediate();
  register int tagStatus = mfc_read_tag_status();

  // Check if the data was sent
  if (tagStatus) {

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Output data for index %d was sent...\n", (int)getSPEID(), msgIndex);
        STATETRACE_OUTPUT(msgIndex);
      }
    #endif

    // Free the local data and message buffers
    if (__builtin_expect(localMemPtr[msgIndex] != NULL, 1)) {

      // Free the LS memory
      //delete [] ((char*)localMemPtr[msgIndex]);
      _free_align(localMemPtr[msgIndex]);

      // Clear the pointers

      // DEBUG (commented out)
      localMemPtr[msgIndex] = NULL;

      readWritePtr[msgIndex] = NULL;
      readOnlyPtr[msgIndex] = NULL;
      writeOnlyPtr[msgIndex] = NULL;
    }

    // If the DMA list was allocated, free it
    if (dmaListSize[msgIndex] > SPE_DMA_LIST_LENGTH) {
      //delete [] dmaList[msgIndex];
      _free_align(dmaList[msgIndex]);
      dmaList[msgIndex] = NULL;
    }

    // Clear the dmaList size so it looks like the dma list has not been set
    dmaListSize[msgIndex] = -1;

    #if SPE_NOTIFY_VIA_MAILBOX != 0

      // Check to see if there is an available entry in the outbound mailbox
      if (spu_stat_out_mbox() > 0) {

        // Clear the entry
        msgState[msgIndex] = SPE_MESSAGE_STATE_CLEAR;

        // STATS
        #if SPE_STATS != 0
          committingPassCounter++;
        #endif

        // Send the index of the entry in the message queue to the PPE
        spu_write_out_mbox(MESSAGE_RETURN_CODE(msgIndex, SPE_MESSAGE_OK));

        // DEBUG
        #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
          wrCompletedCounter++;
        #endif

	// STATS1
        #if SPE_STATS1 != 0
          if (timingIndex == msgIndex) {
            register unsigned int clocks;
            stopTimer(clocks);
            wrClocks += clocks;
            wrClocksDetail[SPE_MESSAGE_STATE_COMMITTING] = clocks;
            wrClocksCounter++;
            timingIndex = -1;

            register int j3;
            for (j3 = 0; j3 < SPE_MESSAGE_NUM_STATES; j3++)
              wrClocksDetailSum[j3] += wrClocksDetail[j3];
	  }
        #endif

      } // end if (spu_stat_out_mbox() > 0)

    #else

      // Update the notify counter to notify the PPE
      register int* notifyQueue = (int*)notifyQueueRaw;
      register int returnCode = SPE_MESSAGE_OK;
      returnCode = (returnCode << 16) | (msgQueue[msgIndex]->counter0 & 0xFFFF);
      notifyQueue[msgIndex] = returnCode;

      // Update the message's state
      msgState[msgIndex] = SPE_MESSAGE_STATE_CLEAR;

      // STATS
      #if SPE_STATS != 0
        committingPassCounter++;
      #endif

      // DEBUG
      #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
        wrCompletedCounter++;
      #endif

      // STATS1
      #if SPE_STATS1 != 0
        if (timingIndex == msgIndex) {
          register unsigned int clocks;
          stopTimer(clocks);
          wrClocks += clocks;
          wrClocksDetail[SPE_MESSAGE_STATE_COMMITTING] = clocks;
          wrClocksCounter++;
          timingIndex = -1;

          register int j3;
          for (j3 = 0; j3 < SPE_MESSAGE_NUM_STATES; j3++)
            wrClocksDetailSum[j3] += wrClocksDetail[j3];
	}
      #endif

    #endif

    // STATS1
    #if SPE_STATS1 != 0
      // DEBUG
      static long long int lastWRClocksCounter = 0;
      if ((wrClocksCounter % 256) == 0 && lastWRClocksCounter != wrClocksCounter) {

        #if 0
        printf("SPE_%d :: [STATS1] :: wrClocks = %f :: wrClocks = %lld, wrClocksCounter = %lld\n",
               (int)getSPEID(), (float)wrClocks / (float)wrClocksCounter, wrClocks, wrClocksCounter
	      );

        printf("SPE_%d :: [STATS1] :: wrClocksDetailSum = { %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f  %d:%.2f }\n",
               (int)getSPEID(),
               SPE_MESSAGE_STATE_CLEAR, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_CLEAR] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_SENT, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_SENT] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_PRE_FETCHING, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_FETCHING, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_PRE_FETCHING_LIST, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING_LIST] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_LIST_READY_LIST, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_LIST_READY_LIST] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_FETCHING_LIST, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING_LIST] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_READY, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_READY] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_EXECUTED, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_EXECUTED_LIST, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED_LIST] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_COMMITTING, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_COMMITTING] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_FINISHED, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_FINISHED] / (float)wrClocksCounter,
               SPE_MESSAGE_STATE_ERROR, (float)wrClocksDetailSum[SPE_MESSAGE_STATE_ERROR] / (float)wrClocksCounter
	      );

        printf("SPE_%d :: [STATS1] :: wrClocksDetail = { %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld  %d:%lld }\n",
               (int)getSPEID(),
               SPE_MESSAGE_STATE_CLEAR, wrClocksDetail[SPE_MESSAGE_STATE_CLEAR],
               SPE_MESSAGE_STATE_SENT, wrClocksDetail[SPE_MESSAGE_STATE_SENT],
               SPE_MESSAGE_STATE_PRE_FETCHING, wrClocksDetail[SPE_MESSAGE_STATE_PRE_FETCHING],
               SPE_MESSAGE_STATE_FETCHING, wrClocksDetail[SPE_MESSAGE_STATE_FETCHING],
               SPE_MESSAGE_STATE_PRE_FETCHING_LIST, wrClocksDetail[SPE_MESSAGE_STATE_PRE_FETCHING_LIST],
               SPE_MESSAGE_STATE_LIST_READY_LIST, wrClocksDetail[SPE_MESSAGE_STATE_LIST_READY_LIST],
               SPE_MESSAGE_STATE_FETCHING_LIST, wrClocksDetail[SPE_MESSAGE_STATE_FETCHING_LIST],
               SPE_MESSAGE_STATE_READY, wrClocksDetail[SPE_MESSAGE_STATE_READY],
               SPE_MESSAGE_STATE_EXECUTED, wrClocksDetail[SPE_MESSAGE_STATE_EXECUTED],
               SPE_MESSAGE_STATE_EXECUTED_LIST, wrClocksDetail[SPE_MESSAGE_STATE_EXECUTED_LIST],
               SPE_MESSAGE_STATE_COMMITTING, wrClocksDetail[SPE_MESSAGE_STATE_COMMITTING],
               SPE_MESSAGE_STATE_FINISHED, wrClocksDetail[SPE_MESSAGE_STATE_FINISHED],
               SPE_MESSAGE_STATE_ERROR, wrClocksDetail[SPE_MESSAGE_STATE_ERROR]
	      );
        #endif

        #if 0 // List

        printf("SPE_%d :: [STATS1] :: wrClocksDetailSum_m = { S:%.2f  PFL:%.2f  LRL:%.2f  F:%.2f  R:%.2f  EL:%.2f  C:%.2f --- E:%.2f  F:%.2f }, %lld samples\n",
               (int)getSPEID(),
               (float)wrClocksDetailSum[SPE_MESSAGE_STATE_SENT] / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING_LIST] - wrClocksDetailSum[SPE_MESSAGE_STATE_SENT]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_LIST_READY_LIST] - wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING_LIST]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING] - wrClocksDetailSum[SPE_MESSAGE_STATE_LIST_READY_LIST]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_READY] - wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED_LIST] - wrClocksDetailSum[SPE_MESSAGE_STATE_READY]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_COMMITTING] - wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED_LIST]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_ERROR] - wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING_LIST]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_FINISHED] - wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING_LIST]) / (float)wrClocksCounter,
               wrClocksCounter
	      );

        #else // Standard

        printf("SPE_%d :: [STATS1] :: wrClocksDetailSum_m = { S:%.2f  PF:%.2f F:%.2f  R:%.2f  E:%.2f  C:%.2f --- F:%.2f} Latency: %.2f - %lld samples\n",
               (int)getSPEID(),
               (float)wrClocksDetailSum[SPE_MESSAGE_STATE_SENT] / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING] - wrClocksDetailSum[SPE_MESSAGE_STATE_SENT]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING] - wrClocksDetailSum[SPE_MESSAGE_STATE_PRE_FETCHING]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_READY] - wrClocksDetailSum[SPE_MESSAGE_STATE_FETCHING]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED] - wrClocksDetailSum[SPE_MESSAGE_STATE_READY]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_COMMITTING] - wrClocksDetailSum[SPE_MESSAGE_STATE_EXECUTED]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_FINISHED] - wrClocksDetailSum[SPE_MESSAGE_STATE_READY]) / (float)wrClocksCounter,
               (float)(wrClocksDetailSum[SPE_MESSAGE_STATE_COMMITTING]) / (float)wrClocksCounter,
               wrClocksCounter
	      );

        #endif

        lastWRClocksCounter = wrClocksCounter;
      }
    #endif

  } // end if (tagStatus)

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    committingClocks += clocks;
    committingClocksCounter++;
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int errorClocks = 0;
  long long int errorClocksCounter = 0;
#endif

// SPE_MESSAGE_STATE_ERROR
inline void processMsgState_error(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_ERROR for index %d (errorCode = %d)...\n",
             (int)getSPEID(), msgIndex, errorCode[msgIndex]
            );
      debug_displayActiveMessageQueue(0x0, msgState, "(*)");
      STATETRACE_OUTPUT(msgIndex);
    }
  #endif

  // NOTE: All clean-up should be taken care of by the code placing the message into the error
  //   state (that way the code here does not have to handle all cases).

  #if SPE_NOTIFY_VIA_MAILBOX != 0

    // Check to see if there is an available entry in the outbound mailbox
    if (spu_stat_out_mbox() > 0) {

      // Clear the entry
      msgState[msgIndex] = SPE_MESSAGE_STATE_CLEAR;

      // Send the index of the entry in the message queue to the PPE along with the ERROR code
      spu_write_out_mbox(MESSAGE_RETURN_CODE(msgIndex, errorCode[msgIndex]));

      // Clear the error
      errorCode[msgIndex] = SPE_MESSAGE_OK;

      // DEBUG
      #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
        wrCompletedCounter++;
      #endif
    }

  #else

    // Update the notify counter to notify the PPE
    register int* notifyQueue = (int*)notifyQueueRaw;
    register int returnCode = errorCode[msgIndex];
    returnCode = (returnCode << 16) | (msgQueue[msgIndex]->counter0 & 0xFFFF);
    notifyQueue[msgIndex] = returnCode;

    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Notify Msg = 0x%08x (0x%08x)...\n",
               (int)getSPEID(), returnCode, notifyQueue[msgIndex]
              );
      }
    #endif

    // Update the message's state
    msgState[msgIndex] = SPE_MESSAGE_STATE_CLEAR;

    // Clear the error
    errorCode[msgIndex] = SPE_MESSAGE_OK;

    // DEBUG
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      wrCompletedCounter++;
    #endif

  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks;
    stopTimer(clocks);
    errorClocks += clocks;
    errorClocksCounter++;
  #endif
}


#if USE_SCHEDULE_LOOP != 0

// Loop scheduler method
inline void speSchedulerInner() {

  register int i;


  // SPE_MESSAGE_STATE_SENT
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    processMsgState_sent(i);
  }

  // SPE_MESSAGE_STATE_PRE_FETCHING_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING_LIST, 0)) {
      processMsgState_preFetchingList(i);
    }
  }

  // SPE_MESSAGE_STATE_FETCHING_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING_LIST, 0)) {
      processMsgState_fetchingList(i);
    }
  }

  // SPE_MESSAGE_STATE_LIST_READY_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_LIST_READY_LIST, 0)) {
      processMsgState_listReadyList(i);
    }
  }

  // SPE_MESSAGE_STATE_PRE_FETCHING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING, 0)) {
      processMsgState_preFetching(i);
    }
  }

  // SPE_MESSAGE_STATE_FETCHING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING, 0)) {
      processMsgState_fetching(i);
    }
  }

  // SPE_MESSAGE_STATE_READY
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_READY, 0)) {
      processMsgState_ready(i);
    }
  }

  // SPE_MESSAGE_STATE_EXECUTED_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED_LIST, 0)) {
      processMsgState_executedList(i);
    }
  }

  // SPE_MESSAGE_STATE_EXECUTED
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED, 0)) {
      processMsgState_executed(i);
    }
  }

  // SPE_MESSAGE_STATE_COMMITTING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_COMMITTING, 0)) {
      processMsgState_committing(i);
    }
  }

  // SPE_MESSAGE_STATE_PRE_ERROR
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_ERROR, 0)) {
      processMsgState_error(i);
    }
  }

}

#else

// STATS
#if SPE_STATS != 0
  //long long int stateCounter = 0;
  //long long int stateCounters[SPE_MESSAGE_NUM_STATES] = { 0 };
  //float stateCountersAvg[SPE_MESSAGE_NUM_STATES] = { 0.0f };
  long long int stateSampleCounter = 0;

  int wrStateCount[SPE_MESSAGE_NUM_STATES] = { 0 };
  int wrInUseQCount = 0;
  int wrInUseLCount = 0;
  int wrInUseCountCounter = 0;
#endif

#define LIMIT_READY  2

// Switch scheduler method
inline void speSchedulerInner() {

  register int i;
  register int tryAgain = 0;

  // STATS
  #if SPE_STATS != 0
    register int stateSampleFlag = 0;
    int wrStateCount_[SPE_MESSAGE_NUM_STATES] = { 0 };
    wrInUseCountCounter++;
  #endif

  // For each message queue entry
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    // STATS
    #if SPE_STATS != 0
      if (msgState[i] != SPE_MESSAGE_STATE_CLEAR) wrInUseLCount++;
      if (msgQueue[i]->state != SPE_MESSAGE_STATE_CLEAR) wrInUseQCount++;
    #endif

    register int msgState_last;

    // Execute the processing function associated with the messaqe queue entry's state
    do {  // while (msgState[i] != msgState_last)

      // STATS
      #if SPE_STATS != 0
        wrStateCount_[msgState[i]]++;
      #endif

      // Record the current state
      msgState_last = msgState[i];

      // Process the message according to the current state
      switch (msgState[i]) {

        case SPE_MESSAGE_STATE_CLEAR:
          processMsgState_sent(i);
          tryAgain = 1;  // Continue on to either PRE_FETCHING or PRE_FETCHING_LIST
          break;

        case SPE_MESSAGE_STATE_PRE_FETCHING:
          processMsgState_preFetching(i);
          tryAgain = 0;  // Just issued the DMA-GET so move on to the next WR
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_PRE_FETCHING_LIST:
          processMsgState_preFetchingList(i);
          tryAgain = 1;  // NOTE: Common case should be that the DMA list length <= SPE_DMA_LIST_LENGTH (so don't wait)
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_FETCHING_LIST:
          processMsgState_fetchingList(i);
          tryAgain = 1;  // Large DMA List just arrived... move on to LIST_READY_LIST
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_LIST_READY_LIST:
          processMsgState_listReadyList(i);
          tryAgain = 0;  // Just issed the DMA-Get so move on to the next WR
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_FETCHING:
          processMsgState_fetching(i);
          tryAgain = 0;  // Input data has arrived... ready to executed
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_READY:
          #if LIMIT_READY == 0
            processMsgState_ready(i);
            tryAgain = 1;  // Execution finished... move on to EXECUTED or EXECUTED_LIST to queue DMA-Put
          #else
	    tryAgain = 0;
          #endif
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_EXECUTED:
          processMsgState_executed(i);
          tryAgain = 0;  // Just issued DMA-PUT so move on to the next WR
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_EXECUTED_LIST:
          processMsgState_executedList(i);
          tryAgain = 0;  // Just issued DMA-Put so move on to the next WR
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_COMMITTING:
          processMsgState_committing(i);
          tryAgain = 0;  // WR finished... no need to continue with it
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_ERROR:
          processMsgState_error(i);
          tryAgain = 0;  // WR finished... no need to continue with it
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        default:
          printf("SPE_%d :: ERROR :: Message queue entry %d in unknown state (%d)...\n",
                 (int)getSPEID(), i, msgState[i]
                );
          break;

      } // end switch(msgState[i])

    } while ((msgState[i] != msgState_last) && (tryAgain != 0));

    #if LIMIT_READY > 0
    {
      static int runIndex = 0;
      register int iOffset;
      register int leftToRun = LIMIT_READY;
      for (iOffset = 0; leftToRun > 0 && iOffset < SPE_MESSAGE_QUEUE_LENGTH; iOffset++) {
        register int ri = (runIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;
        if (msgState[ri] == SPE_MESSAGE_STATE_READY) {
          processMsgState_ready(ri);
          if ((msgQueue[ri]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {
            processMsgState_executed(ri);
	  } else {
            processMsgState_executedList(ri);
	  }
          leftToRun--;
	}
      }
    }
    #endif

  } // end for (i < SPE_MESSAGE_QUEUE_LENGTH)

  // STATS
  #if SPE_STATS != 0
    if (stateSampleFlag > 0) {
      register int i;
      for (i = 0; i < SPE_MESSAGE_NUM_STATES; i++) {
        wrStateCount[i] += wrStateCount_[i];
      }
      stateSampleCounter++;
    }
  #endif

  // STATS
  //#if SPE_STATS != 0
  //  if (stateSampleFlag > 0) {
  //    register int i;
  //    for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
  //      stateCounters[msgState[i]]++;
  //      if (msgState[i] != SPE_MESSAGE_STATE_CLEAR)
  //        stateCounter++;
  //    }
  //    //stateSampleCounter++;
  //  }
  //#endif
}

#endif


// STATS
#if SPE_STATS != 0
  //char displaySampleCounters_buf[1024];
  //inline void displaySampleCounters() {
    //displaySampleCounters_buf[0] = '\0';
    //sprintf(displaySampleCounters_buf, "SPE_%d :: stateSampleCounter = %lld, stateCounter = %lld, stateCounters = { ",
    //       (int)getSPEID(), stateSampleCounter, stateCounter
    //      );
    /*
    register int i;
    for (i = 0; i < SPE_MESSAGE_NUM_STATES; i++) {
      stateCountersAvg[i] = (float)stateCounters[i] / (float)stateSampleCounter;
      sprintf(displaySampleCounters_buf + strlen(displaySampleCounters_buf),
              "%d:%f ",
              i, stateCountersAvg[i]
             );
    }
    */
    //sprintf(displaySampleCounters_buf + strlen(displaySampleCounters_buf), "}\n");
    //printf(displaySampleCounters_buf);

  //}
#endif


// STATS : Display stat data
void displayStatsData() {

  #if SPE_STATS != 0
    printf("SPE_%d :: [STATS] :: Sent_noMsg = %f :: sentClocks_noMsg = %lld, sentClocksCounter_noMsg = %lld...\n", (int)getSPEID(), (float)sentClocks_noMsg / (float)sentClocksCounter_noMsg, sentClocks_noMsg, sentClocksCounter_noMsg);
    printf("SPE_%d :: [STATS] :: Sent_msg = %f :: sentClocks_msg = %lld, sentClocksCounter_msg = %lld...\n", (int)getSPEID(), (float)sentClocks_msg / (float)sentClocksCounter_msg, sentClocks_msg, sentClocksCounter_msg);
    printf("SPE_%d :: [STATS] :: Pre-Fetching = %f :: preFetchingClocks = %lld, preFetchingClocksCounter = %lld...\n", (int)getSPEID(), (float)preFetchingClocks / (float)preFetchingClocksCounter, preFetchingClocks, preFetchingClocksCounter);
    printf("SPE_%d :: [STATS] :: Pre-Fetching List= %f :: preFetchingListClocks = %lld, preFetchingListClocksCounter = %lld...\n", (int)getSPEID(), (float)preFetchingListClocks / (float)preFetchingListClocksCounter, preFetchingListClocks, preFetchingListClocksCounter);
    printf("SPE_%d :: [STATS] :: Fetching List = %f :: fetchingListClocks = %lld, fetchingListClocksCounter = %lld...\n", (int)getSPEID(), (float)fetchingListClocks / (float)fetchingListClocksCounter, fetchingListClocks, fetchingListClocksCounter);
    printf("SPE_%d :: [STATS] :: List-Ready List = %f :: listReadyListClocks = %lld, listReadyListClocksCounter = %lld...\n", (int)getSPEID(), (float)listReadyListClocks / (float)listReadyListClocksCounter, listReadyListClocks, listReadyListClocksCounter);
    printf("SPE_%d :: [STATS] :: Fetching = %f :: fetchingClocks = %lld, fetchingClocksCounter = %lld...\n", (int)getSPEID(), (float)fetchingClocks / (float)fetchingClocksCounter, fetchingClocks, fetchingClocksCounter);
    printf("SPE_%d :: [STATS] :: Ready = %f :: readyClocks = %lld, readyClocksCounter = %lld...\n", (int)getSPEID(), (float)readyClocks / (float)readyClocksCounter, readyClocks, readyClocksCounter);
    printf("SPE_%d :: [STATS] ::   User Code = %f :: userClocks = %lld, userClocksCounter = %lld...\n", (int)getSPEID(), (float)userClocks / (float)userClocksCounter, userClocks, userClocksCounter);
    printf("SPE_%d :: [STATS] :: Executed = %f :: executedClocks = %lld, executedClocksCounter = %lld...\n", (int)getSPEID(), (float)executedClocks / (float)executedClocksCounter, executedClocks, executedClocksCounter);
    printf("SPE_%d :: [STATS] :: Executed List = %f :: executedListClocks = %lld, executedListClocksCounter = %lld...\n", (int)getSPEID(), (float)executedListClocks / (float)executedListClocksCounter, executedListClocks, executedListClocksCounter);
    printf("SPE_%d :: [STATS] :: Committing = %f :: committingClocks = %lld, committingClocksCounter = %lld...\n", (int)getSPEID(), (float)committingClocks / (float)committingClocksCounter, committingClocks, committingClocksCounter);
    printf("SPE_%d :: [STATS] :: Error = %f :: errorClocks = %lld, errorClocksCounter = %lld...\n", (int)getSPEID(), (float)errorClocks / (float)errorClocksCounter, errorClocks, errorClocksCounter);
    printf("SPE_%d :: [STATS] :: In Use (L/Q) = (%f/%f) :: wrInUseLCount = %d, wrInUseQCount = %d, wrInUseCountCounter = %d...\n", (int)getSPEID(), (float)wrInUseLCount / (float)wrInUseCountCounter, (float)wrInUseQCount / (float)wrInUseCountCounter, wrInUseLCount, wrInUseQCount, wrInUseCountCounter);
    printf("SPE_%d :: [STATS] :: States = { 0:%.2f, 1:%.2f, 2:%.2f, 3:%.2f, 4:%.2f, 5:%.2f, 6:%.2f, 7:%.2f, 8:%.2f, 9:%.2f, 10:%.2f, 11:%.2f, 12:%.2f }...\n",
           (int)getSPEID(),
           (float)wrStateCount[ 0] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 1] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 2] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 3] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 4] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 5] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 6] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 7] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 8] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[ 9] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[10] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[11] / (float)stateSampleCounter, //(float)wrInUseCountCounter,
           (float)wrStateCount[12] / (float)stateSampleCounter //(float)wrInUseCountCounter
          );
    printf("SPE_%d :: [STATS] :: stateSampleCounter = %lld\n", (int)getSPEID(), stateSampleCounter);
    printf("SPE_%d :: [STATS] :: Alloc = %f :: wrAllocCount = %d, wrNoAllocCount = %d, wrTryAllocCount = %d\n", (int)getSPEID(), (float)wrAllocCount / (float)wrTryAllocCount, wrAllocCount, wrNoAllocCount, wrTryAllocCount);
    printf("SPE_%d :: [STATS] :: Fetching Pass Rate = %f :: fetchingPassCounter = %lld\n", (int)getSPEID(), (float)fetchingPassCounter / (float)fetchingClocksCounter, fetchingPassCounter);
    printf("SPE_%d :: [STATS] :: Committing Pass Rate = %f :: committingPassCounter = %lld\n", (int)getSPEID(), (float)committingPassCounter / (float)committingClocksCounter, committingPassCounter);
  #endif
}


// STATS
#if SPE_STATS != 0
  long long int schedLoopClocks = 0;
  long long int schedLoopClocksCounter = 0;
#endif

// General scheduler loop wrapper function
void speScheduler_wrapper(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int tagStatus;
  int i;

  // DEBUG
  #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
    int stillAliveCounter = SPE_DEBUG_DISPLAY_STILL_ALIVE;
  #endif

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    printf("SPE_%d :: --==>> Starting SPE Scheduler (Loop)...\n", (int)getSPEID());
  #endif

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Clear out the DMAListEntry array
  //memset((void*)dmaListEntry, 0, sizeof(DMAListEntry) * 2 * SPE_MESSAGE_QUEUE_LENGTH);

  // DEBUG
  #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
    int msgLastStateCount = 0;
    int msgLastState[SPE_MESSAGE_QUEUE_LENGTH];
  #endif

  // TRACE - Clear the trace arrays
  #if ENABLE_TRACE != 0
    for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      STATETRACE_CLEAR(i);
    }
  #endif

  // Initialize the data structures needed by the message queue entries as they progress through states
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    #if DOUBLE_BUFFER_MESSAGE_QUEUE == 0
      msgQueue[i] = (SPEMessage*)(((char*)msgQueueRaw) + (SIZEOF_16(SPEMessage) * i));
    #else
      msgQueue0[i] = (SPEMessage*)(((char*)msgQueueRaw0) + (SIZEOF_16(SPEMessage) * i));
      msgQueue1[i] = (SPEMessage*)(((char*)msgQueueRaw1) + (SIZEOF_16(SPEMessage) * i));
    #endif

    msgState[i] = SPE_MESSAGE_STATE_CLEAR;
    #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
      msgLastState[i] = SPE_MESSAGE_STATE_CLEAR;
    #endif
    readWritePtr[i] = NULL;
    readOnlyPtr[i] = NULL;
    writeOnlyPtr[i] = NULL;
    msgCounter[i] = 0;
    dmaListSize[i] = -1;
    dmaList[i] = NULL;
    localMemPtr[i] = NULL;
    errorCode[i] = SPE_MESSAGE_OK;
  }

  #if DOUBLE_BUFFER_MESSAGE_QUEUE != 0
    msgQueueRaw = msgQueueRaw0;
    msgQueueRawAlt = msgQueueRaw1;
    msgQueue = msgQueue0;
    msgQueueAlt = msgQueue1;
  #endif

  // Once the message queue has been created, check in with the main processor by sending a pointer to it
  spu_write_out_mbox((unsigned int)msgQueueRaw);
  #if SPE_REPORT_END != 0
    spu_write_out_mbox((unsigned int)(&_end));
  #endif

  // Do the intial read of the message queue from main memory
  #if DOUBLE_BUFFER_MESSAGE_QUEUE == 0
    spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);
  #else
    spu_mfcdma32(msgQueueRawAlt, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);
  #endif

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    printf("SPE_%d :: starting scheduler loop...\n", (int)getSPEID());
  #endif

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // The scheduler loop
  while (__builtin_expect(keepLooping != FALSE, 1)) {

    // Check the in mailbox for commands from the PPE
    register int inMBoxCount = spu_stat_in_mbox();
    while (__builtin_expect(inMBoxCount > 0, 0)) {
      int command = spu_read_in_mbox();

      // SPE_MESSAGE_COMMAND_EXIT
      if (command == SPE_MESSAGE_COMMAND_EXIT) {
        keepLooping = FALSE;
      }

      // Reduce the count of remaining entries
      inMBoxCount--;
    }

    // Check the keepLooping flag
    if (__builtin_expect(keepLooping == FALSE, 0))
      continue;

    // Let the user know that the SPE Runtime is still running...
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      if (__builtin_expect(stillAliveCounter == 0, 0)) {
        printf("SPE_%d :: still going... WRs Completed = %d\n", (int)getSPEID(), wrCompletedCounter);
        stillAliveCounter = SPE_DEBUG_DISPLAY_STILL_ALIVE;
      }
      stillAliveCounter--;
    #endif


    // DEBUG
    //static int tagStatusCounter = 0;
    //if (tagStatusCounter > 1000000 && (int)getSPEID() == 0) {
    //  mfc_write_tag_mask(0xFFFFFFFF);
    //  mfc_write_tag_update_immediate();
    //  tagStatus = mfc_read_tag_status();
    //  printf("SPE_%d :: tagStatus = 0x%08x...\n", (int)getSPEID(), tagStatus);
    //  tagStatusCounter = 0;
    //}
    //tagStatusCounter++;


    #if DOUBLE_BUFFER_MESSAGE_QUEUE == 0

      // Wait for the latest message queue read (blocking)
      mfc_write_tag_mask(0x80000000);   // enable only tag group 31 (message queue request)
      mfc_write_tag_update_all();
      mfc_read_tag_status();

    #else

      // Check to see if the latest message queue read has completed (if so, swap the buffers and start the next one)
      mfc_write_tag_mask(0x80000000);   // enable only tag group 31 (message queue request)
      mfc_write_tag_update_immediate();
      tagStatus = mfc_read_tag_status();

      // Message Queue Throttle Counter - Set to 0 to disable or to possitive throttle value
      #define SPE_MESSAGE_QUEUE_THROTTLE_VALUE   61

      #if SPE_MESSAGE_QUEUE_THROTTLE_VALUE > 0

        // Throttle Counter
        static int mqtc = SPE_MESSAGE_QUEUE_THROTTLE_VALUE;
        mqtc--;

        // If finished and there is free DMA queue entry... swap the buffers and read the message queue again
        if (__builtin_expect(mqtc <= 0, 0)) {

      #endif

      if (tagStatus && mfc_stat_cmd_queue() > 0) {

        // Swap the buffers
        volatile char* tmp0 = msgQueueRaw;
        msgQueueRaw = msgQueueRawAlt;
        msgQueueRawAlt = tmp0;
        volatile SPEMessage** tmp1 = msgQueue;
        msgQueue = msgQueueAlt;
        msgQueueAlt = tmp1;

        // Start the next message queue read
        spu_mfcdma32(msgQueueRawAlt, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);

        #if SPE_MESSAGE_QUEUE_THROTTLE_VALUE > 0
          // Reset the message queue throttle counter
          mqtc = SPE_MESSAGE_QUEUE_THROTTLE_VALUE;
        #endif
      }

      #if SPE_MESSAGE_QUEUE_THROTTLE_VALUE > 0
        }
      #endif

    #endif

    // STATS
    #if SPE_STATS != 0
      register unsigned int clocks0;
      stopTimer(clocks0);
      schedLoopClocks += clocks0;
      schedLoopClocksCounter++;
    #endif

    // Call the function containing the scheduling technique
    speSchedulerInner();

    // STATS
    #if SPE_STATS != 0
      startTimer();
    #endif

    // Get the number of remaining DMA queue entries
    register unsigned int numDMAQueueEntries = mfc_stat_cmd_queue();

    #if DOUBLE_BUFFER_MESSAGE_QUEUE == 0
      // Start a new read of the message queue
      if (__builtin_expect(numDMAQueueEntries > 0, 1)) {
        spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);
        numDMAQueueEntries--;
      }
    #endif

    #define SPE_NOTIFY_QUEUE_THROTTLE_VALUE  64

    #if SPE_NOTIFY_QUEUE_THROTTLE_VALUE > 0

      // Throttle Counter
      static int nqtc = SPE_NOTIFY_QUEUE_THROTTLE_VALUE;
      nqtc--;

      if (__builtin_expect(nqtc <= 0, 0)) {

    #endif

    // If there isn't a pending DMA-Put of the notifcation queue, start one
    mfc_write_tag_mask(0x40000000); // only enable tag group 30 (notifcation queue request)
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status();
    if (tagStatus && numDMAQueueEntries > 0)
      spu_mfcdma32(notifyQueueRaw, (PPU_POINTER_TYPE)(speData->notifyQueue), SPE_NOTIFY_QUEUE_BYTE_COUNT, 30, MFC_PUT_CMD);

    #if SPE_NOTIFY_QUEUE_THROTTLE_VALUE > 0

        nqtc = SPE_NOTIFY_QUEUE_THROTTLE_VALUE;
      }

    #endif

    // TRACE
    #if ENABLE_TRACE != 0
      STATETRACE_UPDATE_ALL
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      debug_displayActiveMessageQueue(id, msgState, "(B)");
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      debug_displayStateHistogram(id, msgState, "");
    #endif

    // Check to see if there has been no progress in the message's states
    #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
      int msgLastStateFlag = 0;
      for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
        if (msgState[i] != msgLastState[i]) {
          msgLastState[i] = msgState[i];
          msgLastStateFlag = 1;  // Something changed
	}
      }

      if (msgLastStateFlag != 0) {  // if something changed
        msgLastStateCount = 0;
      } else {                      // otherwise, nothing has changed
        msgLastStateCount++;
        if (msgLastStateCount >= SPE_DEBUG_DISPLAY_NO_PROGRESS) {
          for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
            sim_printf("[0x%llx] :: msgState[%d] = %d\n", id, i, msgState[i]);
	  }
          msgLastStateCount = 0; // reset the count
	}
      }
    #endif

    // STATS
    #if SPE_STATS != 0
      statData.schedulerLoopCount++;
    #endif

  } // end while (keepLooping != FALSE)

  // STATS
  #if SPE_STATS != 0
    register unsigned int clocks0;
    stopTimer(clocks0);
    schedLoopClocks += clocks0;
    schedLoopClocksCounter++;
  #endif
}


int main(unsigned long long id, unsigned long long param) {

  /*volatile*/ SPEData myData;
  void* _heapPtr = NULL;
  #if SPE_DEBUG_DISPLAY >= 1
    void* _breakBefore = NULL;
    void* _breakAfter = NULL;
  #endif

  // Tell the world this SPE is alive
  #if SPE_DEBUG_DISPLAY >= 1
    printf(" --==>> Hello From SPE_%d's Runtime <<==--\n", (int)getSPEID());
    printf("SPE_%d :: [INFO] :: SPE_MESSAGE_QUEUE_LENGTH = %d\n", (int)getSPEID(), SPE_MESSAGE_QUEUE_LENGTH);
    printf("SPE_%d :: [INFO] :: SPE_MESSAGE_QUEUE_BYTE_COUNT = %d\n", (int)getSPEID(), SPE_MESSAGE_QUEUE_BYTE_COUNT);
  #endif

  // Call the user's funcLookup() function with funcIndex of SPE_FUNC_INDEX_INIT
  funcLookup(SPE_FUNC_INDEX_INIT, NULL, 0, NULL, 0, NULL, 0, NULL);

  #if SPE_USE_OWN_MALLOC <= 0

    // From Section 4.3 of library_SDK.pdf : "The local store memory heap is initialized the first
    //   time a memory heap allocation routine is called."... Do this now so it is ready to go.
    register unsigned int memLeft = SPE_TOTAL_MEMORY_SIZE - SPE_RESERVED_STACK_SIZE - (unsigned int)(&_end);
    memLeft -= 128;  // Buffer zone between stack and heap
    memLeft &= 0xFFFFFF00;  // Force it to be a multiple of 256 bytes
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: memLeft = %d\n", (int)getSPEID(), memLeft);
    #endif
    if (memLeft < SPE_MINIMUM_HEAP_SIZE) return -1;
    #if SPE_DEBUG_DISPLAY >= 1
      _breakBefore = sbrk(0);
    #endif
    _heapPtr = sbrk((ptrdiff_t)memLeft);
    #if SPE_DEBUG_DISPLAY >= 1
      _breakAfter = sbrk(0);
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: _end = %p, _breakBefore = %p, _heapPtr = %p, _breakAfter = %p (%p)\n",
             (int)getSPEID(), &_end, _breakBefore, _heapPtr, _breakAfter, sbrk(0)
            );
      //printf("SPE_%d :: _end = %p, _heapPtr = %p\n", (int)getSPEID(), &_end, _heapPtr);
    #endif

  #else

    initMem();

  #endif

  // Initialize the message queue to zeros
  #if DOUBLE_BUFFER_MESSAGE_QUEUE == 0
    memset((void*)msgQueueRaw, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
  #else
    memset((void*)msgQueueRaw0, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
    memset((void*)msgQueueRaw1, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
  #endif

  // Initialize the notify queue to zeros
  #if SPE_NOTIFY_VIA_MAILBOX == 0
    memset((void*)notifyQueueRaw, 0x00, SPE_NOTIFY_QUEUE_BYTE_COUNT);
  #endif

  // STATS
  #if SPE_STATS != 0
    memset(&statData, 0, sizeof(statData));
  #endif

  // Read in the data from main storage
  spu_mfcdma32((void*)&myData,          // LS Pointer
               (unsigned int)param,     // Main-Storage Pointer
               SIZEOF_16(SPEData),      // Number of bytes to copy
               0,                       // Tag ID
               MFC_GET_CMD              // DMA Command
	      );

  // Wait for all transfers to complete.  See "SPU C/C++ Language Extentions", page 64 for details.
  //spu_mfcstat(2);  // Blocks for all outstanding DMA Tags to complete  // <<<===---  !!! This does not seem to work (TODO : test and let IBM know) !!!
  mfc_write_tag_mask(0xFFFFFFFF);
  mfc_write_tag_update_all();
  mfc_read_tag_status();

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("SPE :: myData = { mq = %u, mql = %d, vID = %d }\n", myData.messageQueue, myData.messageQueueLength, myData.vID);
  #endif

  // Set the local vID
  vID = myData.vID;

  // Entry into the SPE's scheduler
  #if 0
    speScheduler(&myData, id);
  #else
    speScheduler_wrapper(&myData, id);
  #endif

  // Display the stats
  #if SPE_STATS != 0
    printf("SPE_%d :: [STATS] :: loopCount = %llu, #WRs = %llu...\n",
           (int)getSPEID(), statData.schedulerLoopCount, statData.numWorkRequestsExecuted
          );
    printf("SPE_%d :: [STATS] :: Sched. Loop = %f :: schedLoopClocks = %lld, schedLoopClocksCounter = %lld...\n", (int)getSPEID(), (float)schedLoopClocks / (float)schedLoopClocksCounter, schedLoopClocks, schedLoopClocksCounter);
    displayStatsData();
  #endif

  // Tell the world this SPE is going away
  #if SPE_DEBUG_DISPLAY >= 1
    printf(" --==>> Goodbye From SPE 0x%llx's Runtime <<==--\n", id);
    printf("  \"I do not regret the things I have done, but those I did not do.\" - Lucas, Empire Records\n");
  #endif

  // Call the user's funcLookup() function with funcIndex of SPE_FUNC_INDEX_CLOSE
  funcLookup(SPE_FUNC_INDEX_CLOSE, NULL, 0, NULL, 0, NULL, 0, NULL);

  return 0;
}


#if 0

void speScheduler(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int runIndex = 0;
  int getIndex = 0;
  int putIndex = 0;
  int commitIndex = 0;
  int tagStatus;
  unsigned int numDMAQueueEntries = 0;
  int i, j, iOffset;

  // DEBUG
  #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
    int stillAliveCounter = SPE_DEBUG_DISPLAY_STILL_ALIVE;
  #endif

  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] --==>> Starting SPE Scheduler ...\n", id);
  #endif

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Clear out the DMAListEntry array
  memset((void*)dmaListEntry, 0, sizeof(DMAListEntry) * 2 * SPE_MESSAGE_QUEUE_LENGTH);

  // Create the local message queue
  #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
    int msgLastStateCount = 0;
    int msgLastState[SPE_MESSAGE_QUEUE_LENGTH];
  #endif


  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    STATETRACE_CLEAR(i);
  }


  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    msgQueue[i] = (SPEMessage*)(((char*)msgQueueRaw) + (SIZEOF_16(SPEMessage) * i));
    msgState[i] = SPE_MESSAGE_STATE_CLEAR;
    #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
      msgLastState[i] = SPE_MESSAGE_STATE_CLEAR;
    #endif
    readWritePtr[i] = NULL;
    readOnlyPtr[i] = NULL;
    writeOnlyPtr[i] = NULL;
    msgCounter[i] = 0;
    dmaListSize[i] = -1;
    dmaList[i] = NULL;
    localMemPtr[i] = NULL;
    errorCode[i] = SPE_MESSAGE_OK;
  }

  // Once the message queue has been created, check in with the main processor by sending a pointer to it
  spu_write_out_mbox((unsigned int)msgQueueRaw);
  #if SPE_REPORT_END != 0
    spu_write_out_mbox((unsigned int)(&_end));
  #endif


  // DEBUG
  //__asm__ ("dsync");

  // Do the intial read of the message queue from main memory
  spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] :: starting scheduler loop...\n", id);
  #endif

  // The scheduler loop
  while (__builtin_expect(keepLooping != FALSE, 1)) {

    // Check the in mailbox for commands from the PPE
    register int inMBoxCount = spu_stat_in_mbox();
    while (__builtin_expect(inMBoxCount > 0, 0)) {
      int command = spu_read_in_mbox();
      if (command == SPE_MESSAGE_COMMAND_EXIT) {
        keepLooping = FALSE;
      }

      inMBoxCount--;
    }
    if (__builtin_expect(keepLooping == FALSE, 0)) continue;


    // DEBUG
    register int displayMessageQueueRead = 0;
    for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgQueue[i]->traceFlag != 0) {
        displayMessageQueueRead = 1;
        break;
      }
    }
    if (displayMessageQueueRead) {
      printf("SPE_%d :: Reading Message Queue...\n", (int)getSPEID());
    }


    // Let the user know that the SPE Runtime is still running...
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      if (__builtin_expect(stillAliveCounter == 0, 0)) {
        printf("[0x%llx] :: SPE_%d :: still going...\n", id, (int)getSPEID());
        stillAliveCounter = SPE_DEBUG_DISPLAY_STILL_ALIVE;
      }
      stillAliveCounter--;
    #endif


    // Wait for the latest message queue read (blocking)
    mfc_write_tag_mask(0x80000000);   // enable only tag group 31 (message queue request)
    #if 0
      mfc_write_tag_update_any();
    #else
      mfc_write_tag_update_all();
    #endif
    tagStatus = mfc_read_tag_status();
    //mfc_write_tag_mask(0x7FFFFFFF);   // enable all tag groups except 31


    // DEBUG
    //__asm__ ("dsync");


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(0)");
    #endif


    // DEBUG
    //__asm__ ("dsync");


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_SENT

    // Check for new messages
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {

      // Check for a new message in this slot
      // Conditions... 1) msgQueue[i]->counter0 == msgQueue[i]->counter1  // Entire entry has been written by PPE
      //               2) msgQueue[i]->counter1 != msgCounter[i]          // Entry has not already been processed
      //               3) msgQueue[i]->state == SPE_MESSAGE_STATE_SENT    // PPE wrote an entry and is waiting for the result
      //               4) msgState[i] == SPE_MESSAGE_STATE_CLEAR          // SPE isn't currently processing this slot
      if (__builtin_expect((msgQueue[i]->counter0 == msgQueue[i]->counter1) &&
                           (msgQueue[i]->counter1 != msgCounter[i]) &&
                           (msgQueue[i]->state == SPE_MESSAGE_STATE_SENT) &&
                           (msgState[i] == SPE_MESSAGE_STATE_CLEAR),
                           0
                          )
         ) {

        STATETRACE_CLEAR(i);

        // Start by checking the command
        int command = msgQueue[i]->command;
        if (__builtin_expect(command == SPE_MESSAGE_COMMAND_EXIT, 0)) {
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf(" --==>> SPE received EXIT command...\n");
          #endif
          keepLooping = FALSE;
          break;
        }

        // Update the state of the message (locally)
        if ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00)
          msgState[i] = SPE_MESSAGE_STATE_PRE_FETCHING;
        else
          msgState[i] = SPE_MESSAGE_STATE_PRE_FETCHING_LIST;
      
        // DEBUG
        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_SENT, msgState[i]);
	#endif

        msgCounter[i] = msgQueue[i]->counter1;

        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Tracing entry at index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}
      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(1)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_PRE_FETCHING_LIST

    // Check for messages that need data fetched (list)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING_LIST, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_PRE_FETCHING_LIST for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Ckeck the size of the dmaList.  If it is less than SPE_DMA_LIST_LENGTH then it will fit
        //   in the preallocated area reserved for lists.  Otherwise, malloc memory to receive the
        //   dma list from main memory.
        if (dmaListSize[i] < 0) {

          dmaListSize[i] = msgQueue[i]->readOnlyLen + msgQueue[i]->readWriteLen + msgQueue[i]->writeOnlyLen;
          register int memNeeded = ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry));

          if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
            #ifdef __cplusplus
            try {
            #endif
              //dmaList[i] = (DMAListEntry*)(new char[memNeeded]);
              dmaList[i] = (DMAListEntry*)(_malloc_align(memNeeded, 4));
            #ifdef __cplusplus
	    } catch (...) {
              dmaList[i] = NULL;
	    }
            #endif

	    // DEBUG
            #if SPE_DEBUG_DISPLAY >= 1
              sim_printf("[0x%llx] :: dmaList[%d] = %p\n", id, i, dmaList[i]);
            #endif

            //if (dmaList[i] != NULL) { delete [] dmaList[i]; }
	    if (__builtin_expect((dmaList[i] == NULL) || 
                                 (((unsigned int)dmaList[i]) < ((unsigned int)(&_end))) ||
                                 (((unsigned int)dmaList[i] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                                 0
                                )
               ) {
              #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
                sim_printf("[0x%llu] :: SPE :: Failed to allocate memory for dmaList[%d] (1)... will try again later...\n", id, i);
                sim_printf("[0x%llx] :: SPE :: dmaList[%d] = %p\n", id, i, dmaList[i]);
	      #endif

              dmaList[i] = NULL;
              dmaListSize[i] = -1;  // Try allocating again next time

              // DEBUG
              //print_block_table();

              continue;  // Skip for now, try again later
            }
          } else {
            dmaList[i] = (DMAListEntry*)(&(dmaListEntry[i * SPE_DMA_LIST_LENGTH]));
          }

          // Zero-out the dmaList
          memset(dmaList[i], 0, memNeeded);
	}

        // Intiate the DMA transfer for the DMA list into dmaList
        if (numDMAQueueEntries > 0 && dmaListSize[i] > 0) {

          spu_mfcdma32(dmaList[i],
                       (unsigned int)(msgQueue[i]->readWritePtr),
                       ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry)),
                       i,
                       MFC_GET_CMD
		      );

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_FETCHING_LIST;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_PRE_FETCHING_LIST, msgState[i]);
          #endif
	}

      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(2)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_FETCHING_LIST

    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_mask(0x7FFFFFFF);
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING_LIST && ((tagStatus & (0x01 << i)) != 0), 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_FETCHING_LIST for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Update the state to show that this message queue entry is ready to be executed
        msgState[i] = SPE_MESSAGE_STATE_LIST_READY_LIST;

        // DEBUG
        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_FETCHING_LIST, msgState[i]);
        #endif

        // Roundup all of the sizes to the next highest multiple of 16
        for (j = 0; j < dmaListSize[i]; j++)
          dmaList[i][j].size = ROUNDUP_16(dmaList[i][j].size);
      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(3)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_LIST_READY_LIST

    // Check for messages that need data fetched (standard)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_LIST_READY_LIST, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_LIST_READY_LIST for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Allocate the memory needed in the LS for this work request
        if (localMemPtr[i] == NULL) {

          // NOTE :Format: The allocated memory will contain a list of pointers (to the buffers) along
          //   with the memory for the buffers themselves following the list (in order).

          // Determine the number of bytes needed
          register unsigned int memNeeded = 0;
          for (j = 0; j < dmaListSize[i]; j++)
            memNeeded += dmaList[i][j].size + sizeof(DMAListEntry);
          if ((dmaListSize[i] & 0x01) != 0x00)  // Force even number of dmaListEntry structures
            memNeeded += sizeof(DMAListEntry);

          // Check the size of the memory needed.  If it is too large for the SPE's LS, then stop this
          //   message with an error code because the memory allocation will never work.
          // TODO : Should also add a define that guesses at what the expected maximum stack size is (that
          //   could also be changed at compile-time by the end user)
          if (__builtin_expect(memNeeded > ((unsigned int)0x40000 - ((unsigned int)(&_end))), 0)) {

            // Clear up the dmaList
            if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
              if (dmaList[i] != NULL) {
                //delete [] dmaList[i];
	        _free_align(dmaList[i]);
	      }
              dmaList[i] = NULL;
              dmaListSize[i] = -1;
	    }

            // Move the message into an error state
            errorCode[i] = SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY;
            msgState[i] = SPE_MESSAGE_STATE_ERROR;

            // Move onto the next message
            continue;
	  }

          // Try to allocate that memory
          #ifdef __cplusplus
          try {
          #endif
            //localMemPtr[i] = (void*)(new char[memNeeded]);
            localMemPtr[i] = (void*)(_malloc_align(memNeeded, 4));
          #ifdef __cplusplus
	  } catch (...) {
            localMemPtr[i] = NULL;
	  }
          #endif

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: localMemPtr[%d] = %p\n", id, i, localMemPtr[i]);
          #endif

	  //if (localMemPtr[i] == NULL || ((unsigned int)localMemPtr[i]) + memNeeded >= (unsigned int)0x40000) {
	  if (__builtin_expect((localMemPtr[i] == NULL) || 
                               (((unsigned int)localMemPtr[i]) < ((unsigned int)(&_end))) ||
                               (((unsigned int)localMemPtr[i] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                               0
                              )
             ) {
            #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	      sim_printf("[0x%llu] :: SPE :: Failed to allocate memory for localMemPtr[%d] (1)... will try again later...\n", id, i);
              sim_printf("[0x%llx] :: SPE :: localMemPtr[%d] = %p\n", id, i, localMemPtr[i]);
	    #endif

	    localMemPtr[i] = NULL;

            // DEBUG
            //print_block_table();

            continue;  // Try again next time
	  }


          // Setup pointers to the buffers
          register unsigned int offset = ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry));
          for (j = 0; j < dmaListSize[i]; j++) {
            ((DMAListEntry*)(localMemPtr[i]))[j].size = ((dmaList[i][j].size) & (0x0000FFFF));  // Force notify and reserved fields to zero
            ((DMAListEntry*)(localMemPtr[i]))[j].ea = (unsigned int)(((char*)(localMemPtr[i])) + offset);
            offset += dmaList[i][j].size;
	  }

          // Zero the memory if needed
          #if SPE_ZERO_WRITE_ONLY_MEMORY != 0
            if (msgQueue[i]->writeOnlyLen > 0) {
	      register unsigned int writeSize = 0;
              for (j = dmaListSize[i] - msgQueue[i]->writeOnlyLen; j < dmaListSize[i]; j++)
                writeSize += dmaList[i][j].size;
              memset(((char*)(localMemPtr[i])) + memNeeded - writeSize, 0, writeSize);
	    }
          #endif

	  // Setup the list pointers for the buffer types (make sure they are NULL)
	  readOnlyPtr[i] = NULL;
          readWritePtr[i] = NULL;
          writeOnlyPtr[i] = NULL;
	}

        // Start the DMA transaction that will grab the data to be read from main memory
        if (numDMAQueueEntries > 0 &&
            (msgQueue[i]->readOnlyLen + msgQueue[i]->readWriteLen) > 0 &&
            localMemPtr[i] != NULL
           ) {

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: Pre-GETL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
                       id, i, ((char*)localMemPtr[i]) + ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry)),
                       (unsigned int)msgQueue[i]->readOnlyPtr,
                       (unsigned int)(dmaList[i]),
                       (msgQueue[i]->readOnlyLen + msgQueue[i]->readWriteLen) * sizeof(DMAListEntry), i
                      );

            for (j = 0; j < dmaListSize[i]; j++) {
              sim_printf("[0x%llx] :: dmaList[%d][%d].size = %d (0x%x), dmaList[%d][%d].ea = 0x%08x (0x%08x)\n",
                         id,
                         i, j, dmaList[i][j].size, ((DMAListEntry*)(localMemPtr[i]))[j].size,
                         i, j, dmaList[i][j].ea, ((DMAListEntry*)(localMemPtr[i]))[j].ea
	                );
            }
	  #endif

	  spu_mfcdma64(((char*)localMemPtr[i]) + ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry)),
                       (unsigned int)msgQueue[i]->readOnlyPtr,  // eah
                       (unsigned int)(dmaList[i]),
                       (msgQueue[i]->readOnlyLen + msgQueue[i]->readWriteLen) * sizeof(DMAListEntry),
                       i,
                       MFC_GETL_CMD
	              );

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_FETCHING;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_LIST_READY_LIST, msgState[i]);
          #endif
	}

      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(4)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_PRE_FETCHING

    // Check for messages that need data fetched (standard)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    int newGetIndex = (getIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    int numGetsLeft = SPE_MAX_GET_PER_LOOP;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (getIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_PRE_FETCHING for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Allocate the memory for the message queue entry (if need be)
        // NOTE: First check to see if it is non-null.  What might have happened was that there was enough memory
        //   last time but the DMA queue was full and a retry was needed.  So this time, the memory is there and
        //   only the DMA needs to be retried.
        if (localMemPtr[i] == NULL) {

          // Allocate the memory and place a pointer to the allocated buffer into localMemPtr.  This buffer will
          //   be divided up into three regions: readOnly, readWrite, write (IN THAT ORDER!; the regions may be
          //   empty if no pointer for the region was supplied by the original sendWorkRequest() call on the PPU).
          register int memNeeded = ROUNDUP_16(msgQueue[i]->readWriteLen) +
                                   ROUNDUP_16(msgQueue[i]->readOnlyLen) +
                                   ROUNDUP_16(msgQueue[i]->writeOnlyLen);
          register int offset = 0;

          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: Allocating Memory for index %d :: memNeeded = %d, msgQueue[%d].totalMem = %d\n",
                       id, i, memNeeded, i, msgQueue[i]->totalMem
                      );
	  #endif

          // Check the size of the memory needed.  If it is too large for the SPE's LS, then stop this
          //   message with an error code because the memory allocation will never work.
          // TODO : Should also add a define that guesses at what the expected maximum stack size is (that
          //   could also be changed at compile-time by the end user)
          if (__builtin_expect(memNeeded > ((int)0x40000 - ((int)(&_end))), 0)) {

            // Move the message into an error state
            errorCode[i] = SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY;
            msgState[i] = SPE_MESSAGE_STATE_ERROR;

            // Move onto the next message
            continue;
	  }

          #ifdef __cplusplus
          try {
          #endif
            //localMemPtr[i] = (void*)(new char[memNeeded]);
            localMemPtr[i] = (void*)(_malloc_align(memNeeded, 4));
          #ifdef __cplusplus
	  } catch (...) {
            localMemPtr[i] = NULL;
	  }
          #endif

          // Check the pointer (if it is bad, then skip this message for now and try again later)
          // TODO: There are probably better checks for this (use _end, etc.)
          //if (localMemPtr[i] == NULL || ((unsigned int)localMemPtr[i] + memNeeded) >= (unsigned int)0x40000) {
          if (__builtin_expect((localMemPtr[i] == NULL) || 
                               (((unsigned int)localMemPtr[i]) < ((unsigned int)(&_end))) ||
                               (((unsigned int)localMemPtr[i] + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                               0
                              )
             ) {
            #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	      sim_printf("[0x%llu] :: SPE :: Failed to allocate memory for localMemPtr[%d] (2)... will try again later...\n", id, i);
              sim_printf("[0x%llx] :: SPE :: localMemPtr[%d] = %p\n", id, i, localMemPtr[i]);
	    #endif

            localMemPtr[i] = NULL;

            // DEBUG
            //print_block_table();

            continue;
	  }

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: localMemPtr[%d] = %p\n", id, i, localMemPtr[i]);
	  #endif

          // Assign the buffer specific pointers
          // NOTE: Order matters here.  Need to allocate the read buffers next to each other and the write
          //   buffers next to each other
          if (msgQueue[i]->readOnlyPtr != (PPU_POINTER_TYPE)NULL) {
            readOnlyPtr[i] = localMemPtr[i];
            offset += ROUNDUP_16(msgQueue[i]->readOnlyLen);
          } else {
            readOnlyPtr[i] = NULL;
	  }

          if (msgQueue[i]->readWritePtr != (PPU_POINTER_TYPE)NULL) {
            readWritePtr[i] = (void*)((char*)localMemPtr[i] + offset);
            offset += ROUNDUP_16(msgQueue[i]->readWriteLen);
          } else {
            readWritePtr[i] = NULL;
	  }

          if (msgQueue[i]->writeOnlyPtr != (PPU_POINTER_TYPE)NULL) {
            writeOnlyPtr[i] = (void*)((char*)localMemPtr[i] + offset);
            #if SPE_ZERO_WRITE_ONLY_MEMORY != 0
              memset(writeOnlyPtr[i], 0, ROUNDUP_16(msgQueue[i]->writeOnlyLen));
            #endif
          } else {
            writeOnlyPtr[i] = NULL;
	  }
	}

        // Check to see if this message does not needs to fetch any data
        if (((msgQueue[i]->readWritePtr == (PPU_POINTER_TYPE)NULL)
             || ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == WORK_REQUEST_FLAGS_RW_IS_WO)
	    )
            && (msgQueue[i]->readOnlyPtr == (PPU_POINTER_TYPE)NULL)
           ) {

          // Update the state (to ready to execute)
          msgState[i] = SPE_MESSAGE_STATE_READY;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_PRE_FETCHING, msgState[i]);
          #endif

          // Done with this one
          continue;
	}

        // Create the DMA list
        // NOTE: Check to see if the dma list has already been created yet or not (if dmaListSize[i] < 0, not created)
        if (dmaListSize[i] < 0) {

          // Count the number of DMA entries needed for the read DMA list
          register int entryCount = 0;
          if ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == 0x00)
            entryCount += (ROUNDUP_16(msgQueue[i]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                          (((msgQueue[i]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
          entryCount += (ROUNDUP_16(msgQueue[i]->readOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->readOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

          // Allocate a larger DMA list if needed
          if (entryCount > SPE_DMA_LIST_LENGTH) {
            #ifdef __cplusplus
            try {
            #endif
              //dmaList[i] = new DMAListEntry[entryCount];
              dmaList[i] = (DMAListEntry*)(_malloc_align(entryCount * sizeof(DMAListEntry), 4));
            #ifdef __cplusplus
	    } catch (...) {
              dmaList[i] = NULL;
	    }
            #endif

	    // DEBUG
            #if SPE_DEBUG_DISPLAY >= 1
              sim_printf("[0x%llx] :: dmaList[%d] = %p\n", id, i, dmaList[i]);
            #endif

	    //if (dmaList[i] == NULL || ((unsigned int)dmaList[i] + (entryCount * sizeof(DMAListEntry))) >= (unsigned int)0x40000) {
            if (__builtin_expect((dmaList[i] == NULL) || 
                                 (((unsigned int)dmaList[i]) < ((unsigned int)(&_end))) ||
                                 (((unsigned int)dmaList[i] + (entryCount * sizeof(DMAListEntry))) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                                 0
				)
               ) {
              #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	        sim_printf("[0x%llu] :: SPE :: Failed to allocate memory for dmaList[%d] (2)... will try again later...\n", id, i);
                sim_printf("[0x%llx] :: SPE :: dmaList[%d] = %p\n", id, i, dmaList[i]);
	      #endif

              dmaList[i] = NULL;

              // DEBUG
              //print_block_table();

              continue;  // Skip for now, try again later
	    }
            memset(dmaList[i], 0, sizeof(DMAListEntry) * entryCount);
	  } else {
            dmaList[i] = (DMAListEntry*)(&(dmaListEntry[i * SPE_DMA_LIST_LENGTH]));
	  }
          dmaListSize[i] = entryCount;

          // Fill in the list
          register int listIndex = 0;

          if (readOnlyPtr[i] != NULL) {
            register int bufferLeft = msgQueue[i]->readOnlyLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->readOnlyPtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
              dmaList[i][listIndex].size = ROUNDUP_16(dmaList[i][listIndex].size);
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }
	  }

          if (((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_WO) == 0x00) && (readWritePtr[i] != NULL)) {
            register int bufferLeft = msgQueue[i]->readWriteLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->readWritePtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
              dmaList[i][listIndex].size = ROUNDUP_16(dmaList[i][listIndex].size);
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }
	  }
	}

        // Initiate the DMA command
        if (numDMAQueueEntries > 0 && dmaListSize[i] > 0) {

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: Pre-GETL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
                   id, i, localMemPtr[i], 0, (unsigned int)(dmaList[i]), dmaListSize[i] * sizeof(DMAListEntry), i
                  );

            for (j = 0; j < dmaListSize[i]; j++) {
              sim_printf("[0x%llx] :: dmaList[%d][%d].size = %d (0x%x), dmaList[%d][%d].ea = 0x%08x (0x%08x)\n",
                         id,
                         i, j, dmaList[i][j].size, ((DMAListEntry*)(localMemPtr[i]))[j].size,
                         i, j, dmaList[i][j].ea, ((DMAListEntry*)(localMemPtr[i]))[j].ea
		        );
	    }
	  #endif

          spu_mfcdma64(localMemPtr[i],
                       0,
                       (unsigned int)(dmaList[i]),
                       dmaListSize[i] * sizeof(DMAListEntry),
                       i,
                       MFC_GETL_CMD
		      );

          // Update the getIndex so it will point to the index directly after the last index that was able
          //   to get the GET through.
          newGetIndex = (i + 1) % SPE_MESSAGE_QUEUE_LENGTH;

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_FETCHING;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_PRE_FETCHING, msgState[i]);
          #endif

	  // Check to see if this maxs out the number of GETs allowed each scheduler loop iteration
	  numGetsLeft--;
          if (numGetsLeft <= 0)
            break;
	}

      }
    }
    getIndex = newGetIndex;

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(5)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_FETCHING

    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_mask(0x7FFFFFFF);
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING && ((tagStatus & (0x01 << i)) != 0), 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_FETCHING for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Update the state to show that this message queue entry is ready to be executed
        msgState[i] = SPE_MESSAGE_STATE_READY;

        // DEBUG
        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_FETCHING, msgState[i]);
        #endif

        // Clean up the dmaList (only if this is NOT a list type work request; if it is a
        //   list type work request (flag has LIST bit set) then do not delete the dma list)
        if ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {
          if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
            //delete [] dmaList[i];
            _free_align(dmaList[i]);
            dmaList[i] = NULL;
	  }
          dmaListSize[i] = -1;  // NOTE: Clear this so data that the dmaList looks like it has not been set now
	}

      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(6)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_READY

    // Execute SPE_MAX_EXECUTE_PER_LOOP ready messages
    register unsigned int numExecLeft = SPE_MAX_EXECUTE_PER_LOOP;
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {

      if (__builtin_expect(msgState[runIndex] == SPE_MESSAGE_STATE_READY, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[runIndex]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_READY for index %d...\n", (int)getSPEID(), runIndex);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        register volatile SPEMessage* msg = msgQueue[runIndex];

        #if SPE_DEBUG_DISPLAY >= 1
	  if (msgQueue[runIndex]->traceFlag)
            printf("SPE_%d :: >>>>> Entering User Code (index %d)...\n", (int)getSPEID(), runIndex);
	#endif


        // DEBUG
        isTracingFlag = ((msg->traceFlag) ? (-1) : (0));


        // Execute the function specified
        if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {

          #if SPE_DEBUG_DISPLAY >= 1
	    if (msgQueue[runIndex]->traceFlag)
	      printf("SPE_%d :: Executing message queue entry as standard entry... fi = %d...\n", (int)getSPEID(), msg->funcIndex);
	  #endif

          funcLookup(msg->funcIndex,
                     readWritePtr[runIndex], msg->readWriteLen,
                     readOnlyPtr[runIndex], msg->readOnlyLen,
                     writeOnlyPtr[runIndex], msg->writeOnlyLen,
                     NULL
                    );
	} else {

          #if SPE_DEBUG_DISPLAY >= 1
	    if (msgQueue[runIndex]->traceFlag)
	      printf("SPE_%d :: Executing message queue entry as list entry... fi = %d...\n", (int)getSPEID(), msg->funcIndex);
	  #endif

          funcLookup(msg->funcIndex,
                     NULL, msg->readWriteLen,
                     NULL, msg->readOnlyLen,
                     NULL, msg->writeOnlyLen,
                     (DMAListEntry*)(localMemPtr[runIndex])
                    );
	}

        #if SPE_DEBUG_DISPLAY >= 1
	  if (msgQueue[runIndex]->traceFlag)
            printf("SPE_%d :: <<<<< Leaving User Code...\n", (int)getSPEID());
	#endif

        #if SPE_STATS != 0
          statData.numWorkRequestsExecuted++;
        #endif

        // Update the state of the message queue entry
        if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == 0x00)
          msgState[runIndex] = SPE_MESSAGE_STATE_EXECUTED;
        else
          msgState[runIndex] = SPE_MESSAGE_STATE_EXECUTED_LIST;

        // DEBUG
        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_READY, msgState[i]);
        #endif

        // Move runIndex to the next message and break from the loop (execute only one)
        runIndex++;
        if (runIndex >= SPE_MESSAGE_QUEUE_LENGTH)
          runIndex = 0;

        // Decrement the count of work request that can still be execute this scheduler loop iteration
        //   and break from this execute loop if the maximum number has already been reached.
        numExecLeft--;
        if (numExecLeft <= 0)
          break;
      }

      // Try the next message
      runIndex++;
      if (runIndex >= SPE_MESSAGE_QUEUE_LENGTH)
        runIndex = 0;
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(7)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_EXECUTED_LIST

    // Check for messages that have been executed but still need data committed to main memory
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED_LIST, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_EXECUTED_LIST for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Initiate the DMA transfer of the readWrite and writeOnly buffer back to main memory
        if ((msgQueue[i]->readWriteLen + msgQueue[i]->writeOnlyLen) > 0 && localMemPtr[i] != NULL) {

          if (numDMAQueueEntries > 0) {

            // Get the offsets in the dmaList and the localMemPtr for the readWrite section
            register unsigned int readWriteOffset = ROUNDUP_16(dmaListSize[i] * sizeof(DMAListEntry));
            for (j = 0; j < msgQueue[i]->readOnlyLen; j++)
              readWriteOffset += dmaList[i][j].size;

            // DEBUG
            #if SPE_DEBUG_DISPLAY >= 1
              sim_printf("[0x%llx] :: Pre-PUTL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
                         id, i, ((char*)localMemPtr[i]) + (readWriteOffset),
                         (unsigned int)msgQueue[i]->readOnlyPtr,
                         (unsigned int)(&(dmaList[i][msgQueue[i]->readOnlyLen])),
                         (msgQueue[i]->readWriteLen + msgQueue[i]->writeOnlyLen) * sizeof(DMAListEntry), i
                        );
	    #endif

            spu_mfcdma64(((char*)localMemPtr[i]) + (readWriteOffset),
                         (unsigned int)msgQueue[i]->readOnlyPtr,  // eah
                         (unsigned int)(&(dmaList[i][msgQueue[i]->readOnlyLen])),
                         (msgQueue[i]->readWriteLen + msgQueue[i]->writeOnlyLen) * sizeof(DMAListEntry),
                         i,
                         MFC_PUTL_CMD
	                );

            // Decrement the counter of available DMA queue entries left
            numDMAQueueEntries--;

            // Update the state of the message queue entry now that the data should be in-flight
            msgState[i] = SPE_MESSAGE_STATE_COMMITTING;

            // DEBUG
            #if SPE_DEBUG_DISPLAY >= 1
              sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_EXECUTED_LIST, msgState[i]);
            #endif
          }

	} else {

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_EXECUTED_LIST, msgState[i]);
          #endif
	}

      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(8)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_EXECUTED

    // Check for messages that have been executed but still need data committed to main memory
    numDMAQueueEntries = mfc_stat_cmd_queue();
    int newPutIndex = (putIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    int numPutsLeft = SPE_MAX_PUT_PER_LOOP;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (putIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_EXECUTED for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Check to see if this message does not need to fetch any data
        if (((msgQueue[i]->readWritePtr == (PPU_POINTER_TYPE)NULL)
             || ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == WORK_REQUEST_FLAGS_RW_IS_RO)
            )
            && (msgQueue[i]->writeOnlyPtr == (PPU_POINTER_TYPE)NULL)
           ) {

          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;  // The index still needs to be passed back to the PPU

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_EXECUTED, msgState[i]);
          #endif

          continue;
	}

        // Create the DMA list
        // NOTE: Check to see if the dma list has already been created yet or not (if dmaListSize[i] < 0, not created)
        if (dmaListSize[i] < 0) {

          // Count the number of DMA entries needed for the read DMA list
          register int entryCount = 0;
          if ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == 0x00)
            entryCount += (ROUNDUP_16(msgQueue[i]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                          (((msgQueue[i]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
          entryCount += (ROUNDUP_16(msgQueue[i]->writeOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->writeOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

          // Allocate a larger DMA list if needed
          if (entryCount > SPE_DMA_LIST_LENGTH) {
            #ifdef __cplusplus
            try {
            #endif
              //dmaList[i] = new DMAListEntry[entryCount];
              dmaList[i] = (DMAListEntry*)(_malloc_align(entryCount * sizeof(DMAListEntry), 4));
            #ifdef __cplusplus
	    } catch (...) {
              dmaList[i] = NULL;
	    }
            #endif

	    // DEBUG
            #if SPE_DEBUG_DISPLAY >= 1
              sim_printf("[0x%llx] :: dmaList[%d] = %p\n", id, i, dmaList[i]);
            #endif

	    //if (dmaList[i] == NULL || ((unsigned int)dmaList[i] + (entryCount * sizeof(DMAListEntry)))>= (unsigned int)0x40000) {
	    if (__builtin_expect((dmaList[i] == NULL) || 
                                 (((unsigned int)dmaList[i]) < ((unsigned int)(&_end))) ||
                                 (((unsigned int)dmaList[i] + (entryCount * sizeof(DMAListEntry))) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                                 0
				)
               ) {
              #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	        sim_printf("[0x%llu] :: SPE :: Failed to allocate memory for dmaList[%d] (3)... will try again later...\n", id, i);
                sim_printf("[0x%llx] :: SPE :: dmaList[%d] = %p\n", id, i, dmaList[i]);
	      #endif

              dmaList[i] = NULL;

              // DEBUG
              //print_block_table();

              continue;  // Skip for now, try again later
            }
            memset(dmaList[i], 0, sizeof(DMAListEntry) * entryCount);
	  } else {
            dmaList[i] = (DMAListEntry*)(&(dmaListEntry[i * SPE_DMA_LIST_LENGTH]));
	  }
          dmaListSize[i] = entryCount;


          // Fill in the list
          readOnlyPtr[i] = NULL;   // Use this pointer to point to the first buffer to be written to memory (don't nead readOnly data anymore)
          register int listIndex = 0;

          if (((msgQueue[i]->flags & WORK_REQUEST_FLAGS_RW_IS_RO) == 0x00) && (readWritePtr[i] != NULL)) {
            register int bufferLeft = msgQueue[i]->readWriteLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->readWritePtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ?
                                             (SPE_DMA_LIST_ENTRY_MAX_LENGTH) :
                                             (bufferLeft)
                                           );
              dmaList[i][listIndex].size = ROUNDUP_16(dmaList[i][listIndex].size);
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }

            // Store the start of the write portion of the localMem buffer in readOnlyPtr
            readOnlyPtr[i] = readWritePtr[i];
	  }

          if (writeOnlyPtr[i] != NULL) {
            register int bufferLeft = msgQueue[i]->writeOnlyLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->writeOnlyPtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
              dmaList[i][listIndex].size = ROUNDUP_16(dmaList[i][listIndex].size);
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }

            // Store the start of the write portion of the localMem buffer in readOnlyPtr (if
            //   it is not set already... i.e. - this buffer isn't first)
            if (readOnlyPtr[i] == NULL) readOnlyPtr[i] = writeOnlyPtr[i];
	  }

	}


        // Initiate the DMA command
        if (numDMAQueueEntries > 0 && dmaListSize[i] > 0) {

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: Pre-PUTL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
                       id, i, readOnlyPtr[i], 0, (unsigned int)(dmaList[i]), dmaListSize[i] * sizeof(DMAListEntry), i
                      );
	  #endif

          spu_mfcdma64(readOnlyPtr[i],  // This pointer is being used to point to the start of the write portion of localMem
                       0,
                       (unsigned int)(dmaList[i]),
                       dmaListSize[i] * sizeof(DMAListEntry),
                       i,
                       MFC_PUTL_CMD
		      );

          // Update the putList heuristic
          newPutIndex = (i + 1) % SPE_MESSAGE_QUEUE_LENGTH;

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_EXECUTED, msgState[i]);
          #endif

	  // Check to see if this maxs out the number of PUTs allowed per scheduler loop iteration
	  numPutsLeft--;
          if (numPutsLeft <= 0)
            break;
	}

      }
    }
    putIndex = newPutIndex;

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(9)");
    #endif


    // Initiate the next message queue read from main memory
    spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_COMMITTING

    // Check for messages that are committed
    mfc_write_tag_mask(0x7FFFFFFF);
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    int commitIndexNext = (commitIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (commitIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_COMMITTING && ((tagStatus & (0x01 << i)) != 0), 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_COMMITTING for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        // Check to see if there is an available entry in the outbound mailbox
        if (spu_stat_out_mbox() > 0) {

          STATETRACE_OUTPUT(i);

          // Free the local data and message buffers
          if (localMemPtr[i] != NULL) {
            //delete [] ((char*)localMemPtr[i]);
            _free_align(localMemPtr[i]);
            localMemPtr[i] = NULL;
            readWritePtr[i] = NULL;
            readOnlyPtr[i] = NULL;
            writeOnlyPtr[i] = NULL;
	  }

          // Clear the entry
          msgState[i] = SPE_MESSAGE_STATE_CLEAR;

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
            sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, SPE_MESSAGE_STATE_COMMITTING, msgState[i]);
          #endif

          // Clear the dmaList size so it looks like the dma list has not been set
          if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
            //delete [] dmaList[i];
            _free_align(dmaList[i]);
            dmaList[i] = NULL;
	  }
          dmaListSize[i] = -1;

          // Send the index of the entry in the message queue to the PPE
          //spu_write_out_mbox((unsigned int)i);
          spu_write_out_mbox(MESSAGE_RETURN_CODE(i, SPE_MESSAGE_OK));

          // Update the commitIndex heuristic
          commitIndexNext = (i + 1) % SPE_MESSAGE_QUEUE_LENGTH;
	}
        
      }
    }
    commitIndex = commitIndexNext;

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(A)");
    #endif


    //////////////////////////////////////////////////////////////////////////////////
    // SPE_MESSAGE_STATE_ERROR

    // Check for any messages that have entered into the ERROR state
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_ERROR, 0)) {


        // DEBUG
        if (__builtin_expect(msgQueue[i]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Processing SPE_MESSAGE_STATE_ERROR for index %d...\n", (int)getSPEID(), i);
          debug_displayActiveMessageQueue(id, msgState, "(*)");
	}


        STATETRACE_OUTPUT(i);


        // NOTE: All clean-up should be taken care of by the code placing the message into the error
        //   state (that way the code here does not have to handle all cases).

        // Check to see if there is an available entry in the outbound mailbox
        if (spu_stat_out_mbox() > 0) {

          // DEBUG
          #if SPE_DEBUG_DISPLAY >= 1
	    sim_printf("[0x%llx] :: msg %d's state going from %d -> %d\n", id, i, msgState[i], SPE_MESSAGE_STATE_CLEAR);
          #endif

          // Clear the entry
          msgState[i] = SPE_MESSAGE_STATE_CLEAR;

          // Send the index of the entry in the message queue to the PPE along with the ERROR code
          spu_write_out_mbox(MESSAGE_RETURN_CODE(i, errorCode[i]));

          // Clear the error
          errorCode[i] = SPE_MESSAGE_OK;
	}
      }
    }

    STATETRACE_UPDATE_ALL

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      debug_displayActiveMessageQueue(id, msgState, "(B)");
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      debug_displayStateHistogram(id, msgState, "");
    #endif

    // Check to see if there has been no progress in the message's states
    #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
      int msgLastStateFlag = 0;
      for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
        if (msgState[i] != msgLastState[i]) {
          msgLastState[i] = msgState[i];
          msgLastStateFlag = 1;  // Something changed
	}
      }

      if (msgLastStateFlag != 0) {  // if something changed
        msgLastStateCount = 0;
      } else {                      // otherwise, nothing has changed
        msgLastStateCount++;
        if (msgLastStateCount >= SPE_DEBUG_DISPLAY_NO_PROGRESS) {
          for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
            sim_printf("[0x%llx] :: msgState[%d] = %d\n", id, i, msgState[i]);
	  }
          msgLastStateCount = 0; // reset the count
	}
      }
    #endif

    #if SPE_STATS != 0
      statData.schedulerLoopCount++;
    #endif


    // DEBUG
    //__asm__("dsync");


    // DEBUG - Moved to here (commented out at original location)
    // Initiate the next message queue read from main memory
    //spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);


  } // end while (keepLooping)

}

#endif



void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str) {
  #if SPE_DEBUG_DISPLAY >= 1

  // DEBUG
  printf("SPE_%d :: Dumping active portion of message queue...\n", (int)getSPEID());

  int tmp;

  for (tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
    //if (msgState[tmp] != SPE_MESSAGE_STATE_CLEAR || msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) {
    if (1) {
      printf("[0x%llx] :: %s%s msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %lu, rwl = %d, ro = %lu, rol = %d, wo = %lu, wol = %d, f = 0x%08X, tm = %u, s = %d(%d), cnt = %d:%d, cmd = %d }\n",
                 id,
                 ((msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) ? ("---===!!! WARNING !!!===--- ") : ("")),
                 ((str == NULL) ? ("") : (str)),
                 tmp,
                 msgQueue[tmp],
                 msgQueue,
                 (volatile int)(msgQueue[tmp]->funcIndex),
                 (volatile PPU_POINTER_TYPE)msgQueue[tmp]->readWritePtr,
                 (volatile int)msgQueue[tmp]->readWriteLen,
                 (volatile PPU_POINTER_TYPE)msgQueue[tmp]->readOnlyPtr,
                 (volatile int)msgQueue[tmp]->readOnlyLen,
                 (volatile PPU_POINTER_TYPE)msgQueue[tmp]->writeOnlyPtr,
                 (volatile int)msgQueue[tmp]->writeOnlyLen,
                 (volatile int)(msgQueue[tmp]->flags),
                 (volatile int)(msgQueue[tmp]->totalMem),
                 (volatile int)(msgQueue[tmp]->state),
                 msgState[tmp],
                 (volatile int)(msgQueue[tmp]->counter0),
                 (volatile int)(msgQueue[tmp]->counter1),
                 (volatile int)(msgQueue[tmp]->command)
                );

      if (msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) {
        printf("***************************************************************************************************\n");
        printf("***************************************************************************************************\n");
        printf("***************************************************************************************************\n");
        printf("[0x%llx] :: msgQueueRaw @ %p, SPE_MESSAGE_QUEUE_BYTE_COUNT = %d\n",
               id, msgQueueRaw, SPE_MESSAGE_QUEUE_BYTE_COUNT
              );
      }

    }
  }

  #endif
}


void debug_displayStateHistogram(unsigned long long id, int* msgState, char* str) {
  #if SPE_DEBUG_DISPLAY >= 1

  char __buffer[2048];
  char* buf = __buffer;
  int somethingToShowFlag = 0;
  int state, i;

  sprintf(buf, "[0x%llx] :: SPE Histogram %s...\n", id, ((str == NULL) ? ("") : (str)));
  buf += strlen(buf);

  for (state = SPE_MESSAGE_STATE_MIN; state <= SPE_MESSAGE_STATE_MAX; state++) {

    sprintf(buf, "     %2d -> ", state);
    buf += strlen(buf);

    for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++)
      if (msgState[i] == state) {
        sprintf(buf, " %2d", i);
        buf += strlen(buf);
        if (state > SPE_MESSAGE_STATE_SENT) somethingToShowFlag++;
      }

    sprintf(buf, "\n");
    buf += strlen(buf);
  }

  if (somethingToShowFlag > 0)
    sim_printf(__buffer);

  #endif
}


#if SPE_USE_OWN_MEMSET != 0
// NOTE : This is meant for DEBUG
void memset(void* ptr, char val, int len) {
  // NOTE: This will actually traverse the memory backwards
  len--;
  while (len >= 0) {
    *((char*)ptr + len) = val;
    len--;
  }
}
#endif


void debug_dumpSPERTState() {

  // Header
  printf("SPE_%d : ----- DUMPING SPE RUNTIME STATE -----\n", (int)getSPEID());
  printf("SPE_%d : Currently Executing Message Queue Entry %d\n", (int)getSPEID(), execIndex);

  // Message queue entry specific data
  register int i, j;
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    printf("SPE_%d : Message Entry %d:\n", (int)getSPEID(), i);
    printf("SPE_%d :   msgState = %d, msgCounter = %d\n", (int)getSPEID(), msgState[i], msgCounter[i]);
    printf("SPE_%d :   localMemPtr = %p, errorCode = %d\n", (int)getSPEID(), localMemPtr[i], errorCode[i]);
    printf("SPE_%d :   readOnlyPtr = %p, readWritePtr = %p, writeOnlyPtr = %p\n", (int)getSPEID(), readOnlyPtr[i], readWritePtr[i], writeOnlyPtr[i]);
    printf("SPE_%d :   dmaList = %p, dmaListSize = %d\n", (int)getSPEID(), dmaList[i], dmaListSize[i]);
    register const DMAListEntry* localDMAList = (DMAListEntry*)(localMemPtr[i]);
    for (j = 0; j < dmaListSize[i]; j++)
      printf("SPE_%d :     entry %d :: { ea = 0x%08x, size = %u } -=> { ea = 0x%08x, size = %u }\n",
             (int)getSPEID(), j,
             dmaList[i][j].ea, dmaList[i][j].size,
             localDMAList[j].ea, localDMAList[j].size
            );
    
  }

  // Footer
  printf("SPE_%d : -------------------------------------\n", (int)getSPEID());

}


#if USE_PRINT_BLOCK == 0

void print_block_table() { }

#else

#ifdef TRUE
#undef TRUE
#endif

#ifdef FALSE
#undef FALSE
#endif

#include "/opt/IBM/cell-sdk-1.1/src/lib/c/malloc_ls.h"

extern void* start_heap_ptr;
extern block_info* block_table;
//extern static int num_blocks;

void print_block_table () {

  int i;
  printf("SPE_%d :: *********info about block table***********\n", (int)getSPEID());
  //for (i = 0; i < num_blocks; i++)
  for (i = 0; i < 50; i++)
  {
    printf("SPE_%d ::   block %d, block_addr=0x%x, status = %d, type = %d, nfree=%d, num_blocks=%d\n",
           (int)getSPEID(),
           i, i*BLOCKSIZE + (int)start_heap_ptr, (int)block_table[i].status, (int)block_table[i].type,
           (int)block_table[i].nfree, (int)block_table[i].num_blocks
          );

  }
  printf("SPE_%d :: ********** end block table info************\n", (int)getSPEID());
}

#endif


#if SPE_USE_OWN_MALLOC >= 1

#define SPE_MEMORY_NUM_BLOCKS  4

typedef struct _blockEntry {
  void* addr;
  int inUse;
} BlockEntry;

BlockEntry memBlocks[SPE_MEMORY_NUM_BLOCKS];

void initMem() {
  register int i;
  register unsigned int startAddr = ROUNDUP_128((unsigned int)(&(_end)));
  //register unsigned int endAddr = (ROUNDUP_128(0x40000 - SPE_RESERVED_STACK_SIZE) - 128);
  register unsigned int endAddr = 0x40000 - SPE_RESERVED_STACK_SIZE;
  endAddr = ROUNDUP_128(endAddr);
  endAddr -= 128;

  if (__builtin_expect(endAddr <= startAddr, 0)) {
    printf("SPE_%d :: [ERROR] :: Initing memory, endAddr <= startAddr... bad things headed your way...\n", (int)getSPEID());
    return;
  }

  register unsigned int stepSize = ROUNDUP_128((endAddr - startAddr) / SPE_MEMORY_NUM_BLOCKS) - 128;

  for (i = 0; i < SPE_MEMORY_NUM_BLOCKS; i++) {
    memBlocks[i].addr = (void*)(startAddr + (i * stepSize));
    memBlocks[i].inUse = 0;
  }

  //#if SPE_DEBUG_DISPLAY >= 1
    printf("SPE_%d :: [INFO] :: initMem() - startAddr = 0x%08x, endAddr = 0x%08x, stepSize = 0x%08x...\n",
           (int)getSPEID(), startAddr, endAddr, stepSize
          );
  //#endif

}

void* _malloc_align(int size, int alignment) {

  // NOTE : Blocks are already aligned to 128 byte boundries in initMem() (that should be the most the user would
  //   ask for.

  if (__builtin_expect(alignment >= 8, 0)) {
    printf("SPE_%d :: [ERROR] :: _malloc_align() called with alignment >= 8 (not implemented yet)...\n", (int)getSPEID());
  }

  register int i;
  register void* rtnValue = NULL;
  for (i = 0; i < SPE_MEMORY_NUM_BLOCKS; i++) {
    if (memBlocks[i].inUse == 0) {
      memBlocks[i].inUse = 1;
      rtnValue = memBlocks[i].addr;
      break;
    }
  }

  return rtnValue;
}

void _free_align(void* addr) {

  register int i;
  register int foundFlag = 0;

  if (__builtin_expect(addr == NULL, 0)) return;

  for (i = 0; i < SPE_MEMORY_NUM_BLOCKS; i++) {
    if (memBlocks[i].addr == addr) {

      if (__builtin_expect(memBlocks[i].inUse != 1, 0)) {
        printf("SPE_%d :: [ERROR] :: free'ing memory that is not in use in _free_align()...\n", (int)getSPEID());
      }

      memBlocks[i].inUse = 0;      
      foundFlag = 1;
    }
  }

  if (foundFlag == 0) {
    printf("SPE_%d :: [ERROR] :: Unable to free block in _free_align()... unknown block...\n", (int)getSPEID());
  }
}

#endif
