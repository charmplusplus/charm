//#ifdef __cplusplus
//extern "C" {
//#endif
  #include <stdlib.h>
  #include <unistd.h>
  #include <stdio.h>
  //#include <free_align.h>
  //#include <sim_printf.h>
//#ifdef __cplusplus
//}
//#endif


#include <spu_intrinsics.h>
#include <spu_mfcio.h>
//#include <cbe_mfc.h>

#if SPE_USE_OWN_MEMSET == 0
  #include <string.h>
#endif

#include "spert_common.h"
#include "spert.h"


#if SPE_USE_OWN_MALLOC <= 0

  #ifdef __cplusplus
  extern "C" {
  #endif
    #include <malloc_align.h>
  #ifdef __cplusplus
  }
  #endif

  #define MALLOC(size) _malloc_align(size, 7)
  #define FREE(addr)   _free_align(addr)

#else

  void local_initMem();
  void* local_malloc(int size, int alignment);
  void local_free(void* addr);

  #define MALLOC(size) local_malloc(size, 7)
  #define FREE(addr)   local_free(addr)

#endif


#define USE_PRINT_BLOCK  0
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


// TODO : FIX ME : Creating a scratch area for the DMA Lists created by the standard work requests.  The
//   static area cannot be used since anything written there will be wipped out by the next message queue
//   DMA-Get transaction.  Possible fix: Have the PPE create the list for standard work requests prior to
//   sending the work request
// NOTE : At most 4 entries should be needed since there can only be 3 buffers (and length of the buffers
//   is not being checked at the moment).
DMAListEntry scratchDMAList[4 * SPE_MESSAGE_QUEUE_LENGTH] __attribute__((aligned(16)));


typedef struct __local_msgq_data {

  int msgState;
  int msgCounter;
  void* localMemPtr;
  int errorCode;

  void* readWritePtr;
  void* readOnlyPtr;
  void* writeOnlyPtr;
  int __pad__0;

  DMAListEntry* dmaList;
  int dmaListSize;
  int __pad__1[2];  // Overall structure size should be a multiple of 16
                    //   so each entry in an array is 16 byte aligned
} LocalMsgQData;

// Local info about the Message Queue Entries
LocalMsgQData localMsgQData[SPE_MESSAGE_QUEUE_LENGTH] __attribute__((aligned(16)));


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

// NOTE : For the timerUpper variable, it is assumed that a "get" or "service" timer call will be made
//   at least once every 2^32 (max unsigned int value) decrementer clocks (i.e. - there aren't two
//   overflows between "get" or "service" timer calls).

// TODO : FIX ME : Modify this code to allow loads and stores to timerUpper_store and lastTimerRead_store
//   to happen faster (i.e. each store does a load, rotate, update, store... or something like that).
unsigned int timerUpper_store = 0;
unsigned int lastTimerRead_store = (unsigned int)0xFFFFFFFF;

// NOTE : The "if" statements in the following code should only mispredict once every 2^32 decrementer
//   cycles (i.e. once per rollover of the decrementer).

#define startTimer() {                                                         \
                       spu_writech(SPU_WrDec, 0xFFFFFFFF);                     \
                       spu_writech(SPU_WrEventMask, MFC_DECREMENTER_EVENT);    \
                       timerUpper_store = 0;                                   \
                       lastTimerRead_store = (unsigned int)0xFFFFFFFF;         \
                     }

#define serviceTimer() {                                                                 \
                         register unsigned int cntr = spu_readch(SPU_RdDec);             \
                         register unsigned int tmp_lastTimerRead = lastTimerRead_store;  \
                         lastTimerRead_store = cntr;                                     \
                         if (__builtin_expect(cntr > tmp_lastTimerRead, 0)) {            \
                           timerUpper_store += 1;                                        \
                         }                                                               \
                       }

// NOTE : Assumes var is 'unsigned int'
#define getTimer(var) {                                                                 \
                        register unsigned int cntr = spu_readch(SPU_RdDec);             \
                        var = ((unsigned int)(0xFFFFFFFF)) - cntr;                      \
                        register unsigned int tmp_lastTimerRead = lastTimerRead_store;  \
                        lastTimerRead_store = cntr;                                     \
                        if (__builtin_expect(cntr > tmp_lastTimerRead, 0)) {            \
                          timerUpper_store += 1;                                        \
                        }                                                               \
                      }

// NOTE : Assumes var is 'unsigned long long int'
#define getTimer64(var) {                                                                                                 \
                          register unsigned int cntr = spu_readch(SPU_RdDec);                                             \
                          register unsigned int var_lower = ((unsigned int)(0xFFFFFFFF)) - cntr;                          \
                          register unsigned int tmp_lastTimerRead = lastTimerRead_store;                                  \
                          lastTimerRead_store = cntr;                                                                     \
                          register unsigned int tmp_timerUpper = timerUpper_store;                                        \
                          if (__builtin_expect(cntr > tmp_lastTimerRead, 0)) {                                            \
                            tmp_timerUpper += 1;                                                                          \
                            timerUpper_store = tmp_timerUpper;                                                            \
                          }                                                                                               \
                          register unsigned long long int var_upper = (((unsigned long long int)(tmp_timerUpper)) << 32); \
                          var = var_upper | ((unsigned long long int)var_lower);                                          \
                        }

// NOTE : Assumes var is 'unsigned int'
#define stopTimer(var) {                                                                 \
                         register unsigned int cntr = spu_readch(SPU_RdDec);             \
                         spu_writech(SPU_WrEventMask, 0);                                \
                         spu_writech(SPU_WrEventAck, MFC_DECREMENTER_EVENT);             \
                         var = ((unsigned int)(0xFFFFFFFF)) - cntr;                      \
                         register unsigned int tmp_lastTimerRead = lastTimerRead_store;  \
                         lastTimerRead_store = cntr;                                     \
                         if (__builtin_expect(cntr > tmp_lastTimerRead, 0)) {            \
                           timerUpper_store += 1;			                 \
                         }                                                               \
                       }

// NOTE : Assumes var is 'unsigned long long int'
#define stopTimer64(var) {                                                                                                 \
                           register unsigned int cntr = spu_readch(SPU_RdDec);                                             \
                           spu_writech(SPU_WrEventMask, 0);                                                                \
                           spu_writech(SPU_WrEventAck, MFC_DECREMENTER_EVENT);                                             \
                           register unsigned int var_lower = ((unsigned int)(0xFFFFFFFF)) - cntr;                          \
                           register unsigned int tmp_lastTimerRead = lastTimerRead_store;                                  \
                           lastTimerRead_store = cntr;                                                                     \
                           register unsigned int tmp_timerUpper = timerUpper_store;                                        \
                           if (__builtin_expect(cntr > tmp_lastTimerRead, 0)) {                                            \
                             tmp_timerUpper += 1;                                                                          \
                             timerUpper_store = tmp_timerUpper;                                                            \
                           }                                                                                               \
                           register unsigned long long int var_upper = (((unsigned long long int)(tmp_timerUpper)) << 32); \
                           var = var_upper | ((unsigned long long int)var_lower);                                          \
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

/*
#define STATETRACE_UPDATE_ALL {					\
                                int iii = 0;                                                        \
                                for (iii = 0; iii < SPE_MESSAGE_QUEUE_LENGTH; iii++) {              \
                                  if (stateTrace[iii][stateTrace_counter[iii]] != msgState[iii]) {  \
                                    stateTrace[iii][stateTrace_counter[iii]] = msgState[iii];       \
                                    stateTrace_counter[iii]++;                                      \
                                    stateTrace[iii][stateTrace_counter[iii]] = msgState[iii];       \
                                  }                                                                 \
                                }                                                                   \
                              }
*/

#define STATETRACE_UPDATE_ALL {                                                                     \
                                int iii = 0;                                                        \
                                for (iii = 0; iii < SPE_MESSAGE_QUEUE_LENGTH; iii++) {              \
                                  if (stateTrace[iii][stateTrace_counter[iii]] != localMsgQData[iii].msgState) {  \
                                    stateTrace[iii][stateTrace_counter[iii]] = localMsgQData[iii].msgState;       \
                                    stateTrace_counter[iii]++;                                      \
                                    stateTrace[iii][stateTrace_counter[iii]] = localMsgQData[iii].msgState;       \
                                  }                                                                 \
                                }                                                                   \
                              }

/*
#define STATETRACE_UPDATE(index) {					\
                                   if (stateTrace[index][stateTrace_counter[index]] != msgState[index]) {  \
                                     stateTrace[index][stateTrace_counter[index]] = msgState[index];       \
                                     stateTrace_counter[index]++;                                          \
                                     stateTrace[index][stateTrace_counter[index]] = msgState[index];       \
                                   }                                                                       \
                                 }
*/

#define STATETRACE_UPDATE(index) {                                                                         \
                                   if (stateTrace[index][stateTrace_counter[index]] != localMsgQData[index].msgState) {  \
                                     stateTrace[index][stateTrace_counter[index]] = localMsgQData[index].msgState;       \
                                     stateTrace_counter[index]++;                                          \
                                     stateTrace[index][stateTrace_counter[index]] = localMsgQData[index].msgState;       \
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

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_recvTimeStart;
    getTimer64(tmp_recvTimeStart);
  #endif


  // Check for a new message in this slot
  // Conditions... 1) msgQueue[i]->counter1 != msgCounter[i]          // Entry has not already been processed (most likely not true, i.e. - test first to reduce cost of overall condition check)
  //               2) msgQueue[i]->counter0 == msgQueue[i]->counter1  // Entire entry has been written by PPE
  //               3) msgQueue[i]->state == SPE_MESSAGE_STATE_SENT    // PPE wrote an entry and is waiting for the result
  //               4) msgState[i] == SPE_MESSAGE_STATE_CLEAR          // SPE isn't currently processing this slot

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_counter1 = tmp_msgQueueEntry->counter1;

  // NOTE : tmp_localData0 = { msgState, msgCounter, localMemPtr, readWritePtr }
  register vector signed int* tmp_localData0Addr = (vector signed int*)(&(localMsgQData[msgIndex]));
  register vector signed int tmp_localData0 = *(tmp_localData0Addr);
  register int tmp_msgCounter = (int)(spu_extract(tmp_localData0, 1));

  // Note: Check if counter1 has changed first because it will be the last thing modified (i.e., if it has
  //   changed, then the other checks are likely to succeed ... unless there is some crazy error... )
  if (__builtin_expect(tmp_counter1 != tmp_msgCounter, 0)) {

    register vector signed int msg_tmp0 = *((vector signed int*)(tmp_msgQueueEntry));
    register int tmp_counter0 = spu_extract(msg_tmp0, 0);
    register int tmp_msgState = spu_extract(msg_tmp0, 1);
    register int tmp_flags = (unsigned int)(spu_extract(msg_tmp0, 2));
    register int tmp_localState = (int)(spu_extract(tmp_localData0, 0));

    if (__builtin_expect(tmp_counter0 == tmp_counter1, 1)) {
      if (__builtin_expect(tmp_msgState == SPE_MESSAGE_STATE_SENT, 1)) {
        if (__builtin_expect(tmp_localState == SPE_MESSAGE_STATE_CLEAR, 1)) {


          // Lastly, check the checksum value
          register int checkSumVal = 0;
          register int* intPtr = (int*)tmp_msgQueueEntry;
          register int jj;
          for (jj = 0; jj < (sizeof(SPEMessage) - sizeof(int)) / sizeof(int); jj++) {
            checkSumVal += intPtr[jj];
	  }
          if (checkSumVal == tmp_msgQueueEntry->checksum) {

            // STATS1
            #if SPE_STATS1 != 0
	      if (timingIndex == -1) {
                startTimer();
                timingIndex = msgIndex;
  	      }
            #endif

            // STATS2
            #if SPE_STATS2 != 0
              #if SPE_STATS2 > 0
              if (msgIndex == SPE_STATS2) {
              #endif
                register unsigned int clock2;
                getTimer(clock2);
                printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_SEND for %d\n", (int)getSPEID(), clock2, msgIndex);
              #if SPE_STATS2 > 0
              }
              #endif
            #endif

            // TRACE
            #if ENABLE_TRACE != 0
              STATETRACE_CLEAR(msgIndex);
            #endif

            // Write the new message state and counter value to the LS
            // Note: tmp_msgStateToAdd = 0 if standard WR or = 1 if scatter/gather(list) WR
            register int tmp_msgStateToAdd = (tmp_flags & WORK_REQUEST_FLAGS_LIST) >> WORK_REQUEST_FLAGS_LIST_SHIFT;
            register int tmp_msgState = SPE_MESSAGE_STATE_PRE_FETCHING + tmp_msgStateToAdd;
            tmp_localData0 = spu_insert(tmp_msgState, tmp_localData0, 0);
            tmp_localData0 = spu_insert(tmp_counter1, tmp_localData0, 1);
            *tmp_localData0Addr = tmp_localData0;

            // TIMING
            #if SPE_TIMING != 0
              register unsigned long long int tmp_endTime;
              getTimer64(tmp_endTime);
              register unsigned int tmp_recvTimeEnd = (unsigned int)(tmp_endTime - tmp_recvTimeStart);
              register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
              register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;
              register vector unsigned int tmp_notifyQueueEntry0 = { 0, 0, 0, 0 };
              tmp_notifyQueueEntry0 = spu_insert((unsigned int)(tmp_recvTimeStart), tmp_notifyQueueEntry0, 1);
              tmp_notifyQueueEntry0 = spu_insert((unsigned int)((tmp_recvTimeStart >> 32) & 0xFFFFFFFF), tmp_notifyQueueEntry0, 0);
              tmp_notifyQueueEntry0 = spu_insert(tmp_recvTimeEnd, tmp_notifyQueueEntry0, 2);
              (*tmp_notifyQueueEntryPtr) = tmp_notifyQueueEntry0;
            #endif

            // TRACE
            #if ENABLE_TRACE != 0
              if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
                unsigned long long int curTime;
                getTimer64(curTime);
                printf("SPE_%d :: [TRACE] @ %llu :: Tracing entry at index %d...\n",
                       (int)getSPEID(), curTime, msgIndex
                      );
                debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
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

          } // end checksum check


          // DEBUG
          //else { printf("SPE_%d :: [DEBUG] :: CHECKSUM FAILURE !!!\n", (int)getSPEID()); }


        }
      }
    }
  }

  // STATS
  #if SPE_STATS
    register unsigned int clocks;
    stopTimer(clocks);
    register int msgNotFoundFlag = (msgFoundFlag ^ 1);
    sentClocks_msg += (clocks * msgFoundFlag);
    sentClocksCounter_msg += msgFoundFlag;
    sentClocks_noMsg += (clocks * msgNotFoundFlag);
    sentClocksCounter_noMsg += msgNotFoundFlag;
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_PRE_FETCHING for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // Get a pointer to the message queue entry and local data
  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_PRE_FETCHING for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x00, NULL/*msgState*/, "(*)");
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_startTime;
    getTimer64(tmp_startTime);
  #endif

  register vector signed int tmp_msgQueueData0 = *(((vector signed int*)tmp_msgQueueEntry) + 0);
  register vector signed int tmp_msgQueueData1 = *(((vector signed int*)tmp_msgQueueEntry) + 1);
  register vector signed int tmp_msgQueueData2 = *(((vector signed int*)tmp_msgQueueEntry) + 2);

  register vector signed int* tmp_localData0Addr = (vector signed int*)(&(localMsgQData[msgIndex]));
  register vector signed int tmp_localData0 = *tmp_localData0Addr;  // = { msgState, msgCounter, localMemPtr, errorCode }
  register vector signed int* tmp_localData1Addr = tmp_localData0Addr + 1;
  register vector signed int tmp_localData1 = *tmp_localData1Addr;  // = { readWritePtr, readOnlyPtr, writeOnlyPtr, xx }
  register vector signed int* tmp_localData2Addr = tmp_localData0Addr + 2;
  register vector signed int tmp_localData2 = *tmp_localData2Addr;  // = { dmaList, dmaListSize, xx, xx }
  register void* tmp_localMemPtr = (void*)spu_extract(tmp_localData0, 2);

  // Allocate the memory for the message queue entry (if need be)
  // NOTE: First check to see if it is non-null.  What might have happened was that there was enough memory
  //   last time but the DMA queue was full and a retry was needed.  So this time, the memory is there and
  //   only the DMA needs to be retried.
  if (__builtin_expect(tmp_localMemPtr == NULL, 1)) {


    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Allocating LS Memory for index %d...\n", (int)getSPEID(), msgIndex);
        printf("SPE_%d :: [TRACE] ::   msgQueue[%d] = { readOnlyPtr = 0x%08lx, readOnlyLen = %d,\n"
               "                                      readWritePtr = 0x%08lx, readWritelen = %d,\n"
               "                                      writeOnlyPtr = 0x%08lx, writeOnlyLen = %d, ...}...\n",
               (int)getSPEID(), msgIndex,
               tmp_msgQueueEntry->readOnlyPtr, tmp_msgQueueEntry->readOnlyLen,
               tmp_msgQueueEntry->readWritePtr, tmp_msgQueueEntry->readWriteLen,
               tmp_msgQueueEntry->writeOnlyPtr, tmp_msgQueueEntry->writeOnlyLen
              );
        debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
      }
    #endif


    // Allocate the memory and place a pointer to the allocated buffer into localMemPtr.  This buffer will
    //   be divided up into three regions: readOnly, readWrite, writeOnly (IN THAT ORDER!; the regions may be
    //   empty if no pointer for the region was supplied by the original sendWorkRequest() call on the PPU).

    // Determine the amount of memory needed
    register int tmp_readWriteLen = spu_extract(tmp_msgQueueData1, 3);
    register int tmp_readOnlyLen = spu_extract(tmp_msgQueueData2, 0);
    register int tmp_writeOnlyLen = spu_extract(tmp_msgQueueData2, 1);
    register int tmp_readWriteLenRU16 = ROUNDUP_16(tmp_readWriteLen);
    register int tmp_readOnlyLenRU16 = ROUNDUP_16(tmp_readOnlyLen);
    register int tmp_writeOnlyLenRU16 = ROUNDUP_16(tmp_writeOnlyLen);
    register int memNeeded = tmp_readWriteLenRU16 + tmp_readOnlyLenRU16 + tmp_writeOnlyLenRU16;

    // Check the size of the memory needed.  If it is too large for the SPE's LS, then stop this
    //   message with an error code because the memory allocation will never work.
    register int endAddr = (int)(&_end);
    register int heapSize = ((int)0x40000) - endAddr - SPE_RESERVED_STACK_SIZE;
    if (__builtin_expect(memNeeded >= heapSize, 0)) {

      // Move the message into the error state
      tmp_localData0 = spu_insert(SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY, tmp_localData0, 3);
      tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_ERROR, tmp_localData0, 0);
      (*tmp_localData0Addr) = tmp_localData0;

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
    //tmp_localMemPtr = (void*)(_malloc_align(memNeeded, 4));
    tmp_localMemPtr = (void*)(MALLOC(memNeeded));

    // STATS
    #if SPE_STATS != 0
      wrTryAllocCount++;
    #endif

    // Check the pointer (if it is bad, then skip this message for now and try again later)
    if (__builtin_expect(tmp_localMemPtr == NULL, 0)
        || __builtin_expect((unsigned int)tmp_localMemPtr < ((unsigned int)endAddr), 0)
        || __builtin_expect((unsigned int)tmp_localMemPtr >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE), 0)
       ) {

      #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
	printf("SPE_%d :: ERROR :: Failed to allocate memory for localMemPtr[%d] (2)... will try again later...\n",
               (int)getSPEID(), msgIndex
              );
        printf("SPE_%d :: ERROR :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, tmp_localMemPtr);
      #endif

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

      return;
    }

    // STATS
    #if SPE_STATS != 0
      wrAllocCount++;
    #endif


    // Update the tmp_localData0 vector with the new local memory pointer
    tmp_localData0 = spu_insert((int)tmp_localMemPtr, tmp_localData0, 2);  // update vector

    // Update the LS with the contents of tmp_localData0
    (*tmp_localData0Addr) = tmp_localData0;


    // Assign the local pointers to the various buffers within the memory just allocated
    // NOTE : Order matters here.  Need to allocate the read buffers next to each other and the write
    //   buffers next to each other (i.e. - read-only, then read-write, then write-only).

    // Do the comparisons (pointer to each buffer type == NULL)
    // NOTE: tmp_msgQueueData1 = { rWPtr, rOPtr, wOPtr, rWLen }, tmp_msgQueueData2 = { rOLen, wOLen, xx, xx }
    register vector unsigned int tmp_ptrCompare = spu_cmpeq(tmp_msgQueueData1, (int)NULL);
    register vector unsigned int tmp_allOnes = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
    register vector unsigned int tmp_ptrCompareNot = spu_sub(tmp_allOnes, tmp_ptrCompare);
    // NOTE: tmp_ptrCompare    = { rWPtr == NULL (-1 if true,  0 otherwise), 'same for rOPtr', 'same for wOPtr', xx }
    // NOTE: tmp_ptrCompareNot = { rWPtr == NULL ( 0 if true, -1 otherwise), 'same for rOPtr', 'same for wOPtr', xx }
    register int tmp_readWriteCompareNot = spu_extract(tmp_ptrCompareNot, 0);
    register int tmp_readOnlyCompareNot = spu_extract(tmp_ptrCompareNot, 1);
    register int tmp_writeOnlyCompareNot = spu_extract(tmp_ptrCompareNot, 2);

    register int tmp_readOnlyOffset = tmp_readOnlyLenRU16 & tmp_readOnlyCompareNot;
    register int tmp_readWriteOffset = tmp_readWriteLenRU16 & tmp_readWriteCompareNot;

    // Set local buffer pointers
    register int tmp_localReadOnlyPtr = (int)tmp_localMemPtr & tmp_readOnlyCompareNot;
    register int tmp_localReadWritePtr = ((int)tmp_localMemPtr + tmp_readOnlyOffset) & tmp_readWriteCompareNot;
    register int tmp_localWriteOnlyPtr = ((int)tmp_localMemPtr + tmp_readOnlyOffset + tmp_readWriteOffset) & tmp_writeOnlyCompareNot;

    // Update the tmp_localData1 vector with the new local buffer pointers
    tmp_localData1 = spu_insert(tmp_localReadWritePtr, tmp_localData1, 0);
    tmp_localData1 = spu_insert(tmp_localReadOnlyPtr, tmp_localData1, 1);
    tmp_localData1 = spu_insert(tmp_localWriteOnlyPtr, tmp_localData1, 2);

    // Update the LS with the contents of tmp_localData1
    (*tmp_localData1Addr) = tmp_localData1;


    // Setup the DMA List

    register int tmp_readWriteCompare = spu_extract(tmp_ptrCompare, 0);
    register int tmp_readOnlyCompare = spu_extract(tmp_ptrCompare, 1);
    register int tmp_readWritePtr = spu_extract(tmp_msgQueueData1, 0);
    register int tmp_readOnlyPtr = spu_extract(tmp_msgQueueData1, 1);
    register int tmp_writeOnlyPtr = spu_extract(tmp_msgQueueData1, 2);

    // TODO : For now, assume that the buffers are =< single DMA command size limit (either add
    //   the code to adjust the list here again or just do it on the PPE)

    // Entry 1
    register int tmp_entry0ROMask = tmp_readOnlyCompareNot;                                               // set if ro
    register int tmp_entry0RWMask = tmp_readOnlyCompare & tmp_readWriteCompareNot;                        // set if !ro & rw
    register int tmp_entry0WOMask = tmp_readOnlyCompare & tmp_readWriteCompare & tmp_writeOnlyCompareNot; // set if !ro & !rw & wo
    register int tmp_entry0Ptr = (  (tmp_readOnlyPtr  & tmp_entry0ROMask)  // rOPtr if rOPtr set, 0x00 otherwise
				  | (tmp_readWritePtr & tmp_entry0RWMask)  // rWPtr if (rOPtr !set && rWPtr set), 0x00 otherwise
                                  | (tmp_writeOnlyPtr & tmp_entry0WOMask)  // rOPtr if (rOPtr !set && rWPtr !set && wOPtr set), 0x00 otherwise
				 );
    register int tmp_entry0Len = (  (tmp_readOnlyLenRU16  & tmp_entry0ROMask)  // rOPtr if rOPtr set, 0x00 otherwise
				  | (tmp_readWriteLenRU16 & tmp_entry0RWMask)  // rWPtr if (rOPtr !set && rWPtr set), 0x00 otherwise
                                  | (tmp_writeOnlyLenRU16 & tmp_entry0WOMask)  // rOPtr if (rOPtr !set && rWPtr !set && wOPtr set), 0x00 otherwise
				 );

    // Entry 2
    register int tmp_entry1RWMask = tmp_readOnlyCompareNot & tmp_readWriteCompareNot; // set if ro & rw
    register int tmp_entry1WOMask = tmp_writeOnlyCompareNot                           // set if (ro ^ rw) & wo
                                    & (tmp_readOnlyCompareNot ^ tmp_readWriteCompareNot);
    register int tmp_entry1Ptr = (  (tmp_readWritePtr & tmp_entry1RWMask)
				  | (tmp_writeOnlyPtr & tmp_entry1WOMask)
                                 );
    register int tmp_entry1Len = (  (tmp_readWriteLenRU16 & tmp_entry1RWMask)
				  | (tmp_writeOnlyLenRU16 & tmp_entry1WOMask)
                                 );

    // Entry 3
    register int tmp_entry2WOMask = tmp_readOnlyCompareNot & tmp_readWriteCompareNot & tmp_writeOnlyCompareNot; // set if ro & rw & wo
    register int tmp_entry2Ptr = tmp_writeOnlyPtr & tmp_entry2WOMask;
    register int tmp_entry2Len = tmp_writeOnlyLenRU16 & tmp_entry2WOMask;

    // TODO : For now, assume that the DMA lists for standard work requests are short enough
    //   that they will fit into the static DMA list area in the message queue.
    //register DMAListEntry* tmp_dmaList = (DMAListEntry*)(((char*)tmp_msgQueueEntry) + 48);
    // TODO : FIX ME : See comment for scratchDMAList declaration
    register DMAListEntry* tmp_dmaList = &(scratchDMAList[4 * msgIndex]);

    // Fill in the DMA list in the LS
    register vector signed int tmp_dmaListData0;
    register vector signed int tmp_dmaListData1 = { 0, 0, 0, 0 };
    tmp_dmaListData0 = spu_insert(tmp_entry0Len, tmp_dmaListData1, 0);
    tmp_dmaListData0 = spu_insert(tmp_entry0Ptr, tmp_dmaListData0, 1);
    tmp_dmaListData0 = spu_insert(tmp_entry1Len, tmp_dmaListData0, 2);
    tmp_dmaListData0 = spu_insert(tmp_entry1Ptr, tmp_dmaListData0, 3);
    *(((vector signed int*)tmp_dmaList) + 0) = tmp_dmaListData0;
    tmp_dmaListData1 = spu_insert(tmp_entry2Len, tmp_dmaListData1, 0);
    tmp_dmaListData1 = spu_insert(tmp_entry2Ptr, tmp_dmaListData1, 1);
    *(((vector signed int*)tmp_dmaList) + 1) = tmp_dmaListData1;


    // Count the read entries (read-only + read-write)
    register unsigned int tmp_flags = spu_extract(tmp_msgQueueData0, 2);
    register int tmp_dmaListReadSize = (0x01 & tmp_readOnlyCompareNot);  // +1 if read-only set
    tmp_dmaListReadSize += ((0x01 & tmp_readWriteCompareNot)             // +1 if read-write set and is not flagged as write-only
                            & (~((tmp_flags & WORK_REQUEST_FLAGS_RW_IS_WO) >> WORK_REQUEST_FLAGS_RW_IS_WO_SHIFT))
		           );

    // Update the LS with the contents of tmp_localData2
    tmp_localData2 = spu_insert((int)tmp_dmaList, tmp_localData2, 0);
    tmp_localData2 = spu_insert(tmp_dmaListReadSize, tmp_localData2, 1);
    (*tmp_localData2Addr) = tmp_localData2;

    
    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] ::   localMemPtr[%d] = %p...\n", (int)getSPEID(), msgIndex, localMsgQData[msgIndex].localMemPtr);
        printf("SPE_%d :: [TRACE] ::   readOnlyPtr[%d] = %p, readWritePtr[%d] = %p, writeOnlyPtr[%d] = %p...\n",
               (int)getSPEID(),
               msgIndex, localMsgQData[msgIndex].readOnlyPtr,
               msgIndex, localMsgQData[msgIndex].readWritePtr,
               msgIndex, localMsgQData[msgIndex].writeOnlyPtr
              );
      }
    #endif

    
    // If the number of read entries is == 0, just just the state to READY and skip DMA-Get
    // NOTE: The typical case should be that a Work Request has input data.
    if (__builtin_expect(tmp_dmaListReadSize == 0, 0)) {

      // Set the message queue entry's state
      tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_READY, tmp_localData0, 0);

      // Update the LS with the contents of tmp_localData0
      (*tmp_localData0Addr) = tmp_localData0;

      // STATS
      #if SPE_STATS != 0
        register unsigned int clocks;
        stopTimer(clocks);
        preFetchingClocks += clocks;
        preFetchingClocksCounter++;
      #endif

      return;
    }

  } // end if (localMsgQData[msgIndex].localMemPtr == NULL)


  // Get the number of free DMA queue entries
  register int numDMAQueueEntries = mfc_stat_cmd_queue();
  
  // Initiate the DMA command if there is at least one free DMA queue entry
  if (__builtin_expect(numDMAQueueEntries > 0, 1)) {  // Stay positive... they'll let me in... I'm a likable guy...

    // Queue the DMA command
    register int tmp_dmaList = spu_extract(tmp_localData2, 0);
    register int tmp_dmaListSize = spu_extract(tmp_localData2, 1);

    spu_mfcdma64(tmp_localMemPtr,
                 0,  // TODO : Update the message queue so the upper 32-bits are also sent
                 (unsigned int)tmp_dmaList,
                 tmp_dmaListSize * sizeof(DMAListEntry),
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
    tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_FETCHING, tmp_localData0, 0);
    (*tmp_localData0Addr) = tmp_localData0;

    // TIMING
    // NOTE : Only keeping the last entry (this code will be executed the last time
    //   this state function is called for any given work request).
    #if SPE_TIMING != 0
      // Get the ending time
      register unsigned long long int tmp_endTime;
      getTimer64(tmp_endTime);

      // Get a pointer to the notify queue entry
      register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
      register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

      // Grab the initial recv time
      register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
      register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);

      // Calculate the start and end offsets from recvStartTime
      register unsigned int tmp_preFetchingStart = (unsigned int)(tmp_startTime - tmp_recvTimeStart);
      register unsigned int tmp_preFetchingEnd = (unsigned int)(tmp_endTime - tmp_recvTimeStart);

      // Write the start and end times to the LS
      register vector unsigned int tmp_notifyQueueEntry1 = { 0, 0, 0, 0 };
      tmp_notifyQueueEntry1 = spu_insert(tmp_preFetchingStart, tmp_notifyQueueEntry1, 0);
      tmp_notifyQueueEntry1 = spu_insert(tmp_preFetchingEnd, tmp_notifyQueueEntry1, 1);
      (*(tmp_notifyQueueEntryPtr + 1)) = tmp_notifyQueueEntry1;
    #endif

    // DEBUG
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: msg %d's state going from %d -> %d\n",
             (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_PRE_FETCHING, localMsgQData[msgIndex].msgState
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_PRE_FETCHING_LIST for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_PRE_FETCHING_LIST for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_dmaListSize = localMsgQData[msgIndex].dmaListSize;

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
          //localMsgQData[msgIndex].dmaList = (DMAListEntry*)(new char[memNeeded]);
          //localMsgQData[msgIndex].dmaList = (DMAListEntry*)(_malloc_align(memNeeded, 4));
          localMsgQData[msgIndex].dmaList = (DMAListEntry*)(MALLOC(memNeeded));
      #ifdef __cplusplus
	} catch (...) {
          localMsgQData[msgIndex].dmaList = NULL;
	}
      #endif

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, localMsgQData[msgIndex].dmaList);
      #endif

      // Verify the pointer returned
      if (__builtin_expect((localMsgQData[msgIndex].dmaList == NULL) || 
                           (((unsigned int)localMsgQData[msgIndex].dmaList) < ((unsigned int)(&_end))) ||
                           (((unsigned int)localMsgQData[msgIndex].dmaList + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
                           0
                          )
         ) {
        #if SPE_NOTIFY_ON_MALLOC_FAILURE != 0
          printf("SPE_%d :: Failed to allocate memory for dmaList[%d] (1)... will try again later...\n",
                 (int)getSPEID(), msgIndex
                );
          printf("SPE_%d :: dmaList[%d] = %p\n", (int)getSPEID(), msgIndex, localMsgQData[msgIndex].dmaList);
	#endif

        localMsgQData[msgIndex].dmaList = NULL;
        localMsgQData[msgIndex].dmaListSize = -1;  // Try allocating again next time

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
      //memset(localMsgQData[msgIndex].dmaList, 0, memNeeded);

    } else {  // DMA list will fit in the static DMA list

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Using static DMA List...\n", (int)getSPEID());
        }
      #endif

      // Point the DMA list pointer at the message queue entry's dma list
      localMsgQData[msgIndex].dmaList = (DMAListEntry*)(tmp_msgQueueEntry->dmaList);

      //// DEBUG
      //printf("SPE_%d :: USING STATIC DMA LIST !!!\n", (int)getSPEID());
      //register int k;
      //for (k = 0; k < localMsgQData[msgIndex].dmaListSize; k++)
      //  printf("SPE_%d ::   dmaList[%d] = { ea = 0x%08x, size = %u }\n",
      //         (int)getSPEID(), k, (localMsgQData[msgIndex].dmaList)[k].ea, (localMsgQData[msgIndex].dmaList)[k].size
      //        );

      //// Set the dmaList pointer
      //localMsgQData[msgIndex].dmaList = (DMAListEntry*)(&(dmaListEntry[msgIndex * SPE_DMA_LIST_LENGTH]));
      //// Copy the list from the message queue entry

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Created DMA List for index %d...\n", (int)getSPEID(), msgIndex);
          printf("SPE_%d :: [TRACE] ::   dmaList[%d] = %p, dmaListSize[%d] = %d...\n",
                 (int)getSPEID(), msgIndex, localMsgQData[msgIndex].dmaList, msgIndex, tmp_dmaListSize
                );
          register int _j0;
          for (_j0 = 0; _j0 < tmp_dmaListSize; _j0++) {
            printf("SPE_%d :: [TRACE] ::    DMA Entry %d = { ea = 0x%08x, size = %d }\n",
                   (int)getSPEID(), _j0,
                   (localMsgQData[msgIndex].dmaList)[_j0].ea, (localMsgQData[msgIndex].dmaList)[_j0].size
                  );
          }
        }
      #endif

      // Update the message queue's state
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_LIST_READY_LIST;
    }

    // Store the DMA list size to the LS
    localMsgQData[msgIndex].dmaListSize = tmp_dmaListSize;

  } // end if (localMsgQData[msgIndex].dmaListSize < 0)

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

      spu_mfcdma32(localMsgQData[msgIndex].dmaList,
                   (unsigned int)(msgQueue[msgIndex]->readWritePtr),
                   ROUNDUP_16(localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry)),
                   msgIndex,
                   MFC_GET_CMD
                  );

      // Update the state of the message queue entry now that the data should be in-flight
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_FETCHING_LIST;
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_FETCHING_LIST for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_FETCHING_LIST for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
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
    localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_LIST_READY_LIST;

    // Roundup all of the sizes to the next highest multiple of 16
    for (j = 0; j < localMsgQData[msgIndex].dmaListSize; j++)
      (localMsgQData[msgIndex].dmaList)[j].size = ROUNDUP_16((localMsgQData[msgIndex].dmaList)[j].size);
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_LIST_READY_LIST for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_LIST_READY_LIST for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register void* tmp_localMemPtr = localMsgQData[msgIndex].localMemPtr;
  register int tmp_dmaListSize = localMsgQData[msgIndex].dmaListSize;

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
    localMsgQData[msgIndex].readOnlyPtr = NULL;
    localMsgQData[msgIndex].readWritePtr = NULL;
    localMsgQData[msgIndex].writeOnlyPtr = NULL;

    // Determine the number of bytes needed
    register int tmp_evenDMAListSize = tmp_dmaListSize + (tmp_dmaListSize & 0x01);
    //register unsigned int memNeeded = sizeof(DMAListEntry) * tmp_dmaListSize;
    register unsigned int memNeeded = ROUNDUP_128(sizeof(DMAListEntry) * tmp_evenDMAListSize);
    register int j0;
    #if 0
      for (j0 = 0; j0 < tmp_dmaListSize; j0++) {
        memNeeded += (localMsgQData[msgIndex].dmaList)[j0].size;
      }
    #else
      register int tmp_0 = 0;
      register const DMAListEntry* tmp_dmaList = localMsgQData[msgIndex].dmaList;
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
      if (__builtin_expect(localMsgQData[msgIndex].dmaListSize > SPE_DMA_LIST_LENGTH, 0)) {
        if (__builtin_expect(localMsgQData[msgIndex].dmaList != NULL, 1)) {
          //delete [] localMsgQData[msgIndex].dmaList;
          //_free_align(localMsgQData[msgIndex].dmaList);
          FREE(localMsgQData[msgIndex].dmaList);
	}
        localMsgQData[msgIndex].dmaList = NULL;
        localMsgQData[msgIndex].dmaListSize = -1;
      }

      // Move the message into the error state
      localMsgQData[msgIndex].errorCode = SPE_MESSAGE_ERROR_NOT_ENOUGH_MEMORY;
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_ERROR;

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
        //localMsgQData[msgIndex].localMemPtr = (void*)(new char[memNeeded]);
        //localMsgQData[msgIndex].localMemPtr = (void*)(_malloc_align(memNeeded, 4));
        //tmp_localMemPtr = (void*)(_malloc_align(memNeeded, 4));
        tmp_localMemPtr = (void*)(MALLOC(memNeeded));
    #ifdef __cplusplus
      } catch (...) {
        //localMsgQData[msgIndex].localMemPtr = NULL;
        tmp_localMemPtr = NULL;
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
      printf("SPE_%d :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMsgQData[msgIndex].localMemPtr);
    #endif

    //// Check the pointer that was returned
    //if (__builtin_expect((localMsgQData[msgIndex].localMemPtr == NULL) || 
    //                     (((unsigned int)localMsgQData[msgIndex].localMemPtr) < ((unsigned int)(&_end))) ||
    //                     (((unsigned int)localMsgQData[msgIndex].localMemPtr + memNeeded) >= ((unsigned int)0x40000 - SPE_RESERVED_STACK_SIZE)),
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
        printf("SPE_%d :: SPE :: localMemPtr[%d] = %p\n", (int)getSPEID(), msgIndex, localMsgQData[msgIndex].localMemPtr);
      #endif

      // NOTE : localMsgQData[msgIndex].localMemPtr is already NULL (was NULL to begin with if this code reached, the invalid
      //   pointer returned by the malloc call is in tmp_localMemPtr)
      //localMsgQData[msgIndex].localMemPtr = NULL;

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
        printf("SPE_%d :: [TRACE] :: Allocated memory for WR at index %d... tmp_localMemPtr = %p\n",
               (int)getSPEID(), msgIndex, tmp_localMemPtr
              );
      }
    #endif

    #if 0

      // Setup pointers to the buffers
      register unsigned int initOffset = ROUNDUP_128(localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry));
      register unsigned int offset = initOffset;
      register int j1;
      //register DMAListEntry* localDMAList = (DMAListEntry*)(localMsgQData[msgIndex].localMemPtr);
      register DMAListEntry* localDMAList = (DMAListEntry*)(tmp_localMemPtr);
      register const DMAListEntry* remoteDMAList = localMsgQData[msgIndex].dmaList;

      //for (j1 = 0; j1 < localMsgQData[msgIndex].dmaListSize; j1++) {
      for (j1 = 0; j1 < tmp_dmaListSize; j1++) {
        register unsigned int size = ((remoteDMAList[j1].size) & 0x0000FFFF);
        localDMAList[j1].ea = ((unsigned int)localMsgQData[msgIndex].localMemPtr) + offset;
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

        register unsigned int initOffset = ROUNDUP_128(localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry));
        initOffset += (unsigned int)(tmp_localMemPtr);
        register vector unsigned int* localDMAList = (vector unsigned int*)(tmp_localMemPtr);
        register vector unsigned int* remoteDMAList = (vector unsigned int*)(localMsgQData[msgIndex].dmaList);
        //register int tmp_evenDMAListSize = tmp_dmaListSize + (tmp_dmaListSize & 0x01);

        register int j1;
        register unsigned int offset = initOffset;
        register vector unsigned int tmp_offset = spu_splats(initOffset);

        register vector unsigned char mask2 = (vector unsigned char){ 0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x00, 0x01, 0x02, 0x03 };
        register vector unsigned char mask3 = (vector unsigned char){ 0x00, 0x01, 0x02, 0x03,  0x14, 0x15, 0x16, 0x17,  0x08, 0x09, 0x0a, 0x0b,  0x1c, 0x1d, 0x1e, 0x1f };

        // TRACE
        #if ENABLE_TRACE >= 2
          if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
            printf("SPE_%d :: [TRACE] :: Creating Local DMA List...\n", (int)getSPEID());
            printf("SPE_%d :: [TRACE] ::   tmp_evenDMAListSize = %d\n", (int)getSPEID(), tmp_evenDMAListSize);
	  }
        #endif

        for (j1 = 0; j1 < tmp_evenDMAListSize; j1 += 2) {

          // Read the remote DMA list entry (2 entries: size+0, ea+0, size+1, ea+1)
          register vector unsigned int tmp_0 = *(remoteDMAList);  // contains { 'size+0', 'ea+0', 'size+1', 'ea+1' }
          remoteDMAList++;

          // TRACE
          #if ENABLE_TRACE >= 2
            if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
              printf("SPE_%d :: [TRACE] ::   tmp_0 = { %u, 0x%08x, %u, 0x%08x }\n",
                     (int)getSPEID(), spu_extract(tmp_0, 0), spu_extract(tmp_0, 1), spu_extract(tmp_0, 2), spu_extract(tmp_0, 3)
                    );
              printf("SPE_%d :: [TRACE] ::   tmp_offset = { %u, %u, %u, %u }\n",
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
              printf("SPE_%d :: [TRACE] ::   tmp_1 = { xx, %u, xx, %u }\n",
                     (int)getSPEID(), spu_extract(tmp_1, 1), spu_extract(tmp_1, 3)
                    );
              printf("SPE_%d :: [TRACE] ::   tmp_offset_1 = { xx, %u, xx, %u }\n",
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
              printf("SPE_%d :: [TRACE] ::   tmp_entry = { %u, 0x%08x, %u, 0x%08x }\n",
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
      register const DMAListEntry* remoteDMAList = localMsgQData[msgIndex].dmaList;

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
        //for (j2 = localMsgQData[msgIndex].dmaListSize - msgQueue[msgIndex]->writeOnlyLen; j2 < localMsgQData[msgIndex].dmaListSize; j2++)
        register const DMAListEntry* tmp_dmaList = localMsgQData[msgIndex].dmaList;
        for (j2 = tmp_dmaListSize - tmp_msgQueueEntry->writeOnlyLen; j2 < tmp_dmaListSize; j2++)
          //writeSize += (localMsgQData[msgIndex].dmaList)[j2].size;
          writeSize += tmp_dmaList[j2].size;
        //memset(((char*)(localMsgQData[msgIndex].localMemPtr)) + memNeeded - writeSize, 0, writeSize);
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
    //localMsgQData[msgIndex].localMemPtr = tmp_localMemPtr;

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

  } // end if (localMsgQData[msgIndex].localMemPtr == NULL)

  // NOTE : If execution reaches here, tmp_localMemPtr contains a valid value (store it to localMemPtr[msgIndex])
  localMsgQData[msgIndex].localMemPtr = tmp_localMemPtr;

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
               (int)getSPEID, msgIndex, ((char*)(localMsgQData[msgIndex].localMemPtr)) + ROUNDUP_16(localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry)),
               (unsigned int)msgQueue[msgIndex]->readOnlyPtr,
               (unsigned int)(localMsgQData[msgIndex].dmaList),
               (msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen) * sizeof(DMAListEntry), msgIndex
              );

        //register int j4;
        //for (j4 = 0; j4 < dmaListSize[i]; j4++) {
        //  printf("SPE_%d :: dmaList[%d][%d].size = %d (0x%x), dmaList[%d][%d].ea = 0x%08x (0x%08x)\n",
        //         (int)getSPEID(),
        //         msgIndex, j4, dmaList[msgIndex][j4].size, ((DMAListEntry*)(localMemPtr[msgIndex]))[j4].size,
        //         msgIndex, j4, dmaList[msgIndex][j4].ea, ((DMAListEntry*)(localMemPtr[msgIndex]))[j4].ea
	//        );
        //}
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
      //register unsigned int lsPtr = (unsigned int)(localMsgQData[msgIndex].localMemPtr);
      register unsigned int lsPtr = (unsigned int)(tmp_localMemPtr);
      //register unsigned int lsOffset = localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry);
      register unsigned int lsOffset = tmp_dmaListSize * sizeof(DMAListEntry);
      lsOffset = ROUNDUP_128(lsOffset);
      lsPtr += lsOffset;
      //spu_mfcdma64((void*)lsPtr,
      //             (unsigned int)msgQueue[msgIndex]->readOnlyPtr,  // eah
      //             (unsigned int)(localMsgQData[msgIndex].dmaList),
      //             (msgQueue[msgIndex]->readOnlyLen + msgQueue[msgIndex]->readWriteLen) * sizeof(DMAListEntry),
      //             msgIndex,
      //             MFC_GETL_CMD
      //            );


      // DEBUG
      {
        register int delay = 0;
        register int jj = 0;
        register DMAListEntry* tmp_dmaListPtr = localMsgQData[msgIndex].dmaList;
        for (jj = 0; jj < tmp_readCount; jj++) {
          register unsigned int ea = tmp_dmaListPtr->ea;
          register unsigned int size = tmp_dmaListPtr->size;
          tmp_dmaListPtr++;
          if (__builtin_expect((ea & 0x7F) != 0x00, 0)) {
            printf("SPE :: [WARNING] :: non-128-byte-aligned ea in DMA List in entry %d... :(\n", jj);
            delay = 100;
	  }
          if (__builtin_expect((size & 0x0F) != 0x00, 0)) {
            printf("SPE :: [WARNING] :: non-128-byte size in DMA List in entry %d... :(\n", jj);
            delay = 100;
	  }
          if (__builtin_expect(size > SPE_DMA_LIST_ENTRY_MAX_LENGTH, 0)) {
            printf("SPE :: [WARNING] :: Size in DMA List in entry %d too large (output)... :(\n", jj);
            delay = 100;
	  }
	}

        while (delay > 0) {
          printf("SPE :: [DELAY] :: ...\n");
          delay--;
	}
      }
      

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {

          printf("SPE_%d :: [TRACE] :: Issuing GETL command for index %d (tmp_readCount = %d)\n",
                 (int)getSPEID(), msgIndex, tmp_readCount
                );

          register DMAListEntry* tmp_dmaListPtr = localMsgQData[msgIndex].dmaList;
          register int jj;
          printf("SPE_%d :: [TRACE] :: GETL DMA List:\n", (int)getSPEID());
          for (jj = 0; jj < tmp_readCount; jj++) {
            printf("SPE_%d :: [TRACE] ::   entry %d = { ea = 0x%08x, size = %u }\n",
                   (int)getSPEID(), jj, tmp_dmaListPtr->ea, tmp_dmaListPtr->size
		  );
            tmp_dmaListPtr++;
	  }
	}
      #endif


      spu_mfcdma64((void*)lsPtr,
                   (unsigned int)tmp_msgQueueEntry->readOnlyPtr,  // eah
                   (unsigned int)(localMsgQData[msgIndex].dmaList),
                   (tmp_readCount * sizeof(DMAListEntry)),
                   msgIndex,
                   MFC_GETL_CMD
	          );

      // Update the state of the message queue entry now that the data should be in-flight 
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_FETCHING;

    } // end if (numDMAQueueEntries > 0)

  } else {  // No input buffers so this work request is ready to execute

    // Update the state (to ready to execute)
    localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_READY;
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_FETCHING for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_FETCHING for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_startTime;
    getTimer64(tmp_startTime);
  #endif

  // Read the tag status to see if the data has arrived for the fetching message entry
  mfc_write_tag_mask(0x1 << msgIndex);
  mfc_write_tag_update_immediate();
  register int tagStatus = mfc_read_tag_status();


  // DEBUG
  #if ENABLE_TRACE != 0
    if (msgQueue[msgIndex]->traceFlag) {
      printf("SPE_%d :: [DEBUG/TRACE] :: tagStatus = 0x%08x\n", (int)getSPEID(), tagStatus);

      mfc_write_tag_mask(0xFFFFFFFF);
      mfc_write_tag_update_immediate();
      register int tmp_ts = mfc_read_tag_status();
      printf("SPE_%d :: [DEBUG/TRACE] :: tmp_ts = 0x%08x\n", (int)getSPEID(), tmp_ts);
    }
  #endif


  // Check if the data has arrived
  if (tagStatus != 0) {

    // Update the state to show that this message queue entry is ready to be executed
    localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_READY;

    // TIMING
    #if SPE_TIMING != 0
      // Get the time
      register unsigned long long int tmp_endTime;
      getTimer64(tmp_endTime);

      // Get a pointer to the notify queue entry
      register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
      register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

      // Grab the initial recv time
      register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
      register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);

      // Calculate the start and end offsets from recvStartTime
      register unsigned int tmp_fetchingStart = (unsigned int)(tmp_startTime - tmp_recvTimeStart);
      register unsigned int tmp_fetchingEnd = (unsigned int)(tmp_endTime - tmp_recvTimeStart);

      // Write the start and end times to the LS
      register vector unsigned int tmp_notifyQueueEntry1 = (*(tmp_notifyQueueEntryPtr + 1));
      tmp_notifyQueueEntry1 = spu_insert(tmp_fetchingStart, tmp_notifyQueueEntry1, 2);
      tmp_notifyQueueEntry1 = spu_insert(tmp_fetchingEnd, tmp_notifyQueueEntry1, 3);
      (*(tmp_notifyQueueEntryPtr + 1)) = tmp_notifyQueueEntry1;
    #endif

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


vector unsigned long long int userAccumTimeStart[2] __attribute__((aligned(16)));
vector unsigned long long int userAccumTime[2] __attribute__((aligned(16)));


// SPE_MESSAGE_STATE_READY
inline void processMsgState_ready(int msgIndex) {

  // STATS
  #if SPE_STATS != 0
    startTimer();
  #endif

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_READY for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_READY for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_readyStartTime;
    getTimer64(tmp_readyStartTime);
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register vector signed int tmp_msgQueueData0 = *(((vector signed int*)tmp_msgQueueEntry) + 0);
  register vector signed int tmp_msgQueueData1 = *(((vector signed int*)tmp_msgQueueEntry) + 1);
  register vector signed int tmp_msgQueueData2 = *(((vector signed int*)tmp_msgQueueEntry) + 2);

  register LocalMsgQData* tmp_localDataEntry = (LocalMsgQData*)(&(localMsgQData[msgIndex]));
  register vector signed int tmp_localData0 = *(((vector signed int*)tmp_localDataEntry) + 0);
  register vector signed int tmp_localData1 = *(((vector signed int*)tmp_localDataEntry) + 1);

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    if (msgQueue[msgIndex]->traceFlag)
      printf("SPE_%d :: >>>>> Entering User Code (index %d)...\n", (int)getSPEID(), msgIndex);
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    //isTracingFlag = ((msg->traceFlag) ? (-1) : (0));
    register int tmp_traceFlag = spu_extract(tmp_msgQueueData2, 3);
    isTracingFlag = tmp_traceFlag;
  #endif

  // STATS
  #if SPE_STATS != 0
    register unsigned int userStartClocks;
    getTimer(userStartClocks);
  #endif

  // Pre-load some values into registers for the funcLookup() call
  register int tmp_funcIndex = spu_extract(tmp_msgQueueData0, 3);
  register int tmp_readWriteLen = spu_extract(tmp_msgQueueData1, 3);
  register int tmp_readOnlyLen = spu_extract(tmp_msgQueueData2, 0);
  register int tmp_writeOnlyLen = spu_extract(tmp_msgQueueData2, 1);

  // Check to see if this is a standard or scatter/gather (list) work request
  register int tmp_flags = spu_extract(tmp_msgQueueData0, 2);
  register int tmp_isListWRFlag = ((tmp_flags & WORK_REQUEST_FLAGS_LIST) >> WORK_REQUEST_FLAGS_LIST_SHIFT); // = 0 (std) or 1 (list)
  register int tmp_isStdWR = tmp_isListWRFlag - 1;  // = -1 (std) or 0 (list)
  register int tmp_isListWR = (-1) - tmp_isStdWR;  // = 0 (std) or -1 (list)

  // Pre-load the remaining values needed for the funcLookup() call
  register int tmp_localReadWritePtr = spu_extract(tmp_localData1, 0) & tmp_isStdWR;  // Only set for std WR
  register int tmp_localReadOnlyPtr = spu_extract(tmp_localData1, 1) & tmp_isStdWR;   // Only set for std WR
  register int tmp_localWriteOnlyPtr = spu_extract(tmp_localData1, 2) & tmp_isStdWR;  // Only set for std WR
  register int tmp_localDMAList = spu_extract(tmp_localData0, 2) & tmp_isListWR;      // Only set (=localMemPtr) for list WR

  // DEBUG - Used for functions that might be called by the user's code (so the SPE Runtime
  //   can determine which work request is being executed and should be accessed).
  execIndex = msgIndex;

  // Clear out the user timers
  {
    register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
    register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;
    register vector unsigned int tmp_notifyQueueEntry3 = (*(tmp_notifyQueueEntryPtr + 3));
    register vector unsigned int tmp_notifyQueueEntry5 = { 0, 0, 0, 0 };
    tmp_notifyQueueEntry3 = spu_insert((unsigned int)0, tmp_notifyQueueEntry3, 2);
    tmp_notifyQueueEntry3 = spu_insert((unsigned int)0, tmp_notifyQueueEntry3, 3);
    (*(tmp_notifyQueueEntryPtr + 3)) = tmp_notifyQueueEntry3;
    (*(tmp_notifyQueueEntryPtr + 5)) = tmp_notifyQueueEntry5;
  }

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Entering user code for index %d...\n", (int)getSPEID(), msgIndex);
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_userStartTime;
    getTimer64(tmp_userStartTime);
  #endif

  // Call into user code via funcLookup()
  funcLookup(tmp_funcIndex,
             (void*)tmp_localReadWritePtr, tmp_readWriteLen,
             (void*)tmp_localReadOnlyPtr, tmp_readOnlyLen,
             (void*)tmp_localWriteOnlyPtr, tmp_writeOnlyLen,
             (DMAListEntry*)tmp_localDMAList
            );

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_userEndTime;
    getTimer64(tmp_userEndTime);
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      printf("SPE_%d :: [TRACE] :: Exiting user code for index %d...\n", (int)getSPEID(), msgIndex);
    }
  #endif

  // Copy the user accum timers into the notify queue
  {
    register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
    register vector unsigned long long int* tmp_notifyQueueEntryPtr = ((vector unsigned long long int*)notifyQueueRaw) + tmp_offset;
    (*(tmp_notifyQueueEntryPtr + 6)) = *(userAccumTime);
    (*(tmp_notifyQueueEntryPtr + 7)) = *(userAccumTime + 1);
  }

  // DEBUG
  execIndex = -1;

  // STATS
  #if SPE_STATS != 0
    register unsigned int userEndClocks;
    getTimer(userEndClocks);
    userClocks += (userEndClocks - userStartClocks);
    userClocksCounter++;
  #endif

  // Update the state of this WR in the LS
  register int tmp_nextState = SPE_MESSAGE_STATE_EXECUTED + tmp_isListWRFlag;
  tmp_localData0 = spu_insert(tmp_nextState, tmp_localData0, 0);
  *(((vector signed int*)tmp_localDataEntry) + 0) = tmp_localData0;

  // TIMING
  #if SPE_TIMING != 0
    // Get the ending time
    register unsigned long long int tmp_readyEndTime;
    getTimer64(tmp_readyEndTime);

    // Get a pointer to the notify queue entry
    register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
    register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

    // Grab the initial recv time
    register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
    register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);

    // Calculate the start and end offsets from recvStartTime
    register unsigned int tmp_readyStart = (unsigned int)(tmp_readyStartTime - tmp_recvTimeStart);
    register unsigned int tmp_readyEnd = (unsigned int)(tmp_readyEndTime - tmp_recvTimeStart);
    register unsigned int tmp_userStart = (unsigned int)(tmp_userStartTime - tmp_recvTimeStart);
    register unsigned int tmp_userEnd = (unsigned int)(tmp_userEndTime - tmp_recvTimeStart);

    // Write the start and end times to the LS
    register vector unsigned int tmp_notifyQueueEntry2 = { 0, 0, 0, 0 };
    tmp_notifyQueueEntry2 = spu_insert(tmp_readyStart, tmp_notifyQueueEntry2, 0);
    tmp_notifyQueueEntry2 = spu_insert(tmp_readyEnd, tmp_notifyQueueEntry2, 1);
    tmp_notifyQueueEntry2 = spu_insert(tmp_userStart, tmp_notifyQueueEntry2, 2);
    tmp_notifyQueueEntry2 = spu_insert(tmp_userEnd, tmp_notifyQueueEntry2, 3);
    (*(tmp_notifyQueueEntryPtr + 2)) = tmp_notifyQueueEntry2;
  #endif

  // SPE Stats Code
  #if SPE_STATS != 0
    statData.numWorkRequestsExecuted++;
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

  // TRACE
  #if ENABLE_TRACE != 0
    STATETRACE_UPDATE(msgIndex);
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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_EXECUTED for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // Get a pointer to the message queue entry
  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(tmp_msgQueueEntry->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_EXECUTED for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_startTime;
    getTimer64(tmp_startTime);
  #endif

  register vector signed int tmp_msgQueueData0 = *(((vector signed int*)tmp_msgQueueEntry) + 0);
  register vector signed int tmp_msgQueueData1 = *(((vector signed int*)tmp_msgQueueEntry) + 1);
  register vector signed int tmp_msgQueueData2 = *(((vector signed int*)tmp_msgQueueEntry) + 2);

  register LocalMsgQData* tmp_localDataEntry = (LocalMsgQData*)(&(localMsgQData[msgIndex]));
  register vector signed int tmp_localData0 = *(((vector signed int*)tmp_localDataEntry) + 0);
  register vector signed int tmp_localData2 = *(((vector signed int*)tmp_localDataEntry) + 2);

  // Calculate the number of write buffers
  register int tmp_flags = spu_extract(tmp_msgQueueData0, 2);
  register vector unsigned int tmp_ptrCompare = spu_cmpeq(tmp_msgQueueData1, (int)NULL);
  register vector unsigned int tmp_allOnes = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
  register vector unsigned int tmp_ptrCompareNot = spu_sub(tmp_allOnes, tmp_ptrCompare);
  // NOTE: tmp_ptrCompare    = { rWPtr == NULL (-1 if true,  0 otherwise), 'same for rOPtr', 'same for wOPtr', xx }
  // NOTE: tmp_ptrCompareNot = { rWPtr == NULL ( 0 if true, -1 otherwise), 'same for rOPtr', 'same for wOPtr', xx }

  register int numWriteBuffers = (spu_extract(tmp_ptrCompareNot, 2)) & 0x01;  // 0 if wOPtr == NULL, 1 otherwise
  register int tmp_RWisROflag = (tmp_flags & WORK_REQUEST_FLAGS_RW_IS_RO) >> WORK_REQUEST_FLAGS_RW_IS_RO_SHIFT;
  register int tmp_RWisROflagNot = 1 - tmp_RWisROflag;  // 1 if flag not set, 0 if flag set
  // NOTE : +1 if 'rWPtr != NULL && RWisRO flag not set', +0 otherwise
  numWriteBuffers += (((spu_extract(tmp_ptrCompareNot, 0)) & 0x01) & (tmp_RWisROflagNot));

  // Check to see if this message does not need to fetch any data
  if (__builtin_expect(numWriteBuffers == 0, 0)) {

    // Update the state of the message queue entry
    // NOTE : Even though the work request is basically finished, the message queue index still needs to be
    //   passed back to the PPE, i.e. go to the committing state.
    tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_COMMITTING, tmp_localData0, 0);
    *(((vector signed int*)tmp_localDataEntry) + 0) = tmp_localData0;

    // STATS
    #if SPE_STATS != 0
      register unsigned int clocks;
      stopTimer(clocks);
      executedClocks += clocks;
      executedClocksCounter++;
    #endif

    return;
  }

  register int tmp_rOBufferMask = spu_extract(tmp_ptrCompareNot, 1);  // check to see if there is a read-only buffer
  register int tmp_rWBufferMask = spu_extract(tmp_ptrCompareNot, 0);

  register int tmp_readWriteLen = spu_extract(tmp_msgQueueData1, 3);
  register int tmp_readWriteLenRU16 = ROUNDUP_16(tmp_readWriteLen);

  // The DMA List already exists so just calculate the pointer to the write portion of the list entries
  register int tmp_dmaListOffset = sizeof(DMAListEntry) * ((tmp_rOBufferMask & 0x01) + (tmp_rWBufferMask & tmp_RWisROflag));
  register int tmp_dmaList = spu_extract(tmp_localData2, 0) + tmp_dmaListOffset;

  // Calculate the LS pointer for the start of the write portion of the WR's memory
  register int tmp_localMemPtr = spu_extract(tmp_localData0, 2);
  register int tmp_readOnlyLen = spu_extract(tmp_msgQueueData2, 0);
  register int tmp_readOnlyLenRU16 = ROUNDUP_16(tmp_readOnlyLen);
  register int tmp_LSPtr = tmp_localMemPtr + (tmp_readOnlyLenRU16 & tmp_rOBufferMask);
  // If there is a read/write buffer and it is being used as a read-only then skip it also
  tmp_LSPtr += ((tmp_readWriteLenRU16 & tmp_rWBufferMask) * tmp_RWisROflag);
  
  // Get the number of free DMA queue entries
  register int numDMAQueueEntries = mfc_stat_cmd_queue();

  // Check to see if there is a free DMA queue entry
  if (__builtin_expect(numDMAQueueEntries > 0, 1)) {

    // Queue the DMA transaction
    spu_mfcdma64((void*)tmp_LSPtr,
                 0,
                 (unsigned int)tmp_dmaList,
                 numWriteBuffers * sizeof(DMAListEntry),
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
    tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_COMMITTING, tmp_localData0, 0);
    *(((vector signed int*)tmp_localDataEntry) + 0) = tmp_localData0;

    // TIMING
    #if SPE_TIMING != 0
      // Get the ending time
      register unsigned long long int tmp_endTime;
      getTimer64(tmp_endTime);

      // Get a pointer to the notify queue entry
      register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
      register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

      // Grab the initial recv time
      register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
      register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);

      // Calculate the start and end offsets from recvStartTime
      register unsigned int tmp_executedStart = (unsigned int)(tmp_startTime - tmp_recvTimeStart);
      register unsigned int tmp_executedEnd = (unsigned int)(tmp_endTime - tmp_recvTimeStart);

      // Write the start and end times to the LS
      register vector unsigned int tmp_notifyQueueEntry3 = { 0, 0, 0, 0 };
      tmp_notifyQueueEntry3 = spu_insert(tmp_executedStart, tmp_notifyQueueEntry3, 0);
      tmp_notifyQueueEntry3 = spu_insert(tmp_executedEnd, tmp_notifyQueueEntry3, 1);
      (*(tmp_notifyQueueEntryPtr + 3)) = tmp_notifyQueueEntry3;
    #endif

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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_EXECUTED_LIST for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_EXECUTED_LIST for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  register SPEMessage* tmp_msgQueueEntry = (SPEMessage*)(msgQueue[msgIndex]);
  register int tmp_readOnlyLen = tmp_msgQueueEntry->readOnlyLen;
  register int tmp_writeCount = tmp_msgQueueEntry->readWriteLen + tmp_msgQueueEntry->writeOnlyLen;
  register void* tmp_localMemPtr = localMsgQData[msgIndex].localMemPtr;

  // Check to see if there is output data that needs to be placed into system memory
  //if ((msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) > 0 && localMsgQData[msgIndex].localMemPtr != NULL) {
  if (__builtin_expect(tmp_writeCount > 0, 1) && __builtin_expect(tmp_localMemPtr != NULL, 1)) {

    // Get the number of free DMA queue entries
    register int numDMAQueueEntries = mfc_stat_cmd_queue();

    // If there is a free DMA queue entry, initiate the DMA transfer of the readWrite and
    //   writeOnly buffer back to main memory
    if (numDMAQueueEntries > 0) {

      // Get the offsets in the DMA list and the localMemPtr for the readWrite section
      register unsigned int readWriteOffset = ROUNDUP_128(localMsgQData[msgIndex].dmaListSize * sizeof(DMAListEntry));
      register int j0;
      register DMAListEntry* tmp_dmaList = localMsgQData[msgIndex].dmaList;
      for (j0 = 0; j0 < tmp_readOnlyLen; j0++)
        readWriteOffset += tmp_dmaList[j0].size;

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: Pre-PUTL :: msgIndex = %d, ls_addr = %p, eah = %d, list_addr = %p, list_size = %d, tag = %d...\n",
               (int)getSPEID(), msgIndex, ((char*)localMsgQData[msgIndex].localMemPtr) + (readWriteOffset),
               (unsigned int)msgQueue[msgIndex]->readOnlyPtr,
               (unsigned int)(&((localMsgQData[msgIndex].dmaList)[msgQueue[msgIndex]->readOnlyLen])),
               (msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) * sizeof(DMAListEntry), msgIndex
              );
      #endif

      // Queue the DMA transfer
      //spu_mfcdma64(((char*)localMsgQData[msgIndex].localMemPtr) + (readWriteOffset),
      //             (unsigned int)msgQueue[msgIndex]->readOnlyPtr,  // eah
      //             (unsigned int)(&((localMsgQData[msgIndex].dmaList)[msgQueue[msgIndex]->readOnlyLen])),
      //             (msgQueue[msgIndex]->readWriteLen + msgQueue[msgIndex]->writeOnlyLen) * sizeof(DMAListEntry),
      //             msgIndex,
      //             MFC_PUTL_CMD
      //          );


      // DEBUG
      {
        register int delay = 0;
        register int jj = 0;
        register DMAListEntry* tmp_dmaListPtr = &(tmp_dmaList[tmp_readOnlyLen]);
        for (jj = 0; jj < tmp_writeCount; jj++) {
          register unsigned int ea = tmp_dmaListPtr->ea;
          register unsigned int size = tmp_dmaListPtr->size;
          tmp_dmaListPtr++;
          if (__builtin_expect((ea & 0x7F) != 0x00, 0)) {
            printf("SPE :: [WARNING] :: non-128-byte-aligned ea in DMA List in entry %d (output)... :(\n", jj);
            delay = 100;
	  }
          if (__builtin_expect((size & 0x0F) != 0x00, 0)) {
            printf("SPE :: [WARNING] :: non-128-byte size in DMA List in entry %d (output)... :(\n", jj);
            delay = 100;
	  }
          if (__builtin_expect(size > SPE_DMA_LIST_ENTRY_MAX_LENGTH, 0)) {
            printf("SPE :: [WARNING] :: Size in DMA List in entry %d too large (output)... :(\n", jj);
            delay = 100;
	  }
	}

        while (delay > 0) {
          printf("SPE :: [DELAY] :: ...\n");
          delay--;
	}
      }


      spu_mfcdma64(((char*)tmp_localMemPtr) + (readWriteOffset),
                   (unsigned int)tmp_msgQueueEntry->readOnlyPtr,  // eah
                   (unsigned int)(&(tmp_dmaList[tmp_readOnlyLen])),
                   (tmp_writeCount) * sizeof(DMAListEntry),
                   msgIndex,
                   MFC_PUTL_CMD
	          );

      // Update the state of the message queue entry now that the data should be in-flight
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_COMMITTING;

      // DEBUG
      #if SPE_DEBUG_DISPLAY >= 1
        printf("SPE_%d :: msg %d's state going from %d -> %d\n",
               (int)getSPEID(), msgIndex, SPE_MESSAGE_STATE_EXECUTED_LIST, localMsgQData[msgIndex].msgState
              );
      #endif

    } // end if (numDMAQueueEntries > 0)

  } else {  // Otherwise, there is no output data

    // Update the state of the message queue entry now that the data should be in-flight
    localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_COMMITTING;

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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_COMMITTING for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_COMMITTING for index %d...\n",
             (int)getSPEID(), curTime, msgIndex
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
    }
  #endif

  // TIMING
  #if SPE_TIMING != 0
    register unsigned long long int tmp_startTime;
    getTimer64(tmp_startTime);
  #endif

  // Read the tag status to see if the data was sent for the committing message entry
  mfc_write_tag_mask(0x1 << msgIndex);
  mfc_write_tag_update_immediate();
  register int tagStatus = mfc_read_tag_status();

  // Check if the data was sent
  if (__builtin_expect(tagStatus, 1)) {

    register vector signed int* tmp_localDataEntry = (vector signed int*)(&(localMsgQData[msgIndex]));
    register vector signed int tmp_localData0 = *(((vector signed int*)tmp_localDataEntry) + 0);
    register vector signed int tmp_localData1 = *(((vector signed int*)tmp_localDataEntry) + 1);
    register vector signed int tmp_localData2 = *(((vector signed int*)tmp_localDataEntry) + 2);

    register void* tmp_localMemPtr = (void*)(spu_extract(tmp_localData0, 2));
    register void* tmp_dmaList = (void*)(spu_extract(tmp_localData2, 0));
    register int tmp_dmaListSize = spu_extract(tmp_localData2, 1);

    // TRACE
    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Output data for index %d was sent...\n", (int)getSPEID(), msgIndex);
        STATETRACE_OUTPUT(msgIndex);
      }
    #endif

    // Free the memory being used by the Work Request
    if (__builtin_expect(tmp_localMemPtr != NULL, 1)) {
      //_free_align(tmp_localMemPtr);
      FREE(tmp_localMemPtr);
    }

    // If the DMA list was allocated, free it
    if (__builtin_expect(tmp_dmaListSize > SPE_DMA_LIST_LENGTH, 0)) {
      //_free_align(tmp_dmaList);
      FREE(tmp_dmaList);
    }

    // Clear the memory and dma related local data fields
    tmp_localData0 = spu_insert((int)NULL, tmp_localData0, 2);  // localMemPtr = NULL
    tmp_localData1 = spu_insert((int)NULL, tmp_localData1, 0);  // readWritePtr = NULL
    tmp_localData1 = spu_insert((int)NULL, tmp_localData1, 1);  // readOnlyPtr = NULL
    tmp_localData1 = spu_insert((int)NULL, tmp_localData1, 2);  // writeOnlyPtr = NULL
    tmp_localData2 = spu_insert((int)NULL, tmp_localData2, 0);  // dmaList = NULL
    tmp_localData2 = spu_insert(-1, tmp_localData2, 1);         // dmaListSize = -1

    #if SPE_NOTIFY_VIA_MAILBOX != 0

      // Check to see if there is an available entry in the outbound mailbox
      if (spu_stat_out_mbox() > 0) {

        // Clear the entry
        //localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_CLEAR;
        tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_CLEAR, tmp_localData0, 0);

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

      // Update the notify counter to notify the PPE that this Work Request has completed
      register vector signed int* tmp_msgQueueEntry = (vector signed int*)(msgQueue[msgIndex]);
      register vector signed int tmp_msgQueueData0 = *tmp_msgQueueEntry;
      register int tmp_counter0 = spu_extract(tmp_msgQueueData0, 0);

      // NOTE : This code will fill in the SPENotify counter, errorCode, and commit timing fields
      //   at the same time.

      // Get a pointer to the notify queue entry
      register int tmp_offset = msgIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
      register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

      register unsigned int returnCode = (SPE_MESSAGE_OK << 16) | (tmp_counter0 & 0xFFFF);
      //register vector unsigned int tmp_notifyQueueEntry4 = (*(tmp_notifyQueueEntryPtr + 4));
      register vector unsigned int tmp_notifyQueueEntry4 = { 0, 0, 0, 0 };
      tmp_notifyQueueEntry4 = spu_insert(returnCode, tmp_notifyQueueEntry4, 2);

      // TIMING
      #if SPE_TIMING != 0
        // Get the ending time
        register unsigned long long int tmp_endTime;
        getTimer64(tmp_endTime);

        // Grab the initial recv time
        register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
        register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);

        // Calculate the start and end offsets from recvStartTime
        register unsigned int tmp_commitStart = (unsigned int)(tmp_startTime - tmp_recvTimeStart);
        register unsigned int tmp_commitEnd = (unsigned int)(tmp_endTime - tmp_recvTimeStart);

        // Add the start and end times to the notifyQueueEntry
        tmp_notifyQueueEntry4 = spu_insert(tmp_commitStart, tmp_notifyQueueEntry4, 0);
        tmp_notifyQueueEntry4 = spu_insert(tmp_commitEnd, tmp_notifyQueueEntry4, 1);
      #endif

      // Write the notification queue entry to the LS
      (*(tmp_notifyQueueEntryPtr + 4)) = tmp_notifyQueueEntry4;

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
          printf("SPE_%d :: [TRACE] :: Notified PPE -> msgIndex = %d, returnCode = 0x%08u\n",
                 (int)getSPEID(), msgIndex, returnCode
                );
	}
      #endif

      // Update the message's state
      tmp_localData0 = spu_insert(SPE_MESSAGE_STATE_CLEAR, tmp_localData0, 0);

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

    // Write the contents of the tmp_localData registers back into the LS
    *(((vector signed int*)tmp_localDataEntry) + 0) = tmp_localData0;
    *(((vector signed int*)tmp_localDataEntry) + 1) = tmp_localData1;
    *(((vector signed int*)tmp_localDataEntry) + 2) = tmp_localData2;

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

  // STATS2
  #if SPE_STATS2 != 0
    #if SPE_STATS2 > 0
    if (msgIndex == SPE_STATS2) {
    #endif
      register unsigned int clock2;
      getTimer(clock2);
      printf("SPE_%d :: [0x%08x] SPE_MESSAGE_STATE_ERROR for %d\n", (int)getSPEID(), clock2, msgIndex);
    #if SPE_STATS2 > 0
    }
    #endif
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
      unsigned long long int curTime;
      getTimer64(curTime);
      printf("SPE_%d :: [TRACE] @ %llu :: Processing SPE_MESSAGE_STATE_ERROR for index %d (errorCode = %d)...\n",
             (int)getSPEID(), curTime, msgIndex, localMsgQData[msgIndex].errorCode
            );
      debug_displayActiveMessageQueue(0x0, NULL/*msgState*/, "(*)");
      STATETRACE_OUTPUT(msgIndex);
    }
  #endif

  // NOTE: All clean-up should be taken care of by the code placing the message into the error
  //   state (that way the code here does not have to handle all cases).

  #if SPE_NOTIFY_VIA_MAILBOX != 0

    // Check to see if there is an available entry in the outbound mailbox
    if (spu_stat_out_mbox() > 0) {

      // Clear the entry
      localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_CLEAR;

      // Send the index of the entry in the message queue to the PPE along with the ERROR code
      spu_write_out_mbox(MESSAGE_RETURN_CODE(msgIndex, localMsgQData[msgIndex].errorCode));

      // Clear the error
      localMsgQData[msgIndex].errorCode = SPE_MESSAGE_OK;

      // DEBUG
      #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
        wrCompletedCounter++;
      #endif
    }

  #else

    // Update the notify counter to notify the PPE
    register unsigned int tmp_errorCode = localMsgQData[msgIndex].errorCode;
    register unsigned int tmp_counter0 = msgQueue[msgIndex]->counter0;
    // NOTE: This code will update the SPENotify's errorCode and counter fields at the same time.
    register vector unsigned int* notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + msgIndex;
    register unsigned int returnCode = (tmp_errorCode << 16) | (tmp_counter0 & 0xFFFF);
    register vector unsigned int notifyQueueEntry = (*notifyQueueEntryPtr);
    notifyQueueEntry = spu_insert(returnCode, notifyQueueEntry, 3);
    (*notifyQueueEntryPtr) = notifyQueueEntry;

    #if ENABLE_TRACE != 0
      if (__builtin_expect(msgQueue[msgIndex]->traceFlag, 0)) {
        printf("SPE_%d :: [TRACE] :: Notify Msg ==> counter:%d errorCode:%d...\n",
               (int)getSPEID(), tmp_counter0, tmp_errorCode
              );
      }
    #endif

    // Update the message's state
    localMsgQData[msgIndex].msgState = SPE_MESSAGE_STATE_CLEAR;

    // Clear the error
    localMsgQData[msgIndex].errorCode = SPE_MESSAGE_OK;

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
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_PRE_FETCHING_LIST, 0)) {
      processMsgState_preFetchingList(i);
    }
  }

  // SPE_MESSAGE_STATE_FETCHING_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_FETCHING_LIST, 0)) {
      processMsgState_fetchingList(i);
    }
  }

  // SPE_MESSAGE_STATE_LIST_READY_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_LIST_READY_LIST, 0)) {
      processMsgState_listReadyList(i);
    }
  }

  // SPE_MESSAGE_STATE_PRE_FETCHING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_PRE_FETCHING, 0)) {
      processMsgState_preFetching(i);
    }
  }

  // SPE_MESSAGE_STATE_FETCHING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_FETCHING, 0)) {
      processMsgState_fetching(i);
    }
  }

  // SPE_MESSAGE_STATE_READY
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_READY, 0)) {
      processMsgState_ready(i);
    }
  }

  // SPE_MESSAGE_STATE_EXECUTED_LIST
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_EXECUTED_LIST, 0)) {
      processMsgState_executedList(i);
    }
  }

  // SPE_MESSAGE_STATE_EXECUTED
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_EXECUTED, 0)) {
      processMsgState_executed(i);
    }
  }

  // SPE_MESSAGE_STATE_COMMITTING
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_COMMITTING, 0)) {
      processMsgState_committing(i);
    }
  }

  // SPE_MESSAGE_STATE_PRE_ERROR
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    if (__builtin_expect(localMsgQData[i].msgState == SPE_MESSAGE_STATE_ERROR, 0)) {
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

void processMsgState_doNothing(int msgIndex) {
  printf("SPE_%d :: [ERROR] :: !!! processMsgState_doNothing() called !!!\n", (int)getSPEID());
}

// NOTE : 4 'unsigned int's per state (function pointer, tryAgain value, xx, xx)
unsigned int stateLookupTable[] = {
  (unsigned int)processMsgState_sent,            0xFFFFFFFF, 0, 0, //  0 = CLEAR
  (unsigned int)processMsgState_doNothing,       0x00000000, 0, 0, //  1 = SENT
  (unsigned int)processMsgState_preFetching,     0x00000000, 0, 0, //  2 = PRE_FETCHING
  (unsigned int)processMsgState_preFetchingList, 0xFFFFFFFF, 0, 0, //  3 = PRE_FETCHING_LIST
  (unsigned int)processMsgState_fetching,        0x00000000, 0, 0, //  4 = FETCHING
  (unsigned int)processMsgState_listReadyList,   0x00000000, 0, 0, //  5 = LIST_READY_LIST
  (unsigned int)processMsgState_fetchingList,    0xFFFFFFFF, 0, 0, //  6 = FETCHING_LIST
  (unsigned int)processMsgState_ready,           0x00000000, 0, 0, //  7 = READY
  (unsigned int)processMsgState_executed,        0x00000000, 0, 0, //  8 = EXECUTED
  (unsigned int)processMsgState_executedList,    0x00000000, 0, 0, //  9 = EXECUTED_LIST
  (unsigned int)processMsgState_committing,      0x00000000, 0, 0, // 10 = COMMITTING
  (unsigned int)processMsgState_doNothing,       0x00000000, 0, 0, // 11 = FINISHED
  (unsigned int)processMsgState_error,           0x00000000, 0, 0  // 12 = ERROR
};


#if SPE_USE_STATE_LOOKUP_TABLE != 0

// Switch scheduler method
inline void speSchedulerInner() {

  register int i;

  // STATS
  #if SPE_STATS != 0
    register int stateSampleFlag = 0;
    int wrStateCount_[SPE_MESSAGE_NUM_STATES] = { 0 };
    wrInUseCountCounter++;
  #endif

  // STATS2
  #if SPE_STATS2 != 0
    static unsigned int clearStateCount = 0;
    printf("SPE_%d :: speSchedulerInner() [clearStateCount:%u] - called...\n", (int)getSPEID(), clearStateCount);
  #endif

  // For each message queue entry
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    // STATS
    #if SPE_STATS != 0
      if (localMsgQData[i].msgState != SPE_MESSAGE_STATE_CLEAR) wrInUseLCount++;
      if (msgQueue[i]->state != SPE_MESSAGE_STATE_CLEAR) wrInUseQCount++;
    #endif

    register int tmp_loopCondition;
    register LocalMsgQData* tmp_localMsgQData = localMsgQData;

    // Execute the processing function associated with the messaqe queue entry's state
    do {  // while (localMsgQData[i].msgState != msgState_last)

      // STATS
      #if SPE_STATS != 0
        wrStateCount_[localMsgQData[i].msgState]++;
      #endif

      // Load the Work Request's state
      register LocalMsgQData* tmp_localDataEntry = (tmp_localMsgQData + i);
      register vector signed int tmp_localData0 = *((vector signed int*)tmp_localDataEntry);
      register int tmp_lastMsgState = spu_extract(tmp_localData0, 0);

      // Load the stateLookupTable entry for the Work Request's state
      register vector unsigned int tmp_stateLookupTableEntry = *(((vector unsigned int*)stateLookupTable) + tmp_lastMsgState);
      register void(*tmp_processMsgState_func)(int) = (void(*)(int))(spu_extract(tmp_stateLookupTableEntry, 0));
      register unsigned int tmp_tryAgainMask = spu_extract(tmp_stateLookupTableEntry, 1);

      // Call the state processing function
      tmp_processMsgState_func(i);

      // Re-Load the Work Request's state
      register vector signed int tmp_localData0_0 = *((vector signed int*)tmp_localDataEntry);
      register int tmp_msgState = spu_extract(tmp_localData0_0, 0);

      // Calculate the loop test
      tmp_loopCondition = (tmp_msgState != tmp_lastMsgState) & tmp_tryAgainMask;

    } while (tmp_loopCondition != 0);

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
}

#else // SPE_USE_STATE_LOOKUP_TABLE != 0

inline void speSchedulerInner() {

  register int i;
  register int tryAgain = 0;

  // STATS
  #if SPE_STATS != 0
    register int stateSampleFlag = 0;
    int wrStateCount_[SPE_MESSAGE_NUM_STATES] = { 0 };
    wrInUseCountCounter++;
  #endif

  // STATS2
  #if SPE_STATS2 != 0
    static unsigned int clearStateCount = 0;
    printf("SPE_%d :: speSchedulerInner() [clearStateCount:%u] - called...\n", (int)getSPEID(), clearStateCount);
  #endif

  // For each message queue entry
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    // STATS
    #if SPE_STATS != 0
      if (localMsgQData[i].msgState != SPE_MESSAGE_STATE_CLEAR) wrInUseLCount++;
      if (msgQueue[i]->state != SPE_MESSAGE_STATE_CLEAR) wrInUseQCount++;
    #endif

    register int msgState_last;

    // Execute the processing function associated with the messaqe queue entry's state
    do {  // while (localMsgQData[i].msgState != msgState_last)

      // STATS
      #if SPE_STATS != 0
        wrStateCount_[localMsgQData[i].msgState]++;
      #endif

      // Record the current state
      msgState_last = localMsgQData[i].msgState;

      // Process the message according to the current state
      switch (localMsgQData[i].msgState) {

        case SPE_MESSAGE_STATE_CLEAR:

          // STATS2
          #if SPE_STATS2 != 0
            clearStateCount++;
          #endif

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
          #if LIMIT_READY <= 0
	    tryAgain = 1;  // Input data has arrived... ready to executed
          #else
            tryAgain = 0;  // READY state handled outside this loop... don't try again as it will not do anything
          #endif
          // STATS
          #if SPE_STATS != 0
            stateSampleFlag++;  // DEBUG
          #endif
          break;

        case SPE_MESSAGE_STATE_READY:
          #if LIMIT_READY <= 0
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
                 (int)getSPEID(), i, localMsgQData[i].msgState
                );
          break;

      } // end switch(localMsgQData[i].msgState)

    } while ((localMsgQData[i].msgState != msgState_last) && (tryAgain != 0));
  } // end for (i < SPE_MESSAGE_QUEUE_LENGTH)

  #if LIMIT_READY > 0
  {
    static int runIndex = 0;
    register int iOffset;
    register int leftToRun = LIMIT_READY;
    for (iOffset = 0; leftToRun > 0 && iOffset < SPE_MESSAGE_QUEUE_LENGTH; iOffset++) {
      register int ri = (runIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;
      if (localMsgQData[ri].msgState == SPE_MESSAGE_STATE_READY) {
        processMsgState_ready(ri);
        if ((msgQueue[ri]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {
          processMsgState_executed(ri);
        } else {
          processMsgState_executedList(ri);
        }
        leftToRun--;
      }
    }
    runIndex = (runIndex + iOffset - 1) % SPE_MESSAGE_QUEUE_LENGTH;
  }
  #endif

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

#endif // SPE_USE_STATE_LOOKUP_TABLE != 0

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

    #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
      msgLastState[i] = SPE_MESSAGE_STATE_CLEAR;
    #endif

    LocalMsgQData* msgQData = &(localMsgQData[i]);
    msgQData->msgState = SPE_MESSAGE_STATE_CLEAR;
    msgQData->msgCounter = 0;
    msgQData->localMemPtr = NULL;
    msgQData->readWritePtr = NULL;
    msgQData->readOnlyPtr = NULL;
    msgQData->writeOnlyPtr = NULL;
    msgQData->dmaList = NULL;
    msgQData->dmaListSize = -1;
    msgQData->errorCode = SPE_MESSAGE_OK;
  }

  #if DOUBLE_BUFFER_MESSAGE_QUEUE != 0
    msgQueueRaw = msgQueueRaw0;
    msgQueueRawAlt = msgQueueRaw1;
    msgQueue = msgQueue0;
    msgQueueAlt = msgQueue1;
  #endif

  // TIMING - Start the timer as close to the notification to the PPE that this SPE Runtime is alive
  #if SPE_TIMING != 0
    startTimer();
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

  // STATS2
  #if SPE_STATS2 != 0
    startTimer();
  #endif


  // DEBUG
  register int speID = (int)getSPEID();


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

      // SPE_MESSAGE_COMMAND_RESET_CLOCK
      if (command == SPE_MESSAGE_COMMAND_RESET_CLOCK) {

        // TIMING
        #if SPE_TIMING != 0
          // NOTE : TODO : The code to stop the timer has been commented out because
          //   compiling a call to stopTimer64() followed by startTimer() with "-O3" seems to
          //   mess up the timer somehow... look into this.  Just using startTimer() seems to work.
          //register unsigned long long int tmp_ignore;
          //stopTimer64(tmp_ignore);
          startTimer();
        #endif
      }

      // Reduce the count of remaining entries
      inMBoxCount--;
    }

    // NOTE : Removed this for performance.  When closing the Offload API, the PPE can wait one more
    //   scheduler outer-loop iteration.
    // Check the keepLooping flag
    //if (__builtin_expect(keepLooping == FALSE, 0))
    //  continue;

    // Let the user know that the SPE Runtime is still running...
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      if (__builtin_expect(stillAliveCounter == 0, 0)) {
        printf("SPE_%d :: still going... WRs Completed = %d\n", (int)getSPEID(), wrCompletedCounter);
        stillAliveCounter = SPE_DEBUG_DISPLAY_STILL_ALIVE;

        // DEBUG
        //register int speid = (int)getSPEID();
        //register int iii = 0;
        //printf("SPE_%d ::   Local MSG Queue Data:\n", speid);
        //for (iii = 0; iii < SPE_MESSAGE_QUEUE_LENGTH; iii++) {
        //  printf("SPE_%d ::     %d: state = %d (PPE: %d)\n",
        //         speid, iii, localMsgQData[iii].msgState, msgQueue[iii]->state
        //        );
	//}

      }
      stillAliveCounter--;
    #endif


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
      #define SPE_MESSAGE_QUEUE_THROTTLE_VALUE   0

      #if SPE_MESSAGE_QUEUE_THROTTLE_VALUE > 0

        // Throttle Counter
        static int mqtc = SPE_MESSAGE_QUEUE_THROTTLE_VALUE;
        mqtc--;

        // If finished and there is free DMA queue entry... swap the buffers and read the message queue again
        if (__builtin_expect(mqtc <= 0, 0)) {

      #endif

      if (tagStatus && mfc_stat_cmd_queue() > 0) {

        // STATS2
        #if SPE_STATS2 != 0
          printf("SPE_%d :: MsgQ arrived, swapping buffers...\n", (int)getSPEID());
        #endif

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

    // TIMER - Service the timer
    #if SPE_TIMING != 0
      serviceTimer();
    #endif

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

    #define SPE_NOTIFY_QUEUE_THROTTLE_VALUE  0

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
    if (tagStatus && numDMAQueueEntries > 0) {

      // STATS2
      #if SPE_STATS2 != 0
        printf("SPE_%d :: Posting Notification Queue...\n", (int)getSPEID());
      #endif

      spu_mfcdma32(notifyQueueRaw, (PPU_POINTER_TYPE)(speData->notifyQueue), SPE_NOTIFY_QUEUE_BYTE_COUNT, 30, MFC_PUT_CMD);
    }

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
        if (localMsgQData[i].msgState != msgLastState[i]) {
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
            sim_printf("[0x%llx] :: msgState[%d] = %d\n", id, i, localMsgQData[i].msgState);
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

  #if SPE_USE_OWN_MALLOC <= 0

    // From Section 4.3 of library_SDK.pdf : "The local store memory heap is initialized the first
    //   time a memory heap allocation routine is called."... Do this now so it is ready to go.
    register unsigned int memLeft = SPE_TOTAL_MEMORY_SIZE - SPE_RESERVED_STACK_SIZE - (unsigned int)(&_end);
    memLeft -= 128;  // Buffer zone between stack and heap
    memLeft &= 0xFFFFFF00;  // Force it to be a multiple of 256 bytes
    #if SPE_DEBUG_DISPLAY >= 1
      printf("SPE_%d :: [DEBUG] :: memLeft = %d\n", (int)getSPEID(), memLeft);
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

    local_initMem();

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

  // Call the user's funcLookup() function with funcIndex of SPE_FUNC_INDEX_INIT
  funcLookup(SPE_FUNC_INDEX_INIT, NULL, 0, NULL, 0, NULL, 0, NULL);

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




void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str) {
  #if SPE_DEBUG_DISPLAY >= 1

  // DEBUG
  printf("SPE_%d :: Dumping active portion of message queue...\n", (int)getSPEID());

  int tmp;
  int speID = (int)getSPEID();

  for (tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
    register int msgS = localMsgQData[tmp].msgState;
    if (msgS != SPE_MESSAGE_STATE_CLEAR || msgS < SPE_MESSAGE_STATE_MIN || msgS > SPE_MESSAGE_STATE_MAX) {
    //if (1) {
      printf("SPE_%d ::   :: %s%s msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %lu, rwl = %d, ro = %lu, rol = %d, wo = %lu, wol = %d, f = 0x%08X, tm = %u, s = %d(%d), cnt = %d:%d, cmd = %d }\n",
	         speID,
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
                 msgS, //msgState[tmp]
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
      if (localMsgQData[i].msgState == state) {
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
    printf("SPE_%d :   msgState = %d, msgCounter = %d\n", (int)getSPEID(), localMsgQData[i].msgState, localMsgQData[i].msgCounter);
    printf("SPE_%d :   localMemPtr = %p, errorCode = %d\n", (int)getSPEID(), localMsgQData[i].localMemPtr, localMsgQData[i].errorCode);
    printf("SPE_%d :   readOnlyPtr = %p, readWritePtr = %p, writeOnlyPtr = %p\n", (int)getSPEID(), localMsgQData[i].readOnlyPtr, localMsgQData[i].readWritePtr, localMsgQData[i].writeOnlyPtr);
    printf("SPE_%d :   dmaList = %p, dmaListSize = %d\n", (int)getSPEID(), localMsgQData[i].dmaList, localMsgQData[i].dmaListSize);
    register const DMAListEntry* localDMAList = (DMAListEntry*)(localMsgQData[i].localMemPtr);
    for (j = 0; j < localMsgQData[i].dmaListSize; j++)
      printf("SPE_%d :     entry %d :: { ea = 0x%08x, size = %u } -=> { ea = 0x%08x, size = %u }\n",
             (int)getSPEID(), j,
             (localMsgQData[i].dmaList)[j].ea, (localMsgQData[i].dmaList)[j].size,
             localDMAList[j].ea, localDMAList[j].size
            );
    
  }

  // Footer
  printf("SPE_%d : -------------------------------------\n", (int)getSPEID());

}


void startUserTime0() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry3 = (*(tmp_notifyQueueEntryPtr + 3));
  tmp_notifyQueueEntry3 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry3, 2);
  (*(tmp_notifyQueueEntryPtr + 3)) = tmp_notifyQueueEntry3;
}

void endUserTime0() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry3 = (*(tmp_notifyQueueEntryPtr + 3));
  tmp_notifyQueueEntry3 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry3, 3);
  (*(tmp_notifyQueueEntryPtr + 3)) = tmp_notifyQueueEntry3;
}

void startUserTime1() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry5 = (*(tmp_notifyQueueEntryPtr + 5));
  tmp_notifyQueueEntry5 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry5, 0);
  (*(tmp_notifyQueueEntryPtr + 5)) = tmp_notifyQueueEntry5;
}

void endUserTime1() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry5 = (*(tmp_notifyQueueEntryPtr + 5));
  tmp_notifyQueueEntry5 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry5, 1);
  (*(tmp_notifyQueueEntryPtr + 5)) = tmp_notifyQueueEntry5;
}

void startUserTime2() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry5 = (*(tmp_notifyQueueEntryPtr + 5));
  tmp_notifyQueueEntry5 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry5, 2);
  (*(tmp_notifyQueueEntryPtr + 5)) = tmp_notifyQueueEntry5;
}

void endUserTime2() {

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get a pointer to the notify queue entry
  register int tmp_offset = execIndex * (sizeof(SPENotify) / sizeof(vector unsigned int));
  register vector unsigned int* tmp_notifyQueueEntryPtr = ((vector unsigned int*)notifyQueueRaw) + tmp_offset;

  // Grab the initial recv time
  register vector unsigned long long int tmp_notifyQueueEntry0 = *((vector unsigned long long int*)tmp_notifyQueueEntryPtr);
  register unsigned long long int tmp_recvTimeStart = spu_extract(tmp_notifyQueueEntry0, 0);
  
  // Calculate the start and end offsets from recvStartTime
  register unsigned int tmp_timeDelta = (unsigned int)(tmp_time - tmp_recvTimeStart);

  // Write the start and end times to the LS
  register vector unsigned int tmp_notifyQueueEntry5 = (*(tmp_notifyQueueEntryPtr + 5));
  tmp_notifyQueueEntry5 = spu_insert(tmp_timeDelta, tmp_notifyQueueEntry5, 3);
  (*(tmp_notifyQueueEntryPtr + 5)) = tmp_notifyQueueEntry5;
}



void clearUserAccumTime(int index) {

  // Force the index to be 0 <= index <= 3
  index &= 0x3;
  
  // Place this value into userAccumTime
  register int tmp_ptr_offset = ((index & 0x2) >> 1);
  register vector unsigned long long int* tmp_ptr = userAccumTime + tmp_ptr_offset;
  register vector unsigned long long int tmp_startTime = *tmp_ptr;
  tmp_startTime = spu_insert((unsigned long long int)0, tmp_startTime, (index & 0x1));
  *tmp_ptr = tmp_startTime;

  // Place this value into userAccumTimeStart
  tmp_ptr = userAccumTimeStart + tmp_ptr_offset;
  tmp_startTime = *tmp_ptr;
  tmp_startTime = spu_insert((unsigned long long int)0, tmp_startTime, (index & 0x1));
  *tmp_ptr = tmp_startTime;
}

void clearUserAccumTimeAll() {
  register vector unsigned long long int tmp_zeros = { 0, 0 };
  register vector unsigned long long int* tmp_ptr = userAccumTime;
  *(tmp_ptr) = tmp_zeros;
  *(tmp_ptr + 1) = tmp_zeros;
}

void startUserAccumTime(int index) {

  // Force the index to be 0 <= index <= 3
  index &= 0x3;

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Place this value into userAccumTimeStart
  register int tmp_ptr_offset = ((index & 0x2) >> 1);
  register vector unsigned long long int* tmp_ptr = userAccumTimeStart + tmp_ptr_offset;
  register vector unsigned long long int tmp_startTime = *tmp_ptr;
  tmp_startTime = spu_insert(tmp_time, tmp_startTime, (index & 0x1));
  *tmp_ptr = tmp_startTime;
}

void endUserAccumTime(int index) {

  // Force the index to be 0 <= index <= 3
  index &= 0x3;

  // Get the time
  register unsigned long long int tmp_time;
  getTimer64(tmp_time);

  // Get the userAccumTimeStart value and subtract it
  register int tmp_ptr_offset = ((index & 0x2) >> 1);
  register vector unsigned long long int* tmp_ptr = userAccumTimeStart + tmp_ptr_offset;
  register vector unsigned long long int tmp_startTime_vec = *tmp_ptr;
  register unsigned long long int tmp_startTime = spu_extract(tmp_startTime_vec, (index & 0x1));
  register unsigned long long int tmp_timeDelta = tmp_time - tmp_startTime;

  // Add the tmp_timeDelta value to the accumulator value
  tmp_ptr = userAccumTime + tmp_ptr_offset;
  tmp_startTime_vec = *tmp_ptr;
  tmp_startTime = spu_extract(tmp_startTime_vec, (index & 0x1));
  tmp_startTime += tmp_timeDelta;
  tmp_startTime_vec = spu_insert(tmp_startTime, tmp_startTime_vec, (index & 0x1));
  *tmp_ptr = tmp_startTime_vec;
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

// TODO : This data structure is too large (but fast)... optimize this for size more
typedef struct __memory_block_record {
  unsigned int blockAddr;
  unsigned int inUseFlag;   // NOTE : 0xFFFFFFFF = Free, 0x00000000 = In Use
  unsigned int blockCount;
  unsigned int __padding__;
} MemBlockRec;


MemBlockRec* memBlockTable = NULL;
int memBlockTableSize = 0;


void local_initMem() {

  register int i;

  // Caclulate the starting and ending addresses of the heap (128 bytes of "buffer" on either side)
  register unsigned int startAddr = ROUNDUP_128((unsigned int)(&(_end))) + 128; // '+ 128' for buffer between heap and code/data
  register unsigned int endAddr = SPE_TOTAL_MEMORY_SIZE - SPE_RESERVED_STACK_SIZE - 128;
  register int memAvail = ((int)endAddr - (int)startAddr) - 128;  // '- 128' for alignment of heap memory blocks

  // Make sure there is enough heap memory (for now, just warn the user if there is not)
  if (__builtin_expect(memAvail <= SPE_MEMORY_BLOCK_SIZE, 0)) {
    printf("SPE_%d :: [ERROR] :: Initing memory, not enough memory for a single block... bad things are headed your way...\n",
           (int)getSPEID()
          );
    return;
  }

  // Calculate the number of blocks that will fit in the heap
  register int memPerBlock = SPE_MEMORY_BLOCK_SIZE + sizeof(MemBlockRec);
  register int numBlocks = memAvail / memPerBlock;
  memBlockTableSize = numBlocks;

  // Setup the memory block table
  memBlockTable = (MemBlockRec*)startAddr;
  register unsigned int blockAddr = startAddr + (numBlocks * sizeof(MemBlockRec));
  blockAddr = ROUNDUP_128(blockAddr);
  for (i = 0; i < numBlocks; i++) {

    // Init the record
    memBlockTable[i].blockAddr = blockAddr;
    memBlockTable[i].inUseFlag = 0xFFFFFFFF;
    memBlockTable[i].blockCount = 0;
    memBlockTable[i].__padding__ = 0;

    // Advance the blockAddr "pointer"
    blockAddr += SPE_MEMORY_BLOCK_SIZE;
  }
}

void* local_malloc(int size, int alignment) {

  register int i;
  register void* rtnAddr = NULL;

  // Verify the parameters
  if (__builtin_expect(size <= 0, 0)) return NULL;
  if (__builtin_expect(alignment <= 0, 0)) return NULL;

  // Calculate the loop step from the alignment
  register int tmp_alignSize = 0x01 << alignment;
  register int tmp_blockSize = SPE_MEMORY_BLOCK_SIZE;
  register int tmp_div = tmp_alignSize / tmp_blockSize;
  if (__builtin_expect(tmp_div <= 0, 1)) tmp_div = 1;  // Common case should be alignSize <= blockSize

  // Search for the blocks in the block table
  register int numBlocks = memBlockTableSize;
  register int consecCount = 0;
  register int blocksNeeded = ROUNDUP(size, SPE_MEMORY_BLOCK_SIZE) / SPE_MEMORY_BLOCK_SIZE;

  // Pre-load the first memory block record
  register vector unsigned int* tmp_memBlockTablePtr = (vector unsigned int*)memBlockTable;
  register vector unsigned int tmp_memBlockRec_next = *tmp_memBlockTablePtr;

  for (i = 0; i < numBlocks; i++) {

    // Start loading the next memory block record
    register vector unsigned int tmp_memBlockRec = tmp_memBlockRec_next;

    // Get the inUseFlag
    register unsigned int tmp_inUseFlag = spu_extract(tmp_memBlockRec, 1);

    // Add one to the consecCount and then "and" with inUseFlag (all 1s or all 0s)
    consecCount = (consecCount + 1) & tmp_inUseFlag;

    // Calculate a skip amount (first block in a series of will have blockCount set
    //   so if an in use block is found, skip ahead the block count)
    register unsigned int tmp_blockCount = spu_extract(tmp_memBlockRec, 2);
    register unsigned int tmp_inUseFlag_inv = ((unsigned int)0xFFFFFFFF) - tmp_inUseFlag;
    register unsigned int skipCount = (tmp_blockCount - 1) & tmp_inUseFlag_inv;

    // Check to see if enough blocks have been found
    if (__builtin_expect(consecCount >= blocksNeeded, 0)) {

      // Set the first block's record (inUseFlag and blockCount)      
      register vector unsigned int* tmp_firstRecPtr = tmp_memBlockTablePtr - (consecCount - 1);
      register vector unsigned int tmp_firstRec = *tmp_firstRecPtr;
      rtnAddr = (void*)(spu_extract(tmp_firstRec, 0));
      tmp_firstRec = spu_insert(0x00000000, tmp_firstRec, 1);
      tmp_firstRec = spu_insert(consecCount, tmp_firstRec, 2);
      *tmp_firstRecPtr = tmp_firstRec;

      break;
    }

    // Start loading the next memory block record
    i += skipCount;  // Add to i before incrementing since i will be incremented from the loop itself
    tmp_memBlockTablePtr += (skipCount + 1);
    tmp_memBlockRec_next = *(tmp_memBlockTablePtr);
  }

  return rtnAddr;
}

void local_free(void* addr) {

  if (__builtin_expect(addr == NULL, 0)) return;

  // Calculate the index into the memory block table
  register vector unsigned int* tmp_memBlockTablePtr = (vector unsigned int*)memBlockTable;
  register vector unsigned int tmp_firstRec = *tmp_memBlockTablePtr;
  register unsigned int tmp_firstBlockAddr = spu_extract(tmp_firstRec, 0);
  register int tableIndex = ((unsigned int)addr - tmp_firstBlockAddr) / SPE_MEMORY_BLOCK_SIZE;

  // Load, clear, and store the record for the first block's record
  register vector unsigned int* tmp_recPtr = tmp_memBlockTablePtr + tableIndex;
  register vector unsigned int tmp_rec = *tmp_recPtr;
  tmp_rec = spu_insert(0xFFFFFFFF, tmp_rec, 1);
  tmp_rec = spu_insert(0x00000000, tmp_rec, 2);
  *tmp_recPtr = tmp_rec;
}

#endif  // SPE_USE_OWN_MALLOC >= 1
