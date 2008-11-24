#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

// DEBUG
#include <sys/time.h>

//extern "C" {
  #include <libspe2.h>
//}

#include "spert_common.h"
#include "spert.h"

extern "C" {
  #include "spert_ppu.h"
}

#include "pthread.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#define WRHANDLES_NUM_INITIAL  (1024 * 8)
#define WRHANDLES_GROW_SIZE    (1024 * 1)

#define WRGROUPHANDLES_NUM_INITIAL  (1024 * 2)
#define WRGROUPHANDLES_GROW_SIZE    (1024 * 1)

#define MESSAGE_RETURN_CODE_INDEX(rc)  ((unsigned int)(rc) & 0x0000FFFF)
#define MESSAGE_RETURN_CODE_ERROR(rc)  (((unsigned int)(rc) >> 16) & 0x0000FFFF)

#define USE_MESSAGE_QUEUE_FREE_LIST   1  // Set to non-zero to use a linked list of free message queue entries.
                                         // Note: This can get a free message queue entry in constant time,
                                         //   i.e. - not f(# SPEs, message queue length), but does not use the
                                         //   heuristics for load-balancing (at the moment).
                                         // Note: Currently, the SPE affinity mask is ignored if this is set.

#define ENABLE_LAST_WR_TIMES          0  // Set to allow a call to displayLastWRTimes() to print the last time
                                         //   (according to gettimeofday()) that a work request finished for
                                         //   each SPE.


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structures

typedef struct __pointer_list {
  void *ptr;
  struct __pointer_list *next;
} PtrList;

typedef struct __msgQEntry_list {
  SPEMessage* msgQEntryPtr;
  int speIndex;
  int entryIndex;
  struct __msgQEntry_list *next;
} MSGQEntryList;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data


/// SDK 2.0 ///
//SPEThread* speThreads[NUM_SPE_THREADS];

/// SDK 2.1 ///
SPEThread** speThreads = NULL;
int numSPEThreads = -1;


unsigned long long int wrCounter = 0;

void (*callbackFunc)(void*) = NULL;
void (*groupCallbackFunc)(void*) = NULL;
void (*errorHandlerFunc)(int, void*, WRHandle) = NULL;

// Work Request Structures
PtrList *allocatedWRHandlesList = NULL;

// Work Request Group Structures
PtrList *allocatedWRGroupHandlesList = NULL;

// A Queue of WorkRequest structures that have been filled but are waiting on an open SPE messageQueue slot
WorkRequest *wrQueuedHead = NULL;
WorkRequest *wrQueuedTail = NULL;

// A Queue of WorkRequest structures that have finished execution on some SPE and are waiting for the
//   caller to call isFinished() using the corresponding handle.
// NOTE: If InitOffloadAPI() was originally called with a callback function, the callback function will be
//   called instead and the WorkRequest structure will be immediately free'd.
WorkRequest *wrFinishedHead = NULL;
WorkRequest *wrFinishedTail = NULL;

// A Queue of WorkRequest structures that are free to be used for future work requests.
WorkRequest *wrFreeHead = NULL;
WorkRequest *wrFreeTail = NULL;

// A Queue of WRGroup structures that are free to be used for future groups
WRGroup *wrGroupFreeHead = NULL;
WRGroup *wrGroupFreeTail = NULL;

// A Queue of free Work Request Entries
#if USE_MESSAGE_QUEUE_FREE_LIST != 0
MSGQEntryList *msgQEntryFreeHead = NULL;
MSGQEntryList *msgQEntryFreeTail = NULL;

/// SDK 2.0 ///
//MSGQEntryList __msgQEntries[NUM_SPE_THREADS * SPE_MESSAGE_QUEUE_LENGTH];
//int msgQListLen = NUM_SPE_THREADS * SPE_MESSAGE_QUEUE_LENGTH; // Initially all free

/// SDK 2.1 ///
MSGQEntryList* __msgQEntries;
int msgQListLen = -1;

#endif

// This is used in an attempt to more evenly distribute the workload amongst all the SPE Threads
int speSendStartIndex = 0;

// A counter used to give each SPE a unique ID
unsigned short vIDCounter = 0;


// DEBUG
int idCounter = 0;

// TRACE
#if ENABLE_TRACE != 0
  int traceFlag = 0;
#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// "Projections" Data Structures and Function Prototypes

#if SPE_TIMING != 0

  #define PROJ_BUF_SIZE   (32 * 1024)

  typedef struct __projections_buffer_entry {

    //unsigned long long int startTime;
    //unsigned int runTime;
    unsigned int speIndex;
    unsigned int funcIndex;

    unsigned long long int recvTimeStart;
    unsigned int recvTimeEnd;
    unsigned int preFetchingTimeStart;
    unsigned int preFetchingTimeEnd;
    unsigned int fetchingTimeStart;
    unsigned int fetchingTimeEnd;
    unsigned int readyTimeStart;
    unsigned int readyTimeEnd;
    unsigned int userTimeStart;
    unsigned int userTimeEnd;
    unsigned int executedTimeStart;
    unsigned int executedTimeEnd;
    unsigned int commitTimeStart;    
    unsigned int commitTimeEnd;

    unsigned int userTime0Start;
    unsigned int userTime0End;
    unsigned int userTime1Start;
    unsigned int userTime1End;
    unsigned int userTime2Start;
    unsigned int userTime2End;

    unsigned long long int userAccumTime0;
    unsigned long long int userAccumTime1;
    unsigned long long int userAccumTime2;
    unsigned long long int userAccumTime3;

  } ProjBufEntry;

  // Buffer to hold timing data until the entries are flushed to the file
  ProjBufEntry projBuf[PROJ_BUF_SIZE];
  int projBufCount = 0;
  int totalProjSampleCount = 0;

  FILE* projFile;

  void openProjFile(char* name);
  void closeProjFile();
  void addProjEntry(SPENotify* notifyEntry, int speIndex, int funcIndex);
  void flushProjBuf();

#endif




//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

SPEThread* createSPEThread(SPEData *speData);
SPEThread** createSPEThreads(SPEThread **speThreads, int numThreads);

int sendSPEMessage(SPEThread* speThread, WorkRequest* wrPtr, int command);
//int sendSPEMessage(SPEThread* speThread, int qIndex, WorkRequest* wrPtr, int command);
//int sendSPECommand(SPEThread* speThread, int command);

WorkRequest* createWRHandles(int numHandles);
WRGroup* createWRGroupHandles(int numHandles);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Offload API Function Bodies


// DEBUG
void displayList(char* listName, WorkRequest* list) {
  printf("PPE :: List \"%s\" = {\n", listName);
  WorkRequest* entry = list;
  while (entry != NULL) {
    printf("PPE ::   %p\n", entry);
    entry = entry->next;
  }
  printf("PPE :: }\n");
}


// DEBUG
//void displayMessageQueue(int speNum) { displayMessageQueue(speThreads[speNum]); }
void displayMessageQueue(SPEThread* speThread) {
  int speIndex = -1;
  int i;

  //for (i = 0; i < NUM_SPE_THREADS; i++) if (speThreads[i] == speThread) speIndex = i;
  for (i = 0; i < numSPEThreads; i++) if (speThreads[i] == speThread) speIndex = i;

  printf("OffloadAPI :: Message Queue for SPE_%d...\n", speIndex);
  for (i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    SPEMessage *msg = (SPEMessage*)((char*)(speThread->speData->messageQueue) + (i * SIZEOF_16(SPEMessage)));
    printf("OffloadAPI ::   [%d]>> state = %d, wrPtr = %p, rw = %lu, ro = %lu, wo = %lu\n", i, msg->state, msg->wrPtr, msg->readWritePtr, msg->readOnlyPtr, msg->writeOnlyPtr);
  }
}


//inline int sendSPEMessage(SPEThread *speThread, int qIndex, WorkRequest *wrPtr, int command) {
inline int sendSPEMessage(SPEThread *speThread, int qIndex, WorkRequest *wrPtr, int command, DMAListEntry* dmaListSrc) {

  // Get a pointer to the message queue entry
  volatile SPEMessage* msg = (SPEMessage*)(((char*)speThread->speData->messageQueue) + (qIndex * SIZEOF_16(SPEMessage)));
  
  // Fill in the message queue entry
  msg->funcIndex = wrPtr->funcIndex;
  msg->readWritePtr = (PPU_POINTER_TYPE)(wrPtr->readWritePtr);
  msg->readWriteLen = wrPtr->readWriteLen;
  msg->readOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->readOnlyPtr);
  msg->readOnlyLen = wrPtr->readOnlyLen;
  msg->writeOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->writeOnlyPtr);
  msg->writeOnlyLen = wrPtr->writeOnlyLen;
  msg->flags = wrPtr->flags;

  // Copy the DMA list (if is list WR and the dma list is small enought... otherwise, don't bother)
  if ((wrPtr->flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST) {
    register int dmaListSize = wrPtr->readWriteLen + wrPtr->readOnlyLen + wrPtr->writeOnlyLen;
    if (__builtin_expect(dmaListSize <= SPE_DMA_LIST_LENGTH, 1)) {
      register volatile DMAListEntry* msgDMAList = msg->dmaList;
      //register DMAListEntry* wrDMAList = (DMAListEntry*)(wrPtr->readWritePtr);
      //register DMAListEntry* wrDMAList = (DMAListEntry*)(wrPtr->dmaList);
      register DMAListEntry* wrDMAList = dmaListSrc;

      // DEBUG
      //printf(" --- Offload API :: wrPtr = %p, msgDMAList = %p, wrDMAList = %p...\n", wrPtr, msgDMAList, wrDMAList);

      register int i;
      for (i = 0; i < dmaListSize; i++) {
        msgDMAList[i].ea = wrDMAList[i].ea;
        msgDMAList[i].size = ROUNDUP_16(wrDMAList[i].size);

        // DEBUG
        //printf(" --- Offload API :: msgDMAList[%d] = { ea = 0x%08x, size = %u }\n", i, msgDMAList[i].ea, msgDMAList[i].size);
      }

      // For the sake of the checksum, clear out the rest of the dma list entries to 0
      for (; i < SPE_DMA_LIST_LENGTH; i++) {
        msgDMAList[i].ea = 0;
        msgDMAList[i].size = 0;
      }

    }
  }

  // Calculate the total amount of memory that will be needed on the SPE for this message/work-request
  #if 0
    if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST) {
      // The memory needed is the size of the DMA list rounded up times 2 (two lists) and the size of each
      //   of the individual entries in that list all rounded up
      register int numEntries = wrPtr->readWriteLen + wrPtr->readOnlyLen + wrPtr->writeOnlyLen;
      msg->totalMem = ROUNDUP_16(sizeof(DMAListEntry) * numEntries);
      msg->totalMem *= 2;  // Second DMA List within SPE's local store (with LS pointers)
      for (int entryIndex = 0; entryIndex < numEntries; entryIndex++)
        msg->totalMem += ROUNDUP_16(((DMAListEntry*)(wrPtr->readWritePtr))[entryIndex].size);
    } else {
      // The memory needed is the size of the sum of the three buffers each rounded up
      msg->totalMem = ROUNDUP_16(wrPtr->readWriteLen) + ROUNDUP_16(wrPtr->readOnlyLen) + ROUNDUP_16(wrPtr->writeOnlyLen);
    }
  #else
    msg->totalMem = 0;
  #endif

  // TRACE
  #if ENABLE_TRACE != 0
    msg->traceFlag = ((__builtin_expect(wrPtr->traceFlag, 0)) ? (-1) : (0));  // force -1 or 0
  #endif

  msg->state = SPE_MESSAGE_STATE_SENT;
  msg->command = command;
  msg->wrPtr = (PPU_POINTER_TYPE)wrPtr;

  // NOTE: Important that the counter be the last then set (the change in this value is what prompts the
  //   SPE to consider the entry a new entry... even if the state has been set to SENT).
  // NOTE: Only change the value of msg->counter once so the SPE is not confused (i.e. - don't increment
  //   and then check msg->counter direclty).
  int tmp0 = msg->counter0;
  int tmp1 = msg->counter1;
  tmp0++; if (__builtin_expect(tmp0 > 0xFFFF, 0)) tmp0 = 0;  // NOTE: Counter value must fit in 16 bits (packed with error code in notify queue)
  tmp1++; if (__builtin_expect(tmp1 > 0xFFFF, 0)) tmp1 = 0;  // NOTE: Counter value must fit in 16 bits (packed with error code in notify queue)
  __asm__ ("sync");
  msg->counter0 = tmp0;
  msg->counter1 = tmp1;
  __asm__ ("sync");


  // Fill in the check sum
  register int checkSumVal = 0;
  register int* intPtr = (int*)msg;
  register int i;
  for (i = 0; i < (sizeof(SPEMessage) - sizeof(int)) / sizeof(int); i++) {
    checkSumVal += intPtr[i];
  }
  msg->checksum = checkSumVal;


  return qIndex;
}


// Returns 0 on success, non-zero otherwise
inline int sendSPECommand(SPEThread *speThread, unsigned int command) {

  if (__builtin_expect(command < SPE_MESSAGE_COMMAND_MIN || command > SPE_MESSAGE_COMMAND_MAX, 0)) return -1;

  /// SDK 2.0 ///
  //
  //while (spe_stat_in_mbox(speThread->speID) == 0);      // Loop while mailbox is full
  //return spe_write_in_mbox(speThread->speID, command);
  //

  /// SDK 2.1 ///
  return spe_in_mbox_write(speThread->speContext, &command, 1, SPE_MBOX_ALL_BLOCKING);
}


extern "C"
int InitOffloadAPI(void (*cbFunc)(void*),
                   void (*gcbFunc)(void*),
                   void (*errorFunc)(int, void*, WRHandle),
                   char* timingFileName
                  ) {

  // Let the user know that the Offload API is being initialized
  #if DEBUG_DISPLAY >= 1
    printf("----- Offload API : Enabled... Initializing -----\n");
  #endif

  // If the caller specified a callback function, set callbackFunc to point to it
  callbackFunc = ((cbFunc != NULL) ? (cbFunc) : (NULL));

  // If the caller specified a group callback function, set groupCallbackFunc to point to it
  groupCallbackFunc = ((gcbFunc != NULL) ? (gcbFunc) : (NULL));

  // If the caller specified an error handler function, set errorHandlerFunc to point to it
  errorHandlerFunc = ((errorFunc != NULL) ? (errorFunc) : (NULL));


  /// SDK 2.1 ///

  // Set numSPEThreads
  #if NUM_SPE_THREADS <= 0
    // TODO : NOTE : For now, the assumption is made that all CPUs have the same number of SPEs and that
    //   all will be used (the SDK does not seem to have a function that will identify which CPU this
    //   process is running on so can't really check how many physical and/or usable SPEs are available
    //   locally to "this" CPU).  For now, just check how many SPEs CPU 0 has and create that many threads.
    numSPEThreads = spe_cpu_info_get(SPE_COUNT_PHYSICAL_SPES, 0);
  #else
    int numSPEs = spe_cpu_info_get(SPE_COUNT_PHYSICAL_SPES, 0);
    numSPEThreads = NUM_SPE_THREADS;
    if (numSPEThreads > numSPEs) {
      fprintf(stderr, "OffloadAPI :: ERROR : %d SPE(s) were requested but there are only %d physical SPEs\n", numSPEThreads, numSPEs);
      exit(EXIT_FAILURE);
    }
  #endif

  // Create and initialize the speThreads array
  speThreads = new SPEThread*[numSPEThreads];
  for (int i = 0; i < numSPEThreads; i++) speThreads[i] = NULL;

  // Create the msgQEntries array (will be initialized later)
  msgQListLen = numSPEThreads * SPE_MESSAGE_QUEUE_LENGTH;
  __msgQEntries = new MSGQEntryList[msgQListLen];


  // Start Creating the SPE threads
  #if DEBUG_DISPLAY >= 1
    printf(" --- Creating SPE Threads ---\n");
  #endif

  #if CREATE_EACH_THREAD_ONE_BY_ONE
    //for (int i = 0; i < NUM_SPE_THREADS; i++) {
    for (int i = 0; i < numSPEThreads; i++) {

      // Create the SPE Thread (with a default SPEData structure)
      // NOTE: The createSPEThread() call is blocking (it will block until the thread is created).  This
      //   could be changed in future versions so that the init time will be shortened.
      speThreads[i] = createSPEThread(NULL);
      if (speThreads[i] == NULL) {
        fprintf(stderr, "ERROR : Failed to create SPE Thread %d... Exiting\n", i);
        exit(EXIT_FAILURE);
      }

      // Display information about the thread that was just created
      #if DEBUG_DISPLAY >= 1
        printf("SPE_%d Created {\n", i);
        printf("  speThreads[%d]->speData->messageQueue = %p\n", i, (void*)(speThreads[i]->speData->messageQueue));
        printf("  speThreads[%d]->speData->messageQueueLength = %d\n", i, speThreads[i]->speData->messageQueueLength);
        printf("  speThreads[%d]->speID = %d\n", i, speThreads[i]->speID);
        printf("}\n");
      #endif
    }
  #else

    //if (createSPEThreads(speThreads, NUM_SPE_THREADS) == NULL) {
    if (createSPEThreads(speThreads, numSPEThreads) == NULL) {
      fprintf(stderr, "OffloadAPI :: ERROR :: createSPEThreads returned NULL... Exiting.\n");
      exit(EXIT_FAILURE);
    }

    // Display information about the threads that were just created
    #if DEBUG_DISPLAY >= 1
      //for (int i = 0; i < NUM_SPE_THREADS; i++) {
      for (int i = 0; i < numSPEThreads; i++) {
        printf("SPE_%d Created {\n", i);
        printf("  speThreads[%d]->speData->messageQueue = %p\n", i, (void*)(speThreads[i]->speData->messageQueue));
        printf("  speThreads[%d]->speData->messageQueueLength = %d\n", i, speThreads[i]->speData->messageQueueLength);
        printf("  speThreads[%d]->speID = %d\n", i, speThreads[i]->speID);
        printf("}\n");
      }
    #endif
  #endif

  // Add some initial WRHandle structures to the wrFree list
  wrFreeHead = createWRHandles(WRHANDLES_NUM_INITIAL);
  wrFreeTail = &(wrFreeHead[WRHANDLES_NUM_INITIAL - 1]);

  // Add some initial WRGroupHandle structures to the wrGroupFree list
  wrGroupFreeHead = createWRGroupHandles(WRGROUPHANDLES_NUM_INITIAL);
  wrGroupFreeTail = &(wrGroupFreeHead[WRGROUPHANDLES_NUM_INITIAL - 1]);

  #if USE_MESSAGE_QUEUE_FREE_LIST != 0
    // Initialize the wrEntryFreeList
    for (int entryI = 0; entryI < SPE_MESSAGE_QUEUE_LENGTH; entryI++) {
      //for (int speI = 0; speI < NUM_SPE_THREADS; speI++) {
      for (int speI = 0; speI < numSPEThreads; speI++) {
	//register int index = speI + (entryI * NUM_SPE_THREADS);
        register int index = speI + (entryI * numSPEThreads);
        __msgQEntries[index].msgQEntryPtr = (SPEMessage*)(((char*)speThreads[speI]->speData->messageQueue) + (entryI * SIZEOF_16(SPEMessage)));
        __msgQEntries[index].speIndex = speI;
        __msgQEntries[index].entryIndex = entryI;
        //__msgQEntries[index].next = ((entryI == (SPE_MESSAGE_QUEUE_LENGTH - 1) && speI == (NUM_SPE_THREADS - 1)) ?
        __msgQEntries[index].next = ((entryI == (SPE_MESSAGE_QUEUE_LENGTH - 1) && speI == (numSPEThreads - 1)) ?
                                      (NULL) :
                                      (&(__msgQEntries[index + 1])));
      }
    }
    msgQEntryFreeHead = &(__msgQEntries[0]);
    //msgQEntryFreeTail = &(__msgQEntries[(NUM_SPE_THREADS * SPE_MESSAGE_QUEUE_LENGTH) - 1]);
    msgQEntryFreeTail = &(__msgQEntries[(numSPEThreads * SPE_MESSAGE_QUEUE_LENGTH) - 1]);
  #endif

  // Open the projections/timing file
  #if SPE_TIMING != 0
    openProjFile(timingFileName);
  #endif

  // Send each of the SPE threads a command to restart their clocks (in an attempt to remove clock
  //   skew from the timing information).
  //for (int i = 0; i < NUM_SPE_THREADS; i++)
  for (int i = 0; i < numSPEThreads; i++)
    sendSPECommand(speThreads[i], SPE_MESSAGE_COMMAND_RESET_CLOCK);


  return 1;
}


// STATS
#if PPE_STATS != 0
  long long int iterCount = 0;
  long long int iterCountCounter = 0;
  double progress1time = 0.0;
  double progress2time = 0.0;
  double progress3time = 0.0;
  double progress4time = 0.0;
  int wrInUseCount = 0;
  int wrFinishedCount = 0;
  int wrImmedIssueCount = 0;
#endif

extern "C"
void CloseOffloadAPI() {

  // STATS
  #if PPE_STATS != 0
    printf(" --- Offload API :: [STATS] :: Progress - Iterations = %f, iterCount = %lld, iterCountCounter = %lld\n",
	   (float)iterCount / (float)iterCountCounter, iterCount, iterCountCounter
          );
    printf(" --- Offload API :: [STATS] :: Progress - 1:%lf, 2:%lf, 3:%lf, 4:%lf\n",
           progress1time / (double)iterCountCounter,
           progress2time / (double)iterCountCounter,
           progress3time / (double)iterCountCounter,
           progress4time / (double)iterCountCounter
          );
    printf(" --- Offload API :: [STATS] :: Progress - WRs Finished %f - wrFinishedCount = %d\n",
           (float)wrFinishedCount / (float)iterCountCounter, wrFinishedCount
          );
    printf(" --- Offload API :: [STATS] :: Progress - Immed. Issue %f - wrImmedIssueCount = %d\n",
           (float)wrImmedIssueCount / (float)iterCountCounter, wrImmedIssueCount
          );
    printf(" --- Offload API :: [STATS] :: Progress - In Use %f - wrInUseCount = %d\n",
           (float)wrInUseCount / (float)iterCountCounter, wrInUseCount
          );
  #endif

  int status;
 
  #if DEBUG_DISPLAY >= 1
    printf(" ---------- CLOSING OFFLOAD API ----------\n");
  #endif

  // Send each of the SPE threads a message to exit
  //for (int i = 0; i < NUM_SPE_THREADS; i++)
  for (int i = 0; i < numSPEThreads; i++)
    sendSPECommand(speThreads[i], SPE_MESSAGE_COMMAND_EXIT);


  /// SDK 2.0 ///
  //
  //// Wait for all the SPE threads to finish
  //for (int i = 0; i < NUM_SPE_THREADS; i++) {
  //
  //  #if DEBUG_DISPLAY >= 1
  //    printf("OffloadAPI :: Waiting for SPE_%d to Exit...\n", i);
  //  #endif
  //
  //  spe_wait(speThreads[i]->speID, &status, 0);
  //
  //  #if DEBUG_DISPLAY >= 1
  //    printf("OffloadAPI :: SPE_%d Finished (status : %d)\n", i, status);
  //  #endif
  //}
  //


  /// SDK 2.1 ///

  // Wait for all the pthreads to complete
  //for (int i = 0; i < NUM_SPE_THREADS; i++) {
  for (int i = 0; i < numSPEThreads; i++) {

    int rtnCode = pthread_join(speThreads[i]->pThread, NULL);
    if (rtnCode != 0) {
      fprintf(stderr, "OffloadAPI :: ERROR : Unable to join pthread\n");
      exit(EXIT_FAILURE);
    }
  }


  // Clean-up any data structures that need cleaned up

  // Clean up the speThreads
  //for (int i = 0; i < NUM_SPE_THREADS; i++) {
  for (int i = 0; i < numSPEThreads; i++) {
    free_aligned((void*)(speThreads[i]->speData->messageQueue));
    free_aligned((void*)(speThreads[i]->speData));
    delete speThreads[i];
  }

  // Clean up the allocated memory for the WRHandles
  PtrList *entry = allocatedWRHandlesList;
  while (entry != NULL) {
    delete [] ((WorkRequest*)(entry->ptr));
    PtrList *tmp = entry;
    entry = entry->next;
    delete tmp;
  }

  // Close the projections/timing file
  #if SPE_TIMING != 0
    closeProjFile();
  #endif

  #if DEBUG_DISPLAY >= 1
    printf(" ---------- CLOSING OFFLOAD API ----------\n");
  #endif
}


extern "C"
WRHandle sendWorkRequest(int funcIndex,
                         void* readWritePtr, int readWriteLen,
                         void* readOnlyPtr, int readOnlyLen,
                         void* writeOnlyPtr, int writeOnlyLen,
                         void* userData,
                         unsigned int flags,
                         void (*callbackFunc)(void*),
                         WRGroupHandle wrGroupHandle,
                         unsigned int speAffinityMask
                        ) {

  int processingSPEIndex = -1;
  int sentIndex = -1;

  // Tell the PPU's portion of the spert to make progress
  //OffloadAPIProgress();
  //if (msgQListLen == 0)
  //  OffloadAPIProgress();

  // Verify the parameters
  #if 1
    if (__builtin_expect(funcIndex < 0, 0)) return INVALID_WRHandle;
    if (__builtin_expect((readWritePtr != NULL && readWriteLen <= 0) || (readWriteLen > 0 && readWritePtr == NULL), 0)) return INVALID_WRHandle;
    if (__builtin_expect((readOnlyPtr  != NULL && readOnlyLen  <= 0) || (readOnlyLen  > 0 && readOnlyPtr  == NULL), 0)) return INVALID_WRHandle;
    if (__builtin_expect((writeOnlyPtr != NULL && writeOnlyLen <= 0) || (writeOnlyLen > 0 && writeOnlyPtr == NULL), 0)) return INVALID_WRHandle;
    if (__builtin_expect((flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST, 0)) {
      #if DEBUG_DISPLAY >= 1
        fprintf(stderr, " --- OffloadAPI :: WARNING :: sendWorkRequest() call made with WORK_REQUEST_FLAGS_LIST flag set... ignoring...\n");
      #endif
      flags &= (~WORK_REQUEST_FLAGS_LIST);  // Clear the work request's list flag
    }
  #endif

  // Ensure that there is at least one free WRHandle structure
  if (__builtin_expect(wrFreeHead == NULL, 0)) {  // Out of free work request structures
    // Add some more WRHandle structures to the wrFree list
    wrFreeHead = createWRHandles(WRHANDLES_GROW_SIZE);
    wrFreeTail = &(wrFreeHead[WRHANDLES_GROW_SIZE - 1]);
  }

  // Grab the first free WRHandle structure and use it for this entry
  WorkRequest *wrEntry = wrFreeHead;
  wrFreeHead = wrFreeHead->next;
  if (wrFreeHead == NULL) wrFreeTail = NULL;

  if (__builtin_expect(wrEntry->state != WORK_REQUEST_STATE_FREE, 0)) {
    fprintf(stderr, " --- Offload API :: ERROR :: Work request struct with state != FREE in free list !!!!!\n");
  }
  wrEntry->state = WORK_REQUEST_STATE_INUSE;

  // Fill in the WRHandle structure
  wrEntry->speAffinityMask = speAffinityMask;
  wrEntry->speIndex = -1;
  wrEntry->entryIndex = -1;
  wrEntry->funcIndex = funcIndex;
  wrEntry->readWritePtr = readWritePtr;
  wrEntry->readWriteLen = readWriteLen;
  wrEntry->readOnlyPtr = readOnlyPtr;
  wrEntry->readOnlyLen = readOnlyLen;
  wrEntry->writeOnlyPtr = writeOnlyPtr;
  wrEntry->writeOnlyLen = writeOnlyLen;
  wrEntry->userData = userData;
  wrEntry->callbackFunc = (volatile void (*)(void*))callbackFunc;
  wrEntry->flags = flags;
  wrEntry->wrGroupHandle = wrGroupHandle;
  wrEntry->next = NULL;


  // DEBUG
  wrEntry->id = idCounter;
  idCounter++;

  // TRACE
  #if ENABLE_TRACE != 0
    wrEntry->traceFlag = traceFlag;
  #endif

  // If this work request is part of a group, increment that group's work request count
  if (wrGroupHandle != NULL) wrGroupHandle->numWRs++;

  // Try to send the message
  // NOTE: Through the use of speSendStartIndex, if the SPE's message queues aren't being overloaded, then
  //   this loop should only iterate once and then break.
  // TODO : Update this so the number of outstanding work requests that have been sent to each SPE Thread
  //   is used to pick the SPE Thread to send this request to (i.e. - Send to the thread will the SPE Thread
  //   with the least full message queue.)  For now, the speSendStartIndex heuristic should help.
  #if USE_MESSAGE_QUEUE_FREE_LIST == 0

    processingSPEIndex = -1;
    //for (int i = 0; i < NUM_SPE_THREADS; i++) {
    for (int i = 0; i < numSPEThreads; i++) {

      // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
      if (((0x01 << i) & speAffinityMask) != 0x00) {

	//register int actualSPEThreadIndex = (i + speSendStartIndex) % NUM_SPE_THREADS;
        register int actualSPEThreadIndex = (i + speSendStartIndex) % numSPEThreads;
        sentIndex = sendSPEMessage(speThreads[actualSPEThreadIndex], wrEntry, SPE_MESSAGE_COMMAND_NONE);

        if (sentIndex >= 0) {
	  //speSendStartIndex = (actualSPEThreadIndex + 1) % NUM_SPE_THREADS;
          speSendStartIndex = (actualSPEThreadIndex + 1) % numSPEThreads;
          processingSPEIndex = actualSPEThreadIndex;

          // Fill in the execution data into the work request (which SPE, which message queue entry)
          wrEntry->speIndex = processingSPEIndex;
          wrEntry->entryIndex = sentIndex;

          break;
        }
      }
    }

    // Check to see if the message was sent or not
    if (processingSPEIndex < 0) {

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(wrEntry->traceFlag, 0)) {
          printf("OffloadAPI :: [TRACE] :: No Slots Open, Queuing Work Request...\n");
        }
      #endif

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif

      // There were no available spe message queue entries to start the work request so just queue it up in the
      //   pending list of work requests
      if (wrQueuedHead == NULL) {
        wrQueuedHead = wrQueuedTail = wrEntry;  // This entry becomes the list (NOTE: wrEntry->next = NULL)
      } else {
        wrQueuedTail->next = wrEntry;  // This entry goes at the end of the current queue (NOTE: wrEntry->next = NULL)
        wrQueuedTail = wrEntry;
      }

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif
      #if DEBUG_DISPLAY >= 1
        printf(" --- Offload API :: Stalled Work Request (SPE Queues FULL: %p) ---\n", wrEntry);
      #endif

    } else {

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(wrEntry->traceFlag, 0)) {
          printf("OffloadAPI :: [TRACE] :: Work Request Passed to SPE_%d, slot %d... wrEntry = %p...\n", processingSPEIndex, sentIndex, wrEntry);
          displayMessageQueue(speThreads[processingSPEIndex]);
        }
      #endif

      // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
      wrEntry->speIndex = processingSPEIndex;
      wrEntry->entryIndex = sentIndex;

      if (__builtin_expect(wrEntry->next != NULL, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR : Sent work request where wrEntry->next != NULL\n");
      }
    }

  #else

    // Check to see of the wrEntryFreeList has an entry
    //processingSPEIndex = -1;
    if (msgQEntryFreeHead != NULL) {

      // Remove the entry from the list
      MSGQEntryList* msgQEntry = msgQEntryFreeHead;
      msgQEntryFreeHead = msgQEntryFreeHead->next;
      if (msgQEntryFreeHead == NULL) msgQEntryFreeTail = NULL;
      msgQEntry->next = NULL;
      msgQListLen--;

      // Fill in the cross-indexes
      register int processingSPEIndex = msgQEntry->speIndex;
      register int sentIndex = msgQEntry->entryIndex;

      // Send the SPE Message
      sendSPEMessage(speThreads[processingSPEIndex], sentIndex, wrEntry, SPE_MESSAGE_COMMAND_NONE, wrEntry->dmaList);

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(wrEntry->traceFlag, 0)) {
          printf("OffloadAPI :: [TRACE] :: Work Request Passed to SPE_%d, slot %d... wrEntry = %p...\n", processingSPEIndex, sentIndex, wrEntry);
          displayMessageQueue(speThreads[processingSPEIndex]);
        }
      #endif

      // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
      wrEntry->speIndex = processingSPEIndex;
      wrEntry->entryIndex = sentIndex;

      if (__builtin_expect(wrEntry->next != NULL, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR : Sent work request where wrEntry->next != NULL\n");
      }

    } else {

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(wrEntry->traceFlag, 0)) {
          printf("OffloadAPI :: [TRACE] :: No Slots Open, Queuing Work Request...\n");
        }
      #endif

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif

      // There were no available spe message queue entries to start the work request so just queue it up in the
      //   pending list of work requests
      if (wrQueuedHead == NULL) {
        wrQueuedHead = wrQueuedTail = wrEntry;  // This entry becomes the list (NOTE: wrEntry->next = NULL)
      } else {
        wrQueuedTail->next = wrEntry;  // This entry goes at the end of the current queue (NOTE: wrEntry->next = NULL)
        wrQueuedTail = wrEntry;
      }

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif
      #if DEBUG_DISPLAY >= 1
        printf(" --- Offload API :: Stalled Work Request (SPE Queues FULL: %p) ---\n", wrEntry);
      #endif

    }

  #endif

  // Return the WorkRequest pointer as the handle
  return wrEntry;
}


extern "C"
WRHandle sendWorkRequest_list(int funcIndex,
                              unsigned int eah,
                              DMAListEntry* dmaList,
                              int numReadOnly, int numReadWrite, int numWriteOnly,
                              void* userData,
                              unsigned int flags,
                              void (*callbackFunc)(void*),
                              WRGroupHandle wrGroupHandle,
                              unsigned int speAffinityMask
                             ) {

  int processingSPEIndex = -1;
  int sentIndex = -1;

  // Tell the PPU's portion of the spert to make progress
  //OffloadAPIProgress();
  //if (msgQListLen == 0)
  //  OffloadAPIProgress();

  // Verify the parameters
  #if 1
    if (__builtin_expect(funcIndex < 0, 0)) return INVALID_WRHandle;
    if (__builtin_expect(dmaList == NULL, 0)) return INVALID_WRHandle;
    if (__builtin_expect(numReadOnly == 0 && numReadWrite == 0 && numWriteOnly == 0, 0)) return INVALID_WRHandle;
    if (__builtin_expect(numReadOnly < 0 || numReadWrite < 0 || numWriteOnly < 0, 0)) return INVALID_WRHandle;
    if (__builtin_expect((flags & (WORK_REQUEST_FLAGS_RW_IS_RO || WORK_REQUEST_FLAGS_RW_IS_WO)) != 0x00, 0)) {
      #if DEBUG_DISPLAY >= 1
        fprintf(stderr, "OffloadAPI :: WARNING :: sendWorkRequest_list() call made with WORK_REQUEST_FLAGS_RW_TO_RO and/or WORK_REQUEST_FLAGS_RW_IS_WO flags set... ignoring...\n");
      #endif
      flags &= (~(WORK_REQUEST_FLAGS_RW_IS_RO || WORK_REQUEST_FLAGS_RW_IS_WO));  // Force these flags to clear
    }
  #endif

  // Ensure that there is at least one free WRHandle structure
  if (__builtin_expect(wrFreeHead == NULL, 0)) {  // Out of free work request structures
    // Add some more WRHandle structures to the wrFree list
    wrFreeHead = createWRHandles(WRHANDLES_GROW_SIZE);
    wrFreeTail = &(wrFreeHead[WRHANDLES_GROW_SIZE - 1]);
  }

  // Grab the first free WRHandle structure and use it for this entry
  WorkRequest *wrEntry = wrFreeHead;
  wrFreeHead = wrFreeHead->next;
  if (__builtin_expect(wrFreeHead == NULL, 0)) wrFreeTail = NULL;

  if (__builtin_expect(wrEntry->state != WORK_REQUEST_STATE_FREE, 0)) {
    fprintf(stderr, " --- Offload API :: ERROR :: Work request struct with state != FREE in free list !!!!!\n");
  }
  wrEntry->state = WORK_REQUEST_STATE_INUSE;

  // Fill in the WRHandle structure
  wrEntry->speAffinityMask = speAffinityMask;
  wrEntry->speIndex = -1;
  wrEntry->entryIndex = -1;
  wrEntry->funcIndex = funcIndex;
  wrEntry->readWritePtr = dmaList;
  wrEntry->readWriteLen = numReadWrite;
  wrEntry->readOnlyPtr = (void*)eah;
  wrEntry->readOnlyLen = numReadOnly;
  wrEntry->writeOnlyPtr = NULL;
  wrEntry->writeOnlyLen = numWriteOnly;
  wrEntry->userData = userData;
  wrEntry->callbackFunc = (volatile void (*)(void*))callbackFunc;
  wrEntry->flags = (flags | WORK_REQUEST_FLAGS_LIST);  // force LIST flag
  wrEntry->wrGroupHandle = wrGroupHandle;
  wrEntry->next = NULL;

  // DEBUG
  wrEntry->id = idCounter;
  idCounter++;

  // TRACE
  #if ENABLE_TRACE != 0
    wrEntry->traceFlag = traceFlag;
  #endif

  // If this work request is part of a group, increment that group's work request count
  if (wrGroupHandle != NULL) wrGroupHandle->numWRs++;

  // Try to send the message
  // NOTE: Through the use of speSendStartIndex, if the SPE's message queues aren't being overloaded, then
  //   this loop should only iterate once and then break.
  // TODO : Update this so the number of outstanding work requests that have been sent to each SPE Thread
  //   is used to pick the SPE Thread to send this request to (i.e. - Send to the thread with the SPE Thread
  //   with the least full message queue.)  For now, the speSendStartIndex heuristic should help.
  #if USE_MESSAGE_QUEUE_FREE_LIST == 0

    processingSPEIndex = -1;
    //for (int i = 0; i < NUM_SPE_THREADS; i++) {
    for (int i = 0; i < numSPEThreads; i++) {

      // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
      if (((0x01 << i) & speAffinityMask) != 0x00) {

	//register int actualSPEThreadIndex = (i + speSendStartIndex) % NUM_SPE_THREADS;
        register int actualSPEThreadIndex = (i + speSendStartIndex) % numSPEThreads;
        sentIndex = sendSPEMessage(speThreads[actualSPEThreadIndex], wrEntry, SPE_MESSAGE_COMMAND_NONE);

        if (sentIndex >= 0) {
	  //speSendStartIndex = (actualSPEThreadIndex + 1) % NUM_SPE_THREADS;
          speSendStartIndex = (actualSPEThreadIndex + 1) % numSPEThreads;
          processingSPEIndex = actualSPEThreadIndex;
          break;
        }
      }
    }

    // Check to see if the message was sent or not
    if (processingSPEIndex < 0) {

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif

      // There were no available spe message queue entries to start the work request so just queue it up in the
      //   pending list of work requests
      if (wrQueuedHead == NULL) {
        wrQueuedHead = wrQueuedTail = wrEntry;  // This entry becomes the list (NOTE: wrEntry->next = NULL)
      } else {
        wrQueuedTail->next = wrEntry;  // This entry goes at the end of the current queue (NOTE: wrEntry->next = NULL)
        wrQueuedTail = wrEntry;
      }

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif
      #if DEBUG_DISPLAY >= 1
        printf(" --- Offload API :: Stalled Work Request (SPE Queues FULL: %p) ---\n", wrEntry);
      #endif

    } else {

      // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
      wrEntry->speIndex = processingSPEIndex;
      wrEntry->entryIndex = sentIndex;

      if (__builtin_expect(wrEntry->next != NULL, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR : Sent work request where wrEntry->next != NULL\n");
      }
    }

  #else

    // Check to see of the wrEntryFreeList has an entry
    processingSPEIndex = -1;
    if (msgQEntryFreeHead != NULL) {

      // Remove the entry from the list
      MSGQEntryList* msgQEntry = msgQEntryFreeHead;
      msgQEntryFreeHead = msgQEntryFreeHead->next;
      if (msgQEntryFreeHead == NULL) msgQEntryFreeTail = NULL;
      msgQEntry->next = NULL;
      msgQListLen--;

      // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
      register int processingSPEIndex = msgQEntry->speIndex;
      register int sentIndex = msgQEntry->entryIndex;
      wrEntry->speIndex = processingSPEIndex;
      wrEntry->entryIndex = sentIndex;
      //wrEntry->speIndex = msgQEntry->speIndex;
      //wrEntry->entryIndex = msgQEntry->entryIndex;

      // DEBUG
      if (__builtin_expect(wrEntry->next != NULL, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR : Sent work request where wrEntry->next != NULL\n");
      }


      // TRACE
      #if ENABLE_TRACE != 0
        if (wrEntry->traceFlag) {

          printf("OffloadAPI :: [TRACE] :: (sendWorkRequest_list) processingSPEIndex = %d, sentIndex = %d\n",
                 processingSPEIndex, sentIndex
		);

          printf("OffloadAPI :: [TRACE] :: (sendWorkRequest_list) dmaList:\n");
          register int jj;
          for (jj = 0; jj < numReadOnly; jj++) {
            printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (RO)\n",
                   jj, dmaList[jj].ea, dmaList[jj].size
                  );
	  }
          for (; jj < numReadOnly + numReadWrite; jj++) {
            printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (RW)\n",
                   jj, dmaList[jj].ea, dmaList[jj].size
                  );
	  }
          for (; jj < numReadOnly + numReadWrite + numWriteOnly; jj++) {
            printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (WO)\n",
                   jj, dmaList[jj].ea, dmaList[jj].size
                  );
	  }
	}
      #endif


      // Send the SPE Message (NOTE: The DMA List source is the user's data structure since this Work
      //   Request is being issued to the SPE immediately.
      sendSPEMessage(speThreads[processingSPEIndex], sentIndex, wrEntry, SPE_MESSAGE_COMMAND_NONE, dmaList);

    } else {

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif

      // Copy the DMA list if it should be copied
      register int dmaListSize = numReadOnly + numReadWrite + numWriteOnly;
      if (__builtin_expect(dmaListSize <= SPE_DMA_LIST_LENGTH, 1)) {
        for (int i = 0; i < dmaListSize; i++) {
          wrEntry->dmaList[i].ea = dmaList[i].ea;
          wrEntry->dmaList[i].size = dmaList[i].size;
        }
      }

      // There were no available spe message queue entries to start the work request so just queue it up in the
      //   pending list of work requests
      if (wrQueuedHead == NULL) {
        wrQueuedHead = wrQueuedTail = wrEntry;  // This entry becomes the list (NOTE: wrEntry->next = NULL)
      } else {
        wrQueuedTail->next = wrEntry;  // This entry goes at the end of the current queue (NOTE: wrEntry->next = NULL)
        wrQueuedTail = wrEntry;
      }

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif
      #if DEBUG_DISPLAY >= 1
        printf(" --- Offload API :: Stalled Work Request (SPE Queues FULL: %p) ---\n", wrEntry);
      #endif

    }

  #endif

  // Return the WorkRequest pointer as the handle
  return wrEntry;
}


// Returns: Non-zero if finished, zero otherwise
extern "C"
int isFinished(WRHandle wrHandle) {

  int rtnCode = 0;  // default to "not finished"

  // Tell the PPE's portion of the spert to make progress
  OffloadAPIProgress();

  // Check to see if the work request has finished
  if (__builtin_expect(wrHandle != INVALID_WRHandle, 1) && wrHandle->state == WORK_REQUEST_STATE_FINISHED) {

    // Add this entry to the free list
    wrHandle->state = WORK_REQUEST_STATE_FREE;
    if (wrFreeTail == NULL) {
      wrFreeTail = wrFreeHead = wrHandle;
    } else {
      wrFreeTail->next = wrHandle;
      wrFreeTail = wrHandle;
    }

    rtnCode = -1;  // change the return code to "finished"
  }
    
  return rtnCode;
}


// TODO : It would be nice to change this from a busy wait to something that busy waits for a
//   short time and then gives up and blocks (Maybe get some OS stuff out of the way while we
//   are just waiting anyway).
// NOTE : This function only blocks if a callbackFunc is not specified.
extern "C"
void waitForWRHandle(WRHandle wrHandle) {

  // Verify the WRHandle
  if (__builtin_expect(wrHandle == INVALID_WRHandle, 0)) return;

  // Check to see if the work request needs to call a callback function of some type (if so, do not block)
  if (callbackFunc == NULL && wrHandle != INVALID_WRHandle && wrHandle->callbackFunc == NULL) {
    int yeildCounter = 2500;
    // Wait for the handle to finish
    while (wrHandle != INVALID_WRHandle && !isFinished(wrHandle)) {
      yeildCounter--;
      if (__builtin_expect(yeildCounter <= 0, 0)) {
        // TODO : Place a yield call here (no threading stuff yet)
        yeildCounter = 100;
      }
      // NOTE: isFinished() makes a call to OffloadAPIProgress()... if it is changed so that it
      //   doesn't, call OffloadAPIProgress() here.  It MUST be called each iteration of this loop!
    }
  }
}


extern "C"
WRGroupHandle createWRGroup(void* userData, void (*callbackFunc)(void*)) {

  // Ensure that there is at least one free WRGroupHandle structure
  if (__builtin_expect(wrGroupFreeHead == NULL, 0)) {
    // Add some more WRGroupHandle structures
    wrGroupFreeHead = createWRGroupHandles(WRGROUPHANDLES_GROW_SIZE);
    wrGroupFreeTail = &(wrGroupFreeHead[WRGROUPHANDLES_GROW_SIZE - 1]);
  }

  // Grab the first free WRGroupHandle structure and use it for this group
  WRGroup* wrGroup = wrGroupFreeHead;
  wrGroupFreeHead = wrGroupFreeHead->next;
  if (__builtin_expect(wrGroupFreeHead == NULL, 0)) wrGroupFreeTail = NULL;
  wrGroup->next = NULL;

  // Check and update the WRGroup's state
  if (__builtin_expect(wrGroup->state != WRGROUP_STATE_FREE, 0)) {
    fprintf(stderr, " --- Offload API :: ERROR :: WRGroup struct with state != FREE in free list !!!!!\n");
  }
  wrGroup->state = WRGROUP_STATE_FILLING;

  // Fill in the WRGroup structure
  wrGroup->numWRs = 0;
  wrGroup->finishedCount = 0;
  wrGroup->userData = userData;
  wrGroup->callbackFunc = callbackFunc;

  // Return the handle
  return wrGroup;
}


extern "C"
void completeWRGroup(WRGroupHandle wrGroupHandle) {

  // Verify the parameter
  if (__builtin_expect(wrGroupHandle == NULL, 0)) return;

  // Check
  if (__builtin_expect(wrGroupHandle->state != WRGROUP_STATE_FILLING, 0)) {
    fprintf(stderr, " --- Offload API :: ERROR :: WRGroup structure with state != FILLING being completed !!!!!\n");
  }

  // Update the state of the group
  wrGroupHandle->state = WRGROUP_STATE_FULL;

  // Check to see if all of the work requests associated with this group have already completed
  if (wrGroupHandle->finishedCount >= wrGroupHandle->numWRs) {

    // DEBUG
    printf(" --- Offload API :: completeGroup() - Immediate complete detected ...\n");

    // Get a pointer to the callback function if there is one
    //register void (*callbackFunc)(void*);
    //if (wrGroupHandle->callbackFunc != NULL)
    //  callbackFunc = wrGroupHandle->callbackFunc;
    //else
    //  callbackFunc = groupCallbackFunc;
    register void (*callbackFunc)(void*) = ((wrGroupHandle->callbackFunc != NULL) ?
                                              (wrGroupHandle->callbackFunc) :
                                              (groupCallbackFunc)
                                           );

    // Check to see if there is a callback function
    if (callbackFunc != NULL) {

      // Call the callback function
      callbackFunc(wrGroupHandle->userData);

      // Clean up the WRGroup structure
      wrGroupHandle->state = WRGROUP_STATE_FREE;

      // Add the WRGroup structure back into the free list
      if (__builtin_expect(wrGroupFreeTail == NULL, 0)) {
        wrGroupFreeHead = wrGroupFreeTail = wrGroupHandle;
      } else {
        wrGroupFreeTail->next = wrGroupHandle;
        wrGroupFreeTail = wrGroupHandle;
      }

    } else {  // Otherwise, there is no callback function

      // Mark the group as finished
      wrGroupHandle->state = WRGROUP_STATE_FINISHED;
    }

  } // end if (all work requests in group have finished)
}


// Returns: Non-zero if finished, zero otherwise
extern "C"
int isWRGroupFinished(WRGroupHandle wrGroupHandle) {

  int rtnCode = 0;  // Assume "not finished"

  // Tell the PPE to make progress
  OffloadAPIProgress();

  // Check to see if the group has finished
  if (__builtin_expect(wrGroupHandle != INVALID_WRGroupHandle, 0) &&
      wrGroupHandle->state == WRGROUP_STATE_FINISHED
     ) {

    // Update the state of the WRGroup structure
    wrGroupHandle->state = WRGROUP_STATE_FREE;

    // Add this WRGroup structure back into the free list
    if (wrGroupFreeTail == NULL) {
      wrGroupFreeHead = wrGroupFreeTail = wrGroupHandle;
    } else {
      wrGroupFreeTail->next = wrGroupHandle;
      wrGroupFreeTail = wrGroupHandle;
    }

    // Set the return code to "finished"
    rtnCode = -1;
  }

  return rtnCode;
}


extern "C"
void waitForWRGroupHandle(WRGroupHandle wrGroupHandle) {

  // Verify the parameter
  if (__builtin_expect(wrGroupHandle != NULL, 1)) {

    // Check to see if a callback function was specified (if so, do not block)
    if (groupCallbackFunc == NULL && wrGroupHandle->callbackFunc == NULL) {

      // TODO : Add a yeild call into this
      while (!isWRGroupFinished(wrGroupHandle));

    }
  }  
}


// DEBUG
#if ENABLE_LAST_WR_TIMES != 0
  timeval lastWRTime[NUM_SPE_THREADS];
  void displayLastWRTimes() {
    for (int i = 0; i < NUM_SPE_THREADS; i++) {
      double timeD = (double)lastWRTime[i].tv_sec + ((double)lastWRTime[i].tv_usec / 1000000.0);
      printf("PPE :: displayLastWRTimes() - SPE_%d -> %.9lf sec\n", i, timeD);
    }
  }
#else
  void displayLastWRTimes() { }
#endif


#if 0

// Original

extern "C"
void OffloadAPIProgress() {

  // Mailbox Statistics
  #define OffloadAPIProgress_statFreq  0
  #if OffloadAPIProgress_statFreq > 0
    static int statCount = OffloadAPIProgress_statFreq;
    int statCount_flag = 0;
    static int statSum_all[8] = { 0 };
    static int statSum_all_count[8] = { 0 };
    static int statSum_nonZero[8] = { 0 };
    static int statSum_nonZero_count[8] = { 0 };
    static int queueSample[NUM_SPE_THREADS][SPE_MESSAGE_NUM_STATES] = { 0 };
    static int queueSample_count = OffloadAPIProgress_statFreq;
  #endif


  #if OffloadAPIProgress_statFreq > 0
    queueSample_count--;
    for (int i = 0; i < NUM_SPE_THREADS; i++) {
      register char* qStart = (char*)(speThreads[i]->speData->messageQueue);
      for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++) {
        register SPEMessage* qEntry = (SPEMessage*)(qStart + (SIZEOF_16(SPEMessage) * j));
        queueSample[i][qEntry->state]++;
      }
    }

    if (queueSample_count <= 0) {
      for (int i = 0; i < NUM_SPE_THREADS; i++) {
        printf("SPE_%d :: ", i);
        for (int j = 0; j < SPE_MESSAGE_NUM_STATES; j++) {
          printf("%f ", ((float)(queueSample[i][j]) / (float)(OffloadAPIProgress_statFreq)));
          queueSample[i][j] = 0;
	}
        printf("\n");
      }
      queueSample_count = OffloadAPIProgress_statFreq;
    }
  #endif


  // Check the mailbox from the SPEs to see if any of the messages have finished (and mark them as such)
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // Get the number of entries in the mailbox from the SPE and then read each entry
    int usedEntries = spe_stat_out_mbox(speThreads[i]->speID);

    // Mailbox Statistics
    #if OffloadAPIProgress_statFreq > 0
      statCount_flag += usedEntries;
      statSum_all[i] += usedEntries;
      statSum_all_count[i]++;
      if (usedEntries > 0) {
        statSum_nonZero[i] += usedEntries;
        statSum_nonZero_count[i]++;
      }
    #endif

    while (usedEntries > 0) {      

      // Read the message queue index that was sent by the SPE from the outbound mailbox
      unsigned int messageReturnCode = spe_read_out_mbox(speThreads[i]->speID);
      unsigned int speMessageQueueIndex = MESSAGE_RETURN_CODE_INDEX(messageReturnCode);
      unsigned int speMessageErrorCode = MESSAGE_RETURN_CODE_ERROR(messageReturnCode);
      SPEMessage *msg = (SPEMessage*)((char*)(speThreads[i]->speData->messageQueue) + (speMessageQueueIndex * SIZEOF_16(SPEMessage)));

      // Get a pointer to the associated work request
      WorkRequest *wrPtr = (WorkRequest*)(msg->wrPtr);

      if (__builtin_expect(wrPtr == NULL, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- Offload API :: ERROR :: Received work request completion with no associated work request !!!!!\n");
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      if (__builtin_expect(wrPtr->next != NULL, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- Offload API :: ERROR :: WorkRequest finished while still linked (msg->wrPtr->next should be NULL) !!!!!!!\n");
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      if (__builtin_expect(msg->state != SPE_MESSAGE_STATE_SENT, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- OffloadAPI :: ERROR :: Invalid message queue index (%d) received from SPE_%d...\n", speMessageQueueIndex, i);
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      // If there was an error returned by the SPE, display it now
      if (__builtin_expect(speMessageErrorCode != SPE_MESSAGE_OK, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR :: SPE_%d returned error code %d for message at index %d...\n",
                i, speMessageErrorCode, speMessageQueueIndex
               );
      }

      // DEBUG - Get time last work request completed from this SPE
      #if ENABLE_LAST_WR_TIMES != 0
        gettimeofday(&(lastWRTime[wrPtr->speIndex]), NULL);
      #endif

      // TRACE
      if (wrPtr->traceFlag) {
        printf("OffloadAPI :: [TRACE] :: spe = %d, qi = %d, ec = %d...\n",
               i, speMessageQueueIndex, speMessageErrorCode
              );
        displayMessageQueue(speThreads[i]);
      }

      // Check to see if this work request should call a callback function upon completion
      if (callbackFunc != NULL || wrPtr->callbackFunc != NULL) {

        // Call the callback function
        if (wrPtr->callbackFunc != NULL)
          (wrPtr->callbackFunc)((void*)wrPtr->userData);  // call work request specific callback function
        else
          callbackFunc((void*)(wrPtr->userData));         // call default work request callback function

        // Clear the fields of the work request as needed
        //wrPtr->speIndex = -1;
        //wrPtr->entryIndex = -1;
        //wrPtr->funcIndex = -1;
        //wrPtr->readWritePtr = NULL;
        //wrPtr->readWriteLen = 0;
        //wrPtr->readOnlyPtr = NULL;
        //wrPtr->readOnlyLen = 0;
        //wrPtr->writeOnlyPtr = NULL;
        //wrPtr->writeOnlyLen = 0;
        //wrPtr->flags = WORK_REQUEST_FLAGS_NONE;
        //wrPtr->userData = NULL;
        //wrPtr->callbackFunc = NULL;
        //wrPtr->next = NULL;

        // Add this entry to the end of the wrFree list
        wrPtr->state = WORK_REQUEST_STATE_FREE;
        if (wrFreeTail == NULL) {
          wrFreeTail = wrFreeHead = wrPtr;
        } else {
          wrFreeTail->next = wrPtr;
          wrFreeTail = wrPtr;
        }

      // Otherwise, just place the WorkRequest into the wrFinished list
      } else {

        // Mark the work request as finished
        wrPtr->state = WORK_REQUEST_STATE_FINISHED;
      }

      // Now that the work request has been moved to either the wrFree list or marked as
      //   finished, set the state of the message queue entry to clear so it can accempt
      //   another work request
      msg->state = SPE_MESSAGE_STATE_CLEAR;

      // Re-add the message queue entry to the msgQEntryFreeList
      #if USE_MESSAGE_QUEUE_FREE_LIST != 0
        //MSGQEntryList* msgQEntry = &(__msgQEntries[speMessageQueueIndex + (i * NUM_SPE_THREADS)]);
        MSGQEntryList* msgQEntry = &(__msgQEntries[i + (speMessageQueueIndex * NUM_SPE_THREADS)]);
        if (__builtin_expect(msgQEntry->next != NULL, 0)) {
          printf(" --- OffloadAPI :: ERROR :: msgQEntry->next != NULL !!!!!\n");
          msgQEntry->next = NULL;
        }
        if (msgQEntryFreeTail != NULL) {
          msgQEntryFreeTail->next = msgQEntry;
          msgQEntryFreeTail = msgQEntry;
        } else {
          msgQEntryFreeHead = msgQEntryFreeTail = msgQEntry;
	}
        msgQListLen++;
      #endif

      // Decrement the count of remaining completion notifications from this SPE
      usedEntries--;

    } // end while (usedEntries > 0)
  } // end for (all SPEs)


  // Mailbox Statistics
  #if OffloadAPIProgress_statFreq > 0
    #if 0
      if (statCount_flag > 0)  // For print frequency, only count calls that find at least one mailbox entry
        statCount--;
      if (statCount <= 0) {
        printf("PPE :: OffloadAPIProgress() - Mailbox Statistics...\n");
        for (int i = 0; i < NUM_SPE_THREADS; i++) {
          printf("PPE :: OffloadAPIProgress() -   SPE_%d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
                 i,
                 ((float)statSum_all[i]) / ((float)statSum_all_count[i]), statSum_all_count[i],
                 ((float)statSum_nonZero[i]) / ((float)statSum_nonZero_count[i]), statSum_nonZero_count[i]
                );
          statSum_all[i] = 0;
          statSum_all_count[i] = 0;
          statSum_nonZero[i] = 0;
          statSum_nonZero_count[i] = 0;
        }
        statCount = OffloadAPIProgress_statFreq;
      }
    #else
      for (int i = 0; i < NUM_SPE_THREADS; i++) {
        if (statSum_nonZero_count[i] >= OffloadAPIProgress_statFreq) {
          printf("PPE :: OffloadAPIProgress() - SPE_%d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
                 i,
                 ((float)statSum_all[i]) / ((float)statSum_all_count[i]), statSum_all_count[i],
                 ((float)statSum_nonZero[i]) / ((float)statSum_nonZero_count[i]), statSum_nonZero_count[i]
                );
          statSum_all[i] = 0;
          statSum_all_count[i] = 0;
          statSum_nonZero[i] = 0;
          statSum_nonZero_count[i] = 0;
        }
      }
    #endif
  #endif


  // Loop through the wrQueued list and try to send outstanding messages
  int sentIndex = -1;
  int processingSPEIndex = -1;
  WorkRequest *wrEntry = wrQueuedHead;
  WorkRequest *wrEntryPrev = NULL;
  while (wrEntry != NULL) {

    register unsigned int speAffinityMask = wrEntry->speAffinityMask;

    #if USE_MESSAGE_QUEUE_FREE_LIST == 0

      // Try each SPE
      for (int i = 0; i < NUM_SPE_THREADS; i++) {

        // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
        if (((0x01 << i) & speAffinityMask) != 0x00) {
          sentIndex = sendSPEMessage(speThreads[i], wrEntry, SPE_MESSAGE_COMMAND_NONE);
          if (sentIndex >= 0) {
            processingSPEIndex = i;
            break;
          }
        }
      }

    #else

      // Pull the first entry of the message queue free list
      if (msgQEntryFreeHead != NULL) {

        // Remove the entry from the list
        MSGQEntryList* msgQEntry = msgQEntryFreeHead;
        msgQEntryFreeHead = msgQEntryFreeHead->next;
        if (msgQEntryFreeHead == NULL) msgQEntryFreeTail = NULL;
        msgQEntry->next = NULL;
        msgQListLen--;

        // Fill in the cross-indexes
        processingSPEIndex = msgQEntry->speIndex;
        sentIndex = msgQEntry->entryIndex;

        // Send the SPE Message
        sendSPEMessage(speThreads[processingSPEIndex], sentIndex, wrEntry, SPE_MESSAGE_COMMAND_NONE, wrEntr->dmaList);
      }

    #endif

    // Check to see if the message was sent (remove it from the wrQueued list if so)
    if (processingSPEIndex >= 0) {

      #if DEBUG_DISPLAY >= 1
        printf(" --- Offload API :: Stalled Work Request Being Issued (%p) ---\n", wrEntry);
        #if DEBUG_DISPLAY >= 2
          if (wrQueuedHead != NULL)
            printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
          else
            printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
        #endif
      #endif

      // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
      wrEntry->speIndex = processingSPEIndex;
      wrEntry->entryIndex = sentIndex;

      // Remove the wrEntry from the wrQueued list
      if (wrEntryPrev == NULL) { // is the head of the list
        wrQueuedHead = wrEntry->next;
        if (wrQueuedHead == NULL) wrQueuedTail = NULL;
      } else {  // is in the middle or at the end of the list
        wrEntryPrev->next = wrEntry->next;
        if (wrEntryPrev->next == NULL) wrQueuedTail = wrEntryPrev;
      }
      wrEntry->next = NULL;

      #if DEBUG_DISPLAY >= 2
        if (wrQueuedHead != NULL)
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
        else
          printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
      #endif

    // Otherwise, there was a work request but no empty slots on any of the SPEs (no need to keep
    //   trying more work requests if the work request's affinity specified all SPEs ok)
    } else {
      if (speAffinityMask == 0xFFFFFFFF) break;
    }

    // Move into the next wrQueued entry
    wrEntryPrev = wrEntry;
    wrEntry = wrEntry->next;
  }

}

#else

// Experimental

#if SPE_NOTIFY_VIA_MAILBOX != 0

extern "C"
void OffloadAPIProgress() {

  // Mailbox Statistics
  #define OffloadAPIProgress_statFreq  0
  #if OffloadAPIProgress_statFreq > 0
    static int statCount = OffloadAPIProgress_statFreq;
    int statCount_flag = 0;
    static int statSum_all[8] = { 0 };
    static int statSum_all_count[8] = { 0 };
    static int statSum_nonZero[8] = { 0 };
    static int statSum_nonZero_count[8] = { 0 };
    static int queueSample[NUM_SPE_THREADS][SPE_MESSAGE_NUM_STATES] = { 0 };
    static int queueSample_count = OffloadAPIProgress_statFreq;
  #endif


  #if OffloadAPIProgress_statFreq > 0
    queueSample_count--;
    for (int i = 0; i < NUM_SPE_THREADS; i++) {
      register char* qStart = (char*)(speThreads[i]->speData->messageQueue);
      for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++) {
        register SPEMessage* qEntry = (SPEMessage*)(qStart + (SIZEOF_16(SPEMessage) * j));
        queueSample[i][qEntry->state]++;
      }
    }

    if (queueSample_count <= 0) {
      for (int i = 0; i < NUM_SPE_THREADS; i++) {
        printf("SPE_%d :: ", i);
        for (int j = 0; j < SPE_MESSAGE_NUM_STATES; j++) {
          printf("%f ", ((float)(queueSample[i][j]) / (float)(OffloadAPIProgress_statFreq)));
          queueSample[i][j] = 0;
	}
        printf("\n");
      }
      queueSample_count = OffloadAPIProgress_statFreq;
    }
  #endif

  // STATS
  #if PPE_STATS != 0
    timeval progress1start;
    gettimeofday(&progress1start, NULL);
  #endif

  // Check the mailbox from the SPEs to see if any of the messages have finished (and mark them as such)
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // STATS
    #if PPE_STATS != 0
      timeval progress3start;
      gettimeofday(&progress3start, NULL);
    #endif

    // Get the number of entries in the mailbox from the SPE and then read each entry
    int usedEntries = spe_stat_out_mbox(speThreads[i]->speID);

    // Mailbox Statistics
    #if OffloadAPIProgress_statFreq > 0
      statCount_flag += usedEntries;
      statSum_all[i] += usedEntries;
      statSum_all_count[i]++;
      if (usedEntries > 0) {
        statSum_nonZero[i] += usedEntries;
        statSum_nonZero_count[i]++;
      }
    #endif

    // STATS
    #if PPE_STATS != 0
      timeval progress3end;
      gettimeofday(&progress3end, NULL);
    #endif

    // STATS
    #if PPE_STATS != 0
      timeval progress4start;
      gettimeofday(&progress4start, NULL);
    #endif

    while (usedEntries > 0) {      

      // Read the message queue index that was sent by the SPE from the outbound mailbox
      unsigned int messageReturnCode = spe_read_out_mbox(speThreads[i]->speID);
      unsigned int speMessageQueueIndex = MESSAGE_RETURN_CODE_INDEX(messageReturnCode);
      unsigned int speMessageErrorCode = MESSAGE_RETURN_CODE_ERROR(messageReturnCode);
      SPEMessage *msg = (SPEMessage*)((char*)(speThreads[i]->speData->messageQueue) + (speMessageQueueIndex * SIZEOF_16(SPEMessage)));

      // Get a pointer to the associated work request
      WorkRequest *wrPtr = (WorkRequest*)(msg->wrPtr);

      if (__builtin_expect(wrPtr == NULL, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- Offload API :: ERROR :: Received work request completion with no associated work request !!!!!\n");
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      if (__builtin_expect(wrPtr->next != NULL, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- Offload API :: ERROR :: WorkRequest finished while still linked (msg->wrPtr->next should be NULL) !!!!!!!\n");
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      if (__builtin_expect(msg->state != SPE_MESSAGE_STATE_SENT, 0)) {
        // Warn the user that something bad has just happened
        fprintf(stderr, " --- OffloadAPI :: ERROR :: Invalid message queue index (%d) received from SPE_%d...\n", speMessageQueueIndex, i);
        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      // If there was an error returned by the SPE, display it now
      if (__builtin_expect(speMessageErrorCode != SPE_MESSAGE_OK, 0)) {
        fprintf(stderr, " --- Offload API :: ERROR :: SPE_%d returned error code %d for message at index %d...\n",
                i, speMessageErrorCode, speMessageQueueIndex
               );
      }

      // DEBUG - Get time last work request completed from this SPE
      #if ENABLE_LAST_WR_TIMES != 0
        gettimeofday(&(lastWRTime[wrPtr->speIndex]), NULL);
      #endif

      // TRACE
      #if ENABLE_TRACE != 0
        if (__builtin_expect(wrPtr->traceFlag, 0)) {
          printf("OffloadAPI :: [TRACE] :: spe = %d, qi = %d, ec = %d...\n",
                 i, speMessageQueueIndex, speMessageErrorCode
                );
          displayMessageQueue(speThreads[i]);
        }
      #endif

      // Check to see if this work request should call a callback function upon completion
      if (callbackFunc != NULL || wrPtr->callbackFunc != NULL) {

        // Call the callback function
        if (wrPtr->callbackFunc != NULL)
          (wrPtr->callbackFunc)((void*)wrPtr->userData);  // call work request specific callback function
        else
          callbackFunc((void*)(wrPtr->userData));         // call default work request callback function

        // Add this entry to the end of the wrFree list
        wrPtr->state = WORK_REQUEST_STATE_FREE;
        if (wrFreeTail == NULL) {
          wrFreeTail = wrFreeHead = wrPtr;
        } else {
          wrFreeTail->next = wrPtr;
          wrFreeTail = wrPtr;
        }

      // Otherwise, just place the WorkRequest into the wrFinished list
      } else {

        // Mark the work request as finished
        wrPtr->state = WORK_REQUEST_STATE_FINISHED;
      }

      // Now that the work request has been moved to either the wrFree list or marked as
      //   finished, set the state of the message queue entry to clear so it can accempt
      //   another work request
      msg->state = SPE_MESSAGE_STATE_CLEAR;

      // Re-add the message queue entry to the msgQEntryFreeList
      #if USE_MESSAGE_QUEUE_FREE_LIST != 0
        //MSGQEntryList* msgQEntry = &(__msgQEntries[speMessageQueueIndex + (i * NUM_SPE_THREADS)]);
        MSGQEntryList* msgQEntry = &(__msgQEntries[i + (speMessageQueueIndex * NUM_SPE_THREADS)]);
        if (__builtin_expect(msgQEntry->next != NULL, 0)) {
          printf(" --- OffloadAPI :: ERROR :: msgQEntry->next != NULL !!!!!\n");
          msgQEntry->next = NULL;
        }
        if (msgQEntryFreeTail != NULL) {
          msgQEntryFreeTail->next = msgQEntry;
          msgQEntryFreeTail = msgQEntry;
        } else {
          msgQEntryFreeHead = msgQEntryFreeTail = msgQEntry;
	}
        msgQListLen++;
      #endif

      // Decrement the count of remaining completion notifications from this SPE
      usedEntries--;

    } // end while (usedEntries > 0)

    // STATS
    #if PPE_STATS != 0
      timeval progress4end;
      gettimeofday(&progress4end, NULL);
    #endif

    // STATS
    #if PPE_STATS != 0
      // Calculate the time taken
      double startTimeD = (double)progress3start.tv_sec + ((double)progress3start.tv_usec / 1000000.0);
      double endTimeD = (double)progress3end.tv_sec + ((double)progress3end.tv_usec / 1000000.0);
      double timeDiff = endTimeD - startTimeD;
      progress3time += timeDiff;

      startTimeD = (double)progress4start.tv_sec + ((double)progress4start.tv_usec / 1000000.0);
      endTimeD = (double)progress4end.tv_sec + ((double)progress4end.tv_usec / 1000000.0);
      timeDiff = endTimeD - startTimeD;
      progress4time += timeDiff;
    #endif

  } // end for (all SPEs)

  // STATS
  #if PPE_STATS != 0
    timeval progress1end;
    gettimeofday(&progress1end, NULL);
  #endif

  // Mailbox Statistics
  #if OffloadAPIProgress_statFreq > 0
    #if 0
      if (statCount_flag > 0)  // For print frequency, only count calls that find at least one mailbox entry
        statCount--;
      if (statCount <= 0) {
        printf("PPE :: OffloadAPIProgress() - Mailbox Statistics...\n");
        for (int i = 0; i < NUM_SPE_THREADS; i++) {
          printf("PPE :: OffloadAPIProgress() -   SPE_%d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
                 i,
                 ((float)statSum_all[i]) / ((float)statSum_all_count[i]), statSum_all_count[i],
                 ((float)statSum_nonZero[i]) / ((float)statSum_nonZero_count[i]), statSum_nonZero_count[i]
                );
          statSum_all[i] = 0;
          statSum_all_count[i] = 0;
          statSum_nonZero[i] = 0;
          statSum_nonZero_count[i] = 0;
        }
        statCount = OffloadAPIProgress_statFreq;
      }
    #else
      for (int i = 0; i < NUM_SPE_THREADS; i++) {
        if (statSum_nonZero_count[i] >= OffloadAPIProgress_statFreq) {
          printf("PPE :: OffloadAPIProgress() - SPE_%d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
                 i,
                 ((float)statSum_all[i]) / ((float)statSum_all_count[i]), statSum_all_count[i],
                 ((float)statSum_nonZero[i]) / ((float)statSum_nonZero_count[i]), statSum_nonZero_count[i]
                );
          statSum_all[i] = 0;
          statSum_all_count[i] = 0;
          statSum_nonZero[i] = 0;
          statSum_nonZero_count[i] = 0;
        }
      }
    #endif
  #endif

  // STATS
  #if PPE_STATS != 0
    iterCountCounter++;
  #endif

  // STATS
  #if PPE_STATS != 0
    timeval progress2start;
    gettimeofday(&progress2start, NULL);
  #endif

  // Loop through the wrQueued list and try to send outstanding messages
  int sentIndex = -1;
  int processingSPEIndex = -1;
  WorkRequest *wrEntry = wrQueuedHead;
  WorkRequest *wrEntryPrev = NULL;
  while (wrEntry != NULL) {

    // STATS
    #if PPE_STATS != 0
      iterCount++;
    #endif

    register unsigned int speAffinityMask = wrEntry->speAffinityMask;

    #if USE_MESSAGE_QUEUE_FREE_LIST == 0

      // Try each SPE
      for (int i = 0; i < NUM_SPE_THREADS; i++) {

        // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
        if (((0x01 << i) & speAffinityMask) != 0x00) {
          sentIndex = sendSPEMessage(speThreads[i], wrEntry, SPE_MESSAGE_COMMAND_NONE);
          if (sentIndex >= 0) {
            processingSPEIndex = i;
            break;
          }
        }
      }

      // Check to see if the message was sent (remove it from the wrQueued list if so)
      if (processingSPEIndex >= 0) {

        #if DEBUG_DISPLAY >= 1
          printf(" --- Offload API :: Stalled Work Request Being Issued (%p) ---\n", wrEntry);
          #if DEBUG_DISPLAY >= 2
            if (wrQueuedHead != NULL)
              printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
            else
              printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
          #endif
        #endif

        // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
        wrEntry->speIndex = processingSPEIndex;
        wrEntry->entryIndex = sentIndex;

        // Remove the wrEntry from the wrQueued list
        if (wrEntryPrev == NULL) { // is the head of the list
          wrQueuedHead = wrEntry->next;
          if (wrQueuedHead == NULL) wrQueuedTail = NULL;
        } else {  // is in the middle or at the end of the list
          wrEntryPrev->next = wrEntry->next;
          if (wrEntryPrev->next == NULL) wrQueuedTail = wrEntryPrev;
        }
        wrEntry->next = NULL;

        #if DEBUG_DISPLAY >= 2
          if (wrQueuedHead != NULL)
            printf(":: wrQueuedHead = %p, wrQueuedTail = %p    wrQueuedHead->next = %p\n", wrQueuedHead, wrQueuedTail, wrQueuedHead->next);
          else
            printf(":: wrQueuedHead = %p, wrQueuedTail = %p\n", wrQueuedHead, wrQueuedTail);
        #endif

      // Otherwise, there was a work request but no empty slots on any of the SPEs (no need to keep
      //   trying more work requests if the work request's affinity specified all SPEs ok)
      } else {
        if (speAffinityMask == 0xFFFFFFFF) break;
      }

    #else

      // Pull the first entry of the message queue free list
      if (msgQEntryFreeHead != NULL) {

        // Remove the entry from the list
        MSGQEntryList* msgQEntry = msgQEntryFreeHead;
        msgQEntryFreeHead = msgQEntryFreeHead->next;
        if (msgQEntryFreeHead == NULL) msgQEntryFreeTail = NULL;
        msgQEntry->next = NULL;
        msgQListLen--;

        // Fill in the cross-indexes
        register int processingSPEIndex = msgQEntry->speIndex;
        register int sentIndex = msgQEntry->entryIndex;

        // Send the SPE Message
        sendSPEMessage(speThreads[processingSPEIndex], sentIndex, wrEntry, SPE_MESSAGE_COMMAND_NONE, wrEntry->dmaList);

        // Set the speIndex and entryIndex of the work request since it is queued in an SPE's message queue
        wrEntry->speIndex = processingSPEIndex;
        wrEntry->entryIndex = sentIndex;

        // Remove the wrEntry from the wrQueued list
        if (wrEntryPrev == NULL) { // is the head of the list
          wrQueuedHead = wrEntry->next;
          if (wrQueuedHead == NULL) wrQueuedTail = NULL;
        } else {  // is in the middle or at the end of the list
          wrEntryPrev->next = wrEntry->next;
          if (wrEntryPrev->next == NULL) wrQueuedTail = wrEntryPrev;
        }
        wrEntry->next = NULL;

      } else {

        break;  // Quit the loop
      }

    #endif

    // Move into the next wrQueued entry
    wrEntryPrev = wrEntry;
    wrEntry = wrEntry->next;
  }

  // STATS
  #if PPE_STATS != 0
    timeval progress2end;
    gettimeofday(&progress2end, NULL);
  #endif

  // STATS
  #if PPE_STATS != 0
    // Calculate the time taken
    double startTimeD = (double)progress1start.tv_sec + ((double)progress1start.tv_usec / 1000000.0);
    double endTimeD = (double)progress1end.tv_sec + ((double)progress1end.tv_usec / 1000000.0);
    double timeDiff = endTimeD - startTimeD;
    progress1time += timeDiff;

    // Calculate the time taken
    startTimeD = (double)progress2start.tv_sec + ((double)progress2start.tv_usec / 1000000.0);
    endTimeD = (double)progress2end.tv_sec + ((double)progress2end.tv_usec / 1000000.0);
    timeDiff = endTimeD - startTimeD;
    progress2time += timeDiff;
  #endif
}

#else

extern "C"
void OffloadAPIProgress() {

  // STATS
  #if PPE_STATS != 0
    timeval progress1start;
    gettimeofday(&progress1start, NULL);
  #endif

  #define PIPELINE_LOADS   1

  #if PIPELINE_LOADS != 0
    // DEBUG - Pipeline Loads
    register SPEData* speData_0 = speThreads[0]->speData;
    register char* msgQueueRaw_0 = (char*)(speData_0->messageQueue);
    //register int* notifyQueue_0 = (int*)(speData_0->notifyQueue);
    register SPENotify* notifyQueue_0 = (SPENotify*)(speData_0->notifyQueue);
    register SPEMessage* msg_0 = (SPEMessage*)(msgQueueRaw_0);
    register int state_0 = msg_0->state;
    register int counter0_0 = msg_0->counter0;
    //register int rtnCode_0 = notifyQueue_0[0];
    //#if SPE_TIMING != 0
    //  register unsigned long long int notify_startTime_0 = notifyQueue_0[0].startTime;
    //  register unsigned int notify_runTime_0 = notifyQueue_0[0].runTime;
    //#endif
    register int notify_errorCode_0 = notifyQueue_0[0].errorCode;
    register int notify_counter_0 = notifyQueue_0[0].counter;
  #endif

  // Check each message queue entry
  for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++) {

    // For each SPE
    //for (int i = 0; i < NUM_SPE_THREADS; i++) {
    for (int i = 0; i < numSPEThreads; i++) {

      #if PIPELINE_LOADS != 0

        // Load this iteration's data
        register SPEData* speData = speData_0;
        register char* msgQueueRaw = msgQueueRaw_0;
        //register int* notifyQueue = notifyQueue_0;
        register SPENotify* notifyQueue = notifyQueue_0;
        register SPEMessage* msg = msg_0;
        register int state = state_0;
        register int counter0 = counter0_0;
        //register int rtnCode = rtnCode_0;
        //#if SPE_TIMING != 0
	//  register unsigned long long int notify_startTime = notify_startTime_0;
        //  register unsigned int notify_runTime = notify_runTime_0;
        //#endif
        register int notify_errorCode = notify_errorCode_0;
        register int notify_counter = notify_counter_0;

        register int i_0 = i + 1;
        register int j_0 = j;
        //if (__builtin_expect(i_0 >= NUM_SPE_THREADS, 0)) {
        if (__builtin_expect(i_0 >= numSPEThreads, 0)) {
          j_0++;
          i_0 = 0;
        }
        if (__builtin_expect(j_0 >= SPE_MESSAGE_QUEUE_LENGTH, 0)) {
          j_0 = 0;
        }

        // Next Iteration
        speData_0 = speThreads[i_0]->speData;
        msgQueueRaw_0 = (char*)(speData_0->messageQueue);
        //notifyQueue_0 = (int*)(speData_0->notifyQueue);
        notifyQueue_0 = (SPENotify*)(speData_0->notifyQueue);
        msg_0 = (SPEMessage*)(msgQueueRaw_0 + (j_0 * SIZEOF_16(SPEMessage)));
        state_0 = msg_0->state;
        counter0_0 = msg_0->counter0;
        //rtnCode_0 = notifyQueue_0[j_0];
        //#if SPE_TIMING != 0
        //  notify_startTime_0 = notifyQueue_0[j_0].startTime;
        //  notify_runTime_0 = notifyQueue_0[j_0].runTime;
        //#endif
        notify_errorCode_0 = notifyQueue_0[j_0].errorCode;
        notify_counter_0 = notifyQueue_0[j_0].counter;

      #else

        register SPEData* speData = speThreads[i]->speData;
        register char* msgQueueRaw = (char*)(speData->messageQueue);
        //register int* notifyQueue = (int*)(speData->notifyQueue);
        register SPENotify* notifyQueue = (SPENotify*)(speData->notifyQueue);

        register SPEMessage* msg = (SPEMessage*)(msgQueueRaw + (j * SIZEOF_16(SPEMessage)));
        register int state = msg->state;
        register int counter0 = msg->counter0;
        //register int rtnCode = notifyQueue[j];
        //#if SPE_TIMING != 0
        //  register int notify_startTime = notifyQueue[j].startTime;
        //  register int notify_runTime = notifyQueue[j].runTime;
        //#endif
        register int notify_errorCode = notifyQueue[j].errorCode;
        register int notify_counter = notifyQueue[j].counter;

      #endif

      // STATS
      #if PPE_STATS != 0
        if (state != SPE_MESSAGE_STATE_CLEAR)
          wrInUseCount++;
      #endif

      // Check to see if this message queue entry is pending a completion notification
      //if ((state == SPE_MESSAGE_STATE_SENT) && ((rtnCode & 0xFFFF) == counter0)) {
      if ((state == SPE_MESSAGE_STATE_SENT) && (notify_counter == counter0)) {

        #if SPE_TIMING != 0

          //// DEBUG
          //printf(" --- Offload API :: [DEBUG] :: WR finished with startTime = %llu, runTime = %u\n",
          //       notify_startTime, notify_runTime
	  //      );

          addProjEntry(&(notifyQueue[j]), i, msg->funcIndex);

        #endif

        // STATS
        #if PPE_STATS != 0
          wrFinishedCount++;
        #endif

        // Get a pointer to the associated work request
        register WorkRequest* wrPtr = (WorkRequest*)(msg->wrPtr);

        // Get the error code
        //register int errorCode = (rtnCode >> 16) & 0xFFFF;

        //// If there was an error returned by the SPE, display it now
        //if (__builtin_expect(errorCode != SPE_MESSAGE_OK, 0)) {
        //  fprintf(stderr, " --- Offload API :: ERROR :: SPE_%d returned error code %d for message at index %d...\n",
        //          i, errorCode, j
        //         );
        //}

        // TRACE
        #if ENABLE_TRACE != 0
          if (__builtin_expect(wrPtr->traceFlag, 0)) {
            //printf("OffloadAPI :: [TRACE] :: rtnCode = 0x%08x, errorCode(%d,%d) = %d...\n",
            //       rtnCode, i, j, errorCode
            //      );
            printf("OffloadAPI :: [TRACE] :: counter(%d,%d) = %d, errorCode(%d,%d) = %d...\n",
                   i, j, notify_counter, i, j, notify_errorCode
                  );
          }
        #endif

        // If there was an error returned by the SPE, call the error handler function now
        //if (__builtin_expect(notify_errorCode != SPE_MESSAGE_OK, 0)) {
        if (__builtin_expect(notify_errorCode != SPE_MESSAGE_OK, 0)) {
          if (errorHandlerFunc != NULL) {
            //errorHandlerFunc(errorCode, wrPtr->userData, wrPtr);
            errorHandlerFunc(notify_errorCode, wrPtr->userData, wrPtr);
	  } else {
            fprintf(stderr, " --- Offload API :: ERROR :: SPE_%d returned error code %d for message at index %d...\n",
                    //i, errorCode, j
                    i, notify_errorCode, j
                   );
	  }
	}

        //if (__builtin_expect(wrPtr == NULL, 0)) {
        //  // Warn the user that something bad has just happened
        //  fprintf(stderr, " --- Offload API :: ERROR :: Received work request completion with no associated work request !!!!!\n");
        //  // Kill self out of shame
        //  exit(EXIT_FAILURE);
        //}

        //if (__builtin_expect(wrPtr->next != NULL, 0)) {
        //  // Warn the user that something bad has just happened
        //  fprintf(stderr, " --- Offload API :: ERROR :: WorkRequest finished while still linked (msg->wrPtr->next should be NULL) !!!!!!!\n");
        //  // Kill self out of shame
        //  exit(EXIT_FAILURE);
        //}

        // DEBUG - Get time last work request completed from this SPE
        #if ENABLE_LAST_WR_TIMES != 0
          gettimeofday(&(lastWRTime[wrPtr->speIndex]), NULL);
        #endif

        // TRACE
        #if ENABLE_TRACE != 0
          if (__builtin_expect(wrPtr->traceFlag, 0)) {
            printf("OffloadAPI :: [TRACE] :: spe = %d, qi = %d, ec = %d...\n",
                   i, j, notify_errorCode
                  );
            displayMessageQueue(speThreads[i]);
          }
        #endif

	// Check to see if this work request is part of a group
	register WRGroup* wrGroup = wrPtr->wrGroupHandle;
	if (wrGroup == INVALID_WRGroupHandle) {

          // Check to see if this work request should call a callback function upon completion
          if (callbackFunc != NULL || wrPtr->callbackFunc != NULL) {

            // Call the callback function
            if (wrPtr->callbackFunc != NULL)
              (wrPtr->callbackFunc)((void*)wrPtr->userData);  // call work request specific callback function
            else
              callbackFunc((void*)(wrPtr->userData));         // call default work request callback function

            // Add this entry to the end of the wrFree list
            wrPtr->state = WORK_REQUEST_STATE_FREE;
            if (wrFreeTail == NULL) {
              wrFreeTail = wrFreeHead = wrPtr;
            } else {
              wrFreeTail->next = wrPtr;
              wrFreeTail = wrPtr;
            }

          // Otherwise, just place the WorkRequest into the wrFinished list
          } else {

            // Mark the work request as finished
            wrPtr->state = WORK_REQUEST_STATE_FINISHED;
          }

	} else {  // Otherwise, this work request is part of a group

          // Increment the group's finished counter
          wrGroup->finishedCount++;

          // Check to see if the individual work request's callback should be called in addition to the group's callback.
          if ((wrPtr->flags & WORK_REQUEST_FLAGS_BOTH_CALLBACKS) == WORK_REQUEST_FLAGS_BOTH_CALLBACKS) {            
            register void (*cbf)(void*) = ((wrPtr->callbackFunc != NULL) ? ((void (*)(void*))wrPtr->callbackFunc) : (callbackFunc));
            if (cbf != NULL) cbf(wrPtr->userData);
	  }

          //// DEBUG
          //printf(" --- Offload API : Work Request Group member finished...\n");
          //printf("       wrGroupHandle = { numWRs = %d, fC = %d, state = %d }\n",
          //       wrGroup->numWRs, wrGroup->finishedCount, wrGroup->state
          //      );
          //printf("       wrQueuedHead = %p\n", wrQueuedHead);

          // Check to see if this is the last work request in the group to complete
          if (wrGroup->state == WRGROUP_STATE_FULL && wrGroup->finishedCount >= wrGroup->numWRs) {

            register void (*cbf)(void*) = ((wrGroup->callbackFunc != NULL) ? (wrGroup->callbackFunc) : (groupCallbackFunc));

            // Check to see if there is a callback function
            if (cbf != NULL) {

              // Call the callback function
              cbf(wrGroup->userData);

              // Clean up the WRGroup structure
              wrGroup->state = WRGROUP_STATE_FREE;

              // Add the WRGroup structure back into the free list
              if (__builtin_expect(wrGroupFreeTail == NULL, 0)) {
                wrGroupFreeHead = wrGroupFreeTail = wrGroup;
              } else {
                wrGroupFreeTail->next = wrGroup;
                wrGroupFreeTail = wrGroup;
              }

	    } else {  // Otherwise, there is no callback function

              // Mark the group as finished
              wrGroup->state = WRGROUP_STATE_FINISHED;
	    }

	  }

          // Clean up the work request structure and add it to the free list
          wrPtr->state = WORK_REQUEST_STATE_FREE;
          if (wrFreeTail == NULL) {
            wrFreeTail = wrFreeHead = wrPtr;
          } else {
            wrFreeTail->next = wrPtr;
            wrFreeTail = wrPtr;
          }

	}

        // Check to see if there is a pending work request in the wrQueued list
        if (wrQueuedHead != NULL) {

          // NOTE : The common case should be that any work request can go to any SPE (optimize for this)

          // Remove the first entry in the list that has affinity for this SPE
          WorkRequest* wrEntry = wrQueuedHead;
          WorkRequest* wrEntryPrev = NULL;
          register int affinity = wrEntry->speAffinityMask;
          register int affinityMask = 0x01 << i;
          while (__builtin_expect((affinity & affinityMask) == 0x00, 0) && __builtin_expect(wrEntry != NULL, 1)) {
            wrEntryPrev = wrEntry;
            wrEntry = wrEntry->next;
            affinity = wrEntry->speAffinityMask;
	  }

          // Check to see if a work request was found
          if (__builtin_expect(wrEntry != NULL, 1)) {

            //// DEBUG
            //printf("       ISSUING IMMEDIATE...\n");

            // STATS
            #if PPE_STATS != 0
              wrImmedIssueCount++;
            #endif

            // Set the speIndex and entryIndex of the work request
            wrEntry->speIndex = i;
            wrEntry->entryIndex = j;


            // TRACE
            #if ENABLE_TRACE != 0
  	      if (wrEntry->traceFlag) {

                register int numReadOnly = wrEntry->readOnlyLen;
                register int numReadWrite = wrEntry->readWriteLen;
                register int numWriteOnly = wrEntry->writeOnlyLen;
                register DMAListEntry* dmaList = wrEntry->dmaList;
                register int jj;

                printf("OffloadAPI :: [TRACE] :: (OffloadAPIProgress) processingSPEIndex = %d, sentIndex = %d\n",
                       i, j
	  	      );

                printf("OffloadAPI :: [TRACE] :: (OffloadAPIProgress) dmaList:\n");
                for (jj = 0; jj < numReadOnly; jj++) {
                  printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (RO)\n",
                         jj, dmaList[jj].ea, dmaList[jj].size
                        );
	        }
                for (; jj < numReadOnly + numReadWrite; jj++) {
                  printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (RW)\n",
                         jj, dmaList[jj].ea, dmaList[jj].size
                        );
	        }
                for (; jj < numReadOnly + numReadWrite + numWriteOnly; jj++) {
                  printf("OffloadAPI :: [TRACE] ::                          entry %d = { ea = 0x%08x, size = %u } (WO)\n",
                         jj, dmaList[jj].ea, dmaList[jj].size
                        );
	        }
	      }
            #endif


            // Send the work request
            sendSPEMessage(speThreads[i], j, wrEntry, SPE_MESSAGE_COMMAND_NONE, wrEntry->dmaList);
            
            // Remove the work request from the queued list
            if (__builtin_expect(wrEntryPrev == NULL, 1)) { // Was the head of the queue
              wrQueuedHead = wrEntry->next;
              if (wrQueuedHead == NULL) wrQueuedTail = NULL;
            } else if (__builtin_expect(wrEntry->next == NULL, 0)) { // was the tail of the queue
              wrQueuedTail = wrEntryPrev;
              if (wrQueuedTail == NULL)
                wrQueuedHead = NULL;
              else
                wrQueuedTail->next = NULL;
	    } else { // was in the middle of the queue
              wrEntryPrev->next = wrEntry->next;
	    }
            wrEntry->next = NULL;

	  } else { // Otherwise, just clear the entry

            //// DEBUG
            //printf("       NOT - ISSUING IMMEDIATE...\n");

            // Now that the work request has been moved to either the wrFree list or marked as
            //   finished, set the state of the message queue entry to clear so it can accempt
            //   another work request
            msg->state = SPE_MESSAGE_STATE_CLEAR;

            // Re-add the message queue entry to the msgQEntryFreeList
            #if USE_MESSAGE_QUEUE_FREE_LIST != 0
              //MSGQEntryList* msgQEntry = &(__msgQEntries[speMessageQueueIndex + (i * NUM_SPE_THREADS)]);
              //MSGQEntryList* msgQEntry = &(__msgQEntries[i + (j * NUM_SPE_THREADS)]);
              MSGQEntryList* msgQEntry = &(__msgQEntries[i + (j * numSPEThreads)]);
              if (__builtin_expect(msgQEntry->next != NULL, 0)) {
                printf(" --- OffloadAPI :: ERROR :: msgQEntry->next != NULL !!!!!\n");
                msgQEntry->next = NULL;
              }
              if (msgQEntryFreeTail != NULL) {
                msgQEntryFreeTail->next = msgQEntry;
                msgQEntryFreeTail = msgQEntry;
              } else {
                msgQEntryFreeHead = msgQEntryFreeTail = msgQEntry;
              }
              msgQListLen++;
            #endif

	  }

	} else {  // Otherwise, there is not a pending work request so just clear the message queue entry

          // Now that the work request has been moved to either the wrFree list or marked as
          //   finished, set the state of the message queue entry to clear so it can accempt
          //   another work request
          msg->state = SPE_MESSAGE_STATE_CLEAR;

          // Re-add the message queue entry to the msgQEntryFreeList
          #if USE_MESSAGE_QUEUE_FREE_LIST != 0
            //MSGQEntryList* msgQEntry = &(__msgQEntries[speMessageQueueIndex + (i * NUM_SPE_THREADS)]);
            //MSGQEntryList* msgQEntry = &(__msgQEntries[i + (j * NUM_SPE_THREADS)]);
            MSGQEntryList* msgQEntry = &(__msgQEntries[i + (j * numSPEThreads)]);
            if (__builtin_expect(msgQEntry->next != NULL, 0)) {
              printf(" --- OffloadAPI :: ERROR :: msgQEntry->next != NULL !!!!!\n");
              msgQEntry->next = NULL;
            }
            if (msgQEntryFreeTail != NULL) {
              msgQEntryFreeTail->next = msgQEntry;
              msgQEntryFreeTail = msgQEntry;
            } else {
              msgQEntryFreeHead = msgQEntryFreeTail = msgQEntry;
            }
            msgQListLen++;
          #endif

	} // end if (wrQueuedHead != NULL)

      } // end if (received notification from SPE)

    } // end for (j < SPE_MESSAGE_QUEUE_LENGTH)
  } // end for (i < SPE_NUM_THREADS)

  // STATS
  #if PPE_STATS != 0
    timeval progress1end;
    gettimeofday(&progress1end, NULL);

    // Calculate the time taken
    double startTimeD = (double)progress1start.tv_sec + ((double)progress1start.tv_usec / 1000000.0);
    double endTimeD = (double)progress1end.tv_sec + ((double)progress1end.tv_usec / 1000000.0);
    double timeDiff = endTimeD - startTimeD;
    progress1time += timeDiff;

    iterCount++;
    iterCountCounter++;
  #endif
}

#endif

#endif


int getWorkRequestID(WRHandle wrHandle) {
  int rtn = ((wrHandle == NULL) ? (-1) : (wrHandle->id));
  return rtn;
}

// TRACE
#if ENABLE_TRACE != 0
  void enableTrace() { traceFlag = -1; }
  void disableTrace() { traceFlag = 0; }
#else
  void enableTrace() { }
  void disableTrace() { }
#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Private Function Bodies


void handleStopAndNotify(SPEThread* speThread, int stopCode) {

  // TODO : Handle them... for now, since there aren't any, warn that one occured
  fprintf(stderr, "OffloadAPI :: WARNING : Unhandled stop-and-notify (stopCode = %d)\n", stopCode);
}


void* pthread_func(void* arg) {

  SPEThread* speThread = (SPEThread*)arg;
  int rtnCode;

  // Load the SPE Program into the SPE Context
  rtnCode = spe_program_load(speThread->speContext, &spert_main);
  if (rtnCode != 0) {
    fprintf(stderr, "OffloadAPI :: ERROR : Unable to load program into SPE Context\n");
    exit(EXIT_FAILURE);
  }

  // Start the SPE run loop
  speThread->speEntry = SPE_DEFAULT_ENTRY;
  do {

    // Start/Continue execution of the SPE Context
    rtnCode = spe_context_run(speThread->speContext,
                              &(speThread->speEntry),
                              0,
                              speThread->speData,
                              NULL,
                              &(speThread->stopInfo)
                             );

    // React to a stop-and-notify from the SPE Context (if this is one)
    if (rtnCode > 0) handleStopAndNotify(speThread, rtnCode);

  } while (rtnCode > 0);

  if (rtnCode < 0) {
    fprintf(stderr, "OffloadAPI :: ERROR : SPE Threaded exited with rtnCode = %d\n", rtnCode);
    exit(EXIT_FAILURE);
  }

  // Destroy the SPE Context
  rtnCode = spe_context_destroy(speThread->speContext);
  if (rtnCode != 0) {
    fprintf(stderr, "OffloadAPI :: ERROR : Unable to destroy SPE Context\n");
    exit(EXIT_FAILURE);
  }
  pthread_exit(NULL);
  return NULL;
}


// Returns NULL on error... otherwise, returns a pointer to SPEData structure that was passed to the
//   created thread.
SPEThread* createSPEThread(SPEData *speData) {

  int speDataCreated = 0;

  // Check to see if the speData is NULL... if so, create a default
  if (speData == NULL) {

    // Create the SPEData structure
    speData = (SPEData*)malloc_aligned(sizeof(SPEData), 16);
    if (speData == NULL) {
      fprintf(stderr, "OffloadAPI :: ERROR : createSPEThread() : error code 1.0\n");
      return NULL;
    }

    // Create the Message Queue
    speData->messageQueue = (PPU_POINTER_TYPE)malloc_aligned(SPE_MESSAGE_QUEUE_BYTE_COUNT, 16);
    if ((void*)(speData->messageQueue) == NULL) {
      fprintf(stderr, " --- Offload API :: ERROR : createSPEThread() : Unable to allocate memory for message queue.\n");
      free_aligned(speData);
      return NULL;
    }
    memset((void*)speData->messageQueue, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
    speData->messageQueueLength = SPE_MESSAGE_QUEUE_LENGTH;

    #if SPE_NOTIFY_VIA_MAILBOX == 0
      // Create the notify queue
      speData->notifyQueue = (PPU_POINTER_TYPE)malloc_aligned(SPE_NOTIFY_QUEUE_BYTE_COUNT, 16);
      if ((void*)(speData->notifyQueue) == NULL) {
        fprintf(stderr, " --- Offload API :: ERROR : createSPEThread() : Unable to allocate memory for notification queue.\n");
        free_aligned((void*)(speData->messageQueue));
        free_aligned(speData);
        return NULL;
      }
      memset((void*)speData->notifyQueue, 0x00, SPE_NOTIFY_QUEUE_BYTE_COUNT);
    #endif

    // Give this SPE a unique number
    speData->vID = vIDCounter;
    vIDCounter++;

    // Set flag indicating that this function malloc'ed the speData structure
    speDataCreated = 1;
  }


  /// SDK 2.0 ///
  //
  //// Create the thread
  //speid_t speID = spe_create_thread((spe_gid_t)NULL,       // spe_gid_t (actually a void*) - The SPE Group ID for the Thread
  //                                  (spe_program_handle_t*)(&spert_main),    // spe_program_handle_t* - Pointer to SPE's function that should be executed (name in embedded object file, not function name from code)
  //                                  speData,                 // void* - Argument List
  //                                  (void*)NULL,             // void* - Evironment List
  //                                  (unsigned long)-1,       // unsigned long - Affinity Mask
  //                                  (int)0                   // int - Flags
  //				   );
  //
  //// Verify that the thread was actually created
  //if (speID == NULL) {
  //
  //  fprintf(stderr, "OffloadAPI :: ERROR : createSPEThread() : error code 2.0 --- spe_create_thread() returned %d...\n", speID);
  //
  //  // Clean up the speData structure if this function created it
  //  if (speDataCreated != 0 && speData != NULL) {
  //    if ((void*)(speData->messageQueue) != NULL) free_aligned((void*)(speData->messageQueue));
  //    free_aligned(speData);
  //  }
  //
  //  // Return failure
  //  return NULL;
  //}
  //
  //// Wait for the SPE thread to check in (with pointer to its message queue in its local store)
  //while (spe_stat_out_mbox(speID) <= 0);
  //unsigned int speMessageQueuePtr = spe_read_out_mbox(speID);
  //if (speMessageQueuePtr == 0) {
  //  fprintf(stderr, "OffloadAPI :: ERROR : SPE checked in with NULL value for Message Queue\n");
  //  exit(EXIT_FAILURE);
  //}
  //
  //// Wait for the thread to report it's _end
  //#if SPE_REPORT_END != 0
  //  while (spe_stat_out_mbox(speID) <= 0);
  //  unsigned int endValue = spe_read_out_mbox(speID);
  //  printf("SPE reported _end = 0x%08x\n", endValue);
  //#endif
  //
  //#if DEBUG_DISPLAY >= 1
  //  printf("---> SPE Checked in with 0x%08X\n", speMessageQueuePtr);
  //#endif
  //
  //// Create the SPEThread structure to be returned
  //SPEThread *rtn = new SPEThread;
  //rtn->speData = speData;
  //rtn->speID = speID;
  //rtn->messageQueuePtr = speMessageQueuePtr;
  //rtn->msgIndex = 0;
  //rtn->counter = 0;
  //
  //return rtn;
  //


  /// SDK 2.1 ///

  SPEThread* speThread = new SPEThread;

  // Create the SPE Context
  speThread->speContext = spe_context_create(0, NULL);
  if (speThread->speContext == NULL) {
    fprintf(stderr, "OffloadAPI :: ERROR : Unable to create SPE Context\n");
    exit(EXIT_FAILURE);
  }

  // Create the pthread for the SPE Thread
  int rtnCode = pthread_create(&(speThread->pThread), NULL, pthread_func, speThread);
  if (rtnCode != 0) {
    fprintf(stderr, "OffloadAPI :: ERROR : Unable to create pthread (rtnCode = %d)\n", rtnCode);
    exit(EXIT_FAILURE);
  }

  // Wait for the SPE thread to check in (with pointer to its message queue in its local store)
  while (spe_out_mbox_status(speThread->speContext) == 0);
  spe_out_mbox_read(speThread->speContext, &(speThread->messageQueuePtr), 1);
  if (speThread->messageQueuePtr == NULL) {
    fprintf(stderr, "OffloadAPI :: Error : SPE checked in with NULL value for Message Queue\n");
    exit(EXIT_FAILURE);
  }

  // Wait for the SPE thread to report it's _end value
  #if SPE_REPORT_END != 0
    unsigned int endValue;
    while (spe_out_mbox_status(speThread->speContext) == 0);
    spe_out_mbox_read(speThread->speContext, &endValue, 1);
    printf("SPE reported _end = 0x%08x\n", endValue);
  #endif

  // Finish filling in the SPEThread structure
  // NOTE: The speEntry and stopInfo fields should not be written to.  The pthread will fillin and use
  //   these fields.  Writing to them here will create a race condition.  Badness!
  speThread->speData = speData;
  speThread->msgIndex = 0;
  speThread->counter = 0;

  return speThread;
}


// Returns NULL on error... otherwise, returns a pointer to SPEData structure that was passed to the
//   created thread.
SPEThread** createSPEThreads(SPEThread **speThreads, int numThreads) {

  int speDataCreated = 0;

  // Verify the parameters
  if (speThreads == NULL || numThreads < 1)
    return NULL;

  // Create the Message Queues
  for (int i = 0; i < numThreads; i++) {

    // Create the SPEThread structure if it does not already exist
    if (speThreads[i] == NULL) speThreads[i] = new SPEThread;
    if (speThreads[i] == NULL) {
      printf("OffloadAPI :: ERROR :: Unable to allocate memory for SPEThread structure... Exiting.\n");
      exit(EXIT_FAILURE);
    }

    // Create the speData structure
    speThreads[i]->speData = (SPEData*)malloc_aligned(SIZEOF_16(SPEData), 16);
    if (speThreads[i]->speData == NULL) {
      printf("OffloadAPI :: ERROR :: Unable to allocate memory for SPEData structure... Exiting.\n");
      exit(EXIT_FAILURE);
    }

    // Create the message queue
    speThreads[i]->speData->messageQueue = (PPU_POINTER_TYPE)malloc_aligned(SPE_MESSAGE_QUEUE_BYTE_COUNT, 16);
    if ((void*)(speThreads[i]->speData->messageQueue) == NULL) {
      fprintf(stderr, " --- Offload API :: ERROR : createSPEThreads() : Unable to allocate memory for message queue.\n");
      exit(EXIT_FAILURE);
    }
    memset((void*)speThreads[i]->speData->messageQueue, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
    speThreads[i]->speData->messageQueueLength = SPE_MESSAGE_QUEUE_LENGTH;

    #if SPE_NOTIFY_VIA_MAILBOX == 0
      // Create the notify queue
      speThreads[i]->speData->notifyQueue = (PPU_POINTER_TYPE)malloc_aligned(SPE_NOTIFY_QUEUE_BYTE_COUNT, 16);
      if ((void*)(speThreads[i]->speData->notifyQueue) == NULL) {
        fprintf(stderr, " --- Offload API :: ERROR : createSPEThreads() : Unable to allocate memory for notification queue.\n");
        exit(EXIT_FAILURE);
      }
      memset((void*)speThreads[i]->speData->notifyQueue, 0x00, SPE_NOTIFY_QUEUE_BYTE_COUNT);
    #endif

    // Give the SPE a unique id
    speThreads[i]->speData->vID = vIDCounter;
    vIDCounter++;
  }

  /// SDK 2.0 ///
  //
  //// Create all the threads at once
  //for (int i = 0; i < numThreads; i++) {
  //
  //  speid_t speID = spe_create_thread((spe_gid_t)NULL,         // spe_gid_t (actually a void*) - The SPE Group ID for the Thread
  //                                    (spe_program_handle_t*)(&spert_main),    // spe_program_handle_t* - Pointer to SPE's function that should be executed (name in embedded object file, not function name from code)
  //                                    (void*)speThreads[i]->speData, // void* - Argument List
  //                                    (void*)NULL,             // void* - Evironment List
  //                                    (unsigned long)-1,       // unsigned long - Affinity Mask
  //                                    (int)0                   // int - Flags
  //                                   );
  //
  //  // Verify that the thread was actually created
  //  if (speID == NULL) {
  //    fprintf(stderr, "OffloadAPI :: ERROR : createSPEThreads() : error code 2.0 --- spe_create_thread() returned %d...\n", speID);
  //    exit(EXIT_FAILURE);
  //  }
  //
  //  // Store the speID
  //  speThreads[i]->speID = speID;
  //}
  //
  //// Wait for all the threads to check in (with pointer to its message queue in its local store)
  //for (int i = 0; i < numThreads; i++) {
  //
  //  while (spe_stat_out_mbox(speThreads[i]->speID) <= 0);
  //  unsigned int speMessageQueuePtr = spe_read_out_mbox(speThreads[i]->speID);
  //  if (speMessageQueuePtr == 0) {
  //   fprintf(stderr, "OffloadAPI :: ERROR : SPE checked in with NULL value for Message Queue\n");
  //   exit(EXIT_FAILURE);
  //  }
  //  speThreads[i]->messageQueuePtr = speMessageQueuePtr;
  //
  //  #if DEBUG_DISPLAY >= 1
  //    printf("---> SPE Checked in with 0x%08X\n", speMessageQueuePtr);
  //  #endif
  //
  //  // Wait for the thread to report it's _end
  //  #if SPE_REPORT_END != 0
  //    while (spe_stat_out_mbox(speThreads[i]->speID) <= 0);
  //    unsigned int endValue = spe_read_out_mbox(speThreads[i]->speID);
  //    printf("SPE_%d reported &(_end) = 0x%08x\n", i, endValue);
  //  #endif
  //}
  //
  //// Finish filling in the speThreads array
  //for (int i = 0; i < numThreads; i++) {
  //  speThreads[i]->msgIndex = 0;
  //  speThreads[i]->counter = 0;
  //}
  //


  /// SDK 2.1 ///

  // For each of the SPE Threads...
  for (int i = 0; i < numThreads; i++) {

    // Create the SPE Context
    speThreads[i]->speContext = spe_context_create(0, NULL);
    if (speThreads[i]->speContext == NULL) {
      printf("OffloadAPI :: ERROR : Unable to create SPE Context\n");
      exit(EXIT_FAILURE);
    }

    // Create the pthread for the SPE Thread
    int rtnCode = pthread_create(&(speThreads[i]->pThread), NULL, pthread_func, speThreads[i]);
    if (rtnCode != 0) {
      fprintf(stderr, "OffloadAPI :: ERROR : Unable to create pthread (rtnCode = %d)\n", rtnCode);
      exit(EXIT_FAILURE);
    }
  }

  // For each of the SPE Threads...
  // NOTE : Done as a separate loop since SPEs need startup time (i.e. don't create and wait for the first
  //   SPE thread before creating the second... create all then wait all... by the time the last is created
  //   hopefully the first has checked in).
  for (int i = 0; i < numThreads; i++) {

    // Wait for the SPE Thread to check in
    while (spe_out_mbox_status(speThreads[i]->speContext) == 0);
    spe_out_mbox_read(speThreads[i]->speContext, &(speThreads[i]->messageQueuePtr), 1);
    if (speThreads[i]->messageQueuePtr == NULL) {
      fprintf(stderr, "OffloadAPI :: Error : SPE checked in with NULL value for Message Queue\n");
      exit(EXIT_FAILURE);
    }

    // Wait for the SPE thread to report it's _end value
    #if SPE_REPORT_END != 0
      unsigned int endValue;
      while (spe_out_mbox_status(speThreads[i]->speContext) == 0);
      spe_out_mbox_read(speThreads[i]->speContext, &endValue, 1);
      printf("SPE reported _end = 0x%08x\n", endValue);
    #endif

    // Finish filling in the SPEThread structure
    // NOTE: The speEntry and stopInfo fields should not be written to.  The pthread will fillin and use
    //   these fields.  Writing to them here will create a race condition.  Badness!
    speThreads[i]->msgIndex = 0;
    speThreads[i]->counter = 0;
  }


  return speThreads;
}


int sendSPEMessage(SPEThread *speThread, WorkRequest *wrPtr, int command) {

  // TODO : Re-write these checks now that command no longer use the message queue

  // Verify the parameters
  //if (speThread == NULL) return -1;

  //if (wrPtr != NULL && wrPtr->funcIndex < 0 && command == SPE_MESSAGE_COMMAND_NONE)
  //  return -1;

  // Force the command value to a valid command
  //if (command < SPE_MESSAGE_COMMAND_MIN || command > SPE_MESSAGE_COMMAND_MAX)
  //  command = SPE_MESSAGE_COMMAND_NONE;

  /*
  if (wrPtr != NULL) {

    // Check to see if there is no work request and no command (if so, return without doing anything)
    if (wrPtr->funcIndex < 0 &&
        wrPtr->readWritePtr == NULL && wrPtr->readOnlyPtr == NULL && wrPtr->writeOnlyPtr == NULL &&
        command == SPE_MESSAGE_COMMAND_NONE
       )
      return -1;

    // Make sure the readWriteLen is non-negative
    if (wrPtr->readWriteLen < 0) {
      #if DEBUG_DISPLAY >= 1
        fprintf(stderr, "OffloadAPI :: WARNING :: sendSPEMessage() received readWriteLen < 0... forcing to 0...\n");
      #endif
      wrPtr->readWriteLen = 0;
    }

    // Make sure the readOnlyLen is non-negative
    if (wrPtr->readOnlyLen < 0) {
      #if DEBUG_DISPLAY >= 1
        fprintf(stderr, "OffloadAPI :: WARNING :: sendSPEMessage() received readOnlyLen < 0... forcing to 0...\n");
      #endif
      wrPtr->readOnlyLen = 0;
    }

    // Make sure the writeOnlyLen is non-negative
    if (wrPtr->writeOnlyLen < 0) {
      #if DEBUG_DISPLAY >= 1
        fprintf(stderr, "OffloadAPI :: WARNING :: sendSPEMessage() received writeOnlyLen < 0... forcing to 0...\n");
      #endif
      wrPtr->writeOnlyLen = 0;
    }

    if ((wrPtr->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {  // standard send
      // Force the xxxPtr and xxxLen pairs so they match (i.e. - 'xxxPtr != NULL && xxxLen <= 0' is not good)
      if (wrPtr->readWritePtr == NULL) wrPtr->readWriteLen = 0;
      if (wrPtr->readWriteLen <= 0) { wrPtr->readWritePtr = NULL; wrPtr->readWriteLen = 0; }
      if (wrPtr->readOnlyPtr == NULL) wrPtr->readOnlyLen = 0;
      if (wrPtr->readOnlyLen <= 0) { wrPtr->readOnlyPtr = NULL; wrPtr->readOnlyLen = 0; }
      if (wrPtr->writeOnlyPtr == NULL) wrPtr->writeOnlyLen = 0;
      if (wrPtr->writeOnlyLen <= 0) { wrPtr->writeOnlyPtr = NULL; wrPtr->writeOnlyLen = 0; }
    } else {  // dma list send
      if (wrPtr->readWritePtr == NULL) return -1;  // DMA list send with no list
      if (wrPtr->readWriteLen <= 0 && wrPtr->readOnlyLen <= 0 && wrPtr->writeOnlyLen <= 0) return -1; // no list item info
    }
  }
  */

  // Find the next available index
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    int index = (speThread->msgIndex + i) % SPE_MESSAGE_QUEUE_LENGTH;
    volatile SPEMessage* msg = (SPEMessage*)(((char*)speThread->speData->messageQueue) + (index * SIZEOF_16(SPEMessage)));
    
    if (msg->state == SPE_MESSAGE_STATE_CLEAR) {

      // TRACE
      #if ENABLE_TRACE != 0
        if (wrPtr->traceFlag) {
          printf("PPE :: [TRACE] :: sendSPEMessage() :: message queue before new entry...\n");
          displayMessageQueue(speThread);
        }
      #endif

      if (__builtin_expect(wrPtr == NULL, 0)) {  // NOTE: Common case should be user's code making work
        msg->funcIndex = -1;                     //   requests (not commands going to the SPE Runtime)
        msg->readWritePtr = (PPU_POINTER_TYPE)NULL;
        msg->readWriteLen = 0;
        msg->readOnlyPtr = (PPU_POINTER_TYPE)NULL;
        msg->readOnlyLen = 0;
        msg->writeOnlyPtr = (PPU_POINTER_TYPE)NULL;
        msg->writeOnlyLen = 0;
        msg->flags = WORK_REQUEST_FLAGS_NONE;
        msg->totalMem = 0;

        // TRACE
        #if ENABLE_TRACE != 0
          msg->traceFlag = 0;
        #endif

      } else {
        msg->funcIndex = wrPtr->funcIndex;
        msg->readWritePtr = (PPU_POINTER_TYPE)(wrPtr->readWritePtr);
        msg->readWriteLen = wrPtr->readWriteLen;
        msg->readOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->readOnlyPtr);
        msg->readOnlyLen = wrPtr->readOnlyLen;
        msg->writeOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->writeOnlyPtr);
        msg->writeOnlyLen = wrPtr->writeOnlyLen;
        msg->flags = wrPtr->flags;

        // Copy the DMA list (if is list WR and the dma list is small enought... otherwise, don't bother)
        if ((wrPtr->flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST) {
          register int dmaListSize = wrPtr->readWriteLen + wrPtr->readOnlyLen + wrPtr->writeOnlyLen;
          if (dmaListSize <= SPE_DMA_LIST_LENGTH) {
            register volatile DMAListEntry* msgDMAList = msg->dmaList;
            register DMAListEntry* wrDMAList = (DMAListEntry*)(wrPtr->readWritePtr);
            for (int i = 0; i < dmaListSize; i++) {
              msgDMAList[i].ea = wrDMAList[i].ea;
              msgDMAList[i].size = wrDMAList[i].size;
            }
          }
        }

        // Calculate the total amount of memory that will be needed on the SPE for this message/work-request
        if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST) {
          // The memory needed is the size of the DMA list rounded up times 2 (two lists) and the size of each
          //   of the individual entries in that list all rounded up
          register int numEntries = wrPtr->readWriteLen + wrPtr->readOnlyLen + wrPtr->writeOnlyLen;
          msg->totalMem = ROUNDUP_16(sizeof(DMAListEntry) * numEntries);
          msg->totalMem *= 2;  // Second DMA List within SPE's local store (with LS pointers)
          for (int entryIndex = 0; entryIndex < numEntries; entryIndex++)
            msg->totalMem += ROUNDUP_16(((DMAListEntry*)(wrPtr->readWritePtr))[entryIndex].size);
	} else {
          // The memory needed is the size of the sum of the three buffers each rounded up
          msg->totalMem = ROUNDUP_16(wrPtr->readWriteLen) + ROUNDUP_16(wrPtr->readOnlyLen) + ROUNDUP_16(wrPtr->writeOnlyLen);
	}

        // TRACE
        #if ENABLE_TRACE != 0
          msg->traceFlag = ((wrPtr->traceFlag) ? (-1) : (0));  // force -1 or 0
        #endif
      }
      msg->state = SPE_MESSAGE_STATE_SENT;
      msg->command = command;
      msg->wrPtr = (PPU_POINTER_TYPE)wrPtr;

      // NOTE: Important that the counter be the last then set (the change in this value is what prompts the
      //   SPE to consider the entry a new entry... even if the state has been set to SENT).
      // NOTE: Only change the value of msg->counter once so the SPE is not confused (i.e. - don't increment
      //   and then check msg->counter direclty).
      int tmp0 = msg->counter0;
      int tmp1 = msg->counter1;
      tmp0++; if (tmp0 > 255) tmp0 = 0;
      tmp1++; if (tmp1 > 255) tmp1 = 0;
      __asm__ ("sync");
      msg->counter0 = tmp0;
      msg->counter1 = tmp1;

      // TRACE
      #if ENABLE_TRACE != 0
        if (wrPtr->traceFlag) {
          printf("  sendSPEMessage() : sending message to queue @ %p, slot %d (@ %p; SIZEOF_16(SPEMessage) = %d)...\n",
                 speThread->speData->messageQueue,
                 index,
                 msg,
                 SIZEOF_16(SPEMessage)
	        );
          printf("  sendSPEMessage() : message[%d] = {", index);
          printf(" funcIndex = %d", msg->funcIndex);
          printf(", readWritePtr = %u", msg->readWritePtr);
          //printf(", readWriteLen = %d", msg->readWriteLen);
          printf(", readOnlyPtr = %u", msg->readOnlyPtr);
          //printf(", readOnlyLen = %d", msg->readOnlyLen);
          printf(", writeOnlyPtr = %u", msg->writeOnlyPtr);
          //printf(", writeOnlyLen = %d", msg->writeOnlyLen);
          //printf(", flags = 0x08x", msg->flags);
          printf(", wrPtr = %u", msg->wrPtr);
          printf(", state = %d", msg->state);
          //printf(", command = %d", msg->command);
          //printf(", counter = %d", msg->counter);
          printf(" }\n");
        }
      #endif

      speThread->msgIndex = (index + 1) % SPE_MESSAGE_QUEUE_LENGTH;

      return index;
    } // end if (msg->state == SPE_MESSAGE_STATE_CLEAR)
  } // end for (loop through message queue entries)

  // If execution reaches here, an available slot in the message queue was not found
  return -1;
}


WorkRequest* createWRHandles(int numHandles) {


  // DEBUG
  //printf("OffloadAPI :: [DEBUG] :: createWRHandles(%d) - Called...\n", numHandles);


  // Verify the parameter
  if (numHandles <= 0) return NULL;

  #if DEBUG_DISPLAY >= 1
    printf(" --- Offload API ::: Creating %d more WRHandles ---\n", numHandles);
  #endif

  // Allocate the memory for all the WRHandles at once
  WorkRequest *workReqs = new WorkRequest[numHandles];

  // Initialize the entries
  memset(workReqs, 0, numHandles * sizeof(WorkRequest));
  for (int i = 0; i < numHandles; i++) {
    workReqs[i].isFirstInSet = ((i == 0) ? (TRUE) : (FALSE));
    workReqs[i].state = WORK_REQUEST_STATE_FREE;
    workReqs[i].next = ((i < (numHandles - 1)) ? (&(workReqs[i+1])) : (NULL));
  }

  // Add this allocated WRHandle array to the allocatedWRHandlesList
  if (allocatedWRHandlesList == NULL) {
    allocatedWRHandlesList = new PtrList;
    allocatedWRHandlesList->ptr = (void*)workReqs;
    allocatedWRHandlesList->next = NULL;
  } else {
    PtrList *entry = allocatedWRHandlesList;
    while (entry->next != NULL) entry = entry->next;
    entry->next = new PtrList;
    entry = entry->next;
    entry->ptr = (void*)workReqs;
    entry->next = NULL;
  }

  return workReqs;
}


WRGroup* createWRGroupHandles(int numHandles) {


  // DEBUG
  //printf("OffloadAPI :: [DEBUG] :: createWRGroupHandles(%d) - Called...\n", numHandles);


  // Verify the parameter
  if (numHandles <= 0) return NULL;

  #if DEBUG_DISPLAY >= 1
    printf(" --- Offload API ::: Creating %d more WRGroupHandles ---\n", numHandles);
  #endif

  // Allocate the memory for all the WRGroupHandles at once
  WRGroup* groups = new WRGroup[numHandles];

  // Initialize the entries
  memset(groups, 0, numHandles * sizeof(WRGroup));
  for (int i = 0; i < numHandles; i++) {
    groups[i].state = WRGROUP_STATE_FREE;
    groups[i].next = ((i < (numHandles - 1)) ? (&(groups[i+1])) : (NULL));
  }

  // Add this allocated WRHandle array to the allocatedWRHandlesList
  if (allocatedWRGroupHandlesList == NULL) {
    allocatedWRGroupHandlesList = new PtrList;
    allocatedWRGroupHandlesList->ptr = (void*)groups;
    allocatedWRGroupHandlesList->next = NULL;
  } else {
    PtrList *entry = allocatedWRGroupHandlesList;
    while (entry->next != NULL) entry = entry->next;
    entry->next = new PtrList;
    entry = entry->next;
    entry->ptr = (void*)groups;
    entry->next = NULL;
  }

  return groups;
}


void OffloadAPIDisplayConfig(FILE* fout) {

  // Make sure fout points somewhere
  if (fout == NULL) fout = stdout;

  // Dump the Offload API's configuration parameters to the file specified
  fprintf(fout, "OffloadAPI :: [CONFIG] :: PPE:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Threads:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     NUM_SPE_THREADS: %d\n", NUM_SPE_THREADS);
  #if NUM_SPE_THREADS <= 0
    fprintf(fout, "OffloadAPI :: [CONFIG] ::     numSPEThreads: %d\n", numSPEThreads);
  #endif
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     CREATE_EACH_THREAD_ONE_BY_ONE: %d\n", CREATE_EACH_THREAD_ONE_BY_ONE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Debugging:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     DEBUG_DISPLAY: %d\n", DEBUG_DISPLAY);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Stats:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     PPE_STATS: %d\n", PPE_STATS);
  fprintf(fout, "OffloadAPI :: [CONFIG] :: SPE:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Queues:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_MESSAGE_QUEUE_LENGTH: %d\n", SPE_MESSAGE_QUEUE_LENGTH);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_MESSAGE_QUEUE_BYTE_COUNT: %d\n", SPE_MESSAGE_QUEUE_BYTE_COUNT);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     DOUBLE_BUFFER_MESSAGE_QUEUE: %d\n", DOUBLE_BUFFER_MESSAGE_QUEUE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_NOTIFY_VIA_MAILBOX: %d\n", SPE_NOTIFY_VIA_MAILBOX);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_NOTIFY_QUEUE_BYTE_COUNT: %d\n", SPE_NOTIFY_QUEUE_BYTE_COUNT);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Static DMA Lists:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_DMA_LIST_LENGTH: %d\n", SPE_DMA_LIST_LENGTH);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_DMA_LIST_ENTRY_MAX_LENGTH: %d\n", SPE_DMA_LIST_ENTRY_MAX_LENGTH);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Memory:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_TOTAL_MEMORY_SIZE: %d\n", SPE_TOTAL_MEMORY_SIZE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_USE_OWN_MEMSET: %d\n", SPE_USE_OWN_MEMSET);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_USE_OWN_MALLOC: %d\n", SPE_USE_OWN_MALLOC);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_MEMORY_BLOCK_SIZE: %d\n", SPE_MEMORY_BLOCK_SIZE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_RESERVED_STACK_SIZE: %d\n", SPE_RESERVED_STACK_SIZE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_MINIMUM_HEAP_SIZE: %d\n", SPE_MINIMUM_HEAP_SIZE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_ZERO_WRITE_ONLY_MEMORY: %d\n", SPE_ZERO_WRITE_ONLY_MEMORY);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Scheduler Controls:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_USE_STATE_LOOKUP_TABLE: %d\n", SPE_USE_STATE_LOOKUP_TABLE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     LIMIT_READY: %d\n", LIMIT_READY);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Tracing:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     ENABLE_TRACE: %d\n", ENABLE_TRACE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Debugging:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_DEBUG_DISPLAY: %d\n", SPE_DEBUG_DISPLAY);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_DEBUG_DISPLAY_STILL_ALIVE: %d\n", SPE_DEBUG_DISPLAY_STILL_ALIVE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_DEBUG_DISPLAY_NO_PROGRESS: %d\n", SPE_DEBUG_DISPLAY_NO_PROGRESS);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_REPORT_END: %d\n", SPE_REPORT_END);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_NOTIFY_ON_MALLOC_FAILURE: %d\n", SPE_NOTIFY_ON_MALLOC_FAILURE);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   Stats:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_TIMING: %d\n", SPE_TIMING);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_STATS: %d\n", SPE_STATS);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_STATS1: %d\n", SPE_STATS1);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_STATS2: %d\n", SPE_STATS2);
  fprintf(fout, "OffloadAPI :: [CONFIG] ::   User Tags:\n");
  fprintf(fout, "OffloadAPI :: [CONFIG] ::     SPE_NUM_USER_TAGS: %d\n", SPE_NUM_USER_TAGS);
}


#if SPE_TIMING != 0

void openProjFile(char* name) {

  char buf[256];
  buf[0] = '\0';

  // Verify the parameter
  if (name == NULL || strlen(name) <= 0)
    name = "default";

  // Create the file name
  sprintf(buf, "%s.cellRaw", name);

  // Open the file for writing
  projFile = fopen(buf, "w+");
  if (__builtin_expect(projFile == NULL, 0)) {
    fprintf(stderr, " --- Offload API :: [WARNING] :: Unable to open timing file (\"%s\")...\n", buf);
    return;
  }
}

void closeProjFile() {

  if (projFile != NULL) {

    // Flush the remaining buffer entries to the file
    flushProjBuf();

    // Output the number of total entries
    register size_t bytesWritten = fwrite((void*)(&totalProjSampleCount), 1, sizeof(int), projFile);
    if (sizeof(int) != bytesWritten) {
      fprintf(stderr,
              " --- Offload API :: [WARNING] :: Incorrect number of bytes written when writing entry count (%d of %d bytes)...\n",
              bytesWritten, sizeof(int)
             );
    }

    // Close the file
    fclose(projFile);
  }
}

void addProjEntry(SPENotify* notifyEntry, int speIndex, int funcIndex) {

  // Make sure the file is open
  if (projFile == NULL) return;

  // Make sure there is room in the buffer for this entry (if not, flush)
  if (projBufCount >= PROJ_BUF_SIZE)
    flushProjBuf();

  // Add the entry to the buffer
  //projBuf[projBufCount].startTime = notifyEntry->startTime;
  //projBuf[projBufCount].runTime = notifyEntry->runTime;
  projBuf[projBufCount].speIndex = speIndex;
  projBuf[projBufCount].funcIndex = funcIndex;

  projBuf[projBufCount].recvTimeStart = notifyEntry->recvTimeStart;
  projBuf[projBufCount].recvTimeEnd = notifyEntry->recvTimeEnd;
  projBuf[projBufCount].preFetchingTimeStart = notifyEntry->preFetchingTimeStart;
  projBuf[projBufCount].preFetchingTimeEnd = notifyEntry->preFetchingTimeEnd;
  projBuf[projBufCount].fetchingTimeStart = notifyEntry->fetchingTimeStart;
  projBuf[projBufCount].fetchingTimeEnd = notifyEntry->fetchingTimeEnd;
  projBuf[projBufCount].readyTimeStart = notifyEntry->readyTimeStart;
  projBuf[projBufCount].readyTimeEnd = notifyEntry->readyTimeEnd;
  projBuf[projBufCount].userTimeStart = notifyEntry->userTimeStart;
  projBuf[projBufCount].userTimeEnd = notifyEntry->userTimeEnd;
  projBuf[projBufCount].executedTimeStart = notifyEntry->executedTimeStart;
  projBuf[projBufCount].executedTimeEnd = notifyEntry->executedTimeEnd;
  projBuf[projBufCount].commitTimeStart = notifyEntry->commitTimeStart;
  projBuf[projBufCount].commitTimeEnd = notifyEntry->commitTimeEnd;

  projBuf[projBufCount].userTime0Start = notifyEntry->userTime0Start;
  projBuf[projBufCount].userTime0End = notifyEntry->userTime0End;
  projBuf[projBufCount].userTime1Start = notifyEntry->userTime1Start;
  projBuf[projBufCount].userTime1End = notifyEntry->userTime1End;
  projBuf[projBufCount].userTime2Start = notifyEntry->userTime2Start;
  projBuf[projBufCount].userTime2End = notifyEntry->userTime2End;

  projBuf[projBufCount].userAccumTime0 = notifyEntry->userAccumTime0;
  projBuf[projBufCount].userAccumTime1 = notifyEntry->userAccumTime1;
  projBuf[projBufCount].userAccumTime2 = notifyEntry->userAccumTime2;
  projBuf[projBufCount].userAccumTime3 = notifyEntry->userAccumTime3;

  // Increment the projBufCount so it points to the next projBuf entry
  projBufCount++;

  // Increment the counter for the number of total samples
  totalProjSampleCount++;
}

void flushProjBuf() {

  // Make sure the file is open
  if (projFile == NULL) return;

  // Calculate the number of bytes in projBuf that contain valid data
  register int byteCount = sizeof(ProjBufEntry) * projBufCount;

  // Write projBuf (or a portion of it) to the projFile
  register size_t bytesWritten = fwrite((void*)projBuf, 1, byteCount, projFile);
  if (bytesWritten != byteCount) {
    fprintf(stderr,
            " --- Offload API :: [WARNING] :: Flush only wrote %d of %d bytes to timing file...\n",
            bytesWritten, byteCount
           );
  }

  // Reset the projBufCount counter
  projBufCount = 0;
}

#endif
