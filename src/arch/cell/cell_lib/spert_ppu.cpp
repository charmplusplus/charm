#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

extern "C" {
  #include <libspe.h>
}

#include "spert_common.h"
#include "spert.h"

extern "C" {
#include "spert_ppu.h"
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#define WRHANDLES_NUM_INITIAL  (64)
#define WRHANDLES_GROW_SIZE    (16)

#define MESSAGE_RETURN_CODE_INDEX(rc)  ((unsigned int)(rc) & 0x0000FFFF)
#define MESSAGE_RETURN_CODE_ERROR(rc)  (((unsigned int)(rc) >> 16) & 0x0000FFFF)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structures

typedef struct __pointer_list {
  void *ptr;
  struct __pointer_list *next;
} PtrList;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

SPEThread* speThreads[NUM_SPE_THREADS];
unsigned long long int wrCounter = 0;

void (*callbackFunc)(void*) = NULL;

// Work Request Structures
PtrList *allocatedWRHandlesList = NULL;

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

// This is used in an attempt to more evenly distribute the workload amongst all the SPE Threads
int speSendStartIndex = 0;

// A counter used to give each SPE a unique ID
unsigned short vIDCounter = 0;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

SPEThread* createSPEThread(SPEData *speData);
SPEThread** createSPEThreads(SPEThread **speThreads, int numThreads);

int sendSPEMessage(SPEThread* speThread, WorkRequest* wrPtr, int command);
int sendSPECommand(SPEThread* speThread, int command);

WorkRequest* createWRHandles(int numHandles);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Offload API Function Bodies

extern "C"
int InitOffloadAPI(void (*cbFunc)(void*)) {

  // Let the user know that the Offload API is being initialized
  #if DEBUG_DISPLAY >= 1
    printf("----- Offload API : Enabled... Initializing -----\n");
  #endif

  // If the caller specified a callback function, set callbackFunc to point to it
  callbackFunc = ((cbFunc != NULL) ? (cbFunc) : (NULL));

  // Start Creating the SPE threads
  #if DEBUG_DISPLAY >= 1
    printf(" --- Creating SPE Threads ---\n");
  #endif

  #if CREATE_EACH_THREAD_ONE_BY_ONE
    for (int i = 0; i < NUM_SPE_THREADS; i++) {

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
        printf("SPE %d Created {\n", i);
        printf("  speThreads[%d]->speData->messageQueue = %p\n", i, (void*)(speThreads[i]->speData->messageQueue));
        printf("  speThreads[%d]->speData->messageQueueLength = %d\n", i, speThreads[i]->speData->messageQueueLength);
        printf("  speThreads[%d]->speID = %d\n", i, speThreads[i]->speID);
        printf("}\n");
      #endif
    }
  #else

    if (createSPEThreads(speThreads, NUM_SPE_THREADS) == NULL) {
      fprintf(stderr, "OffloadAPI :: ERROR :: createSPEThreads returned NULL... Exiting.\n");
      exit(EXIT_FAILURE);
    }

    // Display information about the threads that were just created
    #if DEBUG_DISPLAY >= 1
      for (int i = 0; i < NUM_SPE_THREADS; i++) {
        printf("SPE %d Created {\n", i);
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

  return 1;  // Sucess
}


extern "C"
void CloseOffloadAPI() {

  int status;
 
  #if DEBUG_DISPLAY >= 1
    printf(" ---------- CLOSING OFFLOAD API ----------\n");
  #endif

  // Send each of the SPE threads a message to exit
  for (int i = 0; i < NUM_SPE_THREADS; i++)
    sendSPECommand(speThreads[i], SPE_MESSAGE_COMMAND_EXIT);

  // Wait for all the SPE threads to finish
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    #if DEBUG_DISPLAY >= 1
      printf("OffloadAPI :: Waiting for SPE %d to Exit...\n", i);
    #endif

    spe_wait(speThreads[i]->speID, &status, 0);

    #if DEBUG_DISPLAY >= 1
      printf("OffloadAPI :: SPE %d Finished (status : %d)\n", i, status);
    #endif
  }

  // Clean-up any data structures that need cleaned up

  // Clean up the speThreads
  for (int i = 0; i < NUM_SPE_THREADS; i++) {
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
                         unsigned int speAffinityMask
                        ) {

  int processingSPEIndex = -1;
  int sentIndex = -1;

  // Tell the PPU's portion of the spert to make progress
  OffloadAPIProgress();

  // Verify the parameters
  if (funcIndex < 0) return INVALID_WRHandle;
  if (readWritePtr != NULL && readWriteLen <= 0) return INVALID_WRHandle;
  if (readOnlyPtr != NULL && readOnlyLen <= 0) return INVALID_WRHandle;
  if (writeOnlyPtr != NULL && writeOnlyLen <= 0) return INVALID_WRHandle;
  if ((flags & WORK_REQUEST_FLAGS_LIST) == WORK_REQUEST_FLAGS_LIST) {
    #if DEBUG_DISPLAY >= 1
      fprintf(stderr, "OffloadAPI :: WARNING :: sendWorkRequest() call made with WORK_REQUEST_FLAGS_LIST flag set... ignoring...\n");
    #endif
    flags &= (~WORK_REQUEST_FLAGS_LIST);
  }

  // Ensure that there is at least one free WRHandle structure
  if (wrFreeHead == NULL) {  // Out of free work request structures
    // Add some more WRHandle structures to the wrFree list
    wrFreeHead = createWRHandles(WRHANDLES_GROW_SIZE);
    wrFreeTail = &(wrFreeHead[WRHANDLES_GROW_SIZE - 1]);
  }

  // Grab the first free WRHandle structure and use it for this entry
  WorkRequest *wrEntry = wrFreeHead;
  wrFreeHead = wrFreeHead->next;
  if (wrFreeHead == NULL) wrFreeTail = NULL;
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
  wrEntry->next = NULL;

  // Try to send the message
  // NOTE: Through the use of speSendStartIndex, if the SPE's message queues aren't being overloaded, then
  //   this loop should only iterate once and then break.
  // TODO : Update this so the number of outstanding work requests that have been sent to each SPE Thread
  //   is used to pick the SPE Thread to send this request to (i.e. - Send to the thread will the SPE Thread
  //   with the least full message queue.)  For now, the speSendStartIndex heuristic should help.
  processingSPEIndex = -1;
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
    if (((0x01 << i) & speAffinityMask) != 0x00) {

      // NOTE: Since NUM_SPE_THREADS should be 8, the "% NUM_SPE_THREADS" should be transformed into masks/shifts
      //   by strength reduction in the compiler when optimizations are turned on... for debuging purposes, leave them
      //   as "% NUM_SPE_THREADS" in the code (if NUM_SPE_THREADS is reduced to save startup time in the simulator).
      register int actualSPEThreadIndex = (i + speSendStartIndex) % NUM_SPE_THREADS;
      sentIndex = sendSPEMessage(speThreads[actualSPEThreadIndex], wrEntry, SPE_MESSAGE_COMMAND_NONE);

      if (sentIndex >= 0) {
        speSendStartIndex = (actualSPEThreadIndex + 1) % NUM_SPE_THREADS;
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

    if (wrEntry->next != NULL) {
      printf(" ---- sendWorkRequest() ---- :: ERROR : Sent work request where wrEntry->next != NULL\n");
    }
  }

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
                              unsigned int speAffinityMask
                             ) {

  int processingSPEIndex = -1;
  int sentIndex = -1;

  // Tell the PPU's portion of the spert to make progress
  OffloadAPIProgress();

  // Verify the parameters
  if (funcIndex < 0) return INVALID_WRHandle;
  if (dmaList == NULL) return INVALID_WRHandle;
  if (numReadOnly == 0 && numReadWrite == 0 && numWriteOnly == 0) return INVALID_WRHandle;
  if ((flags & (WORK_REQUEST_FLAGS_RW_IS_RO || WORK_REQUEST_FLAGS_RW_IS_WO)) != 0x00) {
    #if DEBUG_DISPLAY >= 1
      fprintf(stderr, "OffloadAPI :: WARNING :: sendWorkRequest_list() call made with WORK_REQUEST_FLAGS_RW_TO_RO and/or WORK_REQUEST_FLAGS_RW_IS_WO flags set... ignoring...\n");
    #endif
    flags &= (~(WORK_REQUEST_FLAGS_RW_IS_RO || WORK_REQUEST_FLAGS_RW_IS_WO));
  }

  // Ensure that there is at least one free WRHandle structure
  if (wrFreeHead == NULL) {  // Out of free work request structures
    // Add some more WRHandle structures to the wrFree list
    wrFreeHead = createWRHandles(WRHANDLES_GROW_SIZE);
    wrFreeTail = &(wrFreeHead[WRHANDLES_GROW_SIZE - 1]);
  }

  // Grab the first free WRHandle structure and use it for this entry
  WorkRequest *wrEntry = wrFreeHead;
  wrFreeHead = wrFreeHead->next;
  if (wrFreeHead == NULL) wrFreeTail = NULL;
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
  wrEntry->next = NULL;

  // Try to send the message
  // NOTE: Through the use of speSendStartIndex, if the SPE's message queues aren't being overloaded, then
  //   this loop should only iterate once and then break.
  // TODO : Update this so the number of outstanding work requests that have been sent to each SPE Thread
  //   is used to pick the SPE Thread to send this request to (i.e. - Send to the thread with the SPE Thread
  //   with the least full message queue.)  For now, the speSendStartIndex heuristic should help.
  processingSPEIndex = -1;
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // Check the affinity flag (if the bit for this SPE is not set, skip this SPE)
    if (((0x01 << i) & speAffinityMask) != 0x00) {

      register int actualSPEThreadIndex = (i + speSendStartIndex) % NUM_SPE_THREADS;
      sentIndex = sendSPEMessage(speThreads[actualSPEThreadIndex], wrEntry, SPE_MESSAGE_COMMAND_NONE);

      if (sentIndex >= 0) {
        speSendStartIndex = (actualSPEThreadIndex + 1) % NUM_SPE_THREADS;
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

    if (wrEntry->next != NULL) {
      printf(" ---- sendWorkRequest() ---- :: ERROR : Sent work request where wrEntry->next != NULL\n");
    }
  }

  // Return the WorkRequest pointer as the handle
  return wrEntry;
}


// Returns: Non-zero if finished, zero otherwise
// TODO : What should really happen... Once OffloadAPIProgress() detects that a work request has finished, it should
//   remove that work request from the message queue so another message can be sent to the SPE without requiring a
//   call to isFinished() first.  For now, just require a call to isFinished() to clean up the message queue entry
//   to get this working.
extern "C"
int isFinished(WRHandle wrHandle) {

  int rtnCode = 0;  // default to "not finished"

  // Tell the PPU's portion of the spert to make progress
  OffloadAPIProgress();

  // Check to see if the work request has finished
  if (wrHandle != INVALID_WRHandle && wrHandle->state == WORK_REQUEST_STATE_FINISHED) {

    // Clear the entry
    wrHandle->speIndex = -1;
    wrHandle->entryIndex = -1;
    wrHandle->funcIndex = -1;
    wrHandle->readWritePtr = NULL;
    wrHandle->readWriteLen = 0;
    wrHandle->readOnlyPtr = NULL;
    wrHandle->readOnlyLen = 0;
    wrHandle->writeOnlyPtr = NULL;
    wrHandle->writeOnlyLen = 0;
    wrHandle->flags = WORK_REQUEST_FLAGS_NONE;
    wrHandle->callbackFunc = NULL;
    wrHandle->next = NULL;

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

/*
  // Otherwise, if execution reaches here


  // Search the wrFinished list for the handle (remove it if it is there)
  WorkRequest *wrEntry = wrFinishedHead;
  WorkRequest *wrEntryPrev = NULL;
  while (wrEntry != NULL) {
    if (wrEntry == wrHandle) {

      // Remove the entry, clear it, and place it back in the wrFree list
      if (wrEntryPrev == NULL) {  // was the head
        wrFinishedHead = wrEntry->next;
        if (wrFinishedHead == NULL) wrFinishedTail = NULL;
      } else {
        wrEntryPrev->next = wrEntry->next;
        if (wrEntryPrev->next == NULL) wrFinishedTail = wrEntryPrev;
      }
      
      // Clear fields as needed
      wrEntry->speIndex = -1;
      wrEntry->entryIndex = -1;
      wrEntry->funcIndex = -1;
      wrEntry->readWritePtr = NULL;
      wrEntry->readWriteLen = 0;
      wrEntry->readOnlyPtr = NULL;
      wrEntry->readOnlyLen = 0;
      wrEntry->writeOnlyPtr = NULL;
      wrEntry->writeOnlyLen = 0;
      wrEntry->flags = WORK_REQUEST_FLAGS_NONE;
      wrEntry->userData = NULL;
      wrEntry->callbackFunc = NULL;
      wrEntry->next = NULL;

      // Add this entry to the free list now that is done being used
      if (wrFreeTail == NULL) {
        wrFreeTail = wrFreeHead = wrEntry;
      } else {
        wrFreeTail->next = wrEntry;
        wrFreeTail = wrEntry;
      }

      // Found it, so just return non-zero ("finished")
      return -1;
    }

    // Try the next entry
    wrEntryPrev = wrEntry;
    wrEntry = wrEntry->next;
  }

  return 0;  // Not finished
}
*/



// TODO : It would be nice to change this from a busy wait to something that busy waits for a
//   short time and then gives up and blocks (Maybe get some OS stuff out of the way while we
//   are just waiting anyway).
// NOTE : This function only blocks if a callbackFunc is not specified.
extern "C"
void waitForWRHandle(WRHandle wrHandle) {

  // Verify the WRHandle
  //int speIndex = wrHandle->speIndex;
  //int msgIndex = wrHandle->entryIndex;
  //if (speIndex < 0 || speIndex >= NUM_SPE_THREADS) return;
  //if (msgIndex < 0 || msgIndex >= SPE_MESSAGE_QUEUE_LENGTH) return;

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
void OffloadAPIProgress() {


  // DEBUG - Mailbox Statistics
  #define OffloadAPIProgress_statFreq  0
  #if OffloadAPIProgress_statFreq > 0
    static int statCount = OffloadAPIProgress_statFreq;
    int statCount_flag = 0;
    static int statSum_all[8] = { 0 };
    static int statSum_all_count[8] = { 0 };
    static int statSum_nonZero[8] = { 0 };
    static int statSum_nonZero_count[8] = { 0 };
  #endif


  // Check the mailbox from the SPEs to see if any of the messages have finished (and mark them as such)
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // Get the number of entries in the mailbox from the SPE and then read each entry
    int usedEntries = spe_stat_out_mbox(speThreads[i]->speID);

    // DEBUG - Mailbox Statistics
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

      if ((WorkRequest*)(msg->wrPtr) != NULL && ((WorkRequest*)(msg->wrPtr))->next != NULL) {
        fprintf(stderr, " --- Offload API :: ERROR :: WorkRequest finished while still linked (msg->wrPtr->next should be NULL) !!!!!!!\n");

        // Kill self out of shame
        exit(EXIT_FAILURE);
      }

      if (msg->state != SPE_MESSAGE_STATE_SENT) {

        // Warn the user that something bad has just happened
        fprintf(stderr, " --- OffloadAPI :: ERROR :: Invalid message queue index (%d) received from SPE %d...\n", speMessageQueueIndex, i);
        msg->state = SPE_MESSAGE_STATE_CLEAR;

        // Kill self out of shame
        exit(EXIT_FAILURE);

      } else {

	// If there was an error returned by the SPE, display it now
        if (speMessageErrorCode != SPE_MESSAGE_OK) {
          fprintf(stderr, " --- Offload API :: ERROR :: SPE %d returned error code %d for message at index %d...\n",
                  i, speMessageErrorCode, speMessageQueueIndex
                 );
	}

        // TODO : Do any gathering of data on SPE load-balance here (before clearing speIndex and entryIndex)

        // If there is a callback function, call it for this entry and then place the entry back into the wrFree list
        WorkRequest *wrPtr = (WorkRequest*)(msg->wrPtr);
        if (wrPtr != NULL) {

          if (callbackFunc != NULL || wrPtr->callbackFunc != NULL) {

            // Call the callback function
            if (wrPtr->callbackFunc != NULL)
              (wrPtr->callbackFunc)((void*)wrPtr->userData);  // call work request specific callback function
            else
              callbackFunc((void*)(wrPtr->userData));         // call default work request callback function

            // Clear the fields of the work request as needed
            wrPtr->speIndex = -1;
            wrPtr->entryIndex = -1;
            wrPtr->funcIndex = -1;
            wrPtr->readWritePtr = NULL;
            wrPtr->readWriteLen = 0;
            wrPtr->readOnlyPtr = NULL;
            wrPtr->readOnlyLen = 0;
            wrPtr->writeOnlyPtr = NULL;
            wrPtr->writeOnlyLen = 0;
            wrPtr->flags = WORK_REQUEST_FLAGS_NONE;
            wrPtr->userData = NULL;
            wrPtr->callbackFunc = NULL;
            wrPtr->next = NULL;

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

            // TODO : Do any gathering of data on SPE load-balance here (before clearing speIndex and entryIndex)

            // Clear the fields of the work request as needed
            wrPtr->speIndex = -1;
            wrPtr->entryIndex = -1;

            // Mark the work request as finished
            wrPtr->state = WORK_REQUEST_STATE_FINISHED;

            //// Add this work request to the end of the wrFinished list
            //if (wrFinishedTail == NULL) {
            //  wrFinishedTail = wrFinishedHead = wrPtr;
            //} else {
            //  wrFinishedTail->next = wrPtr;
            //  wrFinishedTail = wrPtr;
	    //}
	  }
	}

        // Now that the work request has been moved to either the wrFree list or the wrFinished
        //   list, set the state of the message queue entry to clear so it can accempt more work.
        msg->state = SPE_MESSAGE_STATE_CLEAR;
      }

      usedEntries--;
    } // end while (usedEntries > 0)
  } // end for (all SPEs)


  // DEBUG - Mailbox Statistics
  #if OffloadAPIProgress_statFreq > 0
    #if 0
      if (statCount_flag > 0)  // For print frequency, only count calls that find at least one mailbox entry
        statCount--;
      if (statCount <= 0) {
        printf("PPE :: OffloadAPIProgress() - Mailbox Statistics...\n");
        for (int i = 0; i < NUM_SPE_THREADS; i++) {
          printf("PPE :: OffloadAPIProgress() -   SPE %d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
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
          printf("PPE :: OffloadAPIProgress() - SPE %d Mailbox Stats - all:%.6f(%d), non-zero:%.2f(%d)...\n",
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

    // Try each SPE
    register unsigned int speAffinityMask = wrEntry->speAffinityMask;
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

    // Move into the next wrQueued entry
    wrEntryPrev = wrEntry;
    wrEntry = wrEntry->next;
  }

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Private Function Bodies



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
      fprintf(stderr, "OffloadAPI :: ERROR : createSPEThread() : error code 1.1\n");
      free_aligned(speData);
      return NULL;
    }
    memset((void*)speData->messageQueue, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
    speData->messageQueueLength = SPE_MESSAGE_QUEUE_LENGTH;

    // Give this SPE a unique number
    speData->vID = vIDCounter;
    vIDCounter++;

    // Set flag indicating that this function malloc'ed the speData structure
    speDataCreated = 1;
  }

  // Create the thread
  speid_t speID = spe_create_thread((spe_gid_t)NULL,       // spe_gid_t (actually a void*) - The SPE Group ID for the Thread
                                    (spe_program_handle_t*)(&spert_main),    // spe_program_handle_t* - Pointer to SPE's function that should be executed (name in embedded object file, not function name from code)
                                    speData,                 // void* - Argument List
                                    (void*)NULL,             // void* - Evironment List
                                    (unsigned long)-1,       // unsigned long - Affinity Mask
                                    (int)0                   // int - Flags
				   );

  // Verify that the thread was actually created
  if (speID == NULL) {

    fprintf(stderr, "OffloadAPI :: ERROR : createSPEThread() : error code 2.0 --- spe_create_thread() returned %d...\n", speID);

    // Clean up the speData structure if this function created it
    if (speDataCreated != 0 && speData != NULL) {
      if ((void*)(speData->messageQueue) != NULL) free_aligned((void*)(speData->messageQueue));
      free_aligned(speData);
    }

    // Return failure
    return NULL;
  }


  // Wait for the thread to check in (with pointer to its message queue in its local store)
  while (spe_stat_out_mbox(speID) <= 0);
  unsigned int speMessageQueuePtr = spe_read_out_mbox(speID);
  if (speMessageQueuePtr == 0) {
    fprintf(stderr, "OffloadAPI :: ERROR : SPE checked in with NULL value for Message Queue\n");
    exit(EXIT_FAILURE);
  }

  // Wait for the thread to report it's _end
  #if SPE_REPORT_END != 0
    while (spe_stat_out_mbox(speID) <= 0);
    unsigned int endValue = spe_read_out_mbox(speID);
    printf("SPE reported _end = 0x%08x\n", endValue);
  #endif

  #if DEBUG_DISPLAY >= 1
    printf("---> SPE Checked in with 0x%08X\n", speMessageQueuePtr);
  #endif

  // Create the SPEThread structure to be returned
  SPEThread *rtn = new SPEThread;
  rtn->speData = speData;
  rtn->speID = speID;
  rtn->messageQueuePtr = speMessageQueuePtr;
  rtn->msgIndex = 0;
  rtn->counter = 0;

  return rtn;
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
      fprintf(stderr, "OffloadAPI :: ERROR : createSPEThreads() : error code 1.1\n");
      exit(EXIT_FAILURE);
    }
    memset((void*)speThreads[i]->speData->messageQueue, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);
    speThreads[i]->speData->messageQueueLength = SPE_MESSAGE_QUEUE_LENGTH;

    // Give the SPE a unique id
    speThreads[i]->speData->vID = vIDCounter;
    vIDCounter++;
  }

  // Create all the threads at once
  for (int i = 0; i < numThreads; i++) {
    speid_t speID = spe_create_thread((spe_gid_t)NULL,         // spe_gid_t (actually a void*) - The SPE Group ID for the Thread
                                      (spe_program_handle_t*)(&spert_main),    // spe_program_handle_t* - Pointer to SPE's function that should be executed (name in embedded object file, not function name from code)
                                      (void*)speThreads[i]->speData, // void* - Argument List
                                      (void*)NULL,             // void* - Evironment List
                                      (unsigned long)-1,       // unsigned long - Affinity Mask
                                      (int)0                   // int - Flags
                                     );

    // Verify that the thread was actually created
    if (speID == NULL) {
      fprintf(stderr, "OffloadAPI :: ERROR : createSPEThreads() : error code 2.0 --- spe_create_thread() returned %d...\n", speID);
      exit(EXIT_FAILURE);
    }

    // Store the speID
    speThreads[i]->speID = speID;
  }

  // Wait for all the threads to check in (with pointer to its message queue in its local store)
  for (int i = 0; i < numThreads; i++) {

    while (spe_stat_out_mbox(speThreads[i]->speID) <= 0);
    unsigned int speMessageQueuePtr = spe_read_out_mbox(speThreads[i]->speID);
    if (speMessageQueuePtr == 0) {
      fprintf(stderr, "OffloadAPI :: ERROR : SPE checked in with NULL value for Message Queue\n");
      exit(EXIT_FAILURE);
    }
    speThreads[i]->messageQueuePtr = speMessageQueuePtr;

    #if DEBUG_DISPLAY >= 1
      printf("---> SPE Checked in with 0x%08X\n", speMessageQueuePtr);
    #endif

    // Wait for the thread to report it's _end
    #if SPE_REPORT_END != 0
      while (spe_stat_out_mbox(speThreads[i]->speID) <= 0);
      unsigned int endValue = spe_read_out_mbox(speThreads[i]->speID);
      printf("SPE %d reported _end = 0x%08x\n", i, endValue);
    #endif
  }

  // Finish filling in the speThreads array
  for (int i = 0; i < numThreads; i++) {
    speThreads[i]->msgIndex = 0;
    speThreads[i]->counter = 0;
  }

  return speThreads;
}


int sendSPEMessage(SPEThread *speThread, WorkRequest *wrPtr, int command) {

  // Verify the parameters
  if (speThread == NULL) return -1;

  if (wrPtr != NULL && wrPtr->funcIndex < 0 && command == SPE_MESSAGE_COMMAND_NONE)
    return -1;

  // Force the command value to a valid command
  if (command < SPE_MESSAGE_COMMAND_MIN || command > SPE_MESSAGE_COMMAND_MAX)
    command = SPE_MESSAGE_COMMAND_NONE;

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

  // Find the next available index
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    int index = (speThread->msgIndex + i) % SPE_MESSAGE_QUEUE_LENGTH;
    volatile SPEMessage* msg = (SPEMessage*)(((char*)speThread->speData->messageQueue) + (index * SIZEOF_16(SPEMessage)));
    
    if (msg->state == SPE_MESSAGE_STATE_CLEAR) {

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
      } else {
        msg->funcIndex = wrPtr->funcIndex;
        msg->readWritePtr = (PPU_POINTER_TYPE)(wrPtr->readWritePtr);
        msg->readWriteLen = wrPtr->readWriteLen;
        msg->readOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->readOnlyPtr);
        msg->readOnlyLen = wrPtr->readOnlyLen;
        msg->writeOnlyPtr = (PPU_POINTER_TYPE)(wrPtr->writeOnlyPtr);
        msg->writeOnlyLen = wrPtr->writeOnlyLen;
        msg->flags = wrPtr->flags;

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
      }
      msg->state = SPE_MESSAGE_STATE_SENT;
      msg->command = command;
      msg->wrPtr = (PPU_POINTER_TYPE)wrPtr;

      // NOTE: Important that the counter be the last then set (the change in this value is what prompts the
      //   SPE to consider the entry a new entry... even if the state has been set to SENT).
      // NOTE: Only change the value of msg->counter once so the SPE is not confused (i.e. - don't increment
      //   and then check msg->counter direclty).
      int tmp = msg->counter;
      tmp++;
      if (tmp > 255) tmp = 0;
      msg->counter = tmp;

      // DEBUG
      #if DEBUG_DISPLAY >= 3
        printf("  sendSPEMessage() : sending message to queue @ %p, slot %d (@ %p; SIZEOF_16(SPEMessage) = %d)...\n",
               speThread->speData->messageQueue,
               index,
               msg,
               SIZEOF_16(SPEMessage)
	      );
        printf("  sendSPEMessage() : message[%d] = {", index);
        printf(" funcIndex = %d", msg->funcIndex);
        printf(", readWritePtr = %u", msg->readWritePtr);
        printf(", readWriteLen = %d", msg->readWriteLen);
        printf(", readOnlyPtr = %u", msg->readOnlyPtr);
        printf(", readOnlyLen = %d", msg->readOnlyLen);
        printf(", writeOnlyPtr = %u", msg->writeOnlyPtr);
        printf(", writeOnlyLen = %d", msg->writeOnlyLen);
        printf(", flags = 0x08x", msg->flags);
        printf(", wrPtr = %u", msg->wrPtr);
        printf(", state = %d", msg->state);
        printf(", command = %d", msg->command);
        printf(", counter = %d", msg->counter);
        printf(" }\n");
      #endif

      speThread->msgIndex = (index + 1) % SPE_MESSAGE_QUEUE_LENGTH;

      return index;
    } // end if (msg->state == SPE_MESSAGE_STATE_CLEAR)
  } // end for (loop through message queue entries)

  // If execution reaches here, an available slot in the message queue was not found
  return -1;
}


// Returns 0 on success, non-zero otherwise
int sendSPECommand(SPEThread *speThread, int command) {
  if (command < SPE_MESSAGE_COMMAND_MIN || command > SPE_MESSAGE_COMMAND_MAX) return -1;
  while (spe_stat_in_mbox(speThread->speID) == 0);      // Loop while mailbox is full
  return spe_write_in_mbox(speThread->speID, command);
}


WorkRequest* createWRHandles(int numHandles) {

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
