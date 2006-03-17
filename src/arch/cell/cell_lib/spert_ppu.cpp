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

#define CREATE_WRHandle(speIndex, msgIndex)  ((WRHandle)((speIndex << 16) | (msgIndex & 0xFFFF)))
#define WRHandle_SPE(h)                      ((int)((h >> 16) & 0xFFFF))
#define WRHandle_MSG(h)                      ((int)(h & 0xFFFF))

#define WRHANDLES_NUM_INITIAL  (64)
#define WRHANDLES_GROW_SIZE    (16)


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

WorkRequest *wrQueuedHead = NULL;
WorkRequest *wrQueuedTail = NULL;

WorkRequest *wrFinishedHead = NULL;
WorkRequest *wrFinishedTail = NULL;

WorkRequest *wrFreeHead = NULL;
WorkRequest *wrFreeTail = NULL;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

SPEThread* createSPEThread(SPEData *speData);
SPEThread** createSPEThreads(SPEThread **speThreads, int numThreads);

int sendSPEMessage(SPEThread* speThread, int funcIndex, void* data, int dataLen, void* msg, int msgLen, WorkRequest* wrPtr);
int sendSPEMessage(SPEThread* speThread, int command);
int sendSPEMessage(SPEThread* speThread, int funcIndex, void* data, int dataLen, void* msg, int msgLen, WorkRequest* wrPtr, int command);

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
    //// Create the elements of the speThreads array (createSPEThreads assumes that the threads have already been created)
    //for (int i = 0; i < NUM_SPE_THREADS; i++)
    //  speThreads[i] = new SPEThread;

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
    sendSPEMessage(speThreads[i], SPE_MESSAGE_COMMAND_EXIT);

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
    //#if CREATE_EACH_THREAD_ONE_BY_ONE
      free_aligned((void*)(speThreads[i]->speData));
    //#else
    //  if (i == 0) free_aligned((void*)(speThreads[0]->speData));
    //#endif
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

}



extern "C"
WRHandle sendWorkRequest(int funcIndex, void* dataPtr, int dataLen, void* msgPtr, int msgLen, void* userData) {

  // NOTE: This is used in an attempt to more evenly distribute the workload amongst all the SPE Threads
  // Also NOTE: Since NUM_SPE_THREADS should be 8, the "% NUM_SPE_THREADS" should be transformed into masks/shifts
  //   by strength reduction in the compiler when optimizations are turned on... for debuging purposes, leave them
  //   as "% NUM_SPE_THREADS" in the code (if NUM_SPE_THREADS is reduced to save startup time in the simulator).
  static int speIndex = 0;

  int processingSPEIndex = -1;
  int sentIndex = -1;

  // Tell the PPU's portion of the spert to make progress
  OffloadAPIProgress();

  //printf("sendWorkRequest() - DEBUG 1.0...\n");

  // Verify the parameters
  if (funcIndex < 0) return INVALID_WRHandle;
  if (dataPtr != NULL && dataLen <= 0) return INVALID_WRHandle;
  if (msgPtr != NULL && msgLen <= 0) return INVALID_WRHandle;

  //printf("sendWorkRequest() - DEBUG 2.0...\n");

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

  // Fill in the WRHandle structure
  wrEntry->speIndex = -1;
  wrEntry->entryIndex = -1;
  wrEntry->funcIndex = funcIndex;
  wrEntry->data = dataPtr;
  wrEntry->dataLen = dataLen;
  wrEntry->msg = msgPtr;
  wrEntry->msgLen = msgLen;
  wrEntry->userData = userData;
  wrEntry->next = NULL;

  // Try to send the message
  // NOTE: Through the use of speIndex, if the SPE's message queues aren't being overloaded, then
  //   this loop should only iterate once and then break.
  // TODO : Update this so the number of outstanding work requests that have been sent to each SPE Thread
  //   is used to pick the SPE Thread to send this request to (i.e. - Send to the thread will the SPE Thread
  //   with the least full message queue.)  For now, the speIndex heuristic should help.
  processingSPEIndex = -1;
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    int actualSPEThreadIndex = (i + speIndex) % NUM_SPE_THREADS;

    //printf("sendWorkRequest() - DEBUG 2.1 - Calling sendSPEMessage() on SPE Thread %d...\n", actualSPEThreadIndex);

    sentIndex = sendSPEMessage(speThreads[actualSPEThreadIndex],
                               funcIndex, dataPtr, dataLen, msgPtr, msgLen, wrEntry,
                               SPE_MESSAGE_COMMAND_NONE
                              );
    if (sentIndex >= 0) {
      speIndex = (actualSPEThreadIndex + 1) % NUM_SPE_THREADS;
      processingSPEIndex = actualSPEThreadIndex;
      break;
    }
  }

  //printf("sendWorkRequest() - DEBUG 3.0... processingSPEIndex = %d, sentIndex = %d\n", processingSPEIndex, sentIndex);

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

  // DEBUG
  //printf("sendWorkRequest() - DEBUG 4.0... processingSPEIndex = %d, sentIndex = %d\n", processingSPEIndex, sentIndex);
  //fflush(NULL);

  // For now, return the index in the message queue as the work request handle
  return wrEntry; //CREATE_WRHandle(processingSPEIndex, sentIndex);
}


// Returns: Non-zero if finished, zero otherwise
// TODO : What should really happen... Once OffloadAPIProgress() detects that a work request has finished, it should
//   remove that work request from the message queue so another message can be sent to the SPE without requiring a
//   call to isFinished() first.  For now, just require a call to isFinished() to clean up the message queue entry
//   to get this working.
extern "C"
int isFinished(WRHandle wrHandle) {

  // Tell the PPU's portion of the spert to make progress
  OffloadAPIProgress();

  //// Verify the WRHandle
  //int speIndex = WRHandle_SPE(wrHandle);
  //int msgIndex = WRHandle_MSG(wrHandle);

  //printf("isFinished() - DEBUG 1.0 - (%d) wrHandle = %u, speIndex = %d, msgIndex = %d\n", rand() % 10, wrHandle, speIndex, msgIndex);

  //if (speIndex < 0 || speIndex >= NUM_SPE_THREADS) return 0;
  //if (msgIndex < 0 || msgIndex >= SPE_MESSAGE_QUEUE_LENGTH) return 0;

  // Check if the status of the message entry is FINISHED
  //SPEMessage* msg = (SPEMessage*)(((char*)(speThreads[speIndex]->speData->messageQueue)) + (SIZEOF_16(SPEMessage) * msgIndex));
  //if (msg->state == SPE_MESSAGE_STATE_FINISHED || msg->state == SPE_MESSAGE_STATE_CLEAR) {

    // Reset the state to CLEAR (since the caller requested the information and it is now finished)
    //msg->state = SPE_MESSAGE_STATE_CLEAR;

    // Return non-zero
    //return -1;
  //}

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
      wrEntry->data = NULL;
      wrEntry->dataLen = 0;
      wrEntry->msg = NULL;
      wrEntry->msgLen = 0;
      wrEntry->userData = NULL;
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


// TODO : It would be nice to change this from a busy wait to something that busy waits for a
//   short time and then gives up and blocks (Maybe get some OS stuff out of the way while we
//   are just waiting anyway).
// NOTE : This function only blocks if a callbackFunc is not specified.
extern "C"
void waitForWRHandle(WRHandle wrHandle) {

  // Verify the WRHandle
  int speIndex = wrHandle->speIndex; //WRHandle_SPE(wrHandle);
  int msgIndex = wrHandle->entryIndex; //WRHandle_MSG(wrHandle);

  //printf("WaitForWRHandle() - DEBUG 1.0 - (%d) wrHandle = 0x%08x, speIndex = %d, msgIndex = %d\n", rand() % 10, wrHandle, speIndex, msgIndex);

  if (speIndex < 0 || speIndex >= NUM_SPE_THREADS) return;
  if (msgIndex < 0 || msgIndex >= SPE_MESSAGE_QUEUE_LENGTH) return;

  // Wait for the handle to finish
  while (callbackFunc == NULL && wrHandle != INVALID_WRHandle && !isFinished(wrHandle)) OffloadAPIProgress();
}


extern "C"
void OffloadAPIProgress() {

  // Check the mailbox from the SPEs to see if any of the messages have finished (and mark them as such)
  for (int i = 0; i < NUM_SPE_THREADS; i++) {

    // Get the number of entries in the mailbox from the SPE
    int usedEntries = /* NUM_SPE_OUT_MAILBOX_ENTRIES - */ spe_stat_out_mbox(speThreads[i]->speID);
    while (usedEntries > 0) {

      // Read the message queue index that was sent by the SPE from the outbound mailbox
      unsigned int speMessageQueueIndex = spe_read_out_mbox(speThreads[i]->speID);

      //printf("--==>> PPE Received Message Reply from SPE %d (queue index: %d) <<==--\n", i, speMessageQueueIndex);

      SPEMessage *msg = (SPEMessage*)((char*)(speThreads[i]->speData->messageQueue) + (speMessageQueueIndex * SIZEOF_16(SPEMessage)));

      if ((WorkRequest*)(msg->wrPtr) != NULL && ((WorkRequest*)(msg->wrPtr))->next != NULL) {
        fprintf(stderr, " --- Offload API :: ERROR :: WorkRequest Finished while still Linked (msg->wrPtr->next should be NULL) !!!!!!!\n");
      }

      if (msg->state != SPE_MESSAGE_STATE_SENT) {

        // Warn the user that something bad has just happened
        fprintf(stderr, " --- OffloadAPI :: ERROR :: Invalid message queue index received from SPE %d...\n", i);
        msg->state = SPE_MESSAGE_STATE_CLEAR;

        // Kill self out of shame
        exit(EXIT_FAILURE);

      } else {

        //// If there is a callback function, call it for this entry
        //if (callbackFunc != NULL)
        //  callbackFunc((void*)(msg->userData));
        //
        //msg->state = SPE_MESSAGE_STATE_FINISHED;

        // If there is a callback function, call it for this entry and then place the entry back into the wrFree list
        WorkRequest *wrPtr = (WorkRequest*)(msg->wrPtr);
        if (wrPtr != NULL) {

          if (callbackFunc != NULL) {

            // Call the callback function
            callbackFunc((void*)(wrPtr->userData));

            // Clear the fields of the work request as needed
            wrPtr->speIndex = -1;
            wrPtr->entryIndex = -1;
            wrPtr->funcIndex = -1;
            wrPtr->data = NULL;
            wrPtr->dataLen = 0;
            wrPtr->msg = NULL;
            wrPtr->msgLen = 0;
            wrPtr->userData = NULL;
            wrPtr->next = NULL;

            // Add this entry to the end of the wrFree list
            if (wrFreeTail == NULL) {
              wrFreeTail = wrFreeHead = wrPtr;
            } else {
              wrFreeTail->next = wrPtr;
              wrFreeTail = wrPtr;
            }

          // Otherwise, just place the WorkRequest into the wrFinished list
	  } else {

            // TODO : Do any gathering of data on SPE load-balance here (before clearing speIndex and entryIndex)

            //printf(" --- OffloadAPI :: Adding WRHandle (%p) to wrFinished list...\n", wrPtr);

            // Clear the fields of the work request as needed
            wrPtr->speIndex = -1;
            wrPtr->entryIndex = -1;

            // Add this work request to the end of the wrFinished list
            if (wrFinishedTail == NULL) {
              wrFinishedTail = wrFinishedHead = wrPtr;
            } else {
              wrFinishedTail->next = wrPtr;
              wrFinishedTail = wrPtr;
	    }
	  }
	}

        // Now that the work request has been moved to either the wrFree list or the wrFinished
        //   list, set the state of the message queue entry to clear so it can accempt more work.
        msg->state = SPE_MESSAGE_STATE_CLEAR;
      }

      usedEntries--;
    } // end while (usedEntries > 0)
  } // end for (all SPEs)


  // Loop through the wrQueued list and try to send outstanding messages
  int sentIndex = -1;
  int processingSPEIndex = -1;
  WorkRequest *wrEntry = wrQueuedHead;
  WorkRequest *wrEntryPrev = NULL;
  while (wrEntry != NULL) {

    // Try each SPE
    for (int i = 0; i < NUM_SPE_THREADS; i++) {
      sentIndex = sendSPEMessage(speThreads[i],
                                 wrEntry->funcIndex,
                                 wrEntry->data, wrEntry->dataLen,
                                 wrEntry->msg, wrEntry->msgLen,
                                 wrEntry,
                                 SPE_MESSAGE_COMMAND_NONE
                                );
      if (sentIndex >= 0) {
        processingSPEIndex = i;
        break;
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

    // Otherwise, there was a work request but no empty slots on any of the SPEs (no need to keep trying)
    } else {
      break;
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
  }

  // Finishe filling in the speThreads array
  for (int i = 0; i < numThreads; i++) {
    speThreads[i]->msgIndex = 0;
    speThreads[i]->counter = 0;
  }

  return speThreads;
}


int sendSPEMessage(SPEThread* speThread, int funcIndex, void* data, int dataLen, void* msg, WorkRequest* wrPtr, int msgLen) {
  return sendSPEMessage(speThread, funcIndex, data, dataLen, msg, msgLen, wrPtr, SPE_MESSAGE_COMMAND_NONE);
}

int sendSPEMessage(SPEThread* speThread, int command) {
  return sendSPEMessage(speThread, 0, NULL, 0, NULL, 0, NULL, command);
}

// Returns the index in the message queue that was used (or -1 on failure)
int sendSPEMessage(SPEThread *speThread,
                   int funcIndex,
                   void *data,
                   int dataLen,
                   void *message,
                   int messageLen,
                   WorkRequest *wrPtr,
                   int command
                  ) {

  //printf("  sendSPEMessage() - Called...\n");

  // Verify the parameters
  if (speThread == NULL) return -1;

  //printf("  sendSPEMessage() - DEBUG 1.0...\n");

  if (funcIndex < 0 && command == SPE_MESSAGE_COMMAND_NONE)
    return -1;

  if (command < SPE_MESSAGE_COMMAND_MIN || command > SPE_MESSAGE_COMMAND_MAX)
    command = SPE_MESSAGE_COMMAND_NONE;

  if (funcIndex < 0 && data == NULL && message == NULL && command == SPE_MESSAGE_COMMAND_NONE)
    return -1;

  //printf("  sendSPEMessage() - DEBUG 2.0...\n");

  if (data == NULL) dataLen = 0;
  if (dataLen < 0) { data = NULL; dataLen = 0; }

  if (message == NULL) messageLen = 0;
  if (messageLen < 0) { message = NULL; messageLen = 0; }


  // Find the next available index
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

    int index = (speThread->msgIndex + i) % SPE_MESSAGE_QUEUE_LENGTH;
    volatile SPEMessage* msg = (SPEMessage*)(((char*)speThread->speData->messageQueue) + (index * SIZEOF_16(SPEMessage)));
    
    //printf("  sendSPEMessage() - DEBUG 2.1...  (speThread->msgIndex = %d, i = %d, index = %d)...\n", speThread->msgIndex, i, index);

    if (msg->state == SPE_MESSAGE_STATE_CLEAR) {

      //printf("  sendSPEMessage() - DEBUG 2.1.0 - state = CLEAR...\n");

      msg->funcIndex = funcIndex;
      msg->data = (PPU_POINTER_TYPE)data;
      msg->dataLen = dataLen;
      msg->msg = (PPU_POINTER_TYPE)message;
      msg->msgLen = messageLen;
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
        printf(", data = %u", msg->data);
        printf(", dataLen = %d", msg->dataLen);
        printf(", msg = %u", msg->msg);
        printf(", msgLen = %d", msg->msgLen);
        printf(", wrPtr = %u", msg->wrPtr);
        printf(", state = %d", msg->state);
        printf(", command = %d", msg->command);
        printf(", counter = %d", msg->counter);
        printf(" }\n");
      #endif

      speThread->msgIndex = (index + 1) % SPE_MESSAGE_QUEUE_LENGTH;

      //printf("  sendSPEMessage() - DEBUG 2.2... (index = %d)...\n", index);

      return index;
    } // end if (msg->state == SPE_MESSAGE_STATE_CLEAR)

  } // end for (loop through message queue entries)

  //printf("  sendSPEMessage() - DEBUG 3.0...\n");

  // If execution reaches here, an available slot in the message queue was not found
  return -1;
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
