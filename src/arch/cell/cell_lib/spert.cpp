#ifdef __cplusplus
extern "C" {
#endif
  #include <stdlib.h>
  #include <unistd.h>
  #include <stdio.h>
  #include <malloc_align.h>
  #include <free_align.h>
  #include <sim_printf.h>
#ifdef __cplusplus
}
#endif

#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <cbe_mfc.h>


#if SPE_USE_OWN_MEMSET == 0
  #include <string.h>
#endif

#include "spert_common.h"
#include "spert.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

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

const int SPEData_dmaTransferSize = SIZEOF_16(SPEData);

volatile char* msgQueueRaw[SPE_MESSAGE_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
volatile SPEMessage* msgQueue[SPE_MESSAGE_QUEUE_LENGTH];

// NOTE: Allocate two per entry (two buffers are read in from memory, two are
//   written out to memory, read write does not overlap for a given message).
volatile DMAListEntry dmaListEntry[SPE_DMA_LIST_LENGTH * SPE_MESSAGE_QUEUE_LENGTH] __attribute__((aligned(16)));
int dmaListSize[SPE_MESSAGE_QUEUE_LENGTH];

// Location of the end of the 'data segment'
extern unsigned int _end;


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

void speScheduler(SPEData *speData, unsigned long long id);
void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str);
void debug_displayStateHistogram(unsigned long long id, int* msgState, char* str);

#if SPE_USE_OWN_MEMSET != 0
void memset(void* ptr, char val, int len);
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies

int main(unsigned long long id, unsigned long long param) {

  /*volatile*/ SPEData myData;
  void* _heapPtr = NULL;
  #if SPE_DEBUG_DISPLAY >= 1
    void* _breakBefore = NULL;
    void* _breakAfter = NULL;
  #endif

  // Tell the world this SPE is alive
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf(" --==>> Hello From SPE 0x%llx's Runtime <<==--\n", id);
  #endif

  // Call the user's funcLookup() function with funcIndex of SPE_FUNC_INDEX_INIT
  funcLookup(SPE_FUNC_INDEX_INIT, NULL, 0, NULL, 0, NULL, 0, NULL);

  // From Section 4.3 of library_SDK.pdf : "The local store memory heap is initialized the first
  //   time a memory heap allocation routine is called."... Do this now so it is ready to go.
  register unsigned int memLeft = SPE_TOTAL_MEMORY_SIZE - SPE_RESERVED_STACK_SIZE - (unsigned int)(&_end);
  memLeft -= 128;  // Buffer zone between stack and heap
  memLeft &= 0xFFFFFF00;  // Force it to be a multiple of 256 bytes
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] memLeft = %d\n", id, memLeft);
  #endif
  if (memLeft < SPE_MINIMUM_HEAP_SIZE) return -1;
  #if SPE_DEBUG_DISPLAy >= 1
    _breakBefore = sbrk(0);
  #endif
  _heapPtr = sbrk((ptrdiff_t)memLeft);
  #if SPE_DEBUG_DISPLAy >= 1
    _breakAfter = sbrk(0);
  #endif

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] :: _end = %p, _breakBefore = %p, _heapPtr = %p, _breakAfter = %p (%p)\n",
               id, &_end, _breakBefore, _heapPtr, _breakAfter, sbrk(0)
              );
    //sim_printf("[0x%llx] :: _end = %p, _heapPtr = %p\n",id, &_end, _heapPtr);
  #endif

  // Initialize globals
  memset(msgQueueRaw, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);

  #if SPE_STATS != 0
    memset(&statData, 0, sizeof(statData));
  #endif

  // Read in the data from main storage
  spu_mfcdma32((void*)&myData,          // LS Pointer
               (unsigned int)param,     // Main-Storage Pointer
               SPEData_dmaTransferSize, // Number of bytes to copy
               0,                       // Tag ID
               MFC_GET_CMD              // DMA Command
	      );

  // Wait for all transfers to complete.  See "SPU C/C++ Language Extentions", page 64 for details.
  spu_mfcstat(2);

  // Entry into the SPE's scheduler
  speScheduler(&myData, id);

  // Display stat data
  #if SPE_STATS != 0
    sim_printf("[0x%llx] :: SPE Stats\n", id);
    sim_printf("[0x%llx] ::   Total Number of Scheduler Loop Iterations: %llu\n", id, statData.schedulerLoopCount);
    sim_printf("[0x%llx] ::   Total Number of Work Requests Executed : %llu\n", id, statData.numWorkRequestsExecuted);
  #endif

  // Tell the world this SPE is going away
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf(" --==>> Goodbye From SPE 0x%llx's Runtime <<==--\n", id);
    sim_printf("  \"I do not regret the things I have done, but those I did not do.\" - Lucas, Empire Records\n");
  #endif

  // Call the user's funcLookup() function with funcIndex of SPE_FUNC_INDEX_CLOSE
  funcLookup(SPE_FUNC_INDEX_CLOSE, NULL, 0, NULL, 0, NULL, 0, NULL);

  return 0;
}


void speScheduler(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int fetchIndex = 0;
  int runIndex = 0;
  int getIndex = 0;
  int putIndex = 0;
  int commitIndex = 0;
  int cnt = 0;
  int tagStatus;
  unsigned int numDMAQueueEntries = 0;
  int i, j, iOffset;

  // DEBUG
  #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
    int stillAliveCounter = 0;
  #endif

  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] --==>> Starting SPE Scheduler ...\n", id);
  #endif

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Clear out the DMAListEntry array
  memset((void*)dmaListEntry, 0, sizeof(DMAListEntry) * 2 * SPE_MESSAGE_QUEUE_LENGTH);

  // Create the local message queue
  int msgState[SPE_MESSAGE_QUEUE_LENGTH];
  #if SPE_DEBUG_DISPLAY_NO_PROGRESS >= 1
    int msgLastStateCount = 0;
    int msgLastState[SPE_MESSAGE_QUEUE_LENGTH];
  #endif
  void* readWritePtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* readOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* writeOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* localMemPtr[SPE_MESSAGE_QUEUE_LENGTH];
  int msgCounter[SPE_MESSAGE_QUEUE_LENGTH];
  int errorCode[SPE_MESSAGE_QUEUE_LENGTH];
  DMAListEntry* dmaList[SPE_MESSAGE_QUEUE_LENGTH];
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

  // Do the intial read of the message queue from main memory
  spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);

  // DEBUG
  #if SPE_DEBUG_DISPLAY >= 1
    sim_printf("[0x%llx] :: starting scheduler loop...\n", id);
  #endif

  // The scheduler loop
  while (__builtin_expect(keepLooping != FALSE, 1)) {


    // Wait for the latest message queue read (blocking)
    mfc_write_tag_mask(0x80000000);   // enable only tag group 31 (message queue request)
    mfc_write_tag_update_any();
    tagStatus = mfc_read_tag_status();
    mfc_write_tag_mask(0x7FFFFFFF);   // enable all tag groups except 31


    // DEBUG - Let the user know that the SPE is still alive
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      if ((stillAliveCounter % SPE_DEBUG_DISPLAY_STILL_ALIVE) == 0 && stillAliveCounter != 0) {
        #if 1

          sim_printf("[0x%llx] :: still going... \n", id);

        #else

          for (int tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
            sim_printf("[0x%llx] :: still going... msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %d, rwl = %d, ro = %d, rol = %d, wo = %d, wol = %d, f = %x, s = %d(%d), cnt = %d, cmd = %d }\n",
                       id,
                       tmp,
                       &(msgQueue[tmp]),
                       msgQueue,
                       (volatile int)(msgQueue[tmp]->funcIndex),
                       msgQueue[tmp]->readWritePtr,
                       msgQueue[tmp]->readWriteLen,
                       msgQueue[tmp]->readOnlyPtr,
                       msgQueue[tmp]->readOnlyLen,
                       msgQueue[tmp]->writeOnlyPtr,
                       msgQueue[tmp]->writeOnlyLen,
                       msgQueue[tmp]->flags,
                       (volatile int)(msgQueue[tmp]->state),
                       msgState[tmp],
                       (volatile int)(msgQueue[tmp]->counter),
                       (volatile int)(msgQueue[tmp]->command)
                      );
	  }
          //sim_printf("[%llu] :: raw msgQueue = { ", id);
          //for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
          //  sim_printf("%d ", *(((char*)msgQueue) + ti));
          //}
          //sim_printf("}\n");
          //sim_printf("[%llu] :: raw msgQueueRaw = { ", id);
          //for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
          //  sim_printf("%d ", *(((char*)msgQueueRaw) + ti));
          //}
          //sim_printf("}\n");

        #endif
      }
    #endif


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(0)");
    #endif


    // Check for new messages
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {

      // Check for a new message in this slot
      if (__builtin_expect(msgQueue[i]->state == SPE_MESSAGE_STATE_SENT &&
                           msgState[i] == SPE_MESSAGE_STATE_CLEAR &&
                           msgCounter[i] != msgQueue[i]->counter,
                           0
                          )
         ) {

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

        msgCounter[i] = msgQueue[i]->counter;        
      }
    }


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(1)");
    #endif


    // Check for messages that need data fetched (list)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING_LIST, 0)) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(2)");
    #endif


    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING_LIST && ((tagStatus & (0x01 << i)) != 0), 0)) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(3)");
    #endif


    // Check for messages that need data fetched (standard)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_LIST_READY_LIST, 0)) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(4)");
    #endif


    // Check for messages that need data fetched (standard)
    numDMAQueueEntries = mfc_stat_cmd_queue();
    int newGetIndex = (getIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    int numGetsLeft = SPE_MAX_GET_PER_LOOP;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (getIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING, 0)) {

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
          if (__builtin_expect(memNeeded > ((unsigned int)0x40000 - ((unsigned int)(&_end))), 0)) {

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
            writeOnlyPtr[i] == NULL;
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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(5)");
    #endif


    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_FETCHING && ((tagStatus & (0x01 << i)) != 0), 0)) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(6)");
    #endif


    // Execute SPE_MAX_EXECUTE_PER_LOOP ready messages
    register unsigned int numExecLeft = SPE_MAX_EXECUTE_PER_LOOP;
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {

      if (__builtin_expect(msgState[runIndex] == SPE_MESSAGE_STATE_READY, 0)) {

        register volatile SPEMessage* msg = msgQueue[runIndex];

        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: >>>>> Entering User Code (index %d)...\n", id, runIndex);
	#endif

        // Execute the function specified
        if ((msg->flags & WORK_REQUEST_FLAGS_LIST) == 0x00) {

          #if SPE_DEBUG_DISPLAY >= 1
	    sim_printf("[0x%llx] :: Executing message queue entry as standard entry...\n", id);
          #endif

          funcLookup(msg->funcIndex,
                     readWritePtr[runIndex], msg->readWriteLen,
                     readOnlyPtr[runIndex], msg->readOnlyLen,
                     writeOnlyPtr[runIndex], msg->writeOnlyLen,
                     NULL
                    );
	} else {

          #if SPE_DEBUG_DISPLAY >= 1
	    sim_printf("[0x%llx] :: Executing message queue entry as list entry...\n", id);
          #endif

          funcLookup(msg->funcIndex,
                     NULL, msg->readWriteLen,
                     NULL, msg->readOnlyLen,
                     NULL, msg->writeOnlyLen,
                     (DMAListEntry*)(localMemPtr[runIndex])
                    );
	}

        #if SPE_DEBUG_DISPLAY >= 1
          sim_printf("[0x%llx] :: <<<<< Leaving User Code...\n", id);
	#endif

        #if SPE_STATS != 0
          statData.numWorkRequestsExecuted++;
        #endif

        // Update the state of the message queue entry
        if ((msgQueue[i]->flags & WORK_REQUEST_FLAGS_LIST) == 0x00)
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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(7)");
    #endif


    // Check for messages that have been executed but still need data committed to main memory
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED_LIST, 0)) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(8)");
    #endif


    // Check for messages that have been executed but still need data committed to main memory
    numDMAQueueEntries = mfc_stat_cmd_queue();
    int newPutIndex = (putIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    int numPutsLeft = SPE_MAX_PUT_PER_LOOP;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (putIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_EXECUTED, 0)) {

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
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(9)");
    #endif


    // Initiate the next message queue read from main memory
    spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);


    // Check for messages that are committed
    mfc_write_tag_mask(0x7FFFFFFF);
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    int commitIndexNext = (commitIndex + 1) % SPE_MESSAGE_QUEUE_LENGTH;
    for (iOffset = 0; __builtin_expect(iOffset < SPE_MESSAGE_QUEUE_LENGTH, 1); iOffset++) {

      register int i = (commitIndex + iOffset) % SPE_MESSAGE_QUEUE_LENGTH;

      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_COMMITTING && ((tagStatus * (0x01 << i)) != 0), 0)) {

        // Check to see if there is an available entry in the outbound mailbox
        if (spu_stat_out_mbox() > 0) {

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


    // DEBUG
    #if SPE_DEBUG_DISPLAY != 0
      debug_displayActiveMessageQueue(id, msgState, "(A)");
    #endif


    // Check for any messages that have entered into the ERROR state
    for (i = 0; __builtin_expect(i < SPE_MESSAGE_QUEUE_LENGTH, 1); i++) {
      if (__builtin_expect(msgState[i] == SPE_MESSAGE_STATE_ERROR, 0)) {

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

    // Update the stillAliveCounter
    #if SPE_DEBUG_DISPLAY_STILL_ALIVE >= 1
      stillAliveCounter++;
    #endif

    #if SPE_STATS != 0
      statData.schedulerLoopCount++;
    #endif

  } // end while (keepLooping)

}


void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str) {
  #if SPE_DEBUG_DISPLAY >= 1

  int tmp;

  for (tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
    if (msgState[tmp] != SPE_MESSAGE_STATE_CLEAR || msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) {
      sim_printf("[0x%llx] :: %s%s msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %d, rwl = %d, ro = %d, rol = %d, wo = %d, wol = %d, f = 0x%08X, tm = %u, s = %d(%d), cnt = %d, cmd = %d }\n",
                 id,
                 ((msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) ? ("---===!!! WARNING !!!===--- ") : ("")),
                 ((str == NULL) ? ("") : (str)),
                 tmp,
                 &(msgQueue[tmp]),
                 msgQueue,
                 (volatile)(msgQueue[tmp]->funcIndex),
                 (volatile)msgQueue[tmp]->readWritePtr,
                 (volatile)msgQueue[tmp]->readWriteLen,
                 (volatile)msgQueue[tmp]->readOnlyPtr,
                 (volatile)msgQueue[tmp]->readOnlyLen,
                 (volatile)msgQueue[tmp]->writeOnlyPtr,
                 (volatile)msgQueue[tmp]->writeOnlyLen,
                 (volatile)(msgQueue[tmp]->flags),
                 (volatile)(msgQueue[tmp]->totalMem),
                 (volatile int)(msgQueue[tmp]->state),
                 msgState[tmp],
                 (volatile int)(msgQueue[tmp]->counter),
                 (volatile int)(msgQueue[tmp]->command)
                );

      if (msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) {
        sim_printf("***************************************************************************************************\n");
        sim_printf("***************************************************************************************************\n");
        sim_printf("***************************************************************************************************\n");
        sim_printf("[0x%llx] :: msgQueueRaw @ %p, SPE_MESSAGE_QUEUE_BYTE_COUNT = %d\n",
               id, msgQueueRaw, SPE_MESSAGE_QUEUE_BYTE_COUNT
              );
      }

    }
  }

  #endif
}


#if SPE_DEBUG_DISPLAY >= 1
  char __buffer[2048];
#endif


void debug_displayStateHistogram(unsigned long long id, int* msgState, char* str) {
  #if SPE_DEBUG_DISPLAY >= 1

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
void memset(void* ptr, char val, int len) {
  // NOTE: This will actually traverse the memory backwards
  len--;
  while (len >= 0) {
    *((char*)ptr + len) = val;
    len--;
  }
}
#endif
