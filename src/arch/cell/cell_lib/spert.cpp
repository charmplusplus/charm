#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <cbe_mfc.h>
#include <unistd.h>
#include <string.h>

#include "spert_common.h"
#include "spert.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Required External Function(s) that the User Needs to Provide

//extern void funcLookup(int funcIndex, void* data, int dataLen, void* msg, int msgLen);
extern void funcLookup(int funcIndex,
                       void* readWritePtr, int readWriteLen,
                       void* readOnlyPtr, int readOnlyLen,
                       void* writeOnlyPtr, int writeOnlyPtr
                      );


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures

typedef struct __dma_list_entry {
  unsigned int size;  // The size of the DMA transfer (actually three values (from MSB): notify:1, reserved:15, size:16)
  unsigned int ea;    // Effective address of the data
} DMAListEntry;


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

//const int SPEData_dmaTransferSize = (sizeof(SPEData) & 0xFFFFFFF0) + (0x10);
const int SPEData_dmaTransferSize = SIZEOF_16(SPEData);

volatile char* msgQueueRaw[SPE_MESSAGE_QUEUE_BYTE_COUNT] __attribute__((aligned(128)));
volatile SPEMessage* msgQueue[SPE_MESSAGE_QUEUE_LENGTH];

// NOTE: Allocate two per entry (two buffers are read in from memory, two are
//   written out to memory, read write does not overlap for a given message).
volatile DMAListEntry dmaListEntry[SPE_DMA_LIST_LENGTH * SPE_MESSAGE_QUEUE_LENGTH] __attribute__((aligned(16)));
int dmaListSize[SPE_MESSAGE_QUEUE_LENGTH];


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

void speScheduler(SPEData *speData, unsigned long long id);


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies

int main(unsigned long long id, unsigned long long param) {

  /*volatile*/ SPEData myData;

  // Tell the world this SPE is alive
  printf(" --==>> Hello From SPE 0x%llx's Runtime <<==--\n", id);

  // Initialize globals
  memset(msgQueueRaw, 0x00, SPE_MESSAGE_QUEUE_BYTE_COUNT);

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

  // Tell the world this SPE is going away
  printf(" --==>> Goodbye From SPE 0x%llx's Runtime <<==--\n", id);
  printf("  \"I do not regret the things I have done, but those I did not do.\" - Lucas, Empire Records\n");

  return 0;
}


// DEBUG
void debug_displayActiveMessageQueue(unsigned long long id, int* msgState, char* str) {

  for (int tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
    if (msgState[tmp] != SPE_MESSAGE_STATE_CLEAR || msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) {
      printf("[0x%llx] :: %s%s msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %d, rwl = %d, ro = %d, rol = %d, wo = %d, wol = %d, s = %d(%d), cnt = %d, cmd = %d }\n",
             id,
             ((msgQueue[tmp]->state < SPE_MESSAGE_STATE_MIN || msgQueue[tmp]->state > SPE_MESSAGE_STATE_MAX) ? ("---===!!! WARNING !!!===--- ") : ("")),
             ((str == NULL) ? ("") : (str)),
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
             (volatile int)(msgQueue[tmp]->state),
             msgState[tmp],
             (volatile int)(msgQueue[tmp]->counter),
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
}


void speScheduler(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int fetchIndex = 0;
  int runIndex = 0;
  int cnt = 0;
  int tagStatus;

  // DEBUG
  int debugCounter = 0;

  printf("[0x%llx] --==>> Starting SPE Scheduler ...\n", id);

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Clear out the DMAListEntry array
  memset((void*)dmaListEntry, 0, sizeof(DMAListEntry) * 2 * SPE_MESSAGE_QUEUE_LENGTH);

  // Create the local message queue
  int msgState[SPE_MESSAGE_QUEUE_LENGTH];
  void* readWritePtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* readOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* writeOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* localMemPtr[SPE_MESSAGE_QUEUE_LENGTH];
  int msgCounter[SPE_MESSAGE_QUEUE_LENGTH];
  DMAListEntry* dmaList[SPE_MESSAGE_QUEUE_LENGTH];
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    msgQueue[i] = (SPEMessage*)(((char*)msgQueueRaw) + (SIZEOF_16(SPEMessage) * i));
    msgState[i] = SPE_MESSAGE_STATE_CLEAR;
    readWritePtr[i] = NULL;
    readOnlyPtr[i] = NULL;
    writeOnlyPtr[i] = NULL;
    msgCounter[i] = 0;
    dmaListSize[i] = -1;
    dmaList[i] = NULL;
  }

  // Once the message queue has been created, check in with the main processor by sending a pointer to it
  spu_write_out_mbox((unsigned int)msgQueueRaw);

  // Do the intial read of the message queue from main memory
  spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);

  // The scheduler loop
  while (keepLooping != FALSE) {


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DMK - TODO(s) :
    //   1) Add usages of mfc_stat_cmd_queue() before DMA commands (this will indicate whether or not the DMA
    //      controller's queue is full and, as such, if the DMA command would block).
    //   2) Break up the SPE_MESSAGE_STATE_FETCHING incase one DMA command can get through and the other would
    //      otherwise block (this way, at least some of the data can be inbound while the SPE is free to move on).
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Wait for the latest message queue read (block)
    mfc_write_tag_mask(0x80000000);   // enable only tag group 31 (message queue request)
    mfc_write_tag_update_any();
    tagStatus = mfc_read_tag_status();
    mfc_write_tag_mask(0x7FFFFFFF);   // enable all tag groups except 31


    // DEBUG - Let the user know that the SPE is still alive
    if ((debugCounter % 5000) == 0 && debugCounter != 0) {
      #if 1

        printf("[0x%llx] :: still going... \n", id);

      #else

        for (int tmp = 0; tmp < SPE_MESSAGE_QUEUE_LENGTH; tmp++) {
          printf("[0x%llx] :: still going... msgQueue[%d] @ %p (msgQueue: %p) = { fi = %d, rw = %d, rwl = %d, ro = %d, rol = %d, wo = %d, wol = %d, s = %d(%d), cnt = %d, cmd = %d }\n",
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
                 (volatile int)(msgQueue[tmp]->state),
                 msgState[tmp],
                 (volatile int)(msgQueue[tmp]->counter),
	         (volatile int)(msgQueue[tmp]->command)
	        );
	}
        //printf("[%llu] :: raw msgQueue = { ", id);
        //for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
        //  printf("%d ", *(((char*)msgQueue) + ti));
        //}
        //printf("}\n");
        //printf("[%llu] :: raw msgQueueRaw = { ", id);
        //for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
        //  printf("%d ", *(((char*)msgQueueRaw) + ti));
        //}
        //printf("}\n");

      #endif
    }


    // Check for new messages
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

      //if ((debugCounter % 5000) == 0 && debugCounter != 0)
      //  printf("[%llu] :: msgQueue[%d]->state = %d, msgState[%d] = %d, msgCounter[%d] = %d, msgQueue[%d]->counter = %d\n",
      //         id, i, msgQueue[i]->state, i, msgState[i], i, msgCounter[i], i, msgQueue[i]->counter
      //        );

      // Check for a new message in this slot
      if (msgQueue[i]->state == SPE_MESSAGE_STATE_SENT &&
          msgState[i] == SPE_MESSAGE_STATE_CLEAR &&
          msgCounter[i] != msgQueue[i]->counter
         ) {

        // Start by checking the command
        int command = msgQueue[i]->command;
        if (command == SPE_MESSAGE_COMMAND_EXIT) {
          printf(" --==>> SPE received EXIT command...\n");
          keepLooping = FALSE;
          break;
        }

        // Update the state of the message (locally)
        msgState[i] = SPE_MESSAGE_STATE_PRE_FETCHING;
        msgCounter[i] = msgQueue[i]->counter;        
      }
    }

    
    // Check for messages that need data fetched
    unsigned int numDMAQueueEntries = mfc_stat_cmd_queue();
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING) {

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
          localMemPtr[i] = (void*)(new char[memNeeded]);

          // Check the pointer (if it is bad, then skip this message for now and try again later)
          // TODO: There are probably better checks for this (use _end, etc.)
          if (localMemPtr[i] == NULL || (unsigned int)localMemPtr[i] >= (unsigned int)0x40000) {
            localMemPtr[i] = NULL;
            break;
	  }

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
        if ((msgQueue[i]->readWritePtr == (PPU_POINTER_TYPE)NULL) && (msgQueue[i]->readOnlyPtr == (PPU_POINTER_TYPE)NULL)) {

          // Update the state (to ready to execute)
          msgState[i] = SPE_MESSAGE_STATE_READY;

          // Done with this one
          continue;
	}

        // Create the DMA list
        // NOTE: Check to see if the dma list has already been created yet or not (if dmaListSize[i] < 0, not created)
        if (dmaListSize[i] < 0) {

          // Count the number of DMA entries needed for the read DMA list
          register int entryCount = 0;
          entryCount += (ROUNDUP_16(msgQueue[i]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
          entryCount += (ROUNDUP_16(msgQueue[i]->readOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->readOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

          // Allocate a larger DMA list if needed
          if (entryCount > SPE_DMA_LIST_LENGTH) {
            dmaList[i] = new DMAListEntry[entryCount];
            if (dmaList[i] == NULL || (unsigned int)dmaList[i] >= (unsigned int)0x40000) {
              dmaList[i] = NULL;
              continue;  // Skip for now, try again later
	    }
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
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }
	  }

          if (readWritePtr[i] != NULL) {
            register int bufferLeft = msgQueue[i]->readWriteLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->readWritePtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
              dmaList[i][listIndex].ea = srcOffset;

              bufferLeft -= SPE_DMA_LIST_ENTRY_MAX_LENGTH;
              listIndex++;
              srcOffset += SPE_DMA_LIST_ENTRY_MAX_LENGTH;
	    }
	  }
	}

        // Initiate the DMA command
        if (numDMAQueueEntries > 0 && dmaListSize[i] > 0) {

          spu_mfcdma64(localMemPtr[i],
                       0,
                       (unsigned int)(dmaList[i]),
                       dmaListSize[i] * sizeof(DMAListEntry),
                       i,
                       MFC_GETL_CMD
		      );

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_FETCHING;
	}

      }
    }


    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_FETCHING && ((tagStatus & (0x01 << i)) != 0)) {

        // Update the state to show that this message queue entry is ready to be executed
        msgState[i] = SPE_MESSAGE_STATE_READY;

        // Clean up the dmaList
        if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
          delete [] dmaList[i];
          dmaList[i] = NULL;
	}
        dmaListSize[i] = -1;  // NOTE: Clear this so data that the dmaList looks like it has not been set now
      }
    }


    // Execute a single ready message
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

      if (msgState[runIndex] == SPE_MESSAGE_STATE_READY) {

        register volatile SPEMessage* msg = msgQueue[runIndex];

        // Execute the function specified
        funcLookup(msg->funcIndex,
                   readWritePtr[runIndex], msg->readWriteLen,
                   readOnlyPtr[runIndex], msg->readOnlyLen,
                   writeOnlyPtr[runIndex], msg->writeOnlyLen
                  );

        // Update the state of the message queue entry
        msgState[runIndex] = SPE_MESSAGE_STATE_EXECUTED;

        // Move runIndex to the next message and break from the loop (execute only one)
        runIndex++;
        if (runIndex >= SPE_MESSAGE_QUEUE_LENGTH)
          runIndex = 0;

        break;
      }

      // Try the next message
      runIndex++;
      if (runIndex >= SPE_MESSAGE_QUEUE_LENGTH)
        runIndex = 0;
    }


    // Check for messages that have been executed but still need data committed to main memory
    numDMAQueueEntries = mfc_stat_cmd_queue();
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_EXECUTED) {

        // Check to see if this message does not needs to fetch any data
        if ((msgQueue[i]->readWritePtr == (PPU_POINTER_TYPE)NULL) && (msgQueue[i]->writeOnlyPtr == (PPU_POINTER_TYPE)NULL)) {
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;  // The index still needs to be passed back to the PPU
          continue;
	}

        // Create the DMA list
        // NOTE: Check to see if the dma list has already been created yet or not (if dmaListSize[i] < 0, not created)
        if (dmaListSize[i] < 0) {

          // Count the number of DMA entries needed for the read DMA list
          register int entryCount = 0;
          entryCount += (ROUNDUP_16(msgQueue[i]->readWriteLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->readWriteLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));
          entryCount += (ROUNDUP_16(msgQueue[i]->writeOnlyLen) / SPE_DMA_LIST_ENTRY_MAX_LENGTH) +
                        (((msgQueue[i]->writeOnlyLen & (SPE_DMA_LIST_ENTRY_MAX_LENGTH - 1)) == 0x0) ? (0) : (1));

          // Allocate a larger DMA list if needed
          if (entryCount > SPE_DMA_LIST_LENGTH) {
            dmaList[i] = new DMAListEntry[entryCount];
            if (dmaList[i] == NULL || (unsigned int)dmaList[i] >= (unsigned int)0x40000) {
              dmaList[i] = NULL;
              continue;  // Skip for now, try again later
            }
	  } else {
            dmaList[i] = (DMAListEntry*)(&(dmaListEntry[i * SPE_DMA_LIST_LENGTH]));
	  }
          dmaListSize[i] = entryCount;


          // Fill in the list
          readOnlyPtr[i] = NULL;   // Use this pointer to point to the first buffer to be written to memory (don't nead readOnly data anymore)
          register int listIndex = 0;

          if (readWritePtr[i] != NULL) {
            register int bufferLeft = msgQueue[i]->readWriteLen;
            register unsigned int srcOffset = (unsigned int)(msgQueue[i]->readWritePtr);

            while (bufferLeft > 0) {
              dmaList[i][listIndex].size = ((bufferLeft > SPE_DMA_LIST_ENTRY_MAX_LENGTH) ? (SPE_DMA_LIST_ENTRY_MAX_LENGTH) : (bufferLeft));
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

          spu_mfcdma64(readOnlyPtr[i],  // This pointer is being used to point to the start of the write portion of localMem
                       0,
                       (unsigned int)(dmaList[i]),
                       dmaListSize[i] * sizeof(DMAListEntry),
                       i,
                       MFC_PUTL_CMD
		      );

          // Decrement the counter of available DMA queue entries left
          numDMAQueueEntries--;

          // Update the state of the message queue entry now that the data should be in-flight
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;
	}

      }
    }


    // Initiate the next message queue read from main memory
    spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);


    // Check for messages that are committed
    mfc_write_tag_mask(0x7FFFFFFF);
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_COMMITTING && ((tagStatus * (0x01 << i)) != 0)) {

        // Check to see if there is an available entry in the outbound mailbox
        if (spu_stat_out_mbox() > 0) {

          // Free the local data and message buffers
          if (localMemPtr[i] != NULL) {
            delete [] ((char*)localMemPtr[i]);
            localMemPtr[i] = NULL;
            readWritePtr[i] = NULL;
            readOnlyPtr[i] = NULL;
            writeOnlyPtr[i] = NULL;
	  }

          // Clear the entry
          msgState[i] = SPE_MESSAGE_STATE_CLEAR;

          // Clear the dmaList size so it looks like the dma list has not been set
          if (dmaListSize[i] > SPE_DMA_LIST_LENGTH) {
            delete [] dmaList[i];
            dmaList[i] = NULL;
	  }
          dmaListSize[i] = -1;

          // Send the index of the entry in the message queue to the PPE
          spu_write_out_mbox((unsigned int)i);
	}
        
      }
    }

    // Update the debugCounter
    debugCounter++;


  } // end while (keepLooping)

}
