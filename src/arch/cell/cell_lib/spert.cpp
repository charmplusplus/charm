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
// Global Data

//const int SPEData_dmaTransferSize = (sizeof(SPEData) & 0xFFFFFFF0) + (0x10);
const int SPEData_dmaTransferSize = SIZEOF_16(SPEData);

volatile char* msgQueueRaw[SPE_MESSAGE_QUEUE_BYTE_COUNT];
volatile SPEMessage* msgQueue[SPE_MESSAGE_QUEUE_LENGTH];


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

  // Set the MFC's Tag Mask to enable all bits.  (When the MFC Tag Status is read,
  //   its contents will be ANDed with this mask.)
  //spu_writech(MFC_WrTagMask, (unsigned int)(-1));   // speScheduler will control the tag mask

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


void speScheduler(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int fetchIndex = 0;
  int runIndex = 0;
  int cnt = 0;
  int tagStatus;

  // DEBUG
  int debugCounter = 0;

  printf("[%llx] --==>> Starting SPE Scheduler ...\n", id);

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Create the local message queue
  int msgState[SPE_MESSAGE_QUEUE_LENGTH];
  void* readWritePtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* readOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  void* writeOnlyPtr[SPE_MESSAGE_QUEUE_LENGTH];
  int msgCounter[SPE_MESSAGE_QUEUE_LENGTH];
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    msgQueue[i] = (SPEMessage*)(((char*)msgQueueRaw) + (SIZEOF_16(SPEMessage) * i));
    msgState[i] = SPE_MESSAGE_STATE_CLEAR;
    readWritePtr[i] = NULL;
    readOnlyPtr[i] = NULL;
    writeOnlyPtr[i] = NULL;
    msgCounter[i] = 0;
  }

  // Once the message queue has been created, check in with the main processor by sending a pointer to it
  spu_write_out_mbox((unsigned int)msgQueueRaw);

  // Do the intial read of the message queue from main memory
  spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);
  //spu_mfcstat(2);  // wait for the dma to finish

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

        printf("[%llx] :: still going... \n", id);

      #else

        printf("[%llx] :: still going... msgQueue[0] @ %p (msgQueue: %p) = { fi = %d, rw = %d, rwl = %d, ro = %d, rol = %d, wo = %d, wol = %d, s = %d, cnt = %d, cmd = %d }\n",
               id,
               &(msgQueue[0]),
               msgQueue,
               (volatile int)(msgQueue[0]->funcIndex),
               msgQueue[0]->readWritePtr,
               msgQueue[0]->readWriteLen,
               msgQueue[0]->readOnlyPtr,
               msgQueue[0]->readOnlyLen,
               msgQueue[0]->writeOnlyPtr,
               msgQueue[0]->writeOnlyLen,
               (volatile int)(msgQueue[0]->state),
               (volatile int)(msgQueue[0]->counter),
	       (volatile int)(msgQueue[0]->command)
	      );
        //printf("[%llu] :: raw msgQueue = { ", id);
        //for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
        //  printf("%d ", *(((char*)msgQueue) + ti));
        //}
        //printf("}\n");
        printf("[%llu] :: raw msgQueueRaw = { ", id);
        for (int ti = 0; ti < 2 * sizeof(SPEMessage) /*SPE_MESSAGE_QUEUE_BYTE_COUNT*/; ti++) {
          printf("%d ", *(((char*)msgQueueRaw) + ti));
        }
        printf("}\n");

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
    // NOTE: The for loop condition check to make sure that numDMAQueueEntries is >= 2.  This is because
    //   two DMA entries are needed: one for data and one for the message (scatter/gather lists aren't
    //   being used yet).  Since the queue is 16 entries long, if there isn't enough room, there are still
    //   15 outstanding DMA requests so it is probably OK to delay the current requests while still keeping
    //   the SPE busy.
    // TODO: Change this so if only one DMA command entry is available, use it and then initiate the second
    //   one later.
    unsigned int numDMAQueueEntries = mfc_stat_cmd_queue();
    for (int i = 0; (i < SPE_MESSAGE_QUEUE_LENGTH) /* && (numDMAQueueEntries > 0) */; i++) {

      if (msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING) {

        // Make sure there are enough entries left for the message to use
        // TODO : There might be a better way of doing this (hopefully the common path is that there are
        //   enough queue entries... measure this to see if it is true... if it is, then try to get these
        //   if statements out of the way for something that is faster).
        int numDMAEntriesNeeded = 0;
        if (msgQueue[i]->readWritePtr != (PPU_POINTER_TYPE)NULL) numDMAEntriesNeeded++;
        if (msgQueue[i]->readOnlyPtr != (PPU_POINTER_TYPE)NULL) numDMAEntriesNeeded++;
        if (numDMAEntriesNeeded > numDMAQueueEntries) continue;  // Skip this message for now

        // Fetch the readWrite data
        if (msgQueue[i]->readWritePtr != (PPU_POINTER_TYPE)NULL) {

          // Allocate a buffer locally on the SPE
          int retrieveSize = ROUNDUP_16(msgQueue[i]->readWriteLen);
          readWritePtr[i] = (void*)(new char[retrieveSize]);

          // Verify the pointer that was returned
          if (((int)readWritePtr[i]) > 0x40000) {
            printf("!!!!! ERROR !!!!! : new returned a value greater than the LS size : Expect bad things in the near future !!!!!\n");
	  }
          if (readWritePtr[i] == NULL) {
            printf("===== ERROR ===== : speScheduler() : Unable to allocate memory for readWritePtr... expect bad things soon!...\n");
            continue;
	  }

          // Initiate the transfer
          spu_mfcdma32(readWritePtr[i], (PPU_POINTER_TYPE)(msgQueue[i]->readWritePtr), retrieveSize, i, MFC_GET_CMD);
          numDMAQueueEntries--;
        }

        // Fetch the readOnly data
        if (msgQueue[i]->readOnlyPtr != (PPU_POINTER_TYPE)NULL) {

          // Allocate a buffer locally on the SPE
          int retrieveSize = ROUNDUP_16(msgQueue[i]->readOnlyLen);
          readOnlyPtr[i] = (void*)(new char[retrieveSize]);

          // Verify the pointer that was returned
          if (((int)readOnlyPtr[i]) > 0x40000) {
            printf("!!!!! ERROR !!!!! : new returned a value greater than the LS size : Expect bad things in the near future !!!!!\n");
	  }
          if (readOnlyPtr[i] == NULL) {
            printf("===== ERROR ===== : speScheduler() : Unable to allocate memory for readOnlyPtr... expect bad things soon!...\n");
            continue;
	  }

          // Initiate the transfer
          spu_mfcdma32(readOnlyPtr[i], (PPU_POINTER_TYPE)(msgQueue[i]->readOnlyPtr), retrieveSize, i, MFC_GET_CMD);
          numDMAQueueEntries--;
        }

        // Allocate memory for the writeOnly data
        if (msgQueue[i]->writeOnlyPtr != (PPU_POINTER_TYPE)NULL) {

          // Allocate a buffer locally on the SPE
          // NOTE: The wroteOnly buffer still needs to be aligned so it can be written back later (even through
          //   it is not being DMAed to the SPE now).
          int retrieveSize = ROUNDUP_16(msgQueue[i]->writeOnlyLen);
          writeOnlyPtr[i] = (void*)(new char[retrieveSize]);

          // Verify the pointer that was returned
          if (((int)writeOnlyPtr[i]) > 0x40000) {
            printf("!!!!! ERROR !!!!! : new returned a value greater than the LS size : Expect bad things in the near future !!!!!\n");
	  }
          if (writeOnlyPtr[i] == NULL) {
            printf("===== ERROR ===== : speScheduler() : Unable to allocate memory for writeOnlyPtr... expect bad things soon!...\n");
            continue;
	  }
	}

        // Update the state of the message (locally)
        msgState[i] = SPE_MESSAGE_STATE_FETCHING;
      }
    }


    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_FETCHING && ((tagStatus & (0x01 << i)) != 0))
        msgState[i] = SPE_MESSAGE_STATE_READY;
    }


    // Execute a single ready message
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

      if (msgState[runIndex] == SPE_MESSAGE_STATE_READY) {

        volatile SPEMessage *msg = msgQueue[runIndex];

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

        // Check to see if there is data to be written back to main memory or not
        if (readWritePtr[i] != NULL || writeOnlyPtr[i] != NULL) {

          // Check to see if there are enough DMA entries to write the data back to memory
          int numDMAEntriesNeeded = 0;
          if (readWritePtr[i] != NULL) numDMAEntriesNeeded++;
          if (writeOnlyPtr[i] != NULL) numDMAEntriesNeeded++;
          if (numDMAEntriesNeeded > numDMAQueueEntries) continue;

          // Write the readWrite data back to main memory
          if (readWritePtr[i] != NULL) {
            spu_mfcdma32(readWritePtr[i], (PPU_POINTER_TYPE)(msgQueue[i]->readWritePtr), ROUNDUP_16(msgQueue[i]->readWriteLen), i, MFC_PUT_CMD);
            numDMAQueueEntries--;
	  }

          // Write the writeOnly data back to main memory
          if (writeOnlyPtr[i] != NULL) {
            spu_mfcdma32(writeOnlyPtr[i], (PPU_POINTER_TYPE)(msgQueue[i]->writeOnlyPtr), ROUNDUP_16(msgQueue[i]->writeOnlyLen), i, MFC_PUT_CMD);
            numDMAQueueEntries--;
	  }

          // Advance the state
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;

        } else {
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
          if (readWritePtr[i] != NULL) { delete [] ((char*)readWritePtr[i]); readWritePtr[i] = NULL; }
          if (readOnlyPtr[i] != NULL) { delete [] ((char*)readOnlyPtr[i]); readOnlyPtr[i] = NULL; }
          if (writeOnlyPtr[i] != NULL) { delete [] ((char*)writeOnlyPtr[i]); writeOnlyPtr[i] = NULL; }

          // Clear the entry
          msgState[i] = SPE_MESSAGE_STATE_CLEAR;

          // Send the index of the entry in the message queue to the PPE
          spu_write_out_mbox((unsigned int)i);
	}
        
      }
    }

    // Update the debugCounter
    debugCounter++;

  } // end while (keepLooping)

}
