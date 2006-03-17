#include <stdio.h>
#include <spu_intrinsics.h>
//#include <spu/spu_mfcio.h>
#include <spu_mfcio.h>
//#include <stidc/bpa_mfc.h>
#include <cbe_mfc.h>
#include <unistd.h>
#include <string.h>

#include "spert_common.h"
#include "spert.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Required External Function(s) that the User Needs to Provide

extern void funcLookup(int funcIndex, void* data, int dataLen, void* msg, int msgLen);


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


  //printf("[%llu] DEBUG : ( stack) myData @ %p\n", id, &myData);
  //printf("[%llu] DEBUG : (global) msgQueueRaw @ %p\n", id, msgQueueRaw);
  //printf("[%llu] DEBUG : (global) msgQueue @ %p\n", id, msgQueue);


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

  // DEBUG
  //printf("SPEData @ %p = {\n", &myData);
  //printf("  messageQueue = %u\n", myData.messageQueue);
  //printf("  messageQueueLength = %d\n", myData.messageQueueLength);
  //printf("}\n");

  // Entry into the SPE's scheduler
  speScheduler(&myData, id);

  // Tell the world this SPE is going away
  printf(" --==>> Goodbye From SPE 0x%llx's Runtime <<==--\n", id);
  printf("  \"It's a cruel cruel world...\"\n");

  return 0;
}


void speScheduler(SPEData *speData, unsigned long long id) {

  int keepLooping = TRUE;
  int fetchIndex = 0;
  int runIndex = 0;
  int cnt = 0;
  int tagStatus;

  //printf("[%llu] DEBUG : ( stack) speScheduler::keepLooping @ %p\n", id, &keepLooping);
  //printf("[%llu] DEBUG : ( stack) speScheduler::tagStatus @ %p\n", id, &tagStatus);
  //printf("[%llu] DEBUG : SIZEOF_16(SPEMessage) = %d\n", SIZEOF_16(SPEMessage));

  // DEBUG
  int debugCounter = 0;

  printf("[%llu] --==>> Starting SPE Scheduler ...\n", id);

  // Initialize the tag status registers to all tags enabled
  spu_writech(MFC_WrTagMask, (unsigned int)-1);

  // Create the local message queue
  //volatile char* msgQueueRaw[SPE_MESSAGE_QUEUE_BYTE_COUNT];
  //volatile SPEMessage* msgQueue[SPE_MESSAGE_QUEUE_LENGTH];
  int msgState[SPE_MESSAGE_QUEUE_LENGTH];
  void* msgData[SPE_MESSAGE_QUEUE_LENGTH];
  void* msgMsg[SPE_MESSAGE_QUEUE_LENGTH];
  int msgCounter[SPE_MESSAGE_QUEUE_LENGTH];
  for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
    msgQueue[i] = (SPEMessage*)(((char*)msgQueueRaw) + (SIZEOF_16(SPEMessage) * i));
    msgState[i] = SPE_MESSAGE_STATE_CLEAR;
    msgData[i] = NULL;
    msgMsg[i] = NULL;
    msgCounter[i] = 0;
  }

  //printf("[%llu] :: msgQueueRaw = %p, SIZEOF_16(SPEMessage) = 0x%x\n", id, msgQueueRaw, SIZEOF_16(SPEMessage));
  //printf("[%llu] :: msqQueue = { ", id);
  //for (int k = 0; k < SPE_MESSAGE_QUEUE_LENGTH; k++)
  //  printf("%p ", msgQueue[k]);
  //printf("}\n");

  // Once the message queue has been created, check in with the main processor by sending a pointer to it
  spu_write_out_mbox((unsigned int)msgQueueRaw);

  //printf(" >>> Initial Message Queue Read...\n");
  //printf("    Importing Message Queue @ %p\n", speData->messageQueue);

  // Do the intial read of the message queue from main memory
  spu_mfcdma32(msgQueueRaw, (PPU_POINTER_TYPE)(speData->messageQueue), SPE_MESSAGE_QUEUE_BYTE_COUNT, 31, MFC_GET_CMD);
  spu_mfcstat(2);  // wait for the dma to finish

  //printf(" >>> Finished\n");

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

        printf("[%llu] :: still going... \n", id);

      #else

        printf("[%llu] :: still going... msgQueue[0] @ %p (msgQueue: %p) = { fi = %d, d = %d, dl = %d, m = %d, ml = %d, s = %d, cnt = %d, cmd = %d }\n",
               id,
               &(msgQueue[0]),
               msgQueue,
               (volatile int)(msgQueue[0]->funcIndex),
               msgQueue[0]->data,
               msgQueue[0]->dataLen,
               msgQueue[0]->msg,
               msgQueue[0]->msgLen,
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

        //printf("[%llu] :: Received Message! ---------------------------------------------------------------------\n", id);
        //printf("[%llu] :: New Message (index: %d)...\n", id, i);

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

    //if ((debugCounter % 5000) == 0 && debugCounter != 0)
    //  printf("[%llu] :: numDMAQueueEntries = %d\n", id, numDMAQueueEntries);

    for (int i = 0; (i < SPE_MESSAGE_QUEUE_LENGTH) /* && (numDMAQueueEntries > 0) */; i++) {

      if (msgState[i] == SPE_MESSAGE_STATE_PRE_FETCHING) {

        // Make sure there are enough entries left for the message to use
        // TODO : There might be a better way of doing this (hopefully the common path is that there are
        //   enough queue entries... measure this to see if it is true... if it is, then try to get these
        //   if statements out of the way for something that is faster).
        int numDMAEntriesNeeded = 0;
        if (msgQueue[i]-> data != (PPU_POINTER_TYPE)NULL) numDMAEntriesNeeded++;
        if (msgQueue[i]-> msg != (PPU_POINTER_TYPE)NULL) numDMAEntriesNeeded++;
        if (numDMAEntriesNeeded > numDMAQueueEntries) continue;  // Skip this message for now

        //printf("[%llu] :: Fetching Data for queue entry %d ----------------------------------------------------\n", id, i);

        // Fetch the data
        if (msgQueue[i]->data != (PPU_POINTER_TYPE)NULL) {
          int retrieveSize = ROUNDUP_16(msgQueue[i]->dataLen);
          msgData[i] = (void*)(new char[retrieveSize]);
          //msgData[i] = (void*)malloc_aligned(retrieveSize, 16);
          //printf("[%llu] DEBUG : (   new) speScheduler::msgData allocated @ %p (size: %d)\n", id, msgData[i], retrieveSize);

          if ((int)msgData[i] > 0x40000) {
            printf("!!!!! ERROR !!!!! : new returned a value greater than the LS size : Expect bad things in the near future !!!!!\n");
	  }

          //printf("msgData[%d] = %p\n", i, msgData[i]);

          if (msgData[i] == NULL) {
            printf("===== ERROR ERROR ERROR ===== : speScheduler() : Unable to allocate memory for data... expect bad things soon!...\n");
            continue;
          }
          spu_mfcdma32(msgData[i], (PPU_POINTER_TYPE)(msgQueue[i]->data), retrieveSize, i, MFC_GET_CMD);
          numDMAQueueEntries--;
	}

        else {
          //printf("[%llu] :: NO DATA FOR MESSAGE %d -------------------------------------------------------------\n", id, i);  
	}

        // Fetch the message
        if (msgQueue[i]->msg != (PPU_POINTER_TYPE)NULL) {
          int retrieveSize = ROUNDUP_16(msgQueue[i]->msgLen);
          msgMsg[i] = (void*)(new char[retrieveSize]);
          //msgMsg[i] = (void*)malloc_aligned(retrieveSize, 16);

          //printf("[%llu] DEBUG : (   new) speScheduler::msgMsg allocated @ %p (size: %d)\n", id, msgMsg[i], retrieveSize);

          if ((int)msgMsg[i] > 0x40000) {
            printf("!!!!! ERROR !!!!! : new returned a value greater than the LS size : Expect bad things in the near future !!!!!\n");
	  }

          //printf("msgMsg[%d] = %p\n", i, msgMsg[i]);

          if (msgMsg[i] == NULL) {
            printf("===== ERROR ERROR ERROR ===== : speScheduler() : Unable to allocate memory for message... expect bad things soon!...\n");
            continue;
	  }
          spu_mfcdma32(msgMsg[i], (PPU_POINTER_TYPE)(msgQueue[i]->msg), retrieveSize, i, MFC_GET_CMD);
          numDMAQueueEntries--;
	}

        else {
          //printf("[%llu] :: NO MSG FOR MESSAGE %d --------------------------------------------------------------\n", id, i);  
	}
        
        // Update the state of the message (locally)
        msgState[i] = SPE_MESSAGE_STATE_FETCHING;
      }
    }


    // Read the tag status to see if the data has arrived for any of the fetching message entries
    mfc_write_tag_update_immediate();
    tagStatus = mfc_read_tag_status(); //spu_readch(MFC_RdTagStat);

    //if ((debugCounter % 5000) == 0 && debugCounter != 0)
    //  printf("[%llu] :: Checking for Incoming DMA Completion 0x%08x\n", id, tagStatus);

    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_FETCHING && ((tagStatus & (0x01 << i)) != 0))
        msgState[i] = SPE_MESSAGE_STATE_READY;
    }


    // Execute a single ready message
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {

      if (msgState[runIndex] == SPE_MESSAGE_STATE_READY) {

        volatile SPEMessage *msg = msgQueue[runIndex];

        // DEBUG
        //printf("[%llu] :: --==>> SPE : Execute Message :: msgQueue[%d]->funcIndex = %d\n",
        //       id, runIndex, msgQueue[runIndex]->funcIndex
        //      );

        // TODO : Execute the message here
        funcLookup(msg->funcIndex, msgData[runIndex], msg->dataLen, msgMsg[runIndex], msg->msgLen);

        //// DEBUG
        //printf("[%llu] :: (1) msgState = {", id);
        //for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++)
        //  printf("%d ", msgState[j]);
        //printf("}\n");

        // Update the state of the message queue entry
        msgState[runIndex] = SPE_MESSAGE_STATE_EXECUTED;

        //// DEBUG
        //printf("[%llu] :: (2) msgState = {", id);
        //for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++)
        //  printf("%d ", msgState[j]);
        //printf("}\n");

        //// Try the commit the data to memory
        //if (msgQueue[i]->data != (PPU_POINTER_TYPE)NULL && mfc_stat_cmd_queue() > 0) {
        //  spu_mfcdma32(msgData[i], (PPU_POINTER_TYPE)(msgQueue[i]->data), ROUNDUP_16(msgQueue[i]->dataLen), i, MFC_PUT_CMD);
        //  msgState[runIndex] = SPE_MESSAGE_STATE_COMMITTING;
	//}

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


    // Check for messages that are executed but not committed yet
    for (int i = 0; i < SPE_MESSAGE_QUEUE_LENGTH; i++) {
      if (msgState[i] == SPE_MESSAGE_STATE_EXECUTED) {

        //// DEBUG
        //printf("[%llu] :: (3a) msgState = {", id);
        //for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++)
        //  printf("%d ", msgState[j]);
        //printf("}\n");

        // Check to see if there is data and it should be committed back to memory and there is a DMA command queue entry open
        if (msgQueue[i]->data != (PPU_POINTER_TYPE)NULL && mfc_stat_cmd_queue() > 0) {

          spu_mfcdma32(msgData[i], (PPU_POINTER_TYPE)(msgQueue[i]->data), ROUNDUP_16(msgQueue[i]->dataLen), i, MFC_PUT_CMD);
          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;

        // Otherwise, check to see if there is no data to commit
        } else if (msgQueue[i]->data == (PPU_POINTER_TYPE)NULL) {

          msgState[i] = SPE_MESSAGE_STATE_COMMITTING;
        }

        //// DEBUG
        //printf("[%llu] :: (3b) msgState = {", id);
        //for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++)
        //  printf("%d ", msgState[j]);
        //printf("}\n");


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

        //// DEBUG
        //printf("[%llu] :: (4) msgState = {", id);
        //for (int j = 0; j < SPE_MESSAGE_QUEUE_LENGTH; j++)
        //  printf("%d ", msgState[j]);
        //printf("}\n");

        // Check to see if there is an available entry in the outbound mailbox
        if (spu_stat_out_mbox() > 0) {

          // Free the local data and message buffers
          //if (msgData[i] != NULL) { free_aligned((void*)msgData[i]); msgData[i] = NULL; }
          //if ( msgMsg[i] != NULL) { free_aligned((void*) msgMsg[i]);  msgMsg[i] = NULL; }

          if (msgData[i] != NULL) { delete [] ((char*)msgData[i]); msgData[i] = NULL; }
          if ( msgMsg[i] != NULL) { delete [] ((char*)msgMsg[i]); msgMsg[i] = NULL; }

          // Clear the entry
          msgState[i] = SPE_MESSAGE_STATE_CLEAR;

          // DEBUG
          //printf("[%llu] :: Finished with message %d\n", id, i);

          // Send the index of the entry in the message queue to the PPE
          spu_write_out_mbox((unsigned int)i);
	}
        
      }
    }

    // Update the debugCounter
    debugCounter++;

  } // end while (keepLooping)

  printf(" --==>> Exiting SPE Scheduler ...\n");
}
