#ifndef __SPE_RUNTIME_PPU_H__
#define __SPE_RUNTIME_PPU_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <libspe.h>
#ifdef __cplusplus
}
#endif

#include "spert.h"


///////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// DISPLAY DEBUG DATA (the greater the number, more data is shown)
#define DEBUG_DISPLAY  0

// NOTE: When using the simulator, setting NUM_SPE_THREADS to a number lower than the number
//   of physical SPEs will reduce the ammount of time it takes to initiate all the SPEs (which
//   can be slow on the simulator).
#define NUM_SPE_THREADS  8               // This is limited by the number of physical SPEs present

#define CREATE_EACH_THREAD_ONE_BY_ONE   0  // Set this to non-zero to create and wait for each SPE thread one-by-one


///////////////////////////////////////////////////////////////////////////////////////////////
// Externals

extern spe_program_handle_t spert_main;


///////////////////////////////////////////////////////////////////////////////////////////////
// Data Structrues

// Work Request Handle : Used to keep track of and access outstanding work requests
typedef struct __work_request {

  int isFirstInSet;  // Flag to indicate that this work request handle is the first
                     //   in a contiguous set (free the first and the rest in the set
                     //   are also free'd)

  int speIndex;      // Index of the spe thread this work request was assigned to (-1 means not assigned yet or finished)
  int entryIndex;    // Index in the message queue this work request was assigned to (-1 means not assigned yet or finished)

  int funcIndex;     // These fields contain the information passed in via a call to
  void* readWritePtr;
  int readWriteLen;
  void* readOnlyPtr;
  int readOnlyLen;
  void* writeOnlyPtr;
  int writeOnlyLen;
  void *userData;

  struct __work_request *next;    // Pointer to the next WRHandle in the linked list of WRHandles

} WorkRequest;
typedef WorkRequest* WRHandle;
#define INVALID_WRHandle (NULL)

// SPE Thread : Used to maintain information about a particular SPE thread that is running
typedef struct __spe_thread {
  SPEData *speData;   // A pointer to the SPEData structure that was passed to the thread
  speid_t speID;      // The ID for the thread
  unsigned int messageQueuePtr;  // Pointer (in Local Store) to the SPE's message queue
  int msgIndex;
  int counter;
} SPEThread;


///////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes (Offload API)

extern int InitOffloadAPI(void(*cbFunc)(void*)
#ifdef __cplusplus
  = NULL
#endif
);
extern void CloseOffloadAPI();

extern WRHandle sendWorkRequest(int funcIndex,      // Index of the function to be called
                                void* readWritePtr, // Pointer to a readWrite buffer (read before execution, written after)
                                int readWriteLen,   // Length (in bytes) of the buffer pointed to by readWritePtr
                                void* readOnlyPtr,  // Pointer to a readOnly buffer (read before execution)
                                int readOnlyLen,    // Length (in bytes) of the buffer pointed to by readOnlyPtr
                                void* writeOnlyPtr, // Pointer to a writeOnly buffer (written after execution)
                                int writeOnlyLen,   // Length (in bytes) of the buffer pointed to by writeOnlyPtr
                                void* userData      // A pointer to user defined data that will be passed to the callback function (if there is one) once this request is finished
#ifdef __cplusplus
 = NULL   
#endif
                               );                   // Returns: INVALID_WRHandle on failure, a valid WRHandle otherwise

extern int isFinished(WRHandle wrHandle    // A work request handle returned by sendWorkRequest
                     );                    // Returns: Non-zero on finished, zero otherwize

void waitForWRHandle(WRHandle wrHandle     // A work request handle returned by sendWorkRequest
                    );                     // Returns: nothing

extern void OffloadAPIProgress();


#endif //__SPE_RUNTIME_PPU_H__
