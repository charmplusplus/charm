#ifndef __SPE_RUNTIME_PPU_H__
#define __SPE_RUNTIME_PPU_H__

/**
 @defgroup OffloadAPI Offload API

 This is the Offload API which is used to send <i>work requests</i> to special purpose hardware.
 Currently, only IBM's Cell processor is supported.  As such, the Offload API is structured
 around this use.  The user supplies a function (funcLookup()) in their code for the SPE.  The
 user can then make <i>work requests</i> to the Offload API.  The Offload API will then take
 care of moving user data (buffers), scheduling the SPE's, etc.  Once everything is ready, the
 user's SPE code (through the user provided funcLookup() function) is called and user defined
 code is executed on the SPE.  Once finished, the Offload API takes care of moving all the data
 back to main memory and notifying the main processor (the PPE) that the work request has
 finished.

 The prototype for the user provided funcLookup() functions is as follows.

 void funcLookup(int funcIndex,
                 void* readWriteptr, int readWriteLen,
                 void* readOnlyPtr, int readOnlyLen,
                 void* writeOnlyPtr, int writeOnlyLen,
                 DMAListEntry* dmaList
                );

  For standard work requests, dmaList will be null.  For scatter/gather work requests, the other
  pointer parameters will be NULL. The xxxLen parameters also have different units based on the
  type of the work request (in bytes for standard, in units of DMAListEntry structures for
  scatter/gather).  See the documentation for the WorkRequest data atructure for more
  information on the arguments.

 */
/*@{*/


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

/** Flag to display debug data for the PPE side of the Offload API (the greater the number, more data is shown) */
#define DEBUG_DISPLAY  0   // Set to non-zero to have Offload API display debug info (also see SPE_DEBUG_DISPLAY in spert.h)

/** NOTE: When using the simulator, setting NUM_SPE_THREADS to a number lower than the number
 *    of physical SPEs will reduce the ammount of time it takes to initiate all the SPEs (which
 *    can be slow on the simulator).
 */
#define NUM_SPE_THREADS  8   // Normally, this is limited by (and should be set to) the number of physical SPEs present

/** Create each SPEThread one-by-one (i.e. - create one thread and wait for it to check in
 *    before creating the next one).  If this is not set, all SPEThreads are created and then
 *    their "check-in values" are check to make sure they were created and are ready to
 *    execute work requests.
 */
#define CREATE_EACH_THREAD_ONE_BY_ONE   0  // Set this to non-zero to create and wait for each SPE thread one-by-one


#ifdef __cplusplus
  #define DEFAULT_TO_NULL  = NULL
#else
  #define DEFAULT_TO_NULL
#endif

#ifdef __cplusplus
  #define DEFAULT_TO_NONE  = WORK_REQUEST_FLAGS_NONE
#else
  #define DEFAULT_TO_NONE
#endif


///////////////////////////////////////////////////////////////////////////////////////////////
// Externals

/** The symbol name of the SPE executable that should be linked with the main code.  That is, when
 *  the SPE code is embedded into a PPE object file, the symbol name "spert_main" should be used
 *  for the SPE executable.
 */
extern spe_program_handle_t spert_main;


///////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures

/** WorkRequest is used to keep track of and access outstanding work requests. */
typedef struct __work_request {

  int isFirstInSet;  ///< An internal flag that indicates this work request is the first element of an array of work requests (i.e. - work requests with this flag set should be free'd by the Offload API when CloseOffloadAPI() is called by the user).

  int speIndex;   ///< Index of the spe thread this work request was assigned to (-1 means not assigned yet or finished).
  int entryIndex; ///< Index in the message queue this work request was assigned to (-1 means not assigned yet or finished).

  int funcIndex;      ///< User defined value used to indicated the action to be taken on the SPE.
  void* readWritePtr; ///< Pointer to a single buffer that is DMAed into and out-of the SPE's local store before and after the work request is executed, respectively. For scatter/gather work requests, this actually points to the user's dma list in main memory.  (<b>Must be 16 byte aligned.</b>  128 byte alignment is preferred.)
  int readWriteLen;   ///< Length (in bytes) of the buffer pointed to by readWritePtr.  For scatter/gather work requests, this number is the number of DMAListEntry structures in the user's dma list that are read/write.
  void* readOnlyPtr;  ///< Pointer to a single buffer that is DMAed into the SPE's local store before the work request is executed.  For scatter/gather work requests, this member contains the upper 32-bits of the effective addresses contained in the dma list.  (<b>Must be 16 byte aligned.</b>  128 byte alignment is preferred.)
  int readOnlyLen;    ///< Length (in bytes) of the buffer pointed to by readOnlyPtr.  For scatter/gather work requests, this number is the number of DMAListEntry structures in the user's dma list that are read-only.
  void* writeOnlyPtr; ///< Pointer to a single buffer that is DMAed out-of the SPE's local store after the work request has been executed.  An equal sized buffer will be provied to the funcLookup() function, which may be filled in by the user's code.  Upon completion of the work request, the data will be DMAed into the buffer pointed to by this pointer.  Not used for scatter/gather work requests.  (<b>Must be 16 byte aligned.</b>  128 byte alignment is preferred.)
  int writeOnlyLen;   ///< Length (in bytes) of the buffer pointed to by writeOnlyPtr.  For scatter/gather work requests, this number is the number of DMAListEntry structures in the user's dma list that are write-only.
  unsigned int flags; ///< One or more WORK_REQUEST_FLAGS_xxx bitwise ORed together.
  void *userData;     ///< A user defined pointer that will be passed to the callback function provided to InitOffloadAPI().

  struct __work_request *next; ///< Pointer to the next WRHandle in the linked list of WRHandles

} WorkRequest;

/** WRHandle structure that is used by Offload API calls.  This is what the caller (user of Offload API) keeps track of. */
typedef WorkRequest* WRHandle;

/** A "NULL" WRHandle (similar to a null pointer). */
#define INVALID_WRHandle (NULL)


/** An SPE Thread structures which is Used to maintain information about a particular SPE thread
 *  that is running on an SPE.
 */
typedef struct __spe_thread {
  SPEData *speData;   ///< A pointer to the SPEData structure that was passed to the thread when it was initial created
  speid_t speID;      ///< The ID for the thread
  unsigned int messageQueuePtr;  ///< Pointer (in the SPE's Local Store) to the SPE's message queue
  int msgIndex;       ///< Internal variable used to try and balance the usage of the various message queue entries (heuristic)
  int counter;        ///< A counter used to indicate new messages in the message queue
} SPEThread;


///////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes (Offload API)

/** Initialization function.  This function should be called before any other function in the
 *  Offload API.  It should only be called once.  If no callback function is specified, the
 *  isFinished() call should be used to both detect when a work request is finished and to
 *  clean-up the WRHandle (see the isFinished() function).
 *
 *  \return Non-zero on success.
 */
extern int InitOffloadAPI(void(*cbFunc)(void*) DEFAULT_TO_NULL  ///< Pointer to a function that should be called when a work request completes
                         );

/** Close function.  This function should be called after all other function calls to the
 *  Offload API have been made.  It should only be called once.
 */
extern void CloseOffloadAPI();

/** Creates and sends a work request to one of the SPEs.  This method of sending a work request is considered
 *  the \i standard way of sending work requests.  Only one of each type of buffer (read/write, read-only,
 *  write-only) may be specified (unless one of the WORK_REQUEST_FLAG_RW_IS_RO and WORK_REQUEST_FLAG_RW_IS_WO
 *  flags are used).
 *
 *  \return INVALID_WRHandle on failure, a valid WRHandle otherwise.
 */
extern WRHandle sendWorkRequest(int funcIndex,      ///< Index of the function to be called
                                void* readWritePtr, ///< Pointer to a readWrite buffer (read before execution, written after)
                                int readWriteLen,   ///< Length (in bytes) of the buffer pointed to by readWritePtr
                                void* readOnlyPtr,  ///< Pointer to a readOnly buffer (read before execution)
                                int readOnlyLen,    ///< Length (in bytes) of the buffer pointed to by readOnlyPtr
                                void* writeOnlyPtr, ///< Pointer to a writeOnly buffer (written after execution)
                                int writeOnlyLen,   ///< Length (in bytes) of the buffer pointed to by writeOnlyPtr
                                void* userData DEFAULT_TO_NULL, ///< A pointer to user defined data that will be passed to the callback function (if there is one) once this request is finished
				unsigned int flags DEFAULT_TO_NONE ///< Flags for the work request (see WORK_REQUEST_FLAGS_xxx defines)
                               );

/** Creates and sends a work request to one of the SPEs.  This method of sending a work request is considered
 *  a scatter/gather type of work request.  Zero-or-more of each kind of buffer (read/write, read-only, write-only)
 *  may be specified through the use of the dma list.  The DMAListEntry array pointed to by the dmaList parameter
 *  should have all of the read-only buffers listed first (numReadOnly of them), followed by all of the read/write
 *  buffers (numReadWrite of them), and finally all of the write-only buffers (numWriteOnly of them).
 *
 *  \return INVALID_WRHandle on failure, a valid WRHandle otherwise.
 */
extern WRHandle sendWorkRequest_list(int funcIndex,         ///< Index of the function to be called
                                     unsigned int eah,      ///< Upper 32-bits of the effective addresses in dma list
                                     DMAListEntry* dmaList, ///< List of dma list entries (lower 32-bits of effective address and size)
                                     int numReadOnly,       ///< Number read only pointers in dmaList (should be first in list... i.e. - All readOnly pointers should be before all other types of pointers in dmaList)
                                     int numReadWrite,      ///< Number of read/write pointers in dmaList (should be second in list... i.e. - All read/write pointers should be after all readOnly pointers and before all writeOnly pointers in dmaList)
                                     int numWriteOnly,      ///< Number of write only pointers in dmaList (should be third in list... i.e. - After all other types of pointers in dmaList)
                                     void* userData DEFAULT_TO_NULL, ///< A pointer to user defined data that will be passed to the callback function (if there is one) once this request is finished
                                     unsigned int flags DEFAULT_TO_NONE
                                    );

/** Used to check if the specified work request has finished executing (including out-bound data being
 *  DMAed back into main memory).  If no callback function was specified when InitOffloadAPI() was called,
 *  this function must be called to clean up the WRHandle if the work request has finished (NOTE:
 *  waitForWRHandle() will also clean up
 *  the WRHandle.)  If a callback function was passed to InitOffloadAPI() then this function has no
 *  effect (as the callback function will be called and the WRHandle will be cleaned up automatically.)
 *  This function is non-blocking.
 *
 *  \return Non-zero if the specified WRHandle has finished, zero otherwise
 */
extern int isFinished(WRHandle wrHandle    ///< A work request handle returned by sendWorkRequest
                     );

/** Used to wait for the specified WRHandle to finish.  This call is blocking.  Like isFinished(), this
 *  function will clean-up the WRHandle before returning.
 */
void waitForWRHandle(WRHandle wrHandle     ///< A work request handle returned by sendWorkRequest
                    );

/** Used to allow the Offload API to make progress.  Checks for finished work requests, etc.
  */
extern void OffloadAPIProgress();


/*@}*/

#endif //__SPE_RUNTIME_PPU_H__
