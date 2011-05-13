/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** @file
 * A template machine layer
 * @ingroup Machine
 */
/*@{*/

/*TODO: define the machine layer name, such as lapi, mpi etc. */
#define MACHNAME

#include <stdio.h>
#include <errno.h>
#include "converse.h"
#include <MACHNAME.h>

/*Support for ++debug: */
#if defined(_WIN32) && ! defined(__CYGWIN__)
#include <windows.h>
#include <wincon.h>
#include <sys/types.h>
#include <sys/timeb.h>
static void sleep(int secs) {
    Sleep(1000*secs);
}
#else
#include <unistd.h> /*For getpid()*/
#endif
#include <stdlib.h> /*For sleep()*/

#include "machine.h"

/* TODO: macros regarding redefining locks that will affect pcqueue.h*/
#include "pcqueue.h"

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* TODO: add any that are related */
/* =======End of Definitions of Performance-Specific Macros =======*/


/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
/* TODO: add any that are related */
/* =====End of Definitions of Message-Corruption Related Macros=====*/


/* =====Beginning of Declarations of Machine Specific Variables===== */
/* TODO: add any that are related */
/* =====End of Declarations of Machine Specific Variables===== */


/* =====Beginning of Declarations of Machine Specific Functions===== */
/* Utility functions */
/* TODO: add any that are related */

/* TODO: The machine-specific send function */
/*
static CmiCommHandle MachineSpecificSend(int destPE, int size, char *msg, int mode);
#define CmiMachineSpecificSendFunc "function name defined above"
*/

/* ### Beginning of Machine-startup Related Functions ### */
/* TODO: */
/*
static void MachineInitFor${MACHNAME}(int argc, char **argv, int *numNodes, int *myNodeID);
#define MachineSpecificInit "function name defined above"
*/

/*
static void MachinePreCommonInitFor${MACHNAME}(int everReturn);
static void MachinePostCommonInitFor${MACHNAME}(int everReturn);
#define MachineSpecificPreCommonInit "function name defined above1"
#define MachineSpecificPostCommonInit "function name defined above2"
*/
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
/* TODO:
static void AdvanceCommunicationFor${MACHNAME}();
#define MachineSpecificAdvanceCommunication "function name defined above2"

static void DrainResourcesFor${MACHNAME}(); //used when exit
#define MachineSpecificDrainResources "function name defined above2"


static void MachineExitFor${MACHNAME}();
#define MachineSpecificExit
*/
/* ### End of Machine-running Related Functions ### */

/* ### Beginning of Idle-state Related Functions ### */

/* ### End of Idle-state Related Functions ### */

/* TODO: if there's any related
void MachinePostNonLocalForMPI();
#define MachineSpecificPostNonLocal MachinePostNonLocalForMPI
*/

/* =====End of Declarations of Machine Specific Functions===== */

/**
   * Functions that requires machine specific implementation:
   * 1. void CmiReleaseCommHandle(CmiCommHandle c);
   * 2. int CmiAsyncMsgSent(CmiCommHandle c);
   * 3. char *CmiGetNonLocalNodeQ(void)
   * 4. void *CmiGetNonLocal(void)
   */

#include "machine-common.c"

/* The machine specific msg-sending function */

/* TODO: machine specific send operation */


/* TODO: machine specific functions for driving communication */
/* ######Beginning of functions related with communication progress ###### */
/* ######End of functions related with communication progress ###### */

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
}
#endif

/* TODO: */
/* ######Beginning of functions related with exiting programs###### */
/* 1. functions to drain resources */
/* 2. functions to exit */
/* ######End of functions related with exiting programs###### */


/* ######Beginning of functions related with starting programs###### */
/* TODO */
/* 1. machine init function */
/* 2. machine pre/post common init functions */
/**
 *  Obtain the number of nodes, my node id, and consuming machine layer
 *  specific arguments
 */
/* ######End of functions related with starting programs###### */

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {
}

/**************************  TIMER FUNCTIONS **************************/

/************Barrier Related Functions****************/

/*@}*/

