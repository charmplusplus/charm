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


#include "machine-common.h"
#include "machine-common.c"

static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{}

/* ### Beginning of Machine-startup Related Functions ### */
static void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
}

static void LrtsPreCommonInit(int everReturn)
{
}
static void LrtsPostCommonInit(int everReturn)
{
}
static void LrtsAdvanceCommunication()
{
}
static void LrtsDrainResources() /* used when exit */
{
}
static void LrtsExit()
{
}
static void LrtsPostNonLocal()
{
}

#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
}
#endif

void CmiAbort(const char *message) {
}


/* Other assist function */


