/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile$
 *      $Author$        $Locker$                $State$
 *      $Revision$      $Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include "converse.h"
#include "conv-mach.h"
#include "conv-conds.h"

#define MAX_HANDLERS 512

void        *CmiGetNonLocal();

CpvDeclare(int, disable_sys_msgs);
CpvExtern(int, CcdNumChecks) ;
CpvDeclare(void*, CsdSchedQueue);
CpvDeclare(int,   CsdStopFlag);


/*****************************************************************************
 *
 * Some of the modules use this in their argument parsing.
 *
 *****************************************************************************/

static char *DeleteArg(argv)
  char **argv;
{
  char *res = argv[0];
  if (res==0) { CmiError("Bad arglist."); exit(1); }
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}


/*****************************************************************************
 *
 * Statistics: currently, the following statistics are not updated by converse.
 *
 *****************************************************************************/

CpvDeclare(int, CstatsMaxChareQueueLength);
CpvDeclare(int, CstatsMaxForChareQueueLength);
CpvDeclare(int, CstatsMaxFixedChareQueueLength);
CpvStaticDeclare(int, CstatPrintQueueStatsFlag);
CpvStaticDeclare(int, CstatPrintMemStatsFlag);

void CstatsInit(argv)
char **argv;
{
  CpvInitialize(int, CstatsMaxChareQueueLength);
  CpvInitialize(int, CstatsMaxForChareQueueLength);
  CpvInitialize(int, CstatsMaxFixedChareQueueLength);
  CpvInitialize(int, CstatPrintQueueStatsFlag);
  CpvInitialize(int, CstatPrintMemStatsFlag);

  CpvAccess(CstatsMaxChareQueueLength) = 0;
  CpvAccess(CstatsMaxForChareQueueLength) = 0;
  CpvAccess(CstatsMaxFixedChareQueueLength) = 0;

  while (*argv) {
    if (strcmp(*argv, "+mems") == 0) {
      CpvAccess(CstatPrintMemStatsFlag)=1;
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+qs") == 0) {
      CpvAccess(CstatPrintQueueStatsFlag)=1;
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+qs") == 0) {
      CpvAccess(CstatPrintQueueStatsFlag)=1;
      DeleteArg(argv);
    } else
    argv++;
  }
}

int CstatMemory(i)
int i;
{
  return 0;
}

int CstatPrintQueueStats()
{
  return CpvAccess(CstatPrintQueueStatsFlag);
}

int CstatPrintMemStats()
{
  return CpvAccess(CstatPrintMemStatsFlag);
}

/*****************************************************************************
 *
 * Cmi handler registration
 *
 *****************************************************************************/

CpvDeclare(CmiHandler*, CmiHandlerTable);
CpvStaticDeclare(int, CmiHandlerCount);

int CmiRegisterHandler(handlerf)
CmiHandler handlerf ;
{
  CpvAccess(CmiHandlerTable)[CpvAccess(CmiHandlerCount)] = handlerf;
  CpvAccess(CmiHandlerCount)++ ;
  return CpvAccess(CmiHandlerCount)-1 ;
}

static void CmiHandlerInit()
{
  CpvInitialize(int, CmiHandlerCount);
  CpvInitialize(CmiHandler *, CmiHandlerTable);
  CpvAccess(CmiHandlerCount) = 0;
  CpvAccess(CmiHandlerTable) =
    (CmiHandler *)CmiAlloc((MAX_HANDLERS + 1) * sizeof(CmiHandler)) ;
}

/*****************************************************************************
 *
 * The following are some CMI functions.  Having this code here makes
 * the module boundaries for the Cmi quite blurry.
 *
 *****************************************************************************/

#ifdef CMK_USES_COMMON_CMIDELIVERS

CpvStaticDeclare(int, CmiBufferGrabbed);
CpvExtern(void*,      CmiLocalQueue);

void CmiInit(argv)
char **argv;
{
  void *FIFO_Create();

  CmiHandlerInit();
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
  CmiInitMc(argv);
  CmiSpanTreeInit(argv);
}

void CmiGrabBuffer()
{
  CpvAccess(CmiBufferGrabbed) = 1;
}

int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
  void *msg1, *msg2;
  int counter;
  
  while (1) {
    msg1 = CmiGetNonLocal();
    if (msg1) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg1))(msg1);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg1);
      maxmsgs--; if (maxmsgs==0) break;
    }
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg2);
    if (msg2) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg2))(msg2);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg2);
      maxmsgs--; if (maxmsgs==0) break;
    }
    if ((msg1==0)&&(msg2==0)) break;
  }
  return maxmsgs;
}

/*
 * CmiDeliverSpecificMsg(lang)
 *
 * - waits till a message with the specified handler is received,
 *   then delivers it.
 *
 */

void CmiDeliverSpecificMsg(handler)
int handler;
{
  int msgType;
  int *msg, *first ;
  
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg;
    do {
      if (CmiGetHandler(msg)==handler) {
	CpvAccess(CmiBufferGrabbed)=0;
	(CmiGetHandlerFunction(msg))(msg);
	if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
	return;
      } else {
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      }
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  /* receive message from network */
  while ( 1 ) { /* Loop till proper message is received */
    while ( (msg = CmiGetNonLocal()) == NULL )
        ;
    if ( CmiGetHandler(msg)==handler ) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg))(msg);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
      return;
    } else {
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  }
}

#endif /* CMK_USES_COMMON_CMIDELIVERS */

/***************************************************************************
 *
 * 
 ***************************************************************************/


CsdInit(char **argv)
{
  void *CqsCreate();

  CpvInitialize(int, disable_sys_msgs);
  CpvInitialize(int,   CmiHandlerCount);
  CpvInitialize(void*, CsdSchedQueue);
  CpvInitialize(int,   CsdStopFlag);
  
  CpvAccess(disable_sys_msgs) = 0;
  CpvAccess(CsdSchedQueue) = CqsCreate();
  CpvAccess(CsdStopFlag)  = 0;
}


int CsdScheduler(maxmsgs)
int maxmsgs;
{
  int *msg;
  
  CpvAccess(CsdStopFlag) = 0 ;
  
  while (1) {
    maxmsgs = CmiDeliverMsgs(maxmsgs);
    if (maxmsgs == 0) return maxmsgs;
    
    /* Check Scheduler queue */
    if ( !CqsEmpty(CpvAccess(CsdSchedQueue)) ) {
      CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
      (CmiGetHandlerFunction(msg))(msg);
      if (CpvAccess(CsdStopFlag)) return maxmsgs;
      maxmsgs--; if (maxmsgs==0) return maxmsgs;
    } else { /* Processor is idle */
      CcdRaiseCondition(CcdPROCESSORIDLE) ;
      if (CpvAccess(CsdStopFlag)) return maxmsgs;
    }
    
    if (!CpvAccess(disable_sys_msgs)) {
      if (CpvAccess(CcdNumChecks) > 0) {
	CcdCallBacks();
      }
    }
  }
}
 
/*****************************************************************************
 *
 * Initialization for the memory module.  Of course, there isn't any
 * memory-module right now.
 *
 *****************************************************************************/

static int CmemInit(argv)
    char **argv;
{
  int sysmem;
  /*
   * configure the chare kernel according to command line parameters.
   * by convention, chare kernel parameters begin with '+'.
   */
  while (*argv) {
    if (strcmp(*argv, "+m") == 0) {
      DeleteArg(argv);
      DeleteArg(argv);
    } else
    if (sscanf(*argv, "+m%d", &sysmem) == 1) {
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+mm") == 0) {
      DeleteArg(argv);
      DeleteArg(argv);
    } else
    if (sscanf(*argv, "+mm%d", &sysmem) == 1) {
      DeleteArg(argv);
      DeleteArg(argv);
    } else
    argv++;
  }
}


/*****************************************************************************
 *
 * Fast Interrupt Blocking Device
 *
 * This is totally portable.  Go figure.  (The rest of it is just macros,
 * see converse.h)
 *
 *****************************************************************************/

CpvDeclare(int,       CmiInterruptsBlocked);
CpvDeclare(CthVoidFn, CmiInterruptFuncSaved);

ConverseInit(argv)
char **argv;
{
  CstatsInit(argv);
  conv_condsModuleInit(argv);
  CmemInit(argv);
  CmiInit(argv);
  CsdInit(argv);
#ifdef CMK_CTHINIT_IS_IN_CONVERSEINIT
  CthInit(argv);
#endif
  CthSchedInit();
}

ConverseExit()
{
  CmiExit();
}

