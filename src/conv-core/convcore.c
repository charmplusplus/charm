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
 * The following are the CmiDeliverXXX functions.  A common implementation
 * is provided below.  The machine layer can provide an alternate 
 * implementation if it so desires.
 *
 * void CmiDeliversInit()
 *
 *      - CmiInit promises to call this before calling CmiDeliverMsgs
 *        or any of the other functions in this section.
 *
 * int CmiDeliverMsgs(int maxmsgs)
 *
 *      - CmiDeliverMsgs will retrieve up to maxmsgs that were transmitted
 *        with the Cmi, and will invoke their handlers.  It does not wait
 *        if no message is unavailable.  Instead, it returns the quantity
 *        (maxmsgs-delivered), where delivered is the number of messages it
 *        delivered.
 *
 * void CmiDeliverSpecificMsg(int handlerno)
 *
 *      - Waits for a message with the specified handler to show up, then
 *        invokes the message's handler.  Note that unlike CmiDeliverMsgs,
 *        This function _does_ wait.
 *
 * void CmiGrabBuffer(void **bufptrptr)
 *
 *      - When CmiDeliverMsgs or CmiDeliverSpecificMsgs calls a handler,
 *        the handler receives a pointer to a buffer containing the message.
 *        The buffer does not belong to the handler, eg, the handler may not
 *        free the buffer.  Instead, the buffer will be automatically reused
 *        or freed as soon as the handler returns.  If the handler wishes to
 *        keep a copy of the data after the handler returns, it may do so by
 *        calling CmiGrabBuffer and passing it a pointer to a variable which
 *        in turn contains a pointer to the system buffer.  The variable will
 *        be updated to contain a pointer to a handler-owned buffer containing
 *        the same data as before.  The handler then has the responsibility of
 *        making sure the buffer eventually gets freed.  Example:
 *
 * void myhandler(void *msg)
 * {
 *    CmiGrabBuffer(&msg);      // Claim ownership of the message buffer
 *    ... rest of handler ...
 *    CmiFree(msg);             // I have the right to free it or
 *                              // keep it, as I wish.
 * }
 *
 *
 * For this common implementation to work, the machine layer must provide the
 * following:
 *
 * void *CmiGetNonLocal()
 *
 *      - returns a message just retrieved from some other PE, not from
 *        local.  If no such message exists, returns 0.
 *
 * CpvExtern(FIFO_Queue, CmiLocalQueue);
 *
 *      - a FIFO queue containing all messages from the local processor.
 *
 *****************************************************************************/

#ifdef CMK_USES_COMMON_CMIDELIVERS

CpvStaticDeclare(int, CmiBufferGrabbed);
CpvExtern(void*, CmiLocalQueue);

void CmiDeliversInit()
{
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
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


CsdInit(argv)
  char **argv;
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
 * Fast Interrupt Blocking Device
 *
 * This is totally portable.  Go figure.  (The rest of it is just macros,
 * see converse.h)
 *
 *****************************************************************************/

CpvDeclare(int,       CmiInterruptsBlocked);
CpvDeclare(CthVoidFn, CmiInterruptFuncSaved);

/*****************************************************************************
 *
 * ConverseInit and ConverseExit
 *
 *****************************************************************************/

ConverseInit(argv)
char **argv;
{
  CstatsInit(argv);
  conv_condsModuleInit(argv);
  CmiHandlerInit();
  CmiMemoryInit(argv);
  CmiDeliversInit();
  CmiSpanTreeInit(argv);
  CmiInitMc(argv);
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

