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
CsvDeclare(CmiHandler*, CmiHandlerTable);
CpvExtern(void*, CmiLocalQueue);
CpvExtern(int, CcdNumChecks) ;
CpvStaticDeclare(int, handlerCount);
CpvDeclare(void*, CsdSchedQueue);
CpvDeclare(int,   CsdStopFlag);




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




void convcoreModuleInit()
{
     CpvInitialize(int, disable_sys_msgs);
     CpvAccess(disable_sys_msgs) = 0;

     CpvInitialize(int, CstatsMaxChareQueueLength);
     CpvInitialize(int, CstatsMaxForChareQueueLength);
     CpvInitialize(int, CstatsMaxFixedChareQueueLength);
     CpvInitialize(int, CstatPrintQueueStatsFlag);
     CpvInitialize(int, CstatPrintMemStatsFlag);
     CpvInitialize(int, handlerCount);
     CpvInitialize(void*, CsdSchedQueue);
     CpvInitialize(int,   CsdStopFlag);

     CpvAccess(CstatsMaxChareQueueLength) = 0;
     CpvAccess(CstatsMaxForChareQueueLength) = 0;
     CpvAccess(CstatsMaxFixedChareQueueLength) = 0;
     CpvAccess(handlerCount) = 0;
     CpvAccess(CsdStopFlag)  = 0;
}




int CstatMemory(int i)
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
 * The following are some CMI functions.  Having this code here makes
 * the module boundaries for the Cmi quite blurry.
 *
 *****************************************************************************/


void CmiInit(argv)
char **argv;
{
  void *FIFO_Create();

  if (CmiMyRank() != 0) CmiNodeBarrier();


  if (CmiMyRank() == 0) 
  {
     CsvAccess(CmiHandlerTable) =
        (CmiHandler *)CmiSvAlloc((MAX_HANDLERS + 1) * sizeof(CmiHandler)) ;
     CmiNodeBarrier();
  }
  CmiInitMc(argv);
  CmiSpanTreeInit(argv);
}

void *CmiGetMsg()
{
  void *msg;
  if (!FIFO_Empty(CpvAccess(CmiLocalQueue))) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    return msg;
  }
  msg=CmiGetNonLocal();
  return msg;
}

int CmiRegisterHandler(handlerf)
CmiHandler handlerf ;
{
  CsvAccess(CmiHandlerTable)[CpvAccess(handlerCount)] = handlerf;
  CpvAccess(handlerCount)++ ;
  return CpvAccess(handlerCount)-1 ;
}

void *CmiGetSpecificMsg(lang)
int lang;
{
/* loop till a message for lang is recvd, then return it */
  int msgType;
  int *msg, *first ;
  
  /* the LocalQueue-FIFO can have only thoses msgs enqueued at the
     FIFO_EnQueues in this function */
  /* We assume that there can be no self-messages (messages from my
     processor to myself) for the PVM/MPI msg manager
     (ie. the PVM msg manager checks for a self message while sending
     and keeps it within itself instead of giving it to Converse)  */
  
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	return (void *)msg ;
      else
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  /* receive message from network */
  while ( 1 ) { /* Loop till proper message is received */
    
    while ( (msg = CmiGetNonLocal()) == NULL )
      ;
    
    if ( CmiGetHandler(msg)==lang ) 
      return (void *)msg ;
    else
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  return NULL ;
}


/* This function gets all outstanding messages out of the network, executing
   their handlers if they are for this lang, else enqueing them in the FIFO 
   queue */
int
CmiClearNetworkAndCallHandlers(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	{
	  (CmiGetHandlerFunction(msg))(msg);
	  retval = 1;
	}
      else
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  while ( (msg = CmiGetNonLocal()) != NULL )
    if ( CmiGetHandler(msg)==lang ) 
      {
	(CmiGetHandlerFunction(msg))(msg);
	retval = 1;
      }
    else
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  return retval;
}

/* 
  Same as above function except that it does not execute any handler functions
*/
int
CmiClearNetwork(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	  retval = 1;
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  while ( (msg = CmiGetNonLocal()) != NULL )
    {
      if ( CmiGetHandler(msg)==lang ) 
	retval = 1;
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  return retval;
}

/***************************************************************************
 *
 * 
 ***************************************************************************/


CsdInit(char **argv)
{
  void *CqsCreate();
  CpvAccess(CsdSchedQueue) = CqsCreate();
}


void CsdScheduler(counter)
int counter;
{
	int *msg;

	CpvAccess(CsdStopFlag) = 0 ;
  
	while (1) {
		/* This is CmiDeliverMsgs */
		while ( (msg = CmiGetMsg()) != NULL ) {
			(CmiGetHandlerFunction(msg))(msg);

			if (CpvAccess(CsdStopFlag)) return;
			counter--;
			if (counter==0) return;
		}

		/* Check Scheduler queue */
  		if ( !CqsEmpty(CpvAccess(CsdSchedQueue)) ) {
    			CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
			(CmiGetHandlerFunction(msg))(msg);

			if (CpvAccess(CsdStopFlag)) return;
			counter--;
			if (counter==0) return;
  		}
		else { /* Processor is idle */
			CcdRaiseCondition(CcdPROCESSORIDLE) ;
			if (CpvAccess(CsdStopFlag)) return;
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
 * Convenient initializer/deinitializer for "the whole shebang"
 *
 * Note: This module cannot parse arguments, in the long run.  Each
 * independent module must parse its own arguments.
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

static int ConverseParseOptions(argv)
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
    if (strcmp(*argv, "+mems") == 0) {
      CpvAccess(CstatPrintMemStatsFlag)=1;
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+qs") == 0) {
      CpvAccess(CstatPrintQueueStatsFlag)=1;
      DeleteArg(argv);
    } else
    argv++;
  }
}

ConverseInit(argv)
char **argv;
{
  convcoreModuleInit();
  conv_condsModuleInit() ;

  ConverseParseOptions(argv);
  CmiInit(argv);
  CsdInit(argv);
  CthInit();
}

ConverseExit()
{
  CmiExit();
}
