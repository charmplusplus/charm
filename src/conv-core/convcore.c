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



/*****************************************************************************
 *
 * Statistics: currently, the following statistics are not updated by converse.
 *
 *****************************************************************************/

int CstatsMaxChareQueueLength=0;
int CstatsMaxForChareQueueLength=0;
int CstatsMaxFixedChareQueueLength=0;

int CstatMemory(int i)
{
  return 0;
}

static int CstatPrintQueueStatsFlag;
int CstatPrintQueueStats()
{
  return CstatPrintQueueStatsFlag;
}

static int CstatPrintMemStatsFlag;
int CstatPrintMemStats()
{
  return CstatPrintMemStatsFlag;
}

/*****************************************************************************
 *
 * The following are some CMI functions.  Having this code here makes
 * the module boundaries for the Cmi quite blurry.
 *
 *****************************************************************************/

#define MAX_HANDLERS 512
CmiHandler *CmiHandlerTable;
extern void *CmiLocalQueue;
void        *CmiGetNonLocal();

void CmiInit(argv)
char **argv;
{
  void *FIFO_Create();
  CmiHandlerTable =
    (CmiHandler *)CmiAlloc((MAX_HANDLERS + 1) * sizeof(CmiHandler)) ;
  CmiInitMc(argv);
}

void *CmiGetMsg()
{
  void *msg;
  if (!FIFO_Empty(CmiLocalQueue)) {
    FIFO_DeQueue(CmiLocalQueue, &msg);
    return msg;
  }
  msg=CmiGetNonLocal();
  return msg;
}

int CmiRegisterHandler(handlerf)
CmiHandler handlerf ;
{
  static int handlerCount=0 ;

  CmiHandlerTable[handlerCount] = handlerf;
  handlerCount++ ;
  return handlerCount-1 ;
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
  
  if ( !FIFO_Empty(CmiLocalQueue) ) {
    FIFO_DeQueue(CmiLocalQueue, &msg);
    first = msg;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	return (void *)msg ;
      else
	FIFO_EnQueue(CmiLocalQueue, msg);
      
      FIFO_DeQueue(CmiLocalQueue, &msg);
    } while ( msg != first ) ;
    
    FIFO_EnQueue(CmiLocalQueue, msg);
  }
  
  /* receive message from network */
  while ( 1 ) { /* Loop till proper message is received */
    
    while ( (msg = CmiGetNonLocal()) == NULL )
      ;
    
    if ( CmiGetHandler(msg)==lang ) 
      return (void *)msg ;
    else
      FIFO_EnQueue(CmiLocalQueue, msg);
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
  if ( !FIFO_Empty(CmiLocalQueue) ) {
    
    FIFO_DeQueue(CmiLocalQueue, &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	{
	  (CmiGetHandlerFunction(msg))(msg);
	  retval = 1;
	}
      else
	FIFO_EnQueue(CmiLocalQueue, msg);
      FIFO_DeQueue(CmiLocalQueue, &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CmiLocalQueue, msg);
  }
  
  while ( (msg = CmiGetNonLocal()) != NULL )
    if ( CmiGetHandler(msg)==lang ) 
      {
	(CmiGetHandlerFunction(msg))(msg);
	retval = 1;
      }
    else
      FIFO_EnQueue(CmiLocalQueue, msg);
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
  if ( !FIFO_Empty(CmiLocalQueue) ) {
    
    FIFO_DeQueue(CmiLocalQueue, &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	  retval = 1;
      FIFO_EnQueue(CmiLocalQueue, msg);
      FIFO_DeQueue(CmiLocalQueue, &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CmiLocalQueue, msg);
  }
  while ( (msg = CmiGetNonLocal()) != NULL )
    {
      if ( CmiGetHandler(msg)==lang ) 
	retval = 1;
      FIFO_EnQueue(CmiLocalQueue, msg);
    }
  return retval;
}

/***************************************************************************
 *
 * 
 ***************************************************************************/

void *CsdSchedQueue;
int   CsdStopFlag=0;

CsdInit(char **argv)
{
  void *CqsCreate();
  CsdSchedQueue = CqsCreate();
}

void *CsdGetMsg()
{
  int *msg ;
  
  if ((msg = CmiGetMsg()) != NULL)
    return msg;
  
  if ( !CqsEmpty(CsdSchedQueue) ) {
    CqsDequeue(CsdSchedQueue,&msg);
    return msg ;
  }
  
  return NULL ;
}

void CsdScheduler(counter)
int counter;
{
  int *msg;
  
  while (1) {
    msg = CsdGetMsg();
    if (msg)
      (CmiGetHandlerFunction(msg))(msg);
    if (CsdStopFlag) break;
    counter--;
    if (counter==0) break;
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
      CstatPrintMemStatsFlag=1;
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+qs") == 0) {
      CstatPrintQueueStatsFlag=1;
      DeleteArg(argv);
    } else
    argv++;
  }
}

ConverseInit(argv)
char **argv;
{
  ConverseParseOptions(argv);
  CmiInit(argv);
  CsdInit(argv);
}

ConverseExit()
{
  CmiExit();
}
