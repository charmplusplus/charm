/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.17  1997-10-03 19:51:54  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 2.16  1997/04/25 20:48:14  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 2.15  1997/04/24 22:37:04  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 2.14  1997/03/19 04:31:41  jyelon
 * Redesigned ConverseInit
 *
 * Revision 2.13  1997/02/13 09:31:40  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 2.12  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 2.11  1996/04/18 22:40:35  sanjeev
 * CmiFreeSendFn uses CMMD_send_async
 *
 * Revision 2.8  1995/11/08 23:32:31  sanjeev
 * fixed bug in CmiFreeSendFn for msgs to myself
 *
 * Revision 2.7  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.6  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.5  1995/10/09  19:25:55  sanjeev
 * fixed bugs
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/20  16:00:46  gursoy
 * this time really made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.2  1995/09/20  15:59:06  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.1  1995/07/05  22:14:42  sanjeev
 * Megatest++ runs
 *
 * Revision 2.0  1995/06/15  20:14:47  sanjeev
 * *** empty log message ***
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include <stdio.h>
#include <fcntl.h>
#include <cm/cmmd.h>

/* #include <cm/cmmd-io.h>    not needed in CMMD 3.0 */

#include "converse.h"

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

typedef struct msg_list {
	void *msg ;
	struct msg_list *next ;
} MSG_LIST ;

int Cmi_dim /* used in spantree.c */  ;   

static int msglength=0 ;
static int numpes ;
static MSG_LIST *RecvMsgList = 0 ;

extern void * FIFO_Create() ;
extern void FIFO_EnQueue() ;

CpvDeclare(void *, CmiLocalQueue) ;
CpvDeclare(int, Cmi_mype) ;
CpvDeclare(int, Cmi_numpes) ;


/**************************  TIMER FUNCTIONS **************************/
#define TIMER_ID 1

double CmiWallTimer()
{
  double t ;
  
  CMMD_node_timer_stop(TIMER_ID) ;	
  t = CMMD_node_timer_busy(TIMER_ID) ;  /* returns time in sec */
  CMMD_node_timer_start(TIMER_ID) ;     /* restart immediately */	
  return t ;
}

double CmiCpuTimer()
{
  double t ;
  
  CMMD_node_timer_stop(TIMER_ID) ;	
  t = CMMD_node_timer_busy(TIMER_ID) ;  /* returns time in sec */
  CMMD_node_timer_start(TIMER_ID) ;     /* restart immediately */	
  return t ;
}

double CmiTimer()
{
  double t ;
  
  CMMD_node_timer_stop(TIMER_ID) ;	
  t = CMMD_node_timer_busy(TIMER_ID) ;  /* returns time in sec */
  CMMD_node_timer_start(TIMER_ID) ;     /* restart immediately */	
  return t ;
}

double CmiTimerInit()
{
  CMMD_node_timer_clear(TIMER_ID);
  CMMD_node_timer_start(TIMER_ID);
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
        void *env;
        int msglength;
        MSG_LIST *prev ;
        void *ret ;
        CMMD_mcb mcb1, mcb2 ;
 
        if ( RecvMsgList != 0 ) {
        /* msgs previously received in RecvMsgs but not given to system */
                ret = RecvMsgList->msg ;
                prev = RecvMsgList ;
                RecvMsgList = RecvMsgList->next ;
                CmiFree(prev) ;
                return ret ;
        }
 
        if ( !(mcb1 = CMMD_mcb_pending(CMMD_ANY_NODE,CMMD_ANY_TAG)) )
                return 0 ;
 
        msglength = CMMD_mcb_bytes(mcb1) ;
        CMMD_free_mcb(mcb1) ;
        env = (void *)  CmiAlloc(msglength);
        if (env == 0)
                CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
        mcb2 = CMMD_receive_async(CMMD_ANY_NODE, CMMD_ANY_TAG, env,
                                  msglength, 0, 0) ;
        while ( !CMMD_msg_done(mcb2) )
                ;
        CMMD_free_mcb(mcb2) ;
 
        return env ;
}


void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}
 
/********************* MESSAGE SEND INTERNAL FUNCTIONS ******************/
 
CpvExtern(int, CmiInterruptsBlocked) ;

int * SentMsgList = 0 ;

void FreeMsgs()
{
/* used only to free messages that didnt get freed in the SendHandler */

	int *q, *q2 ;

	q = SentMsgList ;
	while ( q != 0 ) {
		q2 = (int *)(*q) ;
		CmiFree(q) ;
		q = q2 ;
	}
	SentMsgList = 0 ;
}


void SendHandler(mcb, msg)
CMMD_mcb *mcb ;
void *msg ;
{
	int prevstate = CMMD_disable_interrupts() ;

	CMMD_free_mcb(*mcb) ;
	
/* msg is the buffer which contained the message */
	/* if ( ! CmiInsideMem ) {	For now CmiInsideMem is always 1 */
	if ( ! CpvAccess(CmiInterruptsBlocked) ) {
		CmiFree(msg) ;
		FreeMsgs() ;
	}
	else { /* put msg in list of messages that have to be freed */
		int *m = (int *)msg ;
		*m = (int)SentMsgList ;
		SentMsgList = msg ;
	}

	if ( prevstate )
		CMMD_enable_interrupts() ;
}



RecvMsgs()
{
/* Called to clear up network when CmiAsyncSend has run out of MCBs.
   Idea is that when this processor runs out of MCBs because other
   processors arent receiving, its likely to be a communication 
   intensive phase, so we should do some receives to free up MCBs
   on other processors. Also, there is no point just idling
   till some MCBs get free : we spend that time doing receives     */
   
/* Receive messages and put them into local (machine) queue */
/* These msgs are returned to the Charm system via CmiGetMsg later */
	void *env;
        int msglength;
	MSG_LIST *msgptr ;
        CMMD_mcb mcb1, mcb2 ;

        while ( (mcb1 = CMMD_mcb_pending(CMMD_ANY_NODE,CMMD_ANY_TAG)) != 0 ) {
 
                msglength = CMMD_mcb_bytes(mcb1) ;
                CMMD_free_mcb(mcb1) ;
                env = (void *)  CmiAlloc(msglength);
                if (env == 0)
                        CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
                mcb2 = CMMD_receive_async(CMMD_ANY_NODE, CMMD_ANY_TAG, env,
                                        msglength, 0, 0) ;
                while ( !CMMD_msg_done(mcb2) )
                        ;
                CMMD_free_mcb(mcb2) ;
 
                /* add env to the list of received but unprocessed msgs */
                msgptr = (MSG_LIST *)CmiAlloc(sizeof(MSG_LIST)) ;
                msgptr->next = RecvMsgList ;
                msgptr->msg = env ;
                RecvMsgList = msgptr ;
        }
}	



/********************* CONVERSE MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
	if ( destPE == CmiMyPe() ) {
		/* Make copy locally */
		char *m2 = (char *)CmiAlloc(size) ;
		memcpy(m2, msg, size) ;
		FIFO_EnQueue(CpvAccess(CmiLocalQueue), m2);
		
		return ;
	}

	CMMD_send_noblock(destPE, CMMD_DEFAULT_TAG, msg, size) ;
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
        CMMD_mcb mcb ;

	if ( destPE == CmiMyPe() ) {
		/* Make copy locally */
		char *m2 = (char *)CmiAlloc(size) ;
		memcpy(m2, msg, size) ;
		FIFO_EnQueue(CpvAccess(CmiLocalQueue), m2);
		
		return ((CmiCommHandle)CMMD_ERRVAL) ;
	}


trysend:
	mcb = CMMD_send_async(destPE, CMMD_DEFAULT_TAG, msg, size, 0, 0);

	if ( mcb == CMMD_ERRVAL ) {
		RecvMsgs() ;	/* try and receive msgs to unclog network */
		goto trysend ;
	}

	return ((CmiCommHandle)mcb) ;
}


void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
        CMMD_mcb mcb ;
 
        if ( destPE == CmiMyPe() ) {
                FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
                return ;
        }
 
/* Since we can free the msg, we do an asynchronous send, where the msg
   gets freed in the SendHandler interrupt handler */
 
        while ( 1 ) {
                mcb = CMMD_send_async(destPE, CMMD_DEFAULT_TAG, msg, size,
                                         SendHandler, msg) ;
/* the last arg to CMMD_send_async is the value passed to SendHandler
   as the 2nd arg, so the buffer "msg" gets freed after being sent */
 
                if ( mcb != CMMD_ERRVAL )
                        return ;
                else
                        RecvMsgs() ;
        }
}



/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcastFn(size, msg)     /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
	int i ;

	for ( i=CmiMyPe()+1; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	for ( i=0; i<CmiMyPe(); i++ ) 
	        CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;

}

CmiCommHandle CmiAsyncBroadcastFn(size, msg) /*same as SyncBroadcast for now*/
int size;
char * msg;
{
	int i ;

	for ( i=CmiMyPe()+1; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	for ( i=0; i<CmiMyPe(); i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	return((CmiCommHandle)CMMD_ERRVAL) ;
}

void CmiFreeBroadcastFn(size, msg)
int size;
char *msg;
{
    CmiSyncBroadcastFn(size, msg);
    CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	return((CmiCommHandle)CMMD_ERRVAL) ;
}


void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	CmiFree(msg) ;
}

/******************* COMM HANDLE FUNCTIONS ***************************/

int CmiAsyncMsgSent(CmiCommHandle c)
{
	if ( c == (CmiCommHandle)CMMD_ERRVAL )
		return 1 ;
	if ( CMMD_msg_done((int)c) ) 
		return 1 ;
	else
		return 0 ;
}


void CmiReleaseCommHandle(CmiCommHandle c)
{
	if ( c == (CmiCommHandle)CMMD_ERRVAL )
		return ;
	CMMD_free_mcb((int)c) ;
}


/**********************  BACKWARD COMPATIBILITY FNS **********************/

/* Neighbour functions used mainly in LDB : pretend the CM5 is a hypercube */

int CmiNumNeighbours(node)
int node;
{
    return Cmi_dim;
}


void CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
    int i;

    for (i = 0; i < Cmi_dim; i++)
        neighbours[i] = FLIPBIT(node,i);
}


int CmiNeighboursIndex(node, neighbour)
int node, neighbour;
{
    int index = 0;
    int linenum = node ^ neighbour;

    while (linenum > 1)
    {
        linenum = linenum >> 1;
        index++;
    }
    return index;
}




/************************** MAIN ***********************************/

void ConverseExit()
{
  ConverseCommonExit();
  exit(0);
}

void ConverseInit(argc, argv, fn, usched, initret)
int argc;
char **argv;
CmiStartFn fn;
int usched, initret;
{
  int n, i, j ;
  
  CMMD_fset_io_mode(stdin, CMMD_independent) ;
  CMMD_fset_io_mode(stdout, CMMD_independent) ;
  CMMD_fset_io_mode(stderr, CMMD_independent) ;
  
  fcntl(fileno(stdout), F_SETFL, O_APPEND) ;
  fcntl(fileno(stderr), F_SETFL, O_APPEND) ;
  
  CpvAccess(Cmi_mype) = CMMD_self_address() ;
  
  CpvAccess(Cmi_numpes) = numpes = CMMD_partition_size() ;
  
  if ( argc >= 2 ) { /* Check if theres a +p #procs */
    for ( i=1; i<argc; i++ ) {
      if ( strncmp(argv[i], "+p", 2) != 0) 
	continue ;
      if ( strlen(argv[i]) > 2 ) {
	CpvAccess(Cmi_numpes) = numpes = atoi(&(argv[i][2])) ;
	for ( j=i; j<argc-1; j++ )
	  argv[j] = argv[j+1] ;
	argc-- ;
      }
      else {
	CpvAccess(Cmi_numpes) = numpes = atoi(argv[i+1]) ;
	for ( j=i; j<argc-2; j++ )
	  argv[j] = argv[j+2] ;
	argc -= 2 ;
      }
      break ;
    }
  }
  
  if (CpvAccess(Cmi_mype) >= CpvAccess(Cmi_numpes))
    exit(0) ;
  CmiSpanTreeInit(); 
  CpvAccess(CmiLocalQueue) = FIFO_Create() ;
  
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=numpes-1; n>=1; n/=2 )
    Cmi_dim++ ;
  
  /* Initialize timers */
  CmiTimerInit();
  ConverseCommonInit(argv);
  CthInit(argv);
  if (initret==0) {
    fn(argc, argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

/*****************************************************************************
 *
 * CmiAlloc, CmiSize, and CmiFree
 *
 *****************************************************************************/
 
 
void *CmiAlloc(size)
int size;
{
	char *res;
	res =(char *)malloc(size+8);
	if (res==0) printf("Memory allocation failed.");
	((int *)res)[0]=size;
	return (void *)(res+8);
}
 
int CmiSize(blk)
void *blk;
{
	return ( ((int *)((char *)blk - 8))[0] );
}
 
void CmiFree(blk)
void *blk;
{
	free( ((char *)blk) - 8);
}

