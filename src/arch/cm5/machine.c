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
 * Revision 2.3  1995-09-20 16:00:46  gursoy
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

#include "machine.h" 
#include "converse.h"

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

typedef struct msg_list {
	void *msg ;
	struct msg_list *next ;
} MSG_LIST ;

int Cmi_dim /* used in spantree.c */  ;   


static int msglength=0 ;
static int numpes ;
static void * SentMsgList = NULL ;
static MSG_LIST *RecvMsgList = NULL ;

extern void * FIFO_Create() ;
extern void FIFO_EnQueue() ;

static void McSyncReceive() ;
static int McArrivedMsgLength() ;
static int McProbe() ;

void *CmiLocalQueue ;
int Cmi_mype ;
int Cmi_numpe ;


/**************************  TIMER FUNCTIONS **************************/
#define TIMER_ID 1


double CmiTimer()
{
	double t ;

	CMMD_node_timer_stop(TIMER_ID) ;	
	t = CMMD_node_timer_busy(TIMER_ID) ;  /* returns time in sec */
	CMMD_node_timer_start(TIMER_ID) ;     /* restart immediately */	
	
	return t ;
}

/* these 3 timer calls are for backward compatibility */
unsigned int McTimer()
{
	double t ;
	unsigned int tmsec ;

	CMMD_node_timer_stop(TIMER_ID) ;	
	t = CMMD_node_timer_busy(TIMER_ID) ;  /* returns time in sec */
	CMMD_node_timer_start(TIMER_ID) ;     /* restart immediately */	

	tmsec = (int)(t * 1000.0) ;
	
	return(tmsec) ;
}


unsigned int McUTimer()
{
	double t ;
	unsigned int tusec ;

	CMMD_node_timer_stop(TIMER_ID) ;	
	t = CMMD_node_timer_busy(TIMER_ID) ;
	CMMD_node_timer_start(TIMER_ID) ;	

	tusec = (int)(t * 1000000.0) ;
	
	return(tusec) ;
}


McHTimer()
{
	double t ;
	unsigned int th ;

	CMMD_node_timer_stop(TIMER_ID) ;	
	t = CMMD_node_timer_busy(TIMER_ID) ;
	CMMD_node_timer_start(TIMER_ID) ;	

	th = (int)(t / 3600.0) ;
	
	return(th) ;
}



/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
        void *env;
        int msglength;
	MSG_LIST *prev ;
	void *ret ;

	if ( RecvMsgList != NULL ) {
	/* msgs previously received in RecvMsgs but not given to system */
		ret = RecvMsgList->msg ;
		prev = RecvMsgList ;
		RecvMsgList = RecvMsgList->next ;
		CmiFree(prev) ;
		return ret ;
	}	

        if (!McProbe())
                return NULL;
        msglength = McArrivedMsgLength();
        env = (void *)  CmiAlloc(msglength);
        if (env == NULL)
                CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
        McSyncReceive(msglength, env);
        return env;
}

void CmiGrabBuffer(pbuf)
void **pbuf ;
{
}

/* Next 3 fns are internal fns */
static int McProbe()
{
	CMMD_mcb mcb1 ;

        if ( !(mcb1 = CMMD_mcb_pending(CMMD_ANY_NODE,CMMD_ANY_TAG)) )
                return(0) ;

        msglength = CMMD_mcb_bytes( mcb1 ) ;
        CMMD_free_mcb(mcb1) ;
	return(1) ;
}

static int McArrivedMsgLength()
{
        return msglength;
}

static void McSyncReceive(size, buffer)
int size ;
void *buffer ;
{
/* Receive a message, dont return till the full message has been
   received */
	CMMD_mcb mcb2 ;

	mcb2 = CMMD_receive_async(CMMD_ANY_NODE, CMMD_ANY_TAG, buffer,
				  size, NULL, 0) ;
	while ( !CMMD_msg_done(mcb2) )
		;
	CMMD_free_mcb(mcb2) ;
}



/********************* MESSAGE SEND FUNCTIONS ******************/

/***************** CmiAsyncSendFree not used for now ***************
static int CmiInsideMem = 1;

* Next 3 fns are internal fns *
static SendHandler(mcb, msg)
CMMD_mcb *mcb ;
void *msg ;
{
	CMMD_free_mcb(*mcb) ;
	
* msg is the buffer which contained the message *
	if ( ! CmiInsideMem ) {	* For now CmiInsideMem is always TRUE *
		CmiFree(msg) ;
		release_messages() ;
	}
	else { * put msg in list of messages that have to be freed *
		((MSG_LIST *)msg)->next = (MSG_LIST *)SentMsgList ;
		SentMsgList = msg ;
	}
}


static release_messages()
{
* used only to free messages that didnt get freed in the SendHandler *

	MSG_LIST *q, *q2 ;

	q = (MSG_LIST *)SentMsgList ;
	while ( q != NULL ) {
		q2 = q->next ;
		CmiFree(q) ;
		q = q2 ;
	}
	SentMsgList = NULL ;
}


void CmiAsyncSendFree(destPE, size, msg)
int destPE;
int size;
void * msg;
{
	CMMD_mcb mcb ;		

	if ( destPE == CmiMyPe() ) {
		FIFO_EnQueue(CmiLocalQueue, msg);
		return ;
	}

trysend:
	mcb = CMMD_send_async(destPE, CMMD_DEFAULT_TAG, msg, size,
				SendHandler, msg) ;
* the last arg to CMMD_send_async is the value passed to SendHandler
   as the 2nd arg, so the buffer "msg" gets freed after being sent *

	if ( mcb == CMMD_ERRVAL ) {
		RecvMsgs() ;	* try and receive msgs to unclog network *
		goto trysend ;
	}
}
***********************************************************************/



static RecvMsgs()
{
/* Called to clear up network when McAsyncSend has run out of MCBs.
   Idea is that when this processor runs out of MCBs because other
   processors arent receiving, its likely to be a communication 
   intensive phase, so we should do some receives to free up MCBs
   on other processors. Also, there is no point just idling
   till some MCBs get free : we spend that time doing receives     */
   
/* Receive messages and put them into local (machine) queue */
/* These msgs are returned to the Charm system via McGetMsg later */
	void *env;
        int msglength;
	MSG_LIST *msgptr ;

	while ( McProbe() ) {
		msglength = McArrivedMsgLength();
        	env = (void *) CmiAlloc(msglength);
        	if (env == NULL)
                	CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
        	McSyncReceive(msglength, env);

		/* add env to the list of received but unprocessed msgs */
		msgptr = (MSG_LIST *)CmiAlloc(sizeof(MSG_LIST)) ;
		msgptr->next = RecvMsgList ;
		msgptr->msg = env ;
		RecvMsgList = msgptr ;
	}
}	



CmiCommHandle CmiAsyncSend(destPE, size, msg)
int destPE;
int size;
void * msg;
{
        CMMD_mcb mcb ;

	if ( destPE == CmiMyPe() ) {
		/* Make copy locally */
		char *m2 = (char *)CmiAlloc(size) ;
		memcpy(m2, msg, size) ;
		FIFO_EnQueue(CmiLocalQueue, m2);
		
		return ((CmiCommHandle)CMMD_ERRVAL) ;
	}


trysend:
	mcb = CMMD_send_async(destPE, CMMD_DEFAULT_TAG, msg, size,
				NULL, NULL) ;

	if ( mcb == CMMD_ERRVAL ) {
		RecvMsgs() ;	/* try and receive msgs to unclog network */
		goto trysend ;
	}

	return ((CmiCommHandle)mcb) ;
}

void CmiSyncSend(destPE, size, msg)
int destPE;
int size;
void * msg;
{
	if ( destPE == CmiMyPe() ) {
		/* Make copy locally */
		char *m2 = (char *)CmiAlloc(size) ;
		memcpy(m2, msg, size) ;
		FIFO_EnQueue(CmiLocalQueue, m2);
		
		return ;
	}

	CMMD_send_noblock(destPE, CMMD_DEFAULT_TAG, msg, size) ;
}


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


/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcast(size, msg)     /* ALL_EXCEPT_ME  */
int size;
void * msg;
{
	int i ;

	for ( i=CmiMyPe()+1; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	for ( i=0; i<CmiMyPe(); i++ ) 
	        CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;

}

void CmiSyncBroadcastAllAndFree(size, msg)
int size;
void * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	CmiFree(msg) ;
}


void CmiSyncBroadcastAll(size, msg)
int size;
void * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
}


CmiCommHandle CmiAsyncBroadcastAll(size, msg)
int size;
void * msg;
{
	int i ;

	for ( i=0; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	return((CmiCommHandle)CMMD_ERRVAL) ;
}


CmiCommHandle CmiAsyncBroadcast(size, msg)  /* same as SyncBroadcast for now */
int size;
void * msg;
{
	int i ;

	for ( i=CmiMyPe()+1; i<numpes; i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	for ( i=0; i<CmiMyPe(); i++ ) 
                CMMD_send_noblock(i, CMMD_DEFAULT_TAG, msg, size) ;
	return((CmiCommHandle)CMMD_ERRVAL) ;
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

main(argc, argv)
int argc;
char **argv;
{
	int n, i, j ;

	/* program_name(argv[0], "CM5");  cant be called from converse */

        CMMD_fset_io_mode(stdin, CMMD_independent) ;
	CMMD_fset_io_mode(stdout, CMMD_independent) ;
	CMMD_fset_io_mode(stderr, CMMD_independent) ;

        fcntl(fileno(stdout), F_SETFL, O_APPEND) ;
        fcntl(fileno(stderr), F_SETFL, O_APPEND) ;

	Cmi_mype = CMMD_self_address() ;

	if ( argc < 2 ) 
		Cmi_numpe = numpes = CMMD_partition_size() ;
	else { /* Check if theres a +p #procs */
		for ( i=1; i<argc; i++ ) {
			if ( strncmp(argv[i], "+p", 2) != 0) 
				continue ;
			if ( strlen(argv[i]) > 2 ) {
				Cmi_numpe = numpes = atoi(&(argv[i][2])) ;
				for ( j=i; j<argc-1; j++ )
					argv[j] = argv[j+1] ;
				argc-- ;
			}
			else {
				Cmi_numpe = numpes = atoi(argv[i+1]) ;
				for ( j=i; j<argc-2; j++ )
					argv[j] = argv[j+2] ;
				argc -= 2 ;
			}
			break ;
		}
	}

	if ( Cmi_mype >= Cmi_numpe )
		exit(0) ;

	/* find dim = log2(numpes), to pretend we are a hypercube */
	for ( Cmi_dim=0,n=numpes; n>1; n/=2 )
		Cmi_dim++ ;

	/* Initialize timers */
	CMMD_node_timer_clear(TIMER_ID) ;
	CMMD_node_timer_start(TIMER_ID) ;

	charm_main(argc, argv);
}



void CmiInitMc(argv)
char *argv[];
{
	CmiSpanTreeInit(); 
 
	CmiLocalQueue = FIFO_Create() ;
}


void CmiExit()
{}

void CmiDeclareArgs()
{}



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
	return ((int *)(  ((char*blk) - 8) )[0];
}
 
void CmiFree(blk)
void *blk;
{
	free( ((char *)blk) - 8);
}

