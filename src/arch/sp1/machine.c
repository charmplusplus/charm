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
 * Revision 2.1  1995-07-17 17:46:05  knauff
 * Fixed problem with machine.c
 *
 * Revision 2.0  1995/07/10  22:12:39  knauff
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <sys/time.h>

#include "machine.h" 
#include "converse.h"

#include <mpproto.h>

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

int Cmi_mype;
int Cmi_numpe;
int Cmi_dim;

int Cmi_maxpenum ;   

void *CmiLocalQueue;

typedef struct msg_list {
     int msgid;
     char *msg;
     struct msg_list *next;
} MSG_LIST;

static int msglength=0 ;
static int numpes ;
static double itime;

static MSG_LIST *sent_msgs=NULL;
static MSG_LIST *end_sent=NULL;


/**************************  TIMER FUNCTIONS **************************/

static void CmiTimerInit()
{
    struct timestruc_t time;

    gettimer(TIMEOFDAY,&time);
    itime=(double)time.tv_sec + 1.0e-9*((double) time.tv_nsec);
}

CmiUTimerInit()
{
/* Nothing, since we use just one timer for both usec and msec. */
}


double CmiTimer()
{
    double tmsec;
    double t;
    struct timestruc_t time;

    gettimer(TIMEOFDAY,&time);
    t=(double)time.tv_sec + 1.0e-9*((double) time.tv_nsec);
    tmsec = (double) (1.0e3*(t-itime));
    return tmsec / 1000.0;
}


/* These functions are for backward compatibility... */
double CmiUTimer()
{
    unsigned int tusec;
    double t;
    struct timestruc_t time;

    gettimer(TIMEOFDAY,&time);
    t=(double)time.tv_sec + 1.0e-9*((double) time.tv_nsec);
    tusec = (unsigned int) (1.0e6*(t-itime));
    return (double) tusec;
}


double CmiHTimer()
{
    unsigned int thr;
    double t;
    struct timestruc_t time;

    gettimer(TIMEOFDAY,&time);
    t=(double)time.tv_sec + 1.0e-9*((double) time.tv_nsec);
    thr = (unsigned int) ((t-itime)/3600.0);
    return (double) thr;
}


/********************* MESSAGE SEND FUNCTIONS ******************/

CmiCommHandle CmiAsyncSend(destPE, size, msg)
     int destPE;
     int size;
     char * msg;
{
     MSG_LIST *msg_tmp;
     int msgid;
     
     /* Send Async message and add msgid to sent msg list */
     mpc_send(msg, size, destPE, 0, &msgid);
     msg_tmp = (MSG_LIST *) CmiAlloc(sizeof(MSG_LIST));
     msg_tmp->msgid = msgid;
     msg_tmp->msg = msg;
     msg_tmp->next = NULL;
     if(sent_msgs==NULL)
	  sent_msgs = msg_tmp;
     else
	  end_sent->next = msg_tmp;
     end_sent = msg_tmp;
}


/* Used only from broadcast for now */
CmiSyncSend(destPE, size, msg)
     int destPE;
     int size;
     char * msg;
{
     mpc_bsend(msg, size, destPE, 0);
}


CmiAllAsyncMsgsSent()
{
     MSG_LIST *msg_tmp = sent_msgs;
     
     while(msg_tmp!=NULL)
     {
	  if(mpc_status(msg_tmp->msgid)<0)
	       return 0;
	  msg_tmp = msg_tmp->next;
     }
     return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {
     
     MSG_LIST *msg_tmp = sent_msgs;

     while ((msg_tmp) && ((CmiCommHandle)msg_tmp->msgid != c))
	  msg_tmp = msg_tmp->next;
     
     if ((msg_tmp) && (mpc_status(msg_tmp->msgid)<0))
	  return 0;
     else
	  return 1;

}

void CmiReleaseCommHandle(CmiCommHandle c)
{
}


CmiReleaseSentMessages()
{
     MSG_LIST *msg_tmp=sent_msgs;
     MSG_LIST *prev=NULL;
     MSG_LIST *temp;
     
     if(sent_msgs==NULL)
	  return;
     while(msg_tmp!=NULL)
     {
	  if(mpc_status(msg_tmp->msgid)>=0)
	  {
	       /* Release the message */
	       temp = msg_tmp->next;
	       if(prev==NULL)	/* first message */
		    sent_msgs = temp;
	       else
		    prev->next = temp;
	       CmiFree(msg_tmp->msg);
	       CmiFree(msg_tmp);
	       msg_tmp = temp;
	  }
	  else
	  {
	       prev = msg_tmp;
	       msg_tmp = msg_tmp->next;
	  }
     }
     end_sent = msg_tmp;
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
     void *env;
     int msglength;
     
     if (!McProbe())
	  return NULL;
     msglength = McArrivedMsgLength();
     env = (void *)  CmiAlloc(msglength);
     if (env == NULL)
     {
	  CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
	  fflush(stdout);
     }
     McSyncReceive(msglength, env);
     return env;
}

/*
 * mp_probe(src, type, nbytes)
 */
extern mp_probe();	/* Fortran interface. In EUI-H, not in EUI */

/* Internal functions */
McProbe()
{
     int src, type, nbytes;
     
     src = DONTCARE;
     type = 0;
     mp_probe(&src, &type, &nbytes);
     if (nbytes < 0)
	  return 0;
     msglength = nbytes;
     return(1) ;
}

McArrivedMsgLength()
{
     return msglength;
}

McSyncReceive(size, buffer)
     int size ;
     char *buffer ;
{
     int src, type, nbytes;
     
     src = DONTCARE;
     type = 0;
     mpc_brecv(buffer, size, &src, &type, &nbytes);
}

void CmiGrabBuffer(pbuf)
void **pbuf ;
{
}

/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcast(size, msg)     /* ALL_EXCEPT_ME  */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=Cmi_mype+1; i<numpes; i++ ) 
	  CmiSyncSend(i, size,msg) ;
     for ( i=0; i<Cmi_mype; i++ ) 
	  CmiSyncSend(i, size,msg) ;
}


CmiSyncBroadcastAllAndFree(size, msg)  /* All including me */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=0; i<numpes; i++ ) 
	  CmiSyncSend(i,size,msg,0) ;
     CmiFree(msg) ;
}


CmiSyncBroadcastAll(size, msg)        /* All including me */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=0; i<numpes; i++ ) 
	  CmiSyncSend(i,size,msg,0) ;
}



CmiCommHandle CmiAsyncBroadcast(size, msg)  
int size;
char * msg;
{
	int i ;

	for ( i=Cmi_mype+1; i<numpes; i++ ) 
		CmiAsyncSend(i,size,msg) ;
	for ( i=0; i<Cmi_mype; i++ ) 
		CmiAsyncSend(i,size,msg) ;
	return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}


CmiCommHandle CmiAsyncBroadcastALL(size, msg)  
int size;
char * msg;
{
	int i ;

	for ( i=1; i<numpes; i++ ) 
		CmiAsyncSend(i,size,msg) ;
	return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}



/**********************  PE NUMBER FUNCTIONS **********************/

int CmiMainPeNum()
{
     return(0);
}


int CmiHostPeNum()
{
     return numpes;
}



/* Neighbour functions used mainly in LDB : pretend the SP1 is a hypercube */

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

int McFlushPrintfs()
{ }



/************************** MAIN ***********************************/

void CmiInitMc(argv)
char *argv[];
{
     program_name(argv[0], "SP1");   
     
     CmiLocalQueue = FIFO_Create();

     CmiSpanTreeInit();
     CmiTimerInit();
     
}

void CmiExit()
{}


void CmiDeclareArgs()
{}


main(argc, argv)
int argc;
char *argv[];
{
	int n ;


	mpc_environ(&Cmi_numpe, &Cmi_mype);
	numpes = Cmi_numpe;

	/* find dim = log2(numpes), to pretend we are a hypercube */
	for ( Cmi_dim=0,n=numpes; n>1; n/=2 )
		Cmi_dim++ ;

	user_main(argc, argv);
/*	exit(0); */
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
char *blk;
{
	return ((int *)(blk-8))[0];
}
 
void CmiFree(blk)
char *blk;
{
	free(blk-8);
}
