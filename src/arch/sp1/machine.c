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
 * Revision 2.7  1995-11-09 18:23:11  milind
 * Fixed the CmiFreeSendFn bug for messages to self.
 *
 * Revision 2.6  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.5  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/20  16:02:35  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.2  1995/09/08  02:38:26  gursoy
 * Cmi_mype Cmi_numpes CmiLocalQueue accessed thru macros now
 *
 * Revision 2.1  1995/07/17  17:46:05  knauff
 * Fixed problem with machine.c
 *
 * Revision 2.0  1995/07/10  22:12:39  knauff
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <sys/time.h>
#include "converse.h"
#include <mpproto.h>

#define MSG_TYPE 1

#define PROCESS_PID 1

#define _CK_VARSIZE_UNIT 8
#define MAXUSERARGS 20
#define MAXARGLENGTH 50

#define MAX_OUTSTANDING_MSGS	1024

#define STATIC 

/* Scanf related constants */
#define SCANFMSGLENGTH  1024
#define SCANFNVAR       16
#define SCANFBUFFER     8192

#define WORDSIZE 	sizeof(int)

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

CpvDeclare(int, Cmi_mype);
CpvDeclare(int, Cmi_numpes);
CpvDeclare(void*, CmiLocalQueue);


typedef struct msg_list {
     int msgid;
     char *msg;
     struct msg_list *next;
} MSG_LIST;

static int Cmi_dim;
static int msglength=0 ;
static int numpes ;
static double itime;

static MSG_LIST *sent_msgs=0;
static MSG_LIST *end_sent=0;


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


CmiAllAsyncMsgsSent()
{
     MSG_LIST *msg_tmp = sent_msgs;
     
     while(msg_tmp!=0)
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
     MSG_LIST *prev=0;
     MSG_LIST *temp;
     
     if(sent_msgs==0)
	  return;
     while(msg_tmp!=0)
     {
	  if(mpc_status(msg_tmp->msgid)>=0)
	  {
	       /* Release the message */
	       temp = msg_tmp->next;
	       if(prev==0)	/* first message */
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
     
     if (!CmiProbe())
	  return 0;
     msglength = CmiArrivedMsgLength();
     env = (void *)  CmiAlloc(msglength);
     if (env == 0)
     {
	  CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
	  fflush(stdout);
     }
     CmiSyncReceive(msglength, env);
     return env;
}

/*
 * mp_probe(src, type, nbytes)
 */
extern mp_probe();	/* Fortran interface. In EUI-H, not in EUI */

/* Internal functions */
CmiProbe()
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

CmiArrivedMsgLength()
{
     return msglength;
}

CmiSyncReceive(size, buffer)
     int size ;
     char *buffer ;
{
     int src, type, nbytes;
     
     src = DONTCARE;
     type = 0;
     mpc_brecv(buffer, size, &src, &type, &nbytes);
}

/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
     int destPE;
     int size;
     char * msg;
{
     mpc_bsend(msg, size, destPE, 0);
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
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
     msg_tmp->next = 0;
     if(sent_msgs==0)
	  sent_msgs = msg_tmp;
     else
	  end_sent->next = msg_tmp;
     end_sent = msg_tmp;
}

void CmiFreeSendFn(destPE, size, msg)
int destPE, size;
char *msg;
{
	if (CpvAccess(Cmi_mype)==destPE) {
		FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
	} else {
		CmiSyncSendFn(destPE, size, msg);
		CmiFree(msg);
	}
}


/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcastFn(size, msg)     /* ALL_EXCEPT_ME  */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=CpvAccess(Cmi_mype)+1; i<numpes; i++ ) 
	  CmiSyncSendFn(i, size,msg) ;
     for ( i=0; i<CpvAccess(Cmi_mype); i++ ) 
	  CmiSyncSendFn(i, size,msg) ;
}


CmiCommHandle CmiAsyncBroadcastFn(size, msg)  
int size;
char * msg;
{
	int i ;

	for ( i=CpvAccess(Cmi_mype)+1; i<numpes; i++ ) 
		CmiAsyncSendFn(i,size,msg) ;
	for ( i=0; i<CpvAccess(Cmi_mype); i++ ) 
		CmiAsyncSendFn(i,size,msg) ;
	return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastFn(size, msg)
    int size;
    char *msg;
{
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}
 
void CmiSyncBroadcastAllFn(size, msg)        /* All including me */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=0; i<numpes; i++ ) 
	  CmiSyncSendFn(i,size,msg,0) ;
}

CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)  
int size;
char * msg;
{
	int i ;

	for ( i=1; i<numpes; i++ ) 
		CmiAsyncSendFn(i,size,msg) ;
	return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastAllFn(size, msg)  /* All including me */
     int size;
     char * msg;
{
     int i ;
     
     for ( i=0; i<numpes; i++ ) 
	  CmiSyncSendFn(i,size,msg,0) ;
     CmiFree(msg) ;
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

int CmiFlushPrintfs()
{ }



/************************** MAIN ***********************************/

void CmiInitMc(argv)
char *argv[];
{
     CpvAccess(CmiLocalQueue) = FIFO_Create();

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

        CpvInitialize(int, Cmi_mype);
        CpvInitialize(int, Cmi_numpes);

	mpc_environ(&CpvAccess(Cmi_numpes), &CpvAccess(Cmi_mype));
	numpes = CpvAccess(Cmi_numpes);

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
void *blk;
{
	return ((int *)( ((char *)blk)-8))[0];
}
 
void CmiFree(blk)
void *blk;
{
	free( ((char*)blk)-8);
}
