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
 * Revision 2.19  1997-12-10 21:01:36  jyelon
 * *** empty log message ***
 *
 * Revision 2.18  1997/10/03 19:51:57  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 2.17  1997/04/25 20:48:16  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 2.16  1997/04/24 22:37:06  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 2.15  1997/03/19 04:31:45  jyelon
 * Redesigned ConverseInit
 *
 * Revision 2.14  1997/02/13 09:31:43  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 2.13  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 2.12  1995/11/13 22:51:32  gursoy
 * fixed a syntax error
 *
 * Revision 2.11  1995/11/08  23:36:13  gursoy
 * fixed varSize problem
 *
 * Revision 2.10  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.9  1995/10/18  22:23:05  jyelon
 * added MSG_TYPE = 1
 *
 * Revision 2.8  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.7  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.6  1995/09/20  16:00:16  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.5  1995/09/07  22:59:57  gursoy
 * added CpvInitialize calls for Cmi_mype etc for the sake of compleeteness
 *
 * Revision 2.4  1995/09/07  22:51:52  gursoy
 * Cmi_mype Cmi_numpes and CmiLocalQueue are accessed thru macros now
 *
 * Revision 2.3  1995/07/03  17:58:04  gursoy
 * changed charm_main to user_main
 *
 * Revision 2.2  1995/06/13  16:01:23  gursoy
 * fixed a minor syntax error
 *
 * Revision 2.1  1995/06/09  21:23:01  gursoy
 * Cpv macros moved to converse
 *
 * Revision 2.0  1995/06/08  16:39:47  gursoy
 * Reorganized directory structure
 *
 * Revision 1.3  1995/05/04  22:19:02  sanjeev
 * Mc to Cmi changes
 *
 * Revision 1.2  1994/11/22  16:56:31  sanjeev
 * Replaced main by SetupCharm
 *
 * Revision 1.1  1994/11/03  17:35:37  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include "converse.h"

#define MSG_TYPE 1

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))


CpvDeclare(int, Cmi_mype);
CpvDeclare(int,  Cmi_numpes);
CpvDeclare(void*, CmiLocalQueue);

static int Cmi_dim;

static int process, host, cflag, source, type;
static double uclockinitvalue;
extern double amicclk();




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
return ((int *)(((char *)blk)- 8))[0];
}

void CmiFree(blk)
void *blk;
{
free( ((char *)blk) - 8);
}





/**************************  TIMER FUNCTIONS **************************/

double CmiTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

double CmiWallTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

double CmiCpuTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

static void CmiTimerInit()
{
  uclockinitvalue = amicclk();
}

int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
    return 1;
}


void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}




/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
        void *env;
        int  msglength;

        type = MSG_TYPE;
        source = -1;  /* dont care */

        if ( (msglength = ntest(&source, &type)) > 0)
        {
               env = (void *)  CmiAlloc(msglength); 
               if (env == 0) 
                  CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
               CmiSyncReceive(msglength, env);
               return env;
        }
        else
		return 0;
}

void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}
 
CmiSyncReceive(size, buffer)
int size;
char *buffer;
{
    nread(buffer, size, &source, &type, &cflag);
}


/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
    char *temp;
    if (CpvAccess(Cmi_mype) == destPE)
       {
          temp = (char *)CmiAlloc(size) ;
          memcpy(temp, msg, size) ;
          FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp);
       }
    else
          nwrite(msg, size, destPE, MSG_TYPE, &cflag);
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)   /* same as sync send for ncube */
int destPE;
int size;
char * msg;
{
    nwrite(msg, size, destPE, MSG_TYPE, &cflag);
    return 0 ;
}


void CmiFreeSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
    if (CpvAccess(Cmi_mype) == destPE)
        FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    else
      {
        nwrite(msg, size, destPE, MSG_TYPE, &cflag);
        CmiFree(msg);
      }
}

/*********************** BROADCAST FUNCTIONS **********************/


void CmiSyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
	int i;

	for (i=0; i<CpvAccess(Cmi_numpes); i++)
		if (i != CpvAccess(Cmi_mype))
			nwrite(msg, size, i, MSG_TYPE, &cflag);
}


CmiCommHandle CmiAsyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
/* Same as sync broadcast for now */
	int i;

	for (i=0; i<CpvAccess(Cmi_numpes); i++)
		if (i != CpvAccess(Cmi_mype))
			nwrite(msg, size, i, MSG_TYPE, &cflag);
	return 0 ;
}

void CmiFreeBroadcastFn(size, msg)
    int size;
    char *msg;
{
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
    	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
    	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
	return 0 ;
}

void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
	CmiFree(msg) ; 
}





/**********************  LOAD BALANCER NEEDS **********************/

long CmiNumNeighbours(node)
int node;
{
    return Cmi_dim;
}


CmiGetNodeNeighbours(node, neighbours)
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



/************************** SETUP ***********************************/

void ConverseExit()
{
  ConverseCommonExit();
  exit(0);
}

void ConverseInit(argc, argv, fn, usched, initret)
int argc;
char *argv[];
CmiStartFn fn;
int usched, initret;
{
  CpvInitialize(int, Cmi_mype);
  CpvInitialize(int, Cmi_numpes);
  CpvInitialize(void*, CmiLocalQueue);
  whoami(&CpvAccess(Cmi_mype), &process, &host, &Cmi_dim);
  CpvAccess(Cmi_numpes) = (1 << Cmi_dim) ;
  CpvAccess(CmiLocalQueue)= (void *) FIFO_Create();
  CmiSpanTreeInit();
  CmiTimerInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  if (initret==0) {
    fn(argc, argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}



